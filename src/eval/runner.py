import asyncio
import hashlib
import inspect
import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TimeElapsedColumn

from .evaluator import EvaluationSuite
from .extractor import prepare_response
from .record_mode import resolve_record_mode, uses_construction_prompt
from .scoring import (
    EVALUATION_SCHEMA_VERSION,
    TOTAL_MAX_SCORE,
    normalize_answer_score,
    normalize_construction_score,
    normalize_proof_score,
)
from src.llm.prompts import build_response_prompt
from src.llm.token_usage import normalize_token_usage, sum_normalized_token_usage, sum_pre_normalized_token_usage


class EmptyGenerationResponseError(RuntimeError):
    def __init__(self, message: str, details: Dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


EMPTY_GENERATION_TOKEN_LIMIT_THRESHOLD = 155_000


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        yield json.loads(line)


def _read_jsonl_with_line_numbers(path: Path) -> Iterable[tuple[int, Dict[str, Any]]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        yield line_number, json.loads(line)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_prompt(record: Dict[str, Any]) -> str:
    mode = resolve_record_mode(record, warn=False)
    return build_response_prompt(
        query=str(record.get("query", "")),
        instruction=str(record.get("instruction", "")),
        mode=mode,
    )


class EvaluationRunner:
    def __init__(
        self,
        dataset_path: Path,
        output_root: Path,
        line_numbers: List[int] | None,
        generation_mode: str,
        generation_profile: str,
        evaluation_profile: str,
        response_model: Any | None,
        evaluation_suite: EvaluationSuite | None,
        timeout: int,
        semaphore: int,
        judge_answer: bool = False,
        repeat: int = 1,
        force_attempt_ids: Iterable[str] | None = None,
    ) -> None:
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got: {repeat}")
        self.dataset_path = dataset_path
        self.output_root = output_root
        self.line_numbers = line_numbers or []
        self.generation_mode = generation_mode
        self.generation_profile = generation_profile
        self.evaluation_profile = evaluation_profile
        self.response_model = response_model
        self.evaluation_suite = evaluation_suite
        self.timeout = timeout
        self.semaphore = semaphore
        self.judge_answer = judge_answer
        self.repeat = repeat
        self.force_attempt_ids = set(force_attempt_ids or [])

    def run(
        self,
        generate: bool,
        evaluate: bool,
        verbose: str = "none",
        use_generation_cache: bool = True,
        use_evaluation_cache: bool = True,
    ) -> Path:
        dataset_name = self.dataset_path.stem
        if self.line_numbers:
            dataset_name = f"{dataset_name}__lines_{_line_numbers_slug(self.line_numbers)}"
        run_dir = self.output_root / dataset_name / self.generation_mode / self.generation_profile
        _ensure_dir(run_dir)

        generation_dir = run_dir / "generation"
        evaluation_dir = run_dir / "evaluation" / self.evaluation_profile
        _ensure_dir(generation_dir)
        _ensure_dir(evaluation_dir)

        responses_path = generation_dir / "cache.jsonl"
        generation_errors_path = generation_dir / "errors.jsonl"
        generation_meta_path = generation_dir / "run.json"
        responses_dir = generation_dir / "cases"

        results_path = evaluation_dir / "cache.jsonl"
        evaluation_errors_path = evaluation_dir / "errors.jsonl"
        evaluation_meta_path = evaluation_dir / "run.json"
        evaluation_usage_path = evaluation_dir / "usage.json"
        eval_cases_dir = evaluation_dir / "cases"
        results_jsonl_path = evaluation_dir / "results.jsonl"
        results_by_problem_path = evaluation_dir / "results_by_problem.jsonl"

        base_records = self._load_records()
        records = _expand_records(base_records, self.repeat)
        self._validate_force_attempt_ids(records)
        if self.force_attempt_ids and not generate:
            raise ValueError("--force-attempt requires generation so the selected attempts can be regenerated")
        responses: List[str] | None = None
        expected_eval_meta = self._build_evaluation_meta(records)

        if generate and evaluate:
            eval_results = asyncio.run(
                self._run_generate_evaluate_pipeline(
                    records=records,
                    responses_path=responses_path,
                    generation_errors_path=generation_errors_path,
                    generation_meta_path=generation_meta_path,
                    responses_dir=responses_dir,
                    results_path=results_path,
                    evaluation_errors_path=evaluation_errors_path,
                    evaluation_meta_path=evaluation_meta_path,
                    eval_cases_dir=eval_cases_dir,
                    results_jsonl_path=results_jsonl_path,
                    results_by_problem_path=results_by_problem_path,
                    evaluation_usage_path=evaluation_usage_path,
                    expected_eval_meta=expected_eval_meta,
                    use_generation_cache=use_generation_cache,
                    use_evaluation_cache=use_evaluation_cache,
                )
            )
            if verbose in {"summary", "detail"}:
                self._print_eval_results(records, eval_results, verbose)
            return run_dir

        if generate:
            responses = (
                self._load_generation_cache(records, responses_path)
                if use_generation_cache and not self.force_attempt_ids
                else None
            )
            if responses is None:
                if self.response_model is None:
                    raise ValueError("response model is required for generation")
                _ensure_dir(responses_dir)
                cached_responses = (
                    self._load_generation_cache_map(records, responses_path)
                    if use_generation_cache
                    else {}
                )
                cached_responses = self._without_forced_cache(cached_responses)
                if not use_generation_cache and responses_path.exists():
                    responses_path.unlink()
                responses = asyncio.run(
                    self._generate_responses(
                        records=records,
                        responses_path=responses_path,
                        errors_path=generation_errors_path,
                        responses_dir=responses_dir,
                        cached_responses=cached_responses,
                    )
                )
                generation_meta = {
                    "dataset": str(self.dataset_path),
                    "generation_profile": self.generation_profile,
                    "record_count": len(records),
                    "base_record_count": _base_record_count(records),
                    "repeat": self.repeat,
                    "prompt_modes": _collect_prompt_modes(records),
                    "usage": _sum_usage(self._load_generation_usage_cache_map(records, responses_path).values()),
                    "usage_normalized": sum_normalized_token_usage(
                        self._load_generation_usage_cache_map(records, responses_path).values()
                    ),
                }
                if self.line_numbers:
                    generation_meta["line_numbers"] = self.line_numbers
                generation_meta_path.write_text(
                    json.dumps(generation_meta, ensure_ascii=False, indent=2), encoding="utf-8"
                )

        eval_results: List[Dict[str, Any]] | None = None
        if evaluate:
            eval_results = (
                self._load_evaluation_cache(records, results_path, evaluation_meta_path, expected_eval_meta)
                if use_evaluation_cache and not self.force_attempt_ids
                else None
            )
            if eval_results is None:
                if responses is None:
                    responses = self._load_generation_cache(records, responses_path)
                if responses is None:
                    raise ValueError("missing generation cache; run with --generate first")
                if self.evaluation_suite is None:
                    raise ValueError("evaluation suite is required for evaluation")
                eval_results = self.evaluation_suite.check(records, responses)
                _ensure_dir(eval_cases_dir)
                self._write_evaluation_cache(records, eval_results, results_path, eval_cases_dir)
                evaluation_meta_path.write_text(
                    json.dumps(expected_eval_meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            if responses is None:
                responses = self._load_generation_cache(records, responses_path)
            if responses is not None:
                with results_jsonl_path.open("w", encoding="utf-8") as f:
                    for record, response, eval_result in zip(records, responses, eval_results, strict=True):
                        result = self._eval_record(record, response, eval_result)
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                self._write_results_by_problem(records, eval_results, results_by_problem_path)
                self._write_evaluation_usage(eval_results, evaluation_usage_path)

            if verbose in {"summary", "detail"}:
                self._print_eval_results(records, eval_results, verbose)

        return run_dir

    def _validate_force_attempt_ids(self, records: List[Dict[str, Any]]) -> None:
        if not self.force_attempt_ids:
            return
        available = {_record_cache_key(record) for record in records}
        missing = sorted(self.force_attempt_ids - available)
        if missing:
            raise ValueError(f"unknown --force-attempt id(s): {missing}")

    def _without_forced_cache(self, cached_by_id: Dict[Any, Any]) -> Dict[Any, Any]:
        if not self.force_attempt_ids:
            return cached_by_id
        return {
            item_id: value
            for item_id, value in cached_by_id.items()
            if item_id not in self.force_attempt_ids
        }

    def _build_evaluation_meta(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt_modes = _collect_prompt_modes(records)
        evaluators = ["proof", "construction"]
        if self.judge_answer:
            evaluators.insert(0, "answer")

        meta = {
            "dataset": str(self.dataset_path),
            "evaluation_profile": self.evaluation_profile,
            "record_count": len(records),
            "base_record_count": _base_record_count(records),
            "repeat": self.repeat,
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "judge_answer": self.judge_answer,
            "evaluators": evaluators,
            "prompt_modes": prompt_modes,
            "prompt_variants": {
                "response": {
                    "construction": "q1_q2_construct_v2",
                    "plain": "single_solution_v1",
                },
                "answer": "appendix_a5",
                "proof": "appendix_b5",
            },
            "verifier_executor": "prime_code",
            "timeout": self.timeout,
        }
        if self.line_numbers:
            meta["line_numbers"] = self.line_numbers
        return meta

    def _load_records(self) -> List[Dict[str, Any]]:
        if not self.line_numbers:
            records = list(_read_jsonl(self.dataset_path))
        else:
            selected_by_line: Dict[int, Dict[str, Any]] = {}
            wanted = set(self.line_numbers)
            for line_number, record in _read_jsonl_with_line_numbers(self.dataset_path):
                if line_number in wanted:
                    selected_by_line[line_number] = record

            missing = [line for line in self.line_numbers if line not in selected_by_line]
            if missing:
                raise ValueError(f"dataset line(s) not found: {missing}; dataset={self.dataset_path}")
            records = [selected_by_line[line] for line in self.line_numbers]

        for record in records:
            resolve_record_mode(record, warn=True)
        return records

    def _eval_record(self, record: Dict[str, Any], response: str, eval_result: Dict[str, Any]) -> Dict[str, Any]:
        mode = resolve_record_mode(record, warn=False)
        prepared = prepare_response(response, mode)
        return {
            "id": record.get("id"),
            "attempt_id": record.get("_attempt_id"),
            "run_index": record.get("_run_index"),
            "mode": mode,
            "response": response,
            "solution_text": prepared.solution_text,
            "answer_text": prepared.solution_text,
            "construction": prepared.construction,
            "answer": eval_result.get("answer"),
            "proof": eval_result.get("proof"),
            "construction_eval": eval_result.get("construction"),
            "total_score": eval_result.get("total_score"),
            "max_score": eval_result.get("max_score", TOTAL_MAX_SCORE),
        }

    def _load_generation_cache(
        self, records: List[Dict[str, Any]], responses_path: Path
    ) -> List[str] | None:
        cached_by_id = self._load_generation_cache_map(records, responses_path)
        record_ids = [_record_cache_key(r) for r in records]
        if not all(record_id in cached_by_id for record_id in record_ids):
            return None
        return [cached_by_id.get(record_id, "") for record_id in record_ids]

    def _load_generation_cache_map(
        self, records: List[Dict[str, Any]], responses_path: Path
    ) -> Dict[Any, str]:
        if not responses_path.exists():
            return {}
        cached = list(_read_jsonl(responses_path))
        record_ids = [_record_cache_key(r) for r in records]
        wanted_ids = set(record_ids)
        cached_by_id: Dict[Any, str] = {}
        for item in cached:
            item_id = _cache_item_key(item)
            response = item.get("response", "")
            if item_id in wanted_ids and _is_nonempty_response(response):
                cached_by_id[item_id] = response
        return cached_by_id

    def _load_generation_usage_cache_map(
        self, records: List[Dict[str, Any]], responses_path: Path
    ) -> Dict[Any, Dict[str, Any] | None]:
        if not responses_path.exists():
            return {}
        cached = list(_read_jsonl(responses_path))
        record_ids = [_record_cache_key(r) for r in records]
        wanted_ids = set(record_ids)
        usage_by_id: Dict[Any, Dict[str, Any] | None] = {}
        for item in cached:
            item_id = _cache_item_key(item)
            if item_id in wanted_ids and _is_nonempty_response(item.get("response", "")):
                usage = item.get("usage")
                usage_by_id[item_id] = usage if isinstance(usage, dict) else None
        return usage_by_id

    def _write_generation_cache(
        self,
        records: List[Dict[str, Any]],
        responses: List[str],
        responses_path: Path,
        responses_dir: Path,
    ) -> None:
        with responses_path.open("w", encoding="utf-8") as f:
            for record, response in zip(records, responses, strict=True):
                item = _cache_record_payload(record)
                item["response"] = response
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                cache_key = _record_cache_key(record)
                if cache_key is not None:
                    response_path = responses_dir / f"{_safe_id(str(cache_key))}.md"
                    response_path.write_text(response, encoding="utf-8")

    def _load_evaluation_cache(
        self,
        records: List[Dict[str, Any]],
        results_path: Path,
        evaluation_meta_path: Path,
        expected_meta: Dict[str, Any],
    ) -> List[Dict[str, Any]] | None:
        cached_by_id = self._load_evaluation_cache_map(records, results_path, evaluation_meta_path, expected_meta)
        record_ids = [_record_cache_key(r) for r in records]
        if not all(record_id in cached_by_id for record_id in record_ids):
            return None
        return [cached_by_id.get(record_id, {}) for record_id in record_ids]

    def _load_evaluation_cache_map(
        self,
        records: List[Dict[str, Any]],
        results_path: Path,
        evaluation_meta_path: Path,
        expected_meta: Dict[str, Any],
    ) -> Dict[Any, Dict[str, Any]]:
        if not results_path.exists() or not evaluation_meta_path.exists():
            return {}
        meta = json.loads(evaluation_meta_path.read_text(encoding="utf-8"))
        if _cache_validation_meta(meta) != _cache_validation_meta(expected_meta):
            return {}

        cached = list(_read_jsonl(results_path))
        record_ids = [_record_cache_key(r) for r in records]
        wanted_ids = set(record_ids)
        cached_by_id: Dict[Any, Dict[str, Any]] = {}
        for item in cached:
            item_id = _cache_item_key(item)
            eval_result = item.get("eval_result", {})
            if item_id in wanted_ids and eval_result.get("schema_version") == EVALUATION_SCHEMA_VERSION:
                cached_by_id[item_id] = eval_result
        return cached_by_id

    def _write_evaluation_cache(
        self,
        records: List[Dict[str, Any]],
        eval_results: List[Dict[str, Any]],
        results_path: Path,
        cases_dir: Path,
    ) -> None:
        with results_path.open("w", encoding="utf-8") as f:
            for record, eval_result in zip(records, eval_results, strict=True):
                item = _cache_record_payload(record)
                item["eval_result"] = eval_result
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                cache_key = _record_cache_key(record)
                if cache_key is not None:
                    eval_path = cases_dir / f"{_safe_id(str(cache_key))}.json"
                    eval_path.write_text(
                        json.dumps(eval_result, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

    def _append_evaluation_cache_record(
        self,
        record: Dict[str, Any],
        eval_result: Dict[str, Any],
        results_path: Path,
        cases_dir: Path,
    ) -> None:
        cache_key = _record_cache_key(record)
        with results_path.open("a", encoding="utf-8") as f:
            item = _cache_record_payload(record)
            item["eval_result"] = eval_result
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        if cache_key is not None:
            eval_path = cases_dir / f"{_safe_id(str(cache_key))}.json"
            eval_path.write_text(
                json.dumps(eval_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _print_eval_results(
        self, records: List[Dict[str, Any]], eval_results: List[Dict[str, Any]], verbose: str
    ) -> None:
        if verbose == "detail":
            for record, result in zip(records, eval_results, strict=True):
                rid = record.get("id")
                answer = _format_component_summary(result.get("answer"))
                proof = _format_component_summary(result.get("proof"))
                construction = _format_component_summary(result.get("construction"))
                total_score = result.get("total_score")
                total_label = f"{total_score:.4f}" if total_score is not None else "ERR"
                print(
                    f"id={rid} answer={answer} proof={proof} construction={construction} total={total_label}"
                )
            return

        total_scores = [result.get("total_score") for result in eval_results if result.get("total_score") is not None]
        average_total = (sum(total_scores) / len(total_scores)) if total_scores else 0.0
        proof_average = _component_average(eval_results, "proof")
        construction_average = _component_average(
            eval_results,
            "construction",
            allowed_statuses={"scored"},
        )
        print(f"records={len(eval_results)} average_total={average_total:.4f}")
        print(f"proof_average={proof_average:.4f}")
        print(f"construction_average={construction_average:.4f}")
        if self.judge_answer:
            answer_average = _component_average(eval_results, "answer")
            print(f"answer_average={answer_average:.4f}")

    async def _generate_responses(
        self,
        records: List[Dict[str, Any]],
        responses_path: Path,
        errors_path: Path,
        responses_dir: Path,
        cached_responses: Dict[Any, str] | None = None,
    ) -> List[str]:
        semaphore = asyncio.Semaphore(self.semaphore)
        write_lock = asyncio.Lock()
        cached_responses = cached_responses or {}
        responses: List[str] = [""] * len(records)
        missing: List[tuple[int, Dict[str, Any]]] = []
        for idx, record in enumerate(records):
            record_id = _record_cache_key(record)
            if record_id in cached_responses:
                responses[idx] = cached_responses[record_id]
            else:
                missing.append((idx, record))

        progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
        with progress:
            task_id = progress.add_task("responses", total=len(records))
            progress.update(task_id, completed=len(records) - len(missing))

            async def _one(idx: int, record: Dict[str, Any]) -> None:
                try:
                    prompt = _build_prompt(record)
                    async with semaphore:
                        generated = await _call_generate(self.response_model, prompt, record)
                    response, usage, debug = _split_generation_result(generated)
                    _validate_generation_response(response, debug)
                    responses[idx] = response
                    async with write_lock:
                        self._append_generation_cache_record(record, response, usage, responses_path, responses_dir)
                except Exception as exc:
                    async with write_lock:
                        self._append_error_record(record, "generation", exc, errors_path)
                finally:
                    progress.update(task_id, advance=1)

            await asyncio.gather(*[_one(i, record) for i, record in missing])

        return responses

    async def _run_generate_evaluate_pipeline(
        self,
        records: List[Dict[str, Any]],
        responses_path: Path,
        generation_errors_path: Path,
        generation_meta_path: Path,
        responses_dir: Path,
        results_path: Path,
        evaluation_errors_path: Path,
        evaluation_meta_path: Path,
        eval_cases_dir: Path,
        results_jsonl_path: Path,
        results_by_problem_path: Path,
        evaluation_usage_path: Path,
        expected_eval_meta: Dict[str, Any],
        use_generation_cache: bool,
        use_evaluation_cache: bool,
    ) -> List[Dict[str, Any]]:
        if self.response_model is None:
            raise ValueError("response model is required for generation")
        if self.evaluation_suite is None:
            raise ValueError("evaluation suite is required for evaluation")

        _ensure_dir(responses_dir)
        _ensure_dir(eval_cases_dir)
        if not use_generation_cache and responses_path.exists():
            responses_path.unlink()
        if not use_generation_cache and generation_errors_path.exists():
            generation_errors_path.unlink()
        if not use_evaluation_cache and results_path.exists():
            results_path.unlink()
        if not use_evaluation_cache and evaluation_errors_path.exists():
            evaluation_errors_path.unlink()

        cached_responses = (
            self._load_generation_cache_map(records, responses_path)
            if use_generation_cache
            else {}
        )
        cached_responses = self._without_forced_cache(cached_responses)
        cached_eval_results = (
            self._load_evaluation_cache_map(records, results_path, evaluation_meta_path, expected_eval_meta)
            if use_evaluation_cache
            else {}
        )
        cached_eval_results = self._without_forced_cache(cached_eval_results)
        evaluation_meta_path.write_text(
            json.dumps(expected_eval_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        generate_sem = asyncio.Semaphore(self.semaphore)
        generation_write_lock = asyncio.Lock()
        evaluation_write_lock = asyncio.Lock()
        eval_llm_sem = asyncio.Semaphore(self.evaluation_suite.semaphore)
        verify_gate = threading.Semaphore(self.evaluation_suite.semaphore)
        responses: List[str] = [""] * len(records)
        eval_results: List[Dict[str, Any] | None] = [None] * len(records)
        pending: List[tuple[int, Dict[str, Any]]] = []

        for idx, record in enumerate(records):
            record_id = _record_cache_key(record)
            if record_id in cached_responses:
                responses[idx] = cached_responses[record_id]
            if responses[idx] and record_id in cached_eval_results:
                eval_results[idx] = cached_eval_results[record_id]
            if not responses[idx] or eval_results[idx] is None:
                pending.append((idx, record))

        progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
        with progress:
            task_id = progress.add_task("pipeline", total=len(records))
            progress.update(task_id, completed=len(records) - len(pending))

            async def _one(idx: int, record: Dict[str, Any]) -> None:
                response = responses[idx]
                try:
                    if not response:
                        prompt = _build_prompt(record)
                        async with generate_sem:
                            generated = await _call_generate(self.response_model, prompt, record)
                        response, usage, debug = _split_generation_result(generated)
                        _validate_generation_response(response, debug)
                        responses[idx] = response
                        async with generation_write_lock:
                            self._append_generation_cache_record(record, response, usage, responses_path, responses_dir)

                    if eval_results[idx] is None:
                        try:
                            eval_result = await self.evaluation_suite.check_one_async(
                                record,
                                response,
                                llm_sem=eval_llm_sem,
                                verify_gate=verify_gate,
                            )
                        except Exception as exc:
                            eval_result = _build_attempt_error_result(record, "evaluation", exc, self.judge_answer)
                            async with evaluation_write_lock:
                                self._append_error_record(record, "evaluation", exc, evaluation_errors_path)
                        else:
                            async with evaluation_write_lock:
                                self._append_evaluation_cache_record(record, eval_result, results_path, eval_cases_dir)
                        eval_results[idx] = eval_result
                except Exception as exc:
                    eval_results[idx] = _build_attempt_error_result(record, "generation", exc, self.judge_answer)
                    async with generation_write_lock:
                        self._append_error_record(record, "generation", exc, generation_errors_path)
                finally:
                    progress.update(task_id, advance=1)

            await asyncio.gather(*[_one(idx, record) for idx, record in pending])

        final_eval_results = [result or {} for result in eval_results]
        self._write_generation_meta(records, generation_meta_path, responses_path)
        self._write_results_jsonl(records, responses, final_eval_results, results_jsonl_path)
        self._write_results_by_problem(records, final_eval_results, results_by_problem_path)
        self._write_evaluation_usage(final_eval_results, evaluation_usage_path)
        return final_eval_results

    def _write_generation_meta(
        self,
        records: List[Dict[str, Any]],
        generation_meta_path: Path,
        responses_path: Path,
    ) -> None:
        generation_meta = {
            "dataset": str(self.dataset_path),
            "generation_profile": self.generation_profile,
            "record_count": len(records),
            "base_record_count": _base_record_count(records),
            "repeat": self.repeat,
            "prompt_modes": _collect_prompt_modes(records),
            "usage": _sum_usage(self._load_generation_usage_cache_map(records, responses_path).values()),
            "usage_normalized": sum_normalized_token_usage(
                self._load_generation_usage_cache_map(records, responses_path).values()
            ),
        }
        if self.line_numbers:
            generation_meta["line_numbers"] = self.line_numbers
        generation_meta_path.write_text(
            json.dumps(generation_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_results_jsonl(
        self,
        records: List[Dict[str, Any]],
        responses: List[str],
        eval_results: List[Dict[str, Any]],
        results_jsonl_path: Path,
    ) -> None:
        with results_jsonl_path.open("w", encoding="utf-8") as f:
            for record, response, eval_result in zip(records, responses, eval_results, strict=True):
                result = self._eval_record(record, response, eval_result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def _write_results_by_problem(
        self,
        records: List[Dict[str, Any]],
        eval_results: List[Dict[str, Any]],
        results_by_problem_path: Path,
    ) -> None:
        grouped: Dict[Any, List[Dict[str, Any]]] = {}
        for record, eval_result in zip(records, eval_results, strict=True):
            grouped.setdefault(record.get("id"), []).append(eval_result)

        with results_by_problem_path.open("w", encoding="utf-8") as f:
            for record_id, group in grouped.items():
                item = {
                    "id": record_id,
                    "attempts": len(group),
                    "average_total": _average_component(group, "total_score"),
                    "proof_average": _average_nested_component(group, "proof"),
                    "construction_average": _average_nested_component(group, "construction"),
                    "answer_average": _average_nested_component(group, "answer"),
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _write_evaluation_usage(
        self,
        eval_results: List[Dict[str, Any]],
        evaluation_usage_path: Path,
    ) -> None:
        proof_usages = _component_normalized_usages(eval_results, "proof")
        answer_usages = _component_normalized_usages(eval_results, "answer")
        payload = {
            "proof_usage_normalized": sum_pre_normalized_token_usage(proof_usages),
            "answer_usage_normalized": sum_pre_normalized_token_usage(answer_usages),
            "evaluation_usage_normalized": sum_pre_normalized_token_usage(proof_usages + answer_usages),
        }
        evaluation_usage_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _append_generation_cache_record(
        self,
        record: Dict[str, Any],
        response: str,
        usage: Dict[str, Any] | None,
        responses_path: Path,
        responses_dir: Path,
    ) -> None:
        cache_key = _record_cache_key(record)
        with responses_path.open("a", encoding="utf-8") as f:
            item = _cache_record_payload(record)
            item.update({
                "response": response,
                "usage": usage,
                "usage_normalized": normalize_token_usage(usage),
            })
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        if cache_key is not None:
            response_path = responses_dir / f"{_safe_id(str(cache_key))}.md"
            response_path.write_text(response, encoding="utf-8")

    def _append_error_record(
        self,
        record: Dict[str, Any],
        stage: str,
        exc: Exception,
        errors_path: Path,
    ) -> None:
        item = _cache_record_payload(record)
        item.update(
            {
                "stage": stage,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "retryable": True,
            }
        )
        details = getattr(exc, "details", None)
        if isinstance(details, dict) and details:
            item["details"] = details
        with errors_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


async def _call_generate(model: Any, prompt: str, record: Dict[str, Any]) -> Any:
    if inspect.iscoroutinefunction(model.generate):
        return await model.generate(prompt=prompt, record=record)
    return await asyncio.to_thread(model.generate, prompt=prompt, record=record)


def _split_generation_result(result: Any) -> tuple[str, Dict[str, Any] | None, Dict[str, Any] | None]:
    if isinstance(result, str):
        return result, None, None
    if isinstance(result, dict):
        content = result.get("content", result.get("response", ""))
        usage = result.get("usage")
        debug = result.get("stream_debug") or result.get("debug")
        return (
            str(content),
            usage if isinstance(usage, dict) else None,
            debug if isinstance(debug, dict) else None,
        )
    content = getattr(result, "content", None)
    if content is not None:
        usage = getattr(result, "usage", None)
        debug = getattr(result, "stream_debug", None) or getattr(result, "debug", None)
        return (
            str(content),
            usage if isinstance(usage, dict) else None,
            debug if isinstance(debug, dict) else None,
        )
    return str(result), None, None


def _validate_generation_response(response: str, debug: Dict[str, Any] | None = None) -> None:
    if not _is_nonempty_response(response):
        raise EmptyGenerationResponseError("empty generation response", debug)


def _is_nonempty_response(response: Any) -> bool:
    return isinstance(response, str) and bool(response.strip())


def _build_attempt_error_result(
    record: Dict[str, Any],
    stage: str,
    exc: Exception,
    judge_answer: bool,
) -> Dict[str, Any]:
    if (
        stage == "generation"
        and isinstance(exc, EmptyGenerationResponseError)
        and _should_score_empty_generation_zero(exc.details)
    ):
        return _build_empty_generation_score_result(record, exc, judge_answer)

    mode = resolve_record_mode(record, warn=False)
    error = {
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "retryable": True,
    }
    details = getattr(exc, "details", None)
    if isinstance(details, dict) and details:
        error["details"] = details
    skipped = {
        "status": "skipped",
        "raw_score": None,
        "normalized_score": None,
        "raw": None,
        "reason": f"{stage} failed",
        "details": {},
    }
    failed = {
        "status": "error",
        "raw_score": None,
        "normalized_score": None,
        "raw": None,
        "reason": str(exc),
        "details": error,
    }
    construction = skipped
    if uses_construction_prompt(mode):
        construction = failed if stage == "evaluation" else skipped
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "judge_answer": judge_answer,
        "mode": mode,
        "answer": failed if judge_answer and stage == "evaluation" else skipped,
        "proof": failed,
        "construction": construction,
        "prepared_response": {
            "mode": mode,
            "solution_text": None,
            "solution_error": f"{stage} failed: {exc}",
            "construction": None,
            "construction_error": f"{stage} failed: {exc}" if uses_construction_prompt(mode) else None,
        },
        "total_score": None,
        "max_score": TOTAL_MAX_SCORE,
        "error": error,
    }


def _should_score_empty_generation_zero(details: Dict[str, Any] | None) -> bool:
    if not isinstance(details, dict) or not details:
        return True

    finish_reason = details.get("finish_reason")
    if finish_reason == "length":
        return True

    completion_tokens = _usage_int_value(details.get("usage"), "completion_tokens")
    if completion_tokens is not None and completion_tokens >= EMPTY_GENERATION_TOKEN_LIMIT_THRESHOLD:
        return True

    if finish_reason is None and details.get("saw_reasoning_content"):
        return False

    return True


def _usage_int_value(usage: Any, key: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _build_empty_generation_score_result(
    record: Dict[str, Any],
    exc: EmptyGenerationResponseError,
    judge_answer: bool,
) -> Dict[str, Any]:
    mode = resolve_record_mode(record, warn=False)
    error = {
        "stage": "generation",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "retryable": True,
    }
    details = getattr(exc, "details", None)
    if isinstance(details, dict) and details:
        error["details"] = details

    skipped = {
        "status": "skipped",
        "raw_score": None,
        "normalized_score": None,
        "raw": None,
        "reason": "not applicable",
        "details": {},
    }
    proof = {
        "status": "scored",
        "raw_score": 0,
        "normalized_score": normalize_proof_score(0),
        "raw": None,
        "reason": str(exc),
        "details": error,
    }
    construction = skipped
    if uses_construction_prompt(mode):
        construction = {
            "status": "scored",
            "raw_score": 0,
            "normalized_score": normalize_construction_score(0),
            "raw": None,
            "reason": str(exc),
            "details": error,
        }
    answer = skipped
    if judge_answer:
        answer = {
            "status": "scored",
            "raw_score": 0,
            "normalized_score": normalize_answer_score(0),
            "raw": None,
            "reason": str(exc),
            "details": error,
        }

    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "judge_answer": judge_answer,
        "mode": mode,
        "answer": answer,
        "proof": proof,
        "construction": construction,
        "prepared_response": {
            "mode": mode,
            "solution_text": None,
            "solution_error": f"generation failed: {exc}",
            "construction": None,
            "construction_error": f"generation failed: {exc}" if uses_construction_prompt(mode) else None,
        },
        "total_score": 0.0,
        "max_score": TOTAL_MAX_SCORE,
        "error": error,
    }


def _sum_usage(usages: Iterable[Dict[str, Any] | None]) -> Dict[str, Any]:
    totals: Dict[str, Any] = {}
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0) + value
    return totals


def _expand_records(records: List[Dict[str, Any]], repeat: int) -> List[Dict[str, Any]]:
    if repeat == 1:
        return records
    expanded: List[Dict[str, Any]] = []
    for record in records:
        record_id = record.get("id")
        for run_index in range(1, repeat + 1):
            item = dict(record)
            item["_run_index"] = run_index
            item["_attempt_id"] = f"{record_id}__run_{run_index}"
            expanded.append(item)
    return expanded


def _record_cache_key(record: Dict[str, Any]) -> Any:
    return record.get("_attempt_id") or record.get("id")


def _cache_item_key(item: Dict[str, Any]) -> Any:
    return item.get("attempt_id") or item.get("id")


def _cache_record_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"id": record.get("id")}
    if record.get("_attempt_id") is not None:
        payload["attempt_id"] = record.get("_attempt_id")
        payload["run_index"] = record.get("_run_index")
    return payload


def _cache_validation_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    comparable = dict(meta)
    comparable.pop("semaphore", None)
    return comparable


def _base_record_count(records: List[Dict[str, Any]]) -> int:
    return len({record.get("id") for record in records})


def _average_component(eval_results: List[Dict[str, Any]], key: str) -> float | None:
    values = [result.get(key) for result in eval_results if isinstance(result.get(key), (int, float))]
    return (sum(values) / len(values)) if values else None


def _average_nested_component(eval_results: List[Dict[str, Any]], key: str) -> float | None:
    values: List[float] = []
    for result in eval_results:
        component = result.get(key)
        if not isinstance(component, dict):
            continue
        value = component.get("normalized_score")
        if isinstance(value, (int, float)):
            values.append(value)
    return (sum(values) / len(values)) if values else None


def _component_normalized_usages(eval_results: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    usages: List[Dict[str, Any]] = []
    for result in eval_results:
        component = result.get(key)
        if not isinstance(component, dict):
            continue
        details = component.get("details")
        if not isinstance(details, dict):
            continue
        usage = details.get("usage_normalized")
        if isinstance(usage, dict):
            usages.append(usage)
    return usages


def _collect_prompt_modes(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {"construction": 0, "plain": 0, "inconsistent": 0}
    for record in records:
        mode = resolve_record_mode(record, warn=False)
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def _format_component_summary(component: Dict[str, Any] | None) -> str:
    if not component:
        return "ERR"
    status = component.get("status")
    if status == "skipped":
        return "SKIP"
    if status != "scored" or component.get("normalized_score") is None:
        return "ERR"
    raw_score = component.get("raw_score")
    normalized_score = component.get("normalized_score")
    return f"{normalized_score:.4f}(raw={raw_score})"


def _component_average(
    eval_results: List[Dict[str, Any]],
    name: str,
    allowed_statuses: set[str] | None = None,
) -> float:
    scores: List[float] = []
    for result in eval_results:
        component = result.get(name, {})
        status = component.get("status")
        normalized_score = component.get("normalized_score")
        if allowed_statuses is not None and status not in allowed_statuses:
            continue
        if normalized_score is not None:
            scores.append(normalized_score)
    return (sum(scores) / len(scores)) if scores else 0.0


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def _line_numbers_slug(line_numbers: List[int]) -> str:
    raw = "_".join(str(n) for n in line_numbers)
    if len(raw) <= 96:
        return raw
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{line_numbers[0]}_{line_numbers[-1]}_{len(line_numbers)}_{digest}"
