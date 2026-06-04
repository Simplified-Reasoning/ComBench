import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, List

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TimeElapsedColumn

from src.eval.construction_text import make_python_literal_stdin
from src.eval.extractor import PreparedResponse, prepare_response
from src.eval.prime_code_executor import run_prime_code_verifier
from src.eval.record_mode import resolve_record_mode, uses_construction_prompt
from src.eval.score_parsing import parse_answer_score, parse_proof_score
from src.eval.scoring import (
    EVALUATION_SCHEMA_VERSION,
    TOTAL_MAX_SCORE,
    normalize_answer_score,
    normalize_construction_score,
    normalize_proof_score,
)
from src.llm.client import LlmClient
from src.llm.prompts import build_answer_eval_messages, build_proof_eval_messages
from src.llm.token_usage import normalize_token_usage
from src.models.llm.model import build_llm_client
from src.models.profile import load_profile


def _build_component_result(
    *,
    status: str,
    raw_score: int | None,
    normalized_score: float | None,
    raw: str | None = None,
    reason: str | None = None,
    details: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "raw_score": raw_score,
        "normalized_score": normalized_score,
        "raw": raw,
        "reason": reason,
        "details": details or {},
    }


def _usage_details(raw: Dict[str, Any]) -> Dict[str, Any]:
    usage = raw.get("usage")
    return {
        "usage": usage,
        "usage_normalized": normalize_token_usage(usage if isinstance(usage, dict) else None),
    }


class AnswerEvaluator:
    def __init__(self, client: LlmClient, enabled: bool) -> None:
        self.client = client
        self.enabled = enabled

    async def evaluate(
        self, data: Dict[str, Any], prepared: PreparedResponse, llm_sem: asyncio.Semaphore
    ) -> Dict[str, Any]:
        if not self.enabled:
            return _build_component_result(status="skipped", raw_score=None, normalized_score=None, reason="answer judge disabled")
        if prepared.solution_error:
            return _build_component_result(
                status="error",
                raw_score=None,
                normalized_score=None,
                reason=prepared.solution_error,
            )
        try:
            messages = build_answer_eval_messages(
                query=str(data.get("query", "")),
                student_answer=prepared.solution_text or "",
                ref_answer=str(data.get("ref_answer", "")),
            )
        except ValueError as exc:
            return _build_component_result(status="error", raw_score=None, normalized_score=None, reason=str(exc))

        async with llm_sem:
            raw = await self.client.chat(messages)

        content = raw["choices"][0]["message"]["content"]
        score = parse_answer_score(content)
        if score is None:
            return _build_component_result(
                status="error",
                raw_score=None,
                normalized_score=None,
                raw=content,
                reason="invalid answer score output",
                details=_usage_details(raw),
            )
        return _build_component_result(
            status="scored",
            raw_score=score,
            normalized_score=normalize_answer_score(score),
            raw=content,
            details=_usage_details(raw),
        )


class ProofEvaluator:
    def __init__(self, client: LlmClient) -> None:
        self.client = client

    async def evaluate(
        self, data: Dict[str, Any], prepared: PreparedResponse, llm_sem: asyncio.Semaphore
    ) -> Dict[str, Any]:
        if prepared.solution_error:
            return _build_component_result(
                status="error",
                raw_score=None,
                normalized_score=None,
                reason=prepared.solution_error,
            )
        try:
            messages = build_proof_eval_messages(
                query=str(data.get("query", "")),
                student_answer=prepared.solution_text or "",
                grading_guidelines=str(data.get("grading_guidelines", "")),
                ref_solution=data.get("ref_solution"),
            )
        except ValueError as exc:
            return _build_component_result(status="error", raw_score=None, normalized_score=None, reason=str(exc))

        async with llm_sem:
            raw = await self.client.chat(messages)

        content = raw["choices"][0]["message"]["content"]
        score = parse_proof_score(content)
        if score is None:
            return _build_component_result(
                status="error",
                raw_score=None,
                normalized_score=None,
                raw=content,
                reason="invalid proof score output",
                details=_usage_details(raw),
            )
        return _build_component_result(
            status="scored",
            raw_score=score,
            normalized_score=normalize_proof_score(score),
            raw=content,
            details=_usage_details(raw),
        )


class ConstructionEvaluator:
    def __init__(self, timeout: int) -> None:
        self.timeout = timeout

    async def evaluate(
        self, data: Dict[str, Any], prepared: PreparedResponse, verify_gate: threading.Semaphore
    ) -> Dict[str, Any]:
        if not uses_construction_prompt(prepared.mode):
            return _build_component_result(
                status="skipped",
                raw_score=None,
                normalized_score=None,
                reason="construction judge not applicable",
            )

        verify_code = data.get("verify_code")
        if not verify_code:
            return _build_component_result(
                status="error",
                raw_score=None,
                normalized_score=None,
                reason="missing required field: verify_code",
            )
        if prepared.construction_error:
            return _build_component_result(
                status="scored",
                raw_score=0,
                normalized_score=normalize_construction_score(0),
                reason=prepared.construction_error,
            )

        stdin_text = make_python_literal_stdin(prepared.construction or "")

        def _run() -> Dict[str, Any]:
            with verify_gate:
                result = run_prime_code_verifier(
                    verify_code=verify_code,
                    stdin_text=stdin_text,
                    timeout=self.timeout,
                )
            raw_score = 1 if result.passed else 0
            return _build_component_result(
                status="scored",
                raw_score=raw_score,
                normalized_score=normalize_construction_score(raw_score),
                reason=None if result.passed else result.info,
                details={
                    "verifier_score": result.score,
                    "info": result.info,
                    "metadata": result.metadata,
                },
            )

        return await asyncio.to_thread(_run)


class EvaluationSuite:
    def __init__(self, client: LlmClient, timeout: int, semaphore: int, judge_answer: bool = False) -> None:
        self.answer_evaluator = AnswerEvaluator(client, enabled=judge_answer)
        self.proof_evaluator = ProofEvaluator(client)
        self.construction_evaluator = ConstructionEvaluator(timeout)
        self.semaphore = semaphore
        self.judge_answer = judge_answer

    def check(self, data_list: List[Dict[str, Any]], responses: List[str]) -> List[Dict[str, Any]]:
        if len(data_list) != len(responses):
            raise ValueError("data_list and responses length mismatch")
        return asyncio.run(self._check_all_async(data_list, responses))

    async def check_one_async(
        self,
        data: Dict[str, Any],
        response: str,
        llm_sem: asyncio.Semaphore | None = None,
        verify_gate: threading.Semaphore | None = None,
    ) -> Dict[str, Any]:
        llm_sem = llm_sem or asyncio.Semaphore(self.semaphore)
        verify_gate = verify_gate or threading.Semaphore(self.semaphore)
        return await self._check_one_async(data, response, llm_sem, verify_gate)

    async def _check_all_async(
        self, data_list: List[Dict[str, Any]], responses: List[str]
    ) -> List[Dict[str, Any]]:
        llm_sem = asyncio.Semaphore(self.semaphore)
        verify_gate = threading.Semaphore(self.semaphore)
        results: List[Dict[str, Any]] = [None] * len(data_list)  # type: ignore[list-item]

        progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
        with progress:
            task_id = progress.add_task("checks", total=len(data_list))

            async def _one(idx: int, data: Dict[str, Any], response: str) -> None:
                results[idx] = await self._check_one_async(data, response, llm_sem, verify_gate)
                progress.update(task_id, advance=1)

            tasks = [
                _one(i, data, response)
                for i, (data, response) in enumerate(zip(data_list, responses, strict=True))
            ]
            await asyncio.gather(*tasks)

        return results

    async def _check_one_async(
        self,
        data: Dict[str, Any],
        response: str,
        llm_sem: asyncio.Semaphore,
        verify_gate: threading.Semaphore,
    ) -> Dict[str, Any]:
        mode = resolve_record_mode(data, warn=False)
        prepared = prepare_response(response or "", mode)
        answer_task = self.answer_evaluator.evaluate(data, prepared, llm_sem)
        proof_task = self.proof_evaluator.evaluate(data, prepared, llm_sem)
        construction_task = self.construction_evaluator.evaluate(data, prepared, verify_gate)
        answer, proof, construction = await asyncio.gather(answer_task, proof_task, construction_task)
        return self._build_case_result(mode, prepared, answer, proof, construction)

    def _build_case_result(
        self,
        mode: str,
        prepared: PreparedResponse,
        answer: Dict[str, Any],
        proof: Dict[str, Any],
        construction: Dict[str, Any],
    ) -> Dict[str, Any]:
        components = {"answer": answer, "proof": proof, "construction": construction}
        applicable = [component for component in components.values() if component["status"] != "skipped"]
        total_score = None
        if applicable:
            if all(component["status"] == "scored" and component["normalized_score"] is not None for component in applicable):
                total_score = sum(component["normalized_score"] for component in applicable) / len(applicable)

        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "judge_answer": self.judge_answer,
            "mode": mode,
            "answer": answer,
            "proof": proof,
            "construction": construction,
            "prepared_response": {
                "mode": prepared.mode,
                "solution_text": prepared.solution_text,
                "solution_error": prepared.solution_error,
                "construction": prepared.construction,
                "construction_error": prepared.construction_error,
            },
            "total_score": total_score,
            "max_score": TOTAL_MAX_SCORE,
        }


def build_evaluation_suite(
    profile_name: str,
    profiles_dir: str | Any,
    timeout: int,
    semaphore: int,
    judge_answer: bool = False,
) -> EvaluationSuite:
    name, model_type, params = load_profile(profile_name, Path(profiles_dir))
    if model_type != "llm":
        raise ValueError(f"evaluator profile must be llm: {name}")
    required = {"model_name", "api_key_env", "base_url"}
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"evaluator profile missing required fields: {missing}")
    if not params.get("model_name") or not params.get("api_key_env"):
        raise ValueError("evaluator profile requires non-empty model_name and api_key_env")
    kwargs = {k: v for k, v in params.items() if k not in required}
    client = build_llm_client(
        model_name=params["model_name"],
        api_key_env=params["api_key_env"],
        base_url=params["base_url"],
        **kwargs,
    )
    return EvaluationSuite(client=client, timeout=timeout, semaphore=semaphore, judge_answer=judge_answer)
