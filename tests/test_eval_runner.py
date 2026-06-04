import io
import json
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path

from src.eval.runner import EvaluationRunner
from src.eval.scoring import EVALUATION_SCHEMA_VERSION, TOTAL_MAX_SCORE
from src.llm.client import MalformedStreamResponseError


class FakeResponseModel:
    def __init__(self, failures=None, usage_by_id=None, response_by_id=None, debug_by_id=None):
        self.failures = set(failures or [])
        self.usage_by_id = usage_by_id or {}
        self.response_by_id = response_by_id or {}
        self.debug_by_id = debug_by_id or {}
        self.generated_ids = []
        self.generated_keys = []

    async def generate(self, prompt, record):
        record_id = record["id"]
        self.generated_ids.append(record_id)
        self.generated_keys.append(record.get("_attempt_id") or record_id)
        if record_id in self.failures:
            raise RuntimeError(f"boom: {record_id}")
        response = self.response_by_id.get(record_id, f"response for {record_id}")
        usage = self.usage_by_id.get(record_id)
        debug = self.debug_by_id.get(record_id)
        if usage is not None or debug is not None:
            return {"content": response, "usage": usage, "stream_debug": debug}
        return response


class FakeEvaluationSuite:
    def __init__(self, failures=None):
        self.failures = set(failures or [])
        self.evaluated_ids = []
        self.evaluated_keys = []
        self.semaphore = 1

    async def check_one_async(self, data, response, llm_sem=None, verify_gate=None):
        record_id = data["id"]
        self.evaluated_ids.append(record_id)
        self.evaluated_keys.append(data.get("_attempt_id") or record_id)
        if record_id in self.failures:
            raise RuntimeError(f"eval boom: {record_id}")
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "judge_answer": False,
            "mode": "plain",
            "answer": {"status": "skipped", "raw_score": None, "normalized_score": None},
            "proof": {"status": "scored", "raw_score": 7, "normalized_score": 1.0},
            "construction": {"status": "skipped", "raw_score": None, "normalized_score": None},
            "prepared_response": {
                "mode": "plain",
                "solution_text": response,
                "solution_error": None,
                "construction": None,
                "construction_error": None,
            },
            "total_score": 1.0,
            "max_score": TOTAL_MAX_SCORE,
        }


class EvaluationRunnerLineSelectionTests(unittest.TestCase):
    def _write_dataset(self, path: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    '{"id":"id-1","query":"q1","instruction":"i1","verify_code":"print(True)"}',
                    '{"id":"id-2","query":"q2","instruction":"i2","verify_code":"print(True)"}',
                    '{"id":"id-3","query":"q3"}',
                    '{"id":"id-4","query":"q4","instruction":"i4"}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def _make_runner(
        self,
        dataset_path: Path,
        output_root: Path,
        line_numbers,
        judge_answer: bool = False,
        response_model=None,
        evaluation_suite=None,
        semaphore: int = 8,
        repeat: int = 1,
        force_attempt_ids=None,
    ):
        return EvaluationRunner(
            dataset_path=dataset_path,
            output_root=output_root,
            line_numbers=line_numbers,
            generation_mode="mock",
            generation_profile="mock",
            evaluation_profile="gemini",
            response_model=response_model,
            evaluation_suite=evaluation_suite,
            timeout=15,
            semaphore=semaphore,
            judge_answer=judge_answer,
            repeat=repeat,
            force_attempt_ids=force_attempt_ids,
        )

    def test_load_records_uses_selected_lines_in_requested_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            runner = self._make_runner(dataset, root / "outputs", [3, 2])

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                records = runner._load_records()

            self.assertEqual([record["id"] for record in records], ["id-3", "id-2"])

    def test_load_records_raises_when_selected_line_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            runner = self._make_runner(dataset, root / "outputs", [2, 9])

            with self.assertRaises(ValueError):
                runner._load_records()

    def test_run_uses_line_suffix_in_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            output_root = root / "outputs"
            runner = self._make_runner(dataset, output_root, [2, 4])

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                run_dir = runner.run(generate=False, evaluate=False)

            expected = output_root / "sample__lines_2_4" / "mock" / "mock"
            self.assertEqual(run_dir, expected)
            self.assertTrue(run_dir.exists())

    def test_load_records_warns_for_inconsistent_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            runner = self._make_runner(dataset, root / "outputs", [4])

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                runner._load_records()

            self.assertEqual(1, len(caught))
            self.assertIn("inconsistent construction metadata", str(caught[0].message))

    def test_eval_record_emits_solution_construction_and_scores(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            runner = self._make_runner(dataset, root / "outputs", None)

            result = runner._eval_record(
                {"id": "id-1", "instruction": "i1", "verify_code": "print(True)"},
                "\n".join(
                    [
                        "## Solution to Question 1",
                        "answer only",
                        "",
                        "## Solution to Question 2",
                        "construction here",
                        "<construct>",
                        "[[1, 2], [3, 4]]",
                        "</construct>",
                    ]
                ),
                {
                    "answer": {"status": "skipped", "raw_score": None, "normalized_score": None},
                    "proof": {"status": "scored", "raw_score": 7, "normalized_score": 1.0},
                    "construction": {"status": "scored", "raw_score": 1, "normalized_score": 1.0},
                    "total_score": 1.0,
                    "max_score": TOTAL_MAX_SCORE,
                },
            )

            self.assertEqual("answer only", result["solution_text"])
            self.assertEqual("[[1, 2], [3, 4]]", result["construction"])
            self.assertEqual(1.0, result["total_score"])

    def test_load_evaluation_cache_requires_matching_meta_and_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            runner = self._make_runner(dataset, root / "outputs", None)
            with warnings.catch_warnings(record=True):
                records = runner._load_records()[:1]

            cache_path = root / "cache.jsonl"
            meta_path = root / "run.json"
            expected_meta = runner._build_evaluation_meta(records)

            meta_path.write_text(json.dumps(expected_meta), encoding="utf-8")
            cache_path.write_text(
                json.dumps(
                    {
                        "id": "id-1",
                        "eval_result": {"schema_version": EVALUATION_SCHEMA_VERSION},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            cached = runner._load_evaluation_cache(records, cache_path, meta_path, expected_meta)
            self.assertEqual([{"schema_version": EVALUATION_SCHEMA_VERSION}], cached)

            broken_meta = dict(expected_meta)
            broken_meta["judge_answer"] = True
            self.assertIsNone(runner._load_evaluation_cache(records, cache_path, meta_path, broken_meta))

    def test_load_evaluation_cache_ignores_semaphore_meta_difference(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            old_runner = self._make_runner(dataset, root / "outputs", None, semaphore=16)
            new_runner = self._make_runner(dataset, root / "outputs", None, semaphore=4)
            with warnings.catch_warnings(record=True):
                records = new_runner._load_records()[:1]

            cache_path = root / "cache.jsonl"
            meta_path = root / "run.json"
            old_meta = old_runner._build_evaluation_meta(records)
            old_meta["semaphore"] = 16
            expected_meta = new_runner._build_evaluation_meta(records)

            meta_path.write_text(json.dumps(old_meta), encoding="utf-8")
            cache_path.write_text(
                json.dumps(
                    {
                        "id": "id-1",
                        "eval_result": {"schema_version": EVALUATION_SCHEMA_VERSION},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            cached = new_runner._load_evaluation_cache(records, cache_path, meta_path, expected_meta)

            self.assertEqual([{"schema_version": EVALUATION_SCHEMA_VERSION}], cached)
            self.assertNotIn("semaphore", expected_meta)

    def test_load_generation_cache_accepts_out_of_order_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            runner = self._make_runner(dataset, root / "outputs", [1, 2])
            records = runner._load_records()
            cache_path = root / "generation-cache.jsonl"
            cache_path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "id-2", "response": "response 2"}),
                        json.dumps({"id": "id-1", "response": "response 1"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            responses = runner._load_generation_cache(records, cache_path)

            self.assertEqual(["response 1", "response 2"], responses)

    def test_load_generation_cache_ignores_empty_response(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            runner = self._make_runner(dataset, root / "outputs", [1, 2])
            records = runner._load_records()
            cache_path = root / "generation-cache.jsonl"
            cache_path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "id-1", "response": ""}),
                        json.dumps({"id": "id-2", "response": "response 2"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            responses = runner._load_generation_cache(records, cache_path)
            cached_by_id = runner._load_generation_cache_map(records, cache_path)

            self.assertIsNone(responses)
            self.assertNotIn("id-1", cached_by_id)
            self.assertEqual("response 2", cached_by_id["id-2"])

    def test_generation_reuses_partial_cache_and_appends_missing_records(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            output_root = root / "outputs"
            generation_dir = output_root / "sample__lines_1_2_3" / "mock" / "mock" / "generation"
            generation_dir.mkdir(parents=True)
            cache_path = generation_dir / "cache.jsonl"
            cache_path.write_text(
                json.dumps({"id": "id-2", "response": "cached response 2"}) + "\n",
                encoding="utf-8",
            )
            model = FakeResponseModel()
            runner = self._make_runner(
                dataset,
                output_root,
                [1, 2, 3],
                response_model=model,
                semaphore=1,
            )

            runner.run(generate=True, evaluate=False)

            self.assertEqual(["id-1", "id-3"], model.generated_ids)
            responses = runner._load_generation_cache(runner._load_records(), cache_path)
            self.assertEqual(
                ["response for id-1", "cached response 2", "response for id-3"],
                responses,
            )
            self.assertTrue((generation_dir / "cases" / "id-1.md").exists())
            self.assertTrue((generation_dir / "cases" / "id-3.md").exists())

    def test_generation_cache_and_meta_record_token_usage(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            output_root = root / "outputs"
            model = FakeResponseModel(
                usage_by_id={
                    "id-1": {
                        "prompt_tokens": 2,
                        "completion_tokens": 3,
                        "total_tokens": 5,
                        "completion_tokens_details": {"reasoning_tokens": 1},
                    },
                    "id-2": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18},
                }
            )
            runner = self._make_runner(
                dataset,
                output_root,
                [1, 2],
                response_model=model,
                semaphore=1,
            )

            runner.run(generate=True, evaluate=False, use_generation_cache=False)

            generation_dir = output_root / "sample__lines_1_2" / "mock" / "mock" / "generation"
            cache_path = generation_dir / "cache.jsonl"
            cached = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines()]
            usage_by_id = {item["id"]: item.get("usage") for item in cached}
            normalized_by_id = {item["id"]: item.get("usage_normalized") for item in cached}
            self.assertEqual(
                {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                    "completion_tokens_details": {"reasoning_tokens": 1},
                },
                usage_by_id["id-1"],
            )
            self.assertEqual({"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}, usage_by_id["id-2"])
            self.assertEqual(
                {
                    "input_tokens": 2,
                    "output_tokens": 3,
                    "reasoning_tokens": 1,
                    "response_tokens": 2,
                    "total_tokens": 5,
                    "raw": usage_by_id["id-1"],
                },
                normalized_by_id["id-1"],
            )

            meta = json.loads((generation_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(
                {"prompt_tokens": 9, "completion_tokens": 14, "total_tokens": 23},
                meta["usage"],
            )
            self.assertEqual(
                {
                    "input_tokens": 9,
                    "output_tokens": 14,
                    "reasoning_tokens": 1,
                    "response_tokens": 13,
                    "total_tokens": 23,
                },
                meta["usage_normalized"],
            )

    def test_generation_failure_records_error_and_keeps_successful_records_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            output_root = root / "outputs"
            model = FakeResponseModel(failures={"id-2"})
            runner = self._make_runner(
                dataset,
                output_root,
                [1, 2, 3],
                response_model=model,
                semaphore=1,
            )

            runner.run(generate=True, evaluate=False, use_generation_cache=False)

            generation_dir = output_root / "sample__lines_1_2_3" / "mock" / "mock" / "generation"
            cache_path = generation_dir / "cache.jsonl"
            errors_path = generation_dir / "errors.jsonl"
            cached = list(map(json.loads, cache_path.read_text(encoding="utf-8").splitlines()))
            cached_by_id = {item["id"]: item["response"] for item in cached}
            errors = list(map(json.loads, errors_path.read_text(encoding="utf-8").splitlines()))
            self.assertEqual("response for id-1", cached_by_id.get("id-1"))
            self.assertEqual("response for id-3", cached_by_id.get("id-3"))
            self.assertNotIn("id-2", cached_by_id)
            self.assertEqual("id-2", errors[0]["id"])
            self.assertEqual("generation", errors[0]["stage"])
            self.assertEqual("RuntimeError", errors[0]["error_type"])
            self.assertTrue(errors[0]["retryable"])
            self.assertTrue((generation_dir / "cases" / "id-1.md").exists())

    def test_pipeline_writes_generation_and_evaluation_incrementally(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            output_root = root / "outputs"
            model = FakeResponseModel()
            suite = FakeEvaluationSuite()
            runner = self._make_runner(
                dataset,
                output_root,
                [1, 2],
                response_model=model,
                evaluation_suite=suite,
                semaphore=1,
            )

            runner.run(generate=True, evaluate=True, use_generation_cache=False, use_evaluation_cache=False)

            run_dir = output_root / "sample__lines_1_2" / "mock" / "mock"
            generation_cache = run_dir / "generation" / "cache.jsonl"
            evaluation_cache = run_dir / "evaluation" / "gemini" / "cache.jsonl"
            results_jsonl = run_dir / "evaluation" / "gemini" / "results.jsonl"
            self.assertEqual(["id-1", "id-2"], model.generated_ids)
            self.assertEqual(["id-1", "id-2"], suite.evaluated_ids)
            self.assertEqual(2, len(generation_cache.read_text(encoding="utf-8").splitlines()))
            self.assertEqual(2, len(evaluation_cache.read_text(encoding="utf-8").splitlines()))
            self.assertEqual(2, len(results_jsonl.read_text(encoding="utf-8").splitlines()))
            self.assertTrue((run_dir / "generation" / "cases" / "id-1.md").exists())
            self.assertTrue((run_dir / "evaluation" / "gemini" / "cases" / "id-1.json").exists())

    def test_pipeline_reuses_generation_cache_when_only_evaluation_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            output_root = root / "outputs"
            run_dir = output_root / "sample__lines_1_2" / "mock" / "mock"
            generation_dir = run_dir / "generation"
            generation_dir.mkdir(parents=True)
            (generation_dir / "cache.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "id-1", "response": "cached response 1"}),
                        json.dumps({"id": "id-2", "response": "cached response 2"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            model = FakeResponseModel()
            suite = FakeEvaluationSuite()
            runner = self._make_runner(
                dataset,
                output_root,
                [1, 2],
                response_model=model,
                evaluation_suite=suite,
                semaphore=1,
            )

            runner.run(generate=True, evaluate=True)

            self.assertEqual([], model.generated_ids)
            self.assertEqual(["id-1", "id-2"], suite.evaluated_ids)
            evaluation_cache = run_dir / "evaluation" / "gemini" / "cache.jsonl"
            self.assertEqual(2, len(evaluation_cache.read_text(encoding="utf-8").splitlines()))

    def test_pipeline_evaluation_failure_records_error_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            output_root = root / "outputs"
            model = FakeResponseModel()
            suite = FakeEvaluationSuite(failures={"id-2"})
            runner = self._make_runner(
                dataset,
                output_root,
                [1, 2],
                response_model=model,
                evaluation_suite=suite,
                semaphore=1,
            )

            runner.run(generate=True, evaluate=True, use_generation_cache=False, use_evaluation_cache=False)

            run_dir = output_root / "sample__lines_1_2" / "mock" / "mock"
            generation_cache = run_dir / "generation" / "cache.jsonl"
            evaluation_cache = run_dir / "evaluation" / "gemini" / "cache.jsonl"
            evaluation_errors = run_dir / "evaluation" / "gemini" / "errors.jsonl"
            results_jsonl = run_dir / "evaluation" / "gemini" / "results.jsonl"
            generated_by_id = {
                item["id"]: item["response"]
                for item in map(json.loads, generation_cache.read_text(encoding="utf-8").splitlines())
            }
            evaluated_ids = [
                item["id"]
                for item in map(json.loads, evaluation_cache.read_text(encoding="utf-8").splitlines())
            ]
            errors = list(map(json.loads, evaluation_errors.read_text(encoding="utf-8").splitlines()))
            results = list(map(json.loads, results_jsonl.read_text(encoding="utf-8").splitlines()))
            self.assertEqual("response for id-2", generated_by_id.get("id-2"))
            self.assertEqual(["id-1"], evaluated_ids)
            self.assertEqual("id-2", errors[0]["id"])
            self.assertEqual("evaluation", errors[0]["stage"])
            self.assertEqual("RuntimeError", errors[0]["error_type"])
            self.assertIsNone(results[1]["total_score"])
            self.assertEqual("error", results[1]["proof"]["status"])

    def test_pipeline_generation_failure_records_error_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            output_root = root / "outputs"
            model = FakeResponseModel(failures={"id-2"})
            suite = FakeEvaluationSuite()
            runner = self._make_runner(
                dataset,
                output_root,
                [1, 2, 3],
                response_model=model,
                evaluation_suite=suite,
                semaphore=1,
            )

            runner.run(generate=True, evaluate=True, use_generation_cache=False, use_evaluation_cache=False)

            run_dir = output_root / "sample__lines_1_2_3" / "mock" / "mock"
            generation_cache = [
                json.loads(line)
                for line in (run_dir / "generation" / "cache.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            generation_errors = [
                json.loads(line)
                for line in (run_dir / "generation" / "errors.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            evaluation_cache = [
                json.loads(line)
                for line in (run_dir / "evaluation" / "gemini" / "cache.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            results = [
                json.loads(line)
                for line in (run_dir / "evaluation" / "gemini" / "results.jsonl").read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(["id-1", "id-3"], [item["id"] for item in generation_cache])
            self.assertEqual(["id-2"], [item["id"] for item in generation_errors])
            self.assertEqual(["id-1", "id-3"], [item["id"] for item in evaluation_cache])
            self.assertIsNone(results[1]["total_score"])
            self.assertEqual("generation", results[1]["proof"]["details"]["stage"])

    def test_pipeline_empty_generation_response_records_error_and_skips_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            output_root = root / "outputs"
            model = FakeResponseModel(response_by_id={"id-2": "   \n"})
            suite = FakeEvaluationSuite()
            runner = self._make_runner(
                dataset,
                output_root,
                [1, 2, 3],
                response_model=model,
                evaluation_suite=suite,
                semaphore=1,
            )

            runner.run(generate=True, evaluate=True, use_generation_cache=False, use_evaluation_cache=False)

            run_dir = output_root / "sample__lines_1_2_3" / "mock" / "mock"
            generation_cache = [
                json.loads(line)
                for line in (run_dir / "generation" / "cache.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            generation_errors = [
                json.loads(line)
                for line in (run_dir / "generation" / "errors.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            evaluation_cache = [
                json.loads(line)
                for line in (run_dir / "evaluation" / "gemini" / "cache.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            results = [
                json.loads(line)
                for line in (run_dir / "evaluation" / "gemini" / "results.jsonl").read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(["id-1", "id-3"], [item["id"] for item in generation_cache])
            self.assertEqual("id-2", generation_errors[0]["id"])
            self.assertEqual("EmptyGenerationResponseError", generation_errors[0]["error_type"])
            self.assertEqual(["id-1", "id-3"], [item["id"] for item in evaluation_cache])
            self.assertEqual(["id-1", "id-3"], suite.evaluated_ids)
            self.assertEqual(0.0, results[1]["total_score"])
            self.assertEqual("scored", results[1]["proof"]["status"])
            self.assertEqual(0.0, results[1]["proof"]["normalized_score"])
            self.assertEqual("scored", results[1]["construction_eval"]["status"])
            self.assertEqual(0.0, results[1]["construction_eval"]["normalized_score"])

    def test_pipeline_empty_stream_generation_records_debug_details(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            output_root = root / "outputs"
            debug = {
                "saw_content": False,
                "saw_reasoning_content": True,
                "reasoning_content_characters": 13,
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 4, "completion_tokens": 8, "total_tokens": 12},
            }
            model = FakeResponseModel(
                response_by_id={"id-1": ""},
                debug_by_id={"id-1": debug},
            )
            suite = FakeEvaluationSuite()
            runner = self._make_runner(
                dataset,
                output_root,
                [1],
                response_model=model,
                evaluation_suite=suite,
                semaphore=1,
            )

            runner.run(generate=True, evaluate=True, use_generation_cache=False, use_evaluation_cache=False)

            run_dir = output_root / "sample__lines_1" / "mock" / "mock"
            generation_errors = [
                json.loads(line)
                for line in (run_dir / "generation" / "errors.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            results = [
                json.loads(line)
                for line in (run_dir / "evaluation" / "gemini" / "results.jsonl").read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual("EmptyGenerationResponseError", generation_errors[0]["error_type"])
            self.assertEqual(debug, generation_errors[0]["details"])
            self.assertFalse((run_dir / "generation" / "cache.jsonl").exists())
            self.assertEqual([], suite.evaluated_ids)
            self.assertEqual(0.0, results[0]["total_score"])
            self.assertEqual(debug, results[0]["proof"]["details"]["details"])
            self.assertEqual(0.0, results[0]["construction_eval"]["normalized_score"])

    def test_pipeline_incomplete_reasoning_only_stream_stays_retryable_without_zero_score(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            output_root = root / "outputs"
            debug = {
                "saw_content": False,
                "saw_reasoning_content": True,
                "reasoning_content_characters": 82701,
                "finish_reason": None,
                "usage": {"prompt_tokens": 397, "completion_tokens": 0, "total_tokens": 397},
            }
            model = FakeResponseModel(
                response_by_id={"id-1": ""},
                debug_by_id={"id-1": debug},
            )
            suite = FakeEvaluationSuite()
            runner = self._make_runner(
                dataset,
                output_root,
                [1],
                response_model=model,
                evaluation_suite=suite,
                semaphore=1,
            )

            runner.run(generate=True, evaluate=True, use_generation_cache=False, use_evaluation_cache=False)

            run_dir = output_root / "sample__lines_1" / "mock" / "mock"
            generation_errors = [
                json.loads(line)
                for line in (run_dir / "generation" / "errors.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            results = [
                json.loads(line)
                for line in (run_dir / "evaluation" / "gemini" / "results.jsonl").read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual("EmptyGenerationResponseError", generation_errors[0]["error_type"])
            self.assertEqual(debug, generation_errors[0]["details"])
            self.assertEqual([], suite.evaluated_ids)
            self.assertIsNone(results[0]["total_score"])
            self.assertEqual("error", results[0]["proof"]["status"])
            self.assertEqual(debug, results[0]["proof"]["details"]["details"])
            self.assertEqual("skipped", results[0]["construction_eval"]["status"])

    def test_pipeline_empty_stream_generation_at_token_limit_scores_zero(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            output_root = root / "outputs"
            debug = {
                "saw_content": False,
                "saw_reasoning_content": True,
                "reasoning_content_characters": 430000,
                "finish_reason": None,
                "usage": {"prompt_tokens": 397, "completion_tokens": 160000, "total_tokens": 160397},
            }
            model = FakeResponseModel(
                response_by_id={"id-1": ""},
                debug_by_id={"id-1": debug},
            )
            suite = FakeEvaluationSuite()
            runner = self._make_runner(
                dataset,
                output_root,
                [1],
                response_model=model,
                evaluation_suite=suite,
                semaphore=1,
            )

            runner.run(generate=True, evaluate=True, use_generation_cache=False, use_evaluation_cache=False)

            run_dir = output_root / "sample__lines_1" / "mock" / "mock"
            results = [
                json.loads(line)
                for line in (run_dir / "evaluation" / "gemini" / "results.jsonl").read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual([], suite.evaluated_ids)
            self.assertEqual(0.0, results[0]["total_score"])
            self.assertEqual("scored", results[0]["proof"]["status"])
            self.assertEqual(0.0, results[0]["proof"]["normalized_score"])

    def test_pipeline_generation_error_records_exception_details(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            output_root = root / "outputs"
            details = {
                "malformed_stream": True,
                "json_error": "Unterminated string",
                "raw_line_preview": 'data: {"broken"',
                "data_preview": '{"broken"',
            }
            model = FakeResponseModel(failures=["id-1"])
            async def generate(prompt, record):
                raise MalformedStreamResponseError("malformed stream JSON chunk", details)
            model.generate = generate
            suite = FakeEvaluationSuite()
            runner = self._make_runner(
                dataset,
                output_root,
                [1],
                response_model=model,
                evaluation_suite=suite,
                semaphore=1,
            )

            runner.run(generate=True, evaluate=True, use_generation_cache=False, use_evaluation_cache=False)

            run_dir = output_root / "sample__lines_1" / "mock" / "mock"
            generation_errors = [
                json.loads(line)
                for line in (run_dir / "generation" / "errors.jsonl").read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual("MalformedStreamResponseError", generation_errors[0]["error_type"])
            self.assertEqual(details, generation_errors[0]["details"])
            self.assertEqual([], suite.evaluated_ids)

    def test_pipeline_ignores_old_empty_generation_and_old_evaluation_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            output_root = root / "outputs"
            run_dir = output_root / "sample__lines_1_2" / "mock" / "mock"
            generation_dir = run_dir / "generation"
            evaluation_dir = run_dir / "evaluation" / "gemini"
            generation_dir.mkdir(parents=True)
            evaluation_dir.mkdir(parents=True)
            (generation_dir / "cache.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "id-1", "response": "cached response 1"}),
                        json.dumps({"id": "id-2", "response": ""}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            expected_meta_runner = self._make_runner(dataset, output_root, [1, 2], semaphore=1)
            records = expected_meta_runner._load_records()
            evaluation_meta = expected_meta_runner._build_evaluation_meta(records)
            (evaluation_dir / "run.json").write_text(json.dumps(evaluation_meta), encoding="utf-8")
            (evaluation_dir / "cache.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "id-1",
                                "eval_result": {
                                    "schema_version": EVALUATION_SCHEMA_VERSION,
                                    "total_score": 1.0,
                                    "proof": {"normalized_score": 1.0},
                                    "construction": {"status": "skipped", "normalized_score": None},
                                    "answer": {"status": "skipped", "normalized_score": None},
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "id": "id-2",
                                "eval_result": {
                                    "schema_version": EVALUATION_SCHEMA_VERSION,
                                    "total_score": None,
                                    "proof": {"status": "error", "normalized_score": None},
                                    "construction": {"status": "error", "normalized_score": None},
                                    "answer": {"status": "skipped", "normalized_score": None},
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            model = FakeResponseModel()
            suite = FakeEvaluationSuite()
            runner = self._make_runner(
                dataset,
                output_root,
                [1, 2],
                response_model=model,
                evaluation_suite=suite,
                semaphore=1,
            )

            runner.run(generate=True, evaluate=True)

            final_generation_cache = [
                json.loads(line)
                for line in (generation_dir / "cache.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            final_evaluation_cache = [
                json.loads(line)
                for line in (evaluation_dir / "cache.jsonl").read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(["id-2"], model.generated_ids)
            self.assertEqual(["id-2"], suite.evaluated_ids)
            self.assertEqual(
                ["id-1", "id-2"],
                [item["id"] for item in final_generation_cache if item.get("response", "").strip()],
            )
            latest_eval_by_id = {item["id"]: item["eval_result"] for item in final_evaluation_cache}
            self.assertEqual(1.0, latest_eval_by_id["id-1"]["total_score"])
            self.assertEqual(1.0, latest_eval_by_id["id-2"]["total_score"])

    def test_pipeline_retries_previously_failed_attempts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            output_root = root / "outputs"
            first_model = FakeResponseModel(failures={"id-2"})
            first_suite = FakeEvaluationSuite()
            first_runner = self._make_runner(
                dataset,
                output_root,
                [1, 2, 3],
                response_model=first_model,
                evaluation_suite=first_suite,
                semaphore=1,
            )
            first_runner.run(generate=True, evaluate=True, use_generation_cache=False, use_evaluation_cache=False)

            second_model = FakeResponseModel()
            second_suite = FakeEvaluationSuite()
            second_runner = self._make_runner(
                dataset,
                output_root,
                [1, 2, 3],
                response_model=second_model,
                evaluation_suite=second_suite,
                semaphore=1,
            )
            second_runner.run(generate=True, evaluate=True)

            run_dir = output_root / "sample__lines_1_2_3" / "mock" / "mock"
            generation_cache = [
                json.loads(line)
                for line in (run_dir / "generation" / "cache.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            evaluation_cache = [
                json.loads(line)
                for line in (run_dir / "evaluation" / "gemini" / "cache.jsonl").read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(["id-2"], second_model.generated_ids)
            self.assertEqual(["id-2"], second_suite.evaluated_ids)
            self.assertEqual(["id-1", "id-3", "id-2"], [item["id"] for item in generation_cache])
            self.assertEqual(["id-1", "id-3", "id-2"], [item["id"] for item in evaluation_cache])

    def test_repeat_pipeline_uses_attempt_ids_and_writes_problem_aggregates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            output_root = root / "outputs"
            model = FakeResponseModel()
            suite = FakeEvaluationSuite()
            runner = self._make_runner(
                dataset,
                output_root,
                [1, 2],
                response_model=model,
                evaluation_suite=suite,
                semaphore=1,
                repeat=4,
            )

            runner.run(generate=True, evaluate=True, use_generation_cache=False, use_evaluation_cache=False)

            run_dir = output_root / "sample__lines_1_2" / "mock" / "mock"
            generation_cache = [
                json.loads(line)
                for line in (run_dir / "generation" / "cache.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            evaluation_cache = [
                json.loads(line)
                for line in (run_dir / "evaluation" / "gemini" / "cache.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            by_problem = [
                json.loads(line)
                for line in (run_dir / "evaluation" / "gemini" / "results_by_problem.jsonl").read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(8, len(generation_cache))
            self.assertEqual(8, len(evaluation_cache))
            self.assertEqual(8, len({item["attempt_id"] for item in generation_cache}))
            self.assertEqual("id-1__run_1", generation_cache[0]["attempt_id"])
            self.assertEqual(1, generation_cache[0]["run_index"])
            self.assertTrue((run_dir / "generation" / "cases" / "id-1__run_1.md").exists())
            self.assertTrue((run_dir / "evaluation" / "gemini" / "cases" / "id-1__run_1.json").exists())
            self.assertEqual(
                [{"id": "id-1", "attempts": 4, "average_total": 1.0, "proof_average": 1.0, "construction_average": None, "answer_average": None},
                 {"id": "id-2", "attempts": 4, "average_total": 1.0, "proof_average": 1.0, "construction_average": None, "answer_average": None}],
                by_problem,
            )

    def test_repeat_pipeline_resumes_only_missing_attempts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            output_root = root / "outputs"
            run_dir = output_root / "sample__lines_1" / "mock" / "mock"
            generation_dir = run_dir / "generation"
            evaluation_dir = run_dir / "evaluation" / "gemini"
            generation_dir.mkdir(parents=True)
            evaluation_dir.mkdir(parents=True)
            (generation_dir / "cache.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "id-1", "attempt_id": "id-1__run_1", "run_index": 1, "response": "cached 1"}),
                        json.dumps({"id": "id-1", "attempt_id": "id-1__run_2", "run_index": 2, "response": "cached 2"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            expected_meta_runner = self._make_runner(dataset, output_root, [1], semaphore=1, repeat=4)
            records = expected_meta_runner._load_records()
            evaluation_meta = expected_meta_runner._build_evaluation_meta(
                [
                    {**records[0], "_attempt_id": f"id-1__run_{i}", "_run_index": i}
                    for i in range(1, 5)
                ]
            )
            (evaluation_dir / "run.json").write_text(json.dumps(evaluation_meta), encoding="utf-8")
            (evaluation_dir / "cache.jsonl").write_text(
                json.dumps(
                    {
                        "id": "id-1",
                        "attempt_id": "id-1__run_1",
                        "run_index": 1,
                        "eval_result": {
                            "schema_version": EVALUATION_SCHEMA_VERSION,
                            "total_score": 1.0,
                            "proof": {"normalized_score": 1.0},
                            "construction": {"status": "skipped", "normalized_score": None},
                            "answer": {"status": "skipped", "normalized_score": None},
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            model = FakeResponseModel()
            suite = FakeEvaluationSuite()
            runner = self._make_runner(
                dataset,
                output_root,
                [1],
                response_model=model,
                evaluation_suite=suite,
                semaphore=1,
                repeat=4,
            )

            runner.run(generate=True, evaluate=True)

            self.assertEqual(["id-1", "id-1"], model.generated_ids)
            self.assertEqual(["id-1", "id-1", "id-1"], suite.evaluated_ids)
            final_generation_cache = (generation_dir / "cache.jsonl").read_text(encoding="utf-8").splitlines()
            final_evaluation_cache = (evaluation_dir / "cache.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(4, len(final_generation_cache))
            self.assertEqual(4, len(final_evaluation_cache))

    def test_force_attempt_regenerates_only_selected_attempt_and_refreshes_eval(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            output_root = root / "outputs"
            run_dir = output_root / "sample__lines_1" / "mock" / "mock"
            generation_dir = run_dir / "generation"
            evaluation_dir = run_dir / "evaluation" / "gemini"
            generation_dir.mkdir(parents=True)
            evaluation_dir.mkdir(parents=True)
            (generation_dir / "cache.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "id-1", "attempt_id": f"id-1__run_{i}", "run_index": i, "response": f"cached {i}"})
                        for i in range(1, 5)
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            expected_meta_runner = self._make_runner(dataset, output_root, [1], semaphore=1, repeat=4)
            records = expected_meta_runner._load_records()
            expanded_records = [
                {**records[0], "_attempt_id": f"id-1__run_{i}", "_run_index": i}
                for i in range(1, 5)
            ]
            evaluation_meta = expected_meta_runner._build_evaluation_meta(expanded_records)
            (evaluation_dir / "run.json").write_text(json.dumps(evaluation_meta), encoding="utf-8")
            (evaluation_dir / "cache.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "id-1",
                                "attempt_id": f"id-1__run_{i}",
                                "run_index": i,
                                "eval_result": {
                                    "schema_version": EVALUATION_SCHEMA_VERSION,
                                    "total_score": None if i == 3 else 1.0,
                                    "proof": {"normalized_score": None if i == 3 else 1.0},
                                    "construction": {"status": "skipped", "normalized_score": None},
                                    "answer": {"status": "skipped", "normalized_score": None},
                                },
                            }
                        )
                        for i in range(1, 5)
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            model = FakeResponseModel(response_by_id={"id-1": "forced response"})
            suite = FakeEvaluationSuite()
            runner = self._make_runner(
                dataset,
                output_root,
                [1],
                response_model=model,
                evaluation_suite=suite,
                semaphore=1,
                repeat=4,
                force_attempt_ids=["id-1__run_3"],
            )

            runner.run(generate=True, evaluate=True)

            self.assertEqual(["id-1__run_3"], model.generated_keys)
            self.assertEqual(["id-1__run_3"], suite.evaluated_keys)
            final_generation_cache = [
                json.loads(line)
                for line in (generation_dir / "cache.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            final_evaluation_cache = [
                json.loads(line)
                for line in (evaluation_dir / "cache.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(5, len(final_generation_cache))
            self.assertEqual(5, len(final_evaluation_cache))
            self.assertEqual("id-1__run_3", final_generation_cache[-1]["attempt_id"])
            self.assertEqual("forced response", final_generation_cache[-1]["response"])
            self.assertEqual("id-1__run_3", final_evaluation_cache[-1]["attempt_id"])
            self.assertEqual(1.0, final_evaluation_cache[-1]["eval_result"]["total_score"])

    def test_force_attempt_rejects_unknown_attempt_id(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            runner = self._make_runner(
                dataset,
                root / "outputs",
                [1],
                repeat=4,
                force_attempt_ids=["id-1__run_9"],
            )

            with self.assertRaisesRegex(ValueError, "unknown --force-attempt"):
                runner.run(generate=False, evaluate=False)

    def test_print_eval_results_uses_normalized_summary_and_detail(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            runner = self._make_runner(dataset, root / "outputs", None, judge_answer=True)
            with warnings.catch_warnings(record=True):
                records = runner._load_records()[:2]
            eval_results = [
                {
                    "answer": {"status": "scored", "raw_score": 1, "normalized_score": 1.0},
                    "proof": {"status": "scored", "raw_score": 6, "normalized_score": 6 / 7},
                    "construction": {"status": "scored", "raw_score": 1, "normalized_score": 1.0},
                    "total_score": (1.0 + 6 / 7 + 1.0) / 3,
                },
                {
                    "answer": {"status": "scored", "raw_score": 0, "normalized_score": 0.0},
                    "proof": {"status": "scored", "raw_score": 7, "normalized_score": 1.0},
                    "construction": {"status": "skipped", "raw_score": None, "normalized_score": None},
                    "total_score": 0.5,
                },
            ]

            detail_buffer = io.StringIO()
            with redirect_stdout(detail_buffer):
                runner._print_eval_results(records, eval_results, "detail")
            self.assertIn("answer=1.0000(raw=1)", detail_buffer.getvalue())
            self.assertIn(f"proof={6 / 7:.4f}(raw=6)", detail_buffer.getvalue())
            self.assertIn("construction=1.0000(raw=1)", detail_buffer.getvalue())

            summary_buffer = io.StringIO()
            with redirect_stdout(summary_buffer):
                runner._print_eval_results(records, eval_results, "summary")
            summary = summary_buffer.getvalue()
            self.assertIn(f"records=2 average_total={(((1.0 + 6 / 7 + 1.0) / 3) + 0.5) / 2:.4f}", summary)
            self.assertIn(f"proof_average={(6 / 7 + 1.0) / 2:.4f}", summary)
            self.assertIn("construction_average=1.0000", summary)
            self.assertIn("answer_average=0.5000", summary)

    def test_build_evaluation_meta_includes_schema_and_judge_answer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "sample.jsonl"
            self._write_dataset(dataset)
            runner = self._make_runner(dataset, root / "outputs", None, judge_answer=True)

            with warnings.catch_warnings(record=True):
                meta = runner._build_evaluation_meta(runner._load_records()[:1])

            self.assertEqual(EVALUATION_SCHEMA_VERSION, meta["schema_version"])
            self.assertTrue(meta["judge_answer"])
