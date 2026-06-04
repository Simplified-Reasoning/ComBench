import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import data_builder_cli


class FakeChatClient:
    def __init__(self, responses):
        self._responses = list(responses)

    async def chat(self, messages):
        if not self._responses:
            raise AssertionError("no fake LLM responses remaining")
        return {
            "choices": [
                {
                    "message": {
                        "content": self._responses.pop(0),
                    }
                }
            ]
        }


class VerificationParsingTests(unittest.TestCase):
    def test_parse_verification_result_accepts_valid_decision(self):
        parsed = data_builder_cli._parse_verification_result(
            """
            <decision>step2_issue</decision>
            <summary>Verifier coverage is incomplete.</summary>
            <details>verify_code misses one structural constraint.</details>
            <suggestion>Revise the verifier intent and rerun Step2.</suggestion>
            """
        )

        self.assertEqual(parsed["decision"], "step2_issue")
        self.assertIn("coverage", parsed["summary"])
        self.assertIn("verify_code", parsed["details"])

    def test_parse_verification_result_rejects_invalid_decision(self):
        with self.assertRaises(ValueError):
            data_builder_cli._parse_verification_result(
                """
                <decision>maybe</decision>
                <summary>Unknown.</summary>
                <details>Unknown.</details>
                <suggestion>Unknown.</suggestion>
                """
            )

    def test_build_verification_prompt_contains_all_required_inputs(self):
        prompt = data_builder_cli._build_verification_prompt(
            query="Q",
            ref_answer="A",
            construction_goal="Goal",
            instruction="Inst",
            ref_construction="Construct",
            verify_code_plan="Plan",
            verify_code="print(True)",
        )

        self.assertIn("### Query\nQ", prompt)
        self.assertIn("### Ref_answer\nA", prompt)
        self.assertIn("### Construction_goal\nGoal", prompt)
        self.assertIn("### Instruction\nInst", prompt)
        self.assertIn("### Ref_construction\nConstruct", prompt)
        self.assertIn("### Verifier_design_notes\nPlan", prompt)
        self.assertIn("### Verify_code\nprint(True)", prompt)
        self.assertIn("reason from the original problem first", prompt)
        self.assertIn("Treat `ref_answer` as reliable", prompt)
        self.assertIn("genuinely witness the solved mathematical claim", prompt)
        self.assertIn("enforces only a weaker surrogate", prompt)

    def test_step1_prompt_tells_verifier_plan_not_to_parse_construct_tags(self):
        template = data_builder_cli._read_template_file(data_builder_cli.STEP1_TEMPLATE_FILE)

        self.assertIn("The runtime will pass the verifier only the raw construction payload itself.", template)
        self.assertIn(
            "do not instruct the verifier to extract XML/HTML tags",
            template,
        )

    def test_parse_step1_result_keeps_instruction_as_generated(self):
        parsed = data_builder_cli._parse_step1_result(
            "\n".join(
                [
                    "<instruction>Return one construction.</instruction>",
                    "<ref_construction>[[1, 2], [3, 4]]</ref_construction>",
                    "<verify_code_informal_plan>1. Parse input.\n2. Validate constraints.</verify_code_informal_plan>",
                ]
            )
        )
        self.assertEqual(parsed["instruction"], "Return one construction.")


class ProfileRetryTests(unittest.TestCase):
    @patch("pipeline.data_builder_cli.build_llm_client")
    @patch("pipeline.data_builder_cli.load_profile")
    def test_build_llm_client_from_profile_uses_profile_retry_budget_without_stacking(
        self,
        mock_load_profile,
        mock_build_llm_client,
    ):
        fake_client = object()
        mock_load_profile.return_value = (
            "gemini",
            "llm",
            {
                "model_name": "gemini-test",
                "api_key_env": "OPENAI_API_KEY",
                "base_url": "https://example.com/v1",
                "timeout": 90,
                "max_retries": 8,
            },
        )
        mock_build_llm_client.return_value = fake_client

        profile_name, client, retry_budget = data_builder_cli._build_llm_client_from_profile(
            "gemini"
        )

        self.assertEqual("gemini", profile_name)
        self.assertIs(fake_client, client)
        self.assertEqual(8, retry_budget)
        mock_build_llm_client.assert_called_once_with(
            model_name="gemini-test",
            api_key_env="OPENAI_API_KEY",
            base_url="https://example.com/v1",
            timeout=90,
            max_retries=1,
        )

    @patch("pipeline.data_builder_cli.build_llm_client")
    @patch("pipeline.data_builder_cli.load_profile")
    def test_build_llm_client_from_profile_defaults_retry_budget_to_eight(
        self,
        mock_load_profile,
        mock_build_llm_client,
    ):
        mock_load_profile.return_value = (
            "gemini",
            "llm",
            {
                "model_name": "gemini-test",
                "api_key_env": "OPENAI_API_KEY",
                "base_url": "https://example.com/v1",
            },
        )
        mock_build_llm_client.return_value = object()

        _, _, retry_budget = data_builder_cli._build_llm_client_from_profile("gemini")

        self.assertEqual(8, retry_budget)


class Step2VerificationFlowTests(unittest.TestCase):
    def _write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _make_step2_environment(self, root: Path) -> dict[str, object]:
        step1_input = root / "inputs" / "step1"
        step2_input = root / "inputs" / "step2"
        step3_input = root / "inputs" / "step3"
        prompts_dir = root / "prompts"
        templates_dir = root / "templates"

        self._write(step1_input / "intent.md", "Build one explicit valid construction.\n")
        self._write(step2_input / "id.md", "sample-id\n")
        self._write(step2_input / "grading_guidelines.md", "Grade strictly by instruction contract.\n")
        self._write(step2_input / "query.md", "Sample query\n")
        self._write(step2_input / "ref_answer.md", "Sample answer\n")
        self._write(step2_input / "instruction.md", "Return one construction.\n")
        self._write(step2_input / "ref_construction.md", "[[1, 2], [3, 4]]\n")
        self._write(step2_input / "intent.md", "Check shape and correctness.\n")

        self._write(
            templates_dir / "step2_prompt_template.txt",
            "\n".join(
                [
                    "Q=<<QUERY>>",
                    "A=<<REF_ANSWER>>",
                    "I=<<INSTRUCTION>>",
                    "R=<<REF_CONSTRUCTION>>",
                    "T=<<INTENT>>",
                    "F=<<FEW_SHOT_EXAMPLES>>",
                ]
            )
            + "\n",
        )
        self._write(
            templates_dir / "verification_prompt_template.txt",
            "\n".join(
                [
                    "Q=<<QUERY>>",
                    "A=<<REF_ANSWER>>",
                    "G=<<CONSTRUCTION_GOAL>>",
                    "I=<<INSTRUCTION>>",
                    "R=<<REF_CONSTRUCTION>>",
                    "P=<<VERIFY_CODE_PLAN>>",
                    "V=<<VERIFY_CODE>>",
                ]
            )
            + "\n",
        )

        return {
            "STEP2_INPUT_FILES": {
                "id": step2_input / "id.md",
                "grading_guidelines": step2_input / "grading_guidelines.md",
                "query": step2_input / "query.md",
                "ref_answer": step2_input / "ref_answer.md",
                "instruction": step2_input / "instruction.md",
                "ref_construction": step2_input / "ref_construction.md",
                "intent": step2_input / "intent.md",
            },
            "STEP1_VERIFICATION_INPUT_FILES": {
                "construction_goal": step1_input / "intent.md",
            },
            "STEP3_AUTOFILL_FILES": {
                "id": step3_input / "id.md",
                "grading_guidelines": step3_input / "grading_guidelines.md",
                "query": step3_input / "query.md",
                "ref_answer": step3_input / "ref_answer.md",
                "instruction": step3_input / "instruction.md",
                "ref_construction": step3_input / "ref_construction.md",
                "verify_code": step3_input / "verify_code.py",
            },
            "STEP2_OUTPUT_FILE": prompts_dir / "step2_prompt.md",
            "VERIFICATION_OUTPUT_FILE": prompts_dir / "verification_prompt.md",
            "VERIFICATION_RESULT_FILE": prompts_dir / "verification_result.md",
            "STEP2_TEMPLATE_FILE": templates_dir / "step2_prompt_template.txt",
            "VERIFICATION_TEMPLATE_FILE": templates_dir / "verification_prompt_template.txt",
        }

    def test_handle_step2_writes_step3_files_when_verification_reports_step1_issue(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = self._make_step2_environment(Path(tmp_dir))
            client = FakeChatClient(
                [
                    "```python\nprint('True')\n```",
                    "\n".join(
                        [
                            "<decision>step1_issue</decision>",
                            "<summary>Instruction scope is too narrow.</summary>",
                            "<details>instruction omits one requested condition.</details>",
                            "<suggestion>Fix instruction or ref_construction starting from Step1.</suggestion>",
                        ]
                    ),
                ]
            )

            with patch.multiple(data_builder_cli, **env):
                data_builder_cli._handle_step2([], client, "fake-profile")

            verify_code_path = env["STEP3_AUTOFILL_FILES"]["verify_code"]
            guidelines_path = env["STEP3_AUTOFILL_FILES"]["grading_guidelines"]
            verification_result_path = env["VERIFICATION_RESULT_FILE"]
            self.assertTrue(verify_code_path.exists())
            self.assertIn("print('True')", verify_code_path.read_text(encoding="utf-8"))
            self.assertTrue(guidelines_path.exists())
            self.assertEqual(
                guidelines_path.read_text(encoding="utf-8").strip(),
                "Grade strictly by instruction contract.",
            )
            self.assertTrue(verification_result_path.exists())
            self.assertIn(
                "<decision>step1_issue</decision>",
                verification_result_path.read_text(encoding="utf-8"),
            )

    def test_handle_step2_writes_report_when_verification_reports_step2_issue(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = self._make_step2_environment(Path(tmp_dir))
            client = FakeChatClient(
                [
                    "```python\nprint('True')\n```",
                    "\n".join(
                        [
                            "<decision>step2_issue</decision>",
                            "<summary>Verifier misses one constraint.</summary>",
                            "<details>verify_code does not enforce the final uniqueness rule.</details>",
                            "<suggestion>Revise the verifier intent and rerun Step2.</suggestion>",
                        ]
                    ),
                ]
            )

            with patch.multiple(data_builder_cli, **env):
                data_builder_cli._handle_step2([], client, "fake-profile")

            verification_result_path = env["VERIFICATION_RESULT_FILE"]
            self.assertIn(
                "<decision>step2_issue</decision>",
                verification_result_path.read_text(encoding="utf-8"),
            )

    def test_handle_step2_uses_passed_retry_budget_for_each_stage(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = self._make_step2_environment(Path(tmp_dir))
            client = FakeChatClient([])

            with patch.multiple(data_builder_cli, **env):
                with patch(
                    "pipeline.data_builder_cli._call_llm_with_retry_and_parse",
                    side_effect=["print(True)", None],
                ) as mock_call_parse:
                    with patch(
                        "pipeline.data_builder_cli._call_llm_with_retry_and_parse_with_text",
                        return_value=(
                            "<decision>pass</decision>\n<summary>ok</summary>\n<details>ok</details>\n<suggestion>ok</suggestion>",
                            {
                                "decision": "pass",
                                "summary": "ok",
                                "details": "ok",
                                "suggestion": "ok",
                            },
                        ),
                    ) as mock_call_parse_text:
                        data_builder_cli._handle_step2(
                            [],
                            client,
                            "fake-profile",
                            retry_budget=8,
                        )

        self.assertEqual(1, mock_call_parse.call_count)
        self.assertEqual(8, mock_call_parse.call_args.kwargs["max_retries"])
        self.assertEqual(8, mock_call_parse_text.call_args.kwargs["max_retries"])

    def test_handle_step2_raises_when_step1_intent_for_verification_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            env = self._make_step2_environment(root)
            env["STEP1_VERIFICATION_INPUT_FILES"] = {
                "construction_goal": root / "inputs" / "step1" / "missing_intent.md",
            }

            with patch.multiple(data_builder_cli, **env):
                with self.assertRaises(FileNotFoundError):
                    data_builder_cli._handle_step2([], FakeChatClient([]), "fake-profile")

    def test_handle_step2_raises_when_grading_guidelines_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            env = self._make_step2_environment(root)
            env["STEP2_INPUT_FILES"] = {
                **env["STEP2_INPUT_FILES"],
                "grading_guidelines": root / "inputs" / "step2" / "missing_grading_guidelines.md",
            }

            with patch.multiple(data_builder_cli, **env):
                with self.assertRaises(ValueError):
                    data_builder_cli._handle_step2([], FakeChatClient([]), "fake-profile")


class Step1AutofillFlowTests(unittest.TestCase):
    def _write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _make_step1_environment(self, root: Path) -> dict[str, object]:
        step1_input = root / "inputs" / "step1"
        step2_input = root / "inputs" / "step2"
        prompts_dir = root / "prompts"
        templates_dir = root / "templates"

        self._write(step1_input / "id.md", "sample-id\n")
        self._write(step1_input / "grading_guidelines.md", "Grade strictly by instruction contract.\n")
        self._write(step1_input / "query.md", "Sample query\n")
        self._write(step1_input / "ref_answer.md", "Sample answer\n")
        self._write(step1_input / "intent.md", "Construct one explicit witness.\n")
        self._write(
            templates_dir / "step1_prompt_template.txt",
            "\n".join(
                [
                    "Q=<<QUERY>>",
                    "A=<<REF_ANSWER>>",
                    "T=<<INTENT>>",
                    "F=<<FEW_SHOT_EXAMPLES>>",
                ]
            )
            + "\n",
        )

        return {
            "STEP1_INPUT_FILES": {
                "id": step1_input / "id.md",
                "grading_guidelines": step1_input / "grading_guidelines.md",
                "query": step1_input / "query.md",
                "ref_answer": step1_input / "ref_answer.md",
                "intent": step1_input / "intent.md",
            },
            "STEP2_AUTOFILL_FILES": {
                "id": step2_input / "id.md",
                "grading_guidelines": step2_input / "grading_guidelines.md",
                "query": step2_input / "query.md",
                "ref_answer": step2_input / "ref_answer.md",
                "instruction": step2_input / "instruction.md",
                "ref_construction": step2_input / "ref_construction.md",
                "intent": step2_input / "intent.md",
            },
            "STEP1_OUTPUT_FILE": prompts_dir / "step1_prompt.md",
            "STEP1_TEMPLATE_FILE": templates_dir / "step1_prompt_template.txt",
        }

    def test_handle_step1_copies_grading_guidelines_to_step2(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = self._make_step1_environment(Path(tmp_dir))
            client = FakeChatClient(
                [
                    "\n".join(
                        [
                            "<instruction>Return one construction.</instruction>",
                            "<ref_construction>[[1, 2], [3, 4]]</ref_construction>",
                            "<verify_code_informal_plan>1. Parse input.\n2. Validate constraints.</verify_code_informal_plan>",
                        ]
                    )
                ]
            )

            with patch.multiple(data_builder_cli, **env):
                data_builder_cli._handle_step1([], client, "fake-profile")

            guidelines_path = env["STEP2_AUTOFILL_FILES"]["grading_guidelines"]
            instruction_path = env["STEP2_AUTOFILL_FILES"]["instruction"]
            self.assertTrue(guidelines_path.exists())
            self.assertEqual(
                guidelines_path.read_text(encoding="utf-8").strip(),
                "Grade strictly by instruction contract.",
            )
            self.assertTrue(instruction_path.exists())
            self.assertEqual(
                instruction_path.read_text(encoding="utf-8").strip(),
                "Return one construction.",
            )

    def test_handle_step1_uses_passed_retry_budget(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = self._make_step1_environment(Path(tmp_dir))

            with patch.multiple(data_builder_cli, **env):
                with patch(
                    "pipeline.data_builder_cli._call_llm_with_retry_and_parse",
                    return_value={
                        "instruction": "Return one construction.",
                        "ref_construction": "[[1, 2], [3, 4]]",
                        "verify_code_informal_plan": "Plan",
                    },
                ) as mock_call_parse:
                    data_builder_cli._handle_step1(
                        [],
                        FakeChatClient([]),
                        "fake-profile",
                        retry_budget=8,
                    )

        self.assertEqual(8, mock_call_parse.call_args.kwargs["max_retries"])

    def test_handle_step1_raises_when_grading_guidelines_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            env = self._make_step1_environment(root)
            env["STEP1_INPUT_FILES"] = {
                **env["STEP1_INPUT_FILES"],
                "grading_guidelines": root / "inputs" / "step1" / "missing_grading_guidelines.md",
            }

            with patch.multiple(data_builder_cli, **env):
                with self.assertRaises(ValueError):
                    data_builder_cli._handle_step1([], FakeChatClient([]), "fake-profile")


if __name__ == "__main__":
    unittest.main()
