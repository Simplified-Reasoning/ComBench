import unittest

from src.llm.prompts import (
    build_answer_eval_messages,
    build_proof_eval_messages,
    build_response_prompt,
)


class PromptTests(unittest.TestCase):
    def test_build_construction_response_prompt_requires_q1_q2_sections(self):
        prompt = build_response_prompt("Solve x", "Return a matrix", mode="construction")

        self.assertIn("## Question 1", prompt)
        self.assertIn("Problem:\nSolve x", prompt)
        self.assertIn('## Question 2\n"""\nReturn a matrix\n"""', prompt)
        self.assertIn("## Solution to Question 1", prompt)
        self.assertIn("## Solution to Question 2", prompt)
        self.assertIn("Put exactly one <construct>...</construct> block in Solution to Question 2.", prompt)

    def test_build_plain_response_prompt_uses_single_solution_section(self):
        prompt = build_response_prompt("Prove x", "", mode="plain")

        self.assertIn("## Problem", prompt)
        self.assertIn("Problem:\nProve x", prompt)
        self.assertIn("\n## Solution\n[Your complete solution, including the final answer if applicable.]", prompt)
        self.assertIn("Do not use any <construct>...</construct> block.", prompt)

    def test_build_answer_eval_messages_only_includes_solution_text(self):
        messages = build_answer_eval_messages(
            query="What is 1+1?",
            student_answer="The answer is 2.",
            ref_answer="2",
        )

        user_message = messages[0]["content"]
        self.assertIn("**PROBLEM STATEMENT**\nWhat is 1+1?", user_message)
        self.assertIn("**PROPOSED SOLUTION**\nThe answer is 2.", user_message)
        self.assertIn("**GOLDEN ANSWER**\n2", user_message)
        self.assertNotIn("## Instruction", user_message)

    def test_build_proof_eval_messages_omits_ground_truth_section_when_ref_solution_missing(self):
        messages = build_proof_eval_messages(
            query="Prove something",
            student_answer="A proof attempt",
            grading_guidelines="Give 1 point for lemma X.",
            ref_solution=None,
        )

        user_message = messages[0]["content"]
        self.assertIn("**PROBLEM STATEMENT**\nProve something", user_message)
        self.assertIn("**SPECIFIC GRADING GUIDELINES**\nGive 1 point for lemma X.", user_message)
        self.assertIn("**PROPOSED SOLUTION**\nA proof attempt", user_message)
        self.assertNotIn("**GROUND-TRUTH SOLUTION**", user_message)
