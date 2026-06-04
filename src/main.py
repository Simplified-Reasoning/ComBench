import argparse
from pathlib import Path

from src.eval.evaluator import build_evaluation_suite
from src.eval.runner import EvaluationRunner
from src.models.registry import load_response_model


def _parse_line_numbers(value: str) -> list[int]:
    tokens = [part.strip() for part in value.split(",")]
    if not tokens or any(not token for token in tokens):
        raise argparse.ArgumentTypeError("--line must be a comma-separated list of positive integers")

    line_numbers: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        if not token.isdigit():
            raise argparse.ArgumentTypeError(f"--line contains non-integer token: {token!r}")
        number = int(token)
        if number <= 0:
            raise argparse.ArgumentTypeError(f"--line must be >= 1, got: {number}")
        if number in seen:
            continue
        seen.add(number)
        line_numbers.append(number)
    return line_numbers


def _parse_positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be an integer: {value!r}") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be >= 1, got: {number}")
    return number


def _parse_csv_tokens(value: str) -> list[str]:
    tokens = [part.strip() for part in value.split(",")]
    if not tokens or any(not token for token in tokens):
        raise argparse.ArgumentTypeError("value must be a comma-separated list of non-empty tokens")

    parsed: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        parsed.append(token)
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation on a JSONL dataset.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument(
        "--line",
        type=_parse_line_numbers,
        help="Only process selected dataset line numbers, e.g. 2 or 1,2,3,10 (1-based).",
    )
    parser.add_argument("--model-profile", default="mock", help="Response model profile name")
    parser.add_argument("--profiles-dir", default="profiles", help="Directory with model profiles")
    parser.add_argument("--output-root", default="outputs", help="Output directory root")
    parser.add_argument("--timeout", type=int, default=15, help="Timeout in seconds")
    parser.add_argument("--semaphore", type=int, default=128, help="Concurrency semaphore size")
    parser.add_argument(
        "--repeat",
        type=_parse_positive_int,
        default=1,
        help="Number of independent generation/evaluation attempts per dataset record",
    )
    parser.add_argument(
        "--force-attempt",
        type=_parse_csv_tokens,
        default=[],
        help="Comma-separated attempt ids to regenerate and re-evaluate, ignoring existing cache for those attempts.",
    )
    parser.add_argument("--evaluator-profile", default="gemini", help="Evaluator model profile name")
    parser.add_argument(
        "--judge-answer",
        action="store_true",
        help="Enable answer-equivalence judging in addition to proof/construction judging",
    )
    parser.add_argument(
        "--verbose",
        choices=["none", "summary", "detail"],
        default="none",
        help="Print evaluation results",
    )
    parser.add_argument("--generate", action="store_true", help="Run response generation stage")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation stage")
    parser.add_argument(
        "--no-generate-cache",
        action="store_true",
        help="Disable generation cache lookup/use",
    )
    parser.add_argument(
        "--no-eval-cache",
        action="store_true",
        help="Disable evaluation cache lookup/use",
    )
    return parser.parse_args()


def _build_response_model(profile_name: str, profiles_dir: Path):
    return load_response_model(profile_name, profiles_dir)


def main() -> None:
    args = _parse_args()
    generate = args.generate or not args.evaluate
    evaluate = args.evaluate or not args.generate
    dataset_path = Path(args.dataset)
    output_root = Path(args.output_root)
    profiles_dir = Path(args.profiles_dir)

    response_model = None
    if generate:
        _, response_model = _build_response_model(args.model_profile, profiles_dir)

    evaluation_suite = None
    if evaluate:
        evaluation_suite = build_evaluation_suite(
            profile_name=args.evaluator_profile,
            profiles_dir=profiles_dir,
            timeout=args.timeout,
            semaphore=args.semaphore,
            judge_answer=args.judge_answer,
        )

    runner = EvaluationRunner(
        dataset_path=dataset_path,
        output_root=output_root,
        line_numbers=args.line,
        generation_mode=args.model_profile,
        generation_profile=args.model_profile,
        evaluation_profile=args.evaluator_profile,
        response_model=response_model,
        evaluation_suite=evaluation_suite,
        timeout=args.timeout,
        semaphore=args.semaphore,
        judge_answer=args.judge_answer,
        repeat=args.repeat,
        force_attempt_ids=args.force_attempt,
    )
    run_dir = runner.run(
        generate=generate,
        evaluate=evaluate,
        verbose=args.verbose,
        use_generation_cache=not args.no_generate_cache,
        use_evaluation_cache=not args.no_eval_cache,
    )
    print(f"results written to {run_dir}")


if __name__ == "__main__":
    main()
