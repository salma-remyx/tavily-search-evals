"""Generate a task-specific exam and (optionally) score providers on it.

This is the entry point that wires the auto-rag-eval exam generator (adapted
from arXiv:2405.13622) into this repo's existing SimpleQA evaluation pipeline.

Default behavior -- generate a drop-in dataset::

    python generate_exam.py --corpus path/to/docs --topic "My topic" --n_questions 20

produces ``datasets/<topic>_exam.csv`` in the exact SimpleQA schema, so the next
``python run_evaluation.py`` run can be pointed at it (or it can be evaluated
in-process with ``--evaluate``).

With ``--evaluate`` the generated exam is scored by the EXISTING pipeline --
``evaluate_provider_simple_qa`` + ``CorrectnessEvaluator`` are reused wholesale;
only the dataset is swapped for the generated one. See the module docstring in
``utils/exam_generator.py`` for what is ported vs. substituted.
"""

import argparse
import asyncio
import json
import logging
import os

from dotenv import load_dotenv

from utils import EvaluationType, get_output_dir, load_csv_data, prepare_examples
from utils.exam_generator import ExamGenerator, ExamGeneratorConfig

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(_handler)


def _default_output_path(topic: str) -> str:
    slug = (topic or "generated").strip().lower().replace(" ", "_") or "generated"
    return os.path.join("datasets", f"{slug}_exam.csv")


def generate(
    corpus_path: str,
    topic: str,
    n_questions: int,
    model: str,
    output: str,
    filter_items: bool,
) -> str:
    """Generate an exam from a corpus and write a SimpleQA-format CSV.

    Returns the path to the written CSV.
    """
    corpus = ExamGenerator.load_corpus(corpus_path)
    generator = ExamGenerator(ExamGeneratorConfig(model_name=model))
    items = generator.generate_exam(
        documents=corpus,
        n_questions=n_questions,
        topic=topic,
        filter_items=filter_items,
    )
    if not items:
        raise RuntimeError(
            "No exam items survived the quality gate; relax it with --no_filter "
            "or provide a richer corpus."
        )
    return ExamGenerator.write_simple_qa_csv(items, output, topic=topic)


async def _evaluate_generated_exam(csv_path: str, config_path: str, model: str) -> None:
    """Score providers on the generated exam using the existing pipeline.

    Reuses ``evaluate_provider_simple_qa`` + ``CorrectnessEvaluator`` from
    ``run_evaluation`` unchanged. Those read ``output_dir`` / ``evaluation_type``
    as module globals, so we set them here before invoking.
    """
    # Imported lazily so generation works without search-provider keys installed.
    import run_evaluation  # noqa: WPS433 (local import is intentional)

    with open(config_path, "r") as fh:
        search_provider_params = json.load(fh)

    examples = load_csv_data(csv_path)
    examples = prepare_examples(
        examples,
        list(search_provider_params.keys()),
        rerun=False,
        results_dir="results",
        evaluation_type=EvaluationType.SIMPLEQA,
    )

    handlers = await run_evaluation.get_search_handlers(search_provider_params, token_model=model)
    output_dir = get_output_dir(EvaluationType.SIMPLEQA, output_dir="results")
    os.makedirs(output_dir, exist_ok=True)

    # evaluate_provider_simple_qa resolves these as module globals.
    run_evaluation.evaluation_type = EvaluationType.SIMPLEQA
    run_evaluation.output_dir = output_dir

    from utils import PostProcessor  # noqa: WPS433

    post_processor = PostProcessor(llm_model=model)
    for handler, provider_name in zip(handlers, search_provider_params.keys()):
        result = await run_evaluation.evaluate_provider_simple_qa(
            provider_name,
            handler,
            examples[provider_name],
            post_processor,
            evaluator_model=model,
        )
        print(f"[{provider_name}] accuracy: {result['accuracy']:.2%} "
              f"({result['correct_count']}/{result['total_count']})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a task-specific exam (auto-rag-eval) and optionally score providers on it.",
    )
    parser.add_argument("--corpus", required=True, help="Path to a corpus file or directory of text/markdown docs.")
    parser.add_argument("--topic", default="", help="Task/topic label for the generated exam.")
    parser.add_argument("--n_questions", type=int, default=10, help="Number of questions to generate (default: 10).")
    parser.add_argument("--model", default="gpt-4.1", help="Model for exam generation (default: gpt-4.1).")
    parser.add_argument("--output", default=None, help="Output CSV path (default: datasets/<topic>_exam.csv).")
    parser.add_argument("--no_filter", action="store_true", help="Disable the item-quality gate.")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="After generating, score providers on the exam using the existing SimpleQA pipeline.",
    )
    parser.add_argument("--config", default="configs/config.json", help="Provider config for --evaluate.")
    args = parser.parse_args()

    output = args.output or _default_output_path(args.topic)
    csv_path = generate(
        corpus_path=args.corpus,
        topic=args.topic,
        n_questions=args.n_questions,
        model=args.model,
        output=output,
        filter_items=not args.no_filter,
    )
    print(f"Wrote generated exam to {csv_path}")

    if args.evaluate:
        asyncio.run(_evaluate_generated_exam(csv_path, args.config, args.model))


if __name__ == "__main__":
    main()
