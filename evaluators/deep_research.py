"""Deep-research (multi-hop) evaluation type.

Adapted from "From Simple QA to Deep Research: A Verifiable Benchmark
Constructed through Iterative Task Evolution" (arXiv:2608.02163).

Core mechanism kept from the paper
----------------------------------
A simple seed question is evolved into a *deep-research task* represented
as a DAG of atomic steps, each carrying a fact-grounded *checkpoint* (the
expected intermediate answer). Steps are resolved in dependency order:
each step's query is formed by substituting the answers resolved by its
predecessors, and every checkpoint is graded pointwise. A task counts as
solved only when *all* of its checkpoints are met. A step whose required
predecessor produced no answer is *blocked*, modelling the DAG dependency
semantics the paper relies on to discriminate models on multi-hop tasks.

This is a Mode 2 (adapted port). The auxiliary components below were
substituted with target-native equivalents; the core (evolve-a-seed-into-
a-checkpoint-DAG, resolve + grade pointwise) is kept at fidelity:

  * The paper's automatic Explorer-Formalizer-Challenger LLM evolution
    pipeline is replaced by a small catalog of *manually evolved* tasks
    seeded from SimpleQA-style questions. The SPEC's suggested experiment
    is literally "manually evolve it"; the catalog shipped here is the
    *output* of that evolution (the tasks + their checkpoint DAGs), which
    is the paper's actual deliverable. The deterministic ``render_question``
    substitution stands in for the runtime answer-forwarding the paper's
    pipeline performs between atomic steps.
  * The paper's separate benchmark/eval framework is cut entirely. This
    runs through the repo's existing ``EvaluationType`` dispatch and reuses
    the existing ``CorrectnessEvaluator`` as the per-checkpoint grader and
    ``PostProcessor.extract_answer`` for answer extraction -- mirroring how
    ``document_relevance`` already uses a non-CorrectnessEvaluator scorer.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from evaluators.correctness_evaluator import CorrectnessConfig, CorrectnessEvaluator
from utils import EvaluationType, PostProcessor, save_result

logger = logging.getLogger(__name__)


@dataclass
class ResearchStep:
    """One atomic step in a deep-research task DAG.

    ``question_template`` may reference resolved answers of predecessor
    steps with ``{step_id}`` placeholders, which ``render_question``
    substitutes at resolve time.
    """

    step_id: str
    question_template: str
    checkpoint: str
    depends_on: List[str] = field(default_factory=list)


@dataclass
class DeepResearchTask:
    """A seed question evolved into a checkpoint DAG."""

    task_id: str
    seed_question: str
    seed_answer: str
    steps: List[ResearchStep]
    topic: str = ""
    index: int = 0

    def topo_order(self) -> List[ResearchStep]:
        """Return steps ordered so every dependency precedes its dependents.

        Preserves input order among ready steps for determinism. Steps on a
        cycle (or with a missing dependency) are appended in input order
        rather than dropped, so grading still produces a row for them.
        """
        ordered: List[ResearchStep] = []
        done = set()
        while len(ordered) < len(self.steps):
            progressed = False
            for step in self.steps:
                if step.step_id in done:
                    continue
                if all(dep in done for dep in step.depends_on):
                    ordered.append(step)
                    done.add(step.step_id)
                    progressed = True
            if not progressed:
                for step in self.steps:
                    if step.step_id not in done:
                        ordered.append(step)
                        done.add(step.step_id)
                break
        return ordered


def render_question(step: ResearchStep, resolved: Dict[str, str]) -> str:
    """Substitute resolved predecessor answers into a step's query template."""
    query = step.question_template
    for step_id, answer in resolved.items():
        query = query.replace("{" + step_id + "}", answer)
    return query


# Manually-evolved deep-research tasks. Each starts from a SimpleQA-style
# seed and chains dependent sub-questions whose checkpoints are well
# established, so the pointwise grader has reliable gold to grade against.
DEEP_RESEARCH_TASKS: List[DeepResearchTask] = [
    DeepResearchTask(
        task_id="mona_lisa",
        seed_question="Who painted the Mona Lisa?",
        seed_answer="Leonardo da Vinci",
        topic="Art and history",
        steps=[
            ResearchStep("s1", "Who painted the Mona Lisa?", "Leonardo da Vinci"),
            ResearchStep(
                "s2", "In which year was {s1} born?", "1452", depends_on=["s1"]
            ),
        ],
    ),
    DeepResearchTask(
        task_id="hamlet",
        seed_question="Who wrote the play Hamlet?",
        seed_answer="William Shakespeare",
        topic="Literature",
        steps=[
            ResearchStep("s1", "Who wrote the play Hamlet?", "William Shakespeare"),
            ResearchStep(
                "s2", "In which year was {s1} born?", "1564", depends_on=["s1"]
            ),
        ],
    ),
    DeepResearchTask(
        task_id="relativity",
        seed_question="Who developed the theory of general relativity?",
        seed_answer="Albert Einstein",
        topic="Science and technology",
        steps=[
            ResearchStep(
                "s1", "Who developed the theory of general relativity?", "Albert Einstein"
            ),
            ResearchStep(
                "s2",
                "In which year did {s1} publish his paper on general relativity?",
                "1915",
                depends_on=["s1"],
            ),
            ResearchStep(
                "s3", "In which country was {s1} born?", "Germany", depends_on=["s1"]
            ),
        ],
    ),
    DeepResearchTask(
        task_id="first_president",
        seed_question="Who was the first president of the United States?",
        seed_answer="George Washington",
        topic="History",
        steps=[
            ResearchStep(
                "s1", "Who was the first president of the United States?", "George Washington"
            ),
            ResearchStep(
                "s2",
                "In which year did {s1} take office as president?",
                "1789",
                depends_on=["s1"],
            ),
            ResearchStep(
                "s3",
                "Who served as vice president under {s1}?",
                "John Adams",
                depends_on=["s1"],
            ),
        ],
    ),
]


def load_deep_research_tasks(
    start_index: int = 0,
    end_index: Optional[int] = None,
    random_sample: Optional[int] = None,
) -> List[DeepResearchTask]:
    """Load the evolved deep-research catalog, applying slice/sample selectors.

    Mirrors the (start_index, end_index, random_sample) contract of
    ``load_csv_data`` so the existing CLI flags drive this evaluation type.
    Each returned task carries an ``index`` for result-bookkeeping.
    """
    tasks = list(DEEP_RESEARCH_TASKS)
    total = len(tasks)

    if random_sample is not None and random_sample > 0:
        size = min(random_sample, total)
        tasks = random.sample(tasks, size)
        for i, task in enumerate(tasks):
            task.index = i
    else:
        if end_index is None:
            end_index = total
        if total:
            start_index = max(0, min(start_index, total - 1))
            end_index = max(start_index + 1, min(end_index, total))
        tasks = tasks[start_index:end_index]
        for i, task in enumerate(tasks):
            task.index = start_index + i

    return tasks


async def evaluate_provider_deep_research(
    provider_name: str,
    search_handler,
    tasks: List[DeepResearchTask],
    post_processor: Optional[PostProcessor] = None,
    evaluator_model: str = "gpt-4.1",
    batch_size: int = 3,
    evaluator: Optional[CorrectnessEvaluator] = None,
    output_dir: Optional[str] = None,
    evaluation_type: EvaluationType = EvaluationType.DEEP_RESEARCH,
) -> Dict:
    """Evaluate a single search provider on the deep-research catalog.

    For each task the checkpoint DAG is resolved in dependency order: a
    step's query is rendered with the answers its predecessors resolved,
    the provider searches + extracts an answer (reusing the SimpleQA path),
    and the existing ``CorrectnessEvaluator`` grades it against the step's
    checkpoint. A task is solved only when every checkpoint is met; a step
    is *blocked* when a required predecessor produced no answer.

    ``evaluator`` and ``output_dir`` are injectable so the resolve-and-grade
    loop is exercisable without network access or API keys.
    """
    if evaluator is None:
        evaluator = CorrectnessEvaluator(CorrectnessConfig(model_name=evaluator_model))

    results: List[Dict] = []
    tasks_passed = 0

    async def process_task(task: DeepResearchTask) -> None:
        nonlocal tasks_passed
        resolved: Dict[str, str] = {}
        step_rows: List[Dict] = []

        for step in task.topo_order():
            blocked = any(
                dep not in resolved or not resolved[dep] for dep in step.depends_on
            )
            query = render_question(step, resolved)

            if blocked:
                row = {
                    "index": task.index,
                    "task_id": task.task_id,
                    "seed_question": task.seed_question,
                    "step_id": step.step_id,
                    "step_question": query,
                    "checkpoint": step.checkpoint,
                    "predicted_answer": "",
                    "is_correct": False,
                    "grade": "BLOCKED",
                    "blocked": True,
                    "token_count": 0,
                    "token_avg": 0,
                }
                step_rows.append(row)
                if output_dir:
                    save_result(row, provider_name, output_dir, evaluation_type)
                continue

            search_result = await search_handler.search(query)
            original_answer = search_result.get("answer", "")
            is_llm_response = getattr(search_handler, "is_llm_response", False)
            token_count = 0
            token_avg = 0
            if is_llm_response:
                search_ans = original_answer
            else:
                search_ans, token_count, token_avg = await search_handler.post_process(
                    search_result
                )

            if post_processor is not None:
                answer = post_processor.extract_answer(
                    query=query,
                    is_llm_response=is_llm_response,
                    search_result=search_ans,
                )
            else:
                answer = search_ans

            evaluation_result = await evaluator.evaluate(
                {"question": query},
                {"answer": answer},
                {"answer": step.checkpoint},
            )
            is_correct = evaluation_result["score"] == 1.0
            resolved[step.step_id] = answer

            row = {
                "index": task.index,
                "task_id": task.task_id,
                "seed_question": task.seed_question,
                "step_id": step.step_id,
                "step_question": query,
                "checkpoint": step.checkpoint,
                "predicted_answer": answer,
                "is_correct": is_correct,
                "grade": evaluation_result["value"],
                "blocked": False,
                "token_count": token_count,
                "token_avg": token_avg,
            }
            step_rows.append(row)
            if output_dir:
                save_result(row, provider_name, output_dir, evaluation_type)
            logger.info(
                f"[{provider_name}] task {task.task_id} step {step.step_id}: "
                f"{evaluation_result['value']}"
            )

        if step_rows and all(r["is_correct"] for r in step_rows):
            tasks_passed += 1
        results.extend(step_rows)

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        await asyncio.gather(*[process_task(task) for task in batch])
        if output_dir:
            time.sleep(3.0)  # avoid rate limiting on real runs

    checkpoint_total = len(results)
    checkpoint_correct = sum(1 for r in results if r["is_correct"])
    total_tasks = len(tasks)
    task_accuracy = round(tasks_passed / total_tasks, 3) if total_tasks else 0
    step_accuracy = (
        round(checkpoint_correct / checkpoint_total, 3) if checkpoint_total else 0
    )

    return {
        "provider": provider_name,
        "results": results,
        "task_accuracy": task_accuracy,
        "step_accuracy": step_accuracy,
        "tasks_passed": tasks_passed,
        "total_tasks": total_tasks,
    }
