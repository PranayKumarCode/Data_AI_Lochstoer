import asyncio
import json
import os
from typing import Dict, List, Any, Iterable

from datasets import load_dataset
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

CONFIG_PATH = "ai_config.json"


def load_training_plan() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"{CONFIG_PATH} not found. Create it with dataset_sources before running teach_ai.py."
        )
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def save_examples_with_context(context_snippets: List[str]) -> None:
    """Attach the supplied context snippets to the most recent example."""
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    if config.get("examples"):
        config["examples"][-1]["search_results"] = context_snippets[:5]
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def resolve_field(column_names: List[str], overrides: Dict[str, Any], key: str, keywords: List[str]):
    override_value = overrides.get(key)
    if isinstance(override_value, list):
        override_value = override_value[0] if override_value else None
    if override_value:
        return override_value
    lower_columns = {name.lower(): name for name in column_names}
    for keyword in keywords:
        for lower_name, original_name in lower_columns.items():
            if keyword in lower_name:
                return original_name
    return None


def build_context(row: Dict[str, Any], fields: Iterable[str]) -> List[str]:
    snippets = []
    for field in fields:
        if not field:
            continue
        value = row.get(field)
        if value is None or value == "":
            continue
        if isinstance(value, list):
            snippets.extend([f"{field}: {str(v)}" for v in value if v])
        elif isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if sub_val:
                    snippets.append(f"{field}.{sub_key}: {sub_val}")
        else:
            snippets.append(f"{field}: {value}")
    return snippets[:5]


def iter_dataset_examples(source: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    dataset = load_dataset(source["name"], source.get("subset"))
    split_name = source.get("split", "train")
    dataset_split = dataset[split_name] if isinstance(dataset, dict) else dataset

    max_examples = min(source.get("max_examples", 50), len(dataset_split))
    overrides = source.get("field_overrides", {})

    question_field = resolve_field(
        dataset_split.column_names,
        overrides,
        "question",
        ["question", "prompt", "problem", "query"]
    )
    answer_field = resolve_field(
        dataset_split.column_names,
        overrides,
        "answer",
        ["answer", "solution", "response", "label"]
    )
    explanation_field = resolve_field(
        dataset_split.column_names,
        overrides,
        "explanation",
        ["explanation", "rationale", "justification", "context", "text"]
    )

    context_fields = overrides.get("context")
    if isinstance(context_fields, str):
        context_fields = [context_fields]
    elif not context_fields:
        context_fields = []

    if explanation_field and explanation_field not in context_fields:
        context_fields.append(explanation_field)

    if not question_field or not answer_field:
        print(f"Skipping dataset {source['name']}: unable to identify question/answer fields.")
        return

    for idx, row in enumerate(dataset_split):
        if idx >= max_examples:
            break

        question = row.get(question_field)
        answer = row.get(answer_field)
        explanation = row.get(explanation_field) if explanation_field else None

        if not question or not answer:
            continue

        explanation_text = (
            explanation
            if isinstance(explanation, str) and explanation.strip()
            else f"Answer sourced from {source['name']}."
        )

        contexts = build_context(row, context_fields or [])
        if not contexts:
            contexts = [f"{question_field}: {question}", f"{answer_field}: {answer}"]

        yield {
            "question": question.strip(),
            "answer": answer.strip() if isinstance(answer, str) else str(answer),
            "explanation": explanation_text.strip(),
            "contexts": contexts
        }


async def teach_with_curated_datasets():
    """Teach the AI using lecture-aligned Hugging Face datasets."""
    training_plan = load_training_plan()
    dataset_sources = training_plan.get("dataset_sources", [])

    if not dataset_sources:
        print("No dataset_sources configured in ai_config.json. Nothing to teach.")
        return

    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server_v2.py"],
        env=None
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("=== Lecture & Dataset Training ===\n")
            total_examples = 0

            for source in dataset_sources:
                dataset_name = source["name"]
                print(f"→ Loading {dataset_name} ({source.get('split', 'train')} split)")
                try:
                    example_iter = list(iter_dataset_examples(source))
                except Exception as exc:
                    print(f"  ✗ Failed to prepare {dataset_name}: {exc}")
                    continue

                if not example_iter:
                    print(f"  ⚠ No usable rows found for {dataset_name}")
                    continue

                for idx, example in enumerate(example_iter, 1):
                    teach_result = await session.call_tool(
                        "teach_example",
                        arguments={
                            "question": example["question"],
                            "correct_answer": example["answer"],
                            "explanation": example["explanation"]
                        }
                    )

                    response = json.loads(teach_result.content[0].text)
                    if response.get("status") == "success":
                        save_examples_with_context(example["contexts"])
                        total_examples += 1
                        print(f"  ✓ [{idx}/{len(example_iter)}] {example['question'][:80]}...")
                    else:
                        print(f"  ✗ Failed to teach example: {response.get('message')}")

            print("\n" + "=" * 50)
            print(f"Training complete. Stored {total_examples} curated examples.")
            print("The assistant now learns from lecture PDFs and Hugging Face datasets only.")
            print("=" * 50)


if __name__ == "__main__":
    asyncio.run(teach_with_curated_datasets())