from typing import Optional
import json


def format_output_entry(
    original_data: dict,
    a1_result: dict,
    a3_result: dict,
    a4_result: Optional[dict],
    a5_result: dict,
    token_usage: Optional[dict] = None
) -> dict:
    """
    Construct a standard sample result entry for output to unified JSON file.
    """

    # Step 1: Construct basic fields (insert a1 and a3 first)
    output = {
        "id": original_data.get("id", ""),
        "question": original_data.get("question", ""),
        "context": original_data.get("context", ""),
        "a1": a1_result,
        "a3": a3_result,
    }

    # Step 2: Insert a4 (should be after a3, before a5)
    if a4_result is not None:
        output["a4"] = a4_result

    # Step 3: Insert a5 and prediction result
    output["a5"] = a5_result
    output["predicted_answer"] = a5_result.get("s1_truth", "Uncertain")
    output["answer"] = original_data.get("answer", "")

    # Step 4: Insert token usage record
    if token_usage:
        output["token_usage"] = {
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0)
        }

    return output
