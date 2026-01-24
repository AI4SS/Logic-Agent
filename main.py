import os
import json
import asyncio

from tqdm.asyncio import tqdm
from metagpt.schema import Message
from role.phil import ReasoningAgent
import utils.global_vars as global_vars


async def main():
    print(f"[INFO] Running PhilAgent with dataset: {global_vars.DATASET_NAME}")

    # Load dataset
    with open(global_vars.DATASET_PATH, "r", encoding="utf-8") as f:
        global_vars.dataset = json.load(f)

    print(f"[INFO] Loaded {global_vars.DATASET_NAME} with {len(global_vars.dataset)} samples")

    # Create save directory if it doesn't exist
    os.makedirs(global_vars.RESULT_DIR, exist_ok=True)

    # Output path
    save_path = global_vars.RESULT_JSON_PATH

    # Clear old file
    with open(save_path, "w", encoding="utf-8"):
        pass

    # Progress bar with tqdm
    for i, sample in enumerate(tqdm(global_vars.dataset, desc="Processing Samples", unit="sample")):
        global_vars.current_sample_id = sample.get("id", f"sample_{i}")
        max_retries = 10
        retry_count = 0

        while retry_count <= max_retries:
            agent = ReasoningAgent(dataset_name=global_vars.DATASET_NAME)
            msg = Message(content=json.dumps(sample, ensure_ascii=False))

            try:
                result = await agent.run(msg)

                if isinstance(result, dict) and "error" in result:
                    print(f"[✗] Sample {sample['id']} failed with error: {result['error']}")
                else:
                    print(f"[✓] Sample {sample['id']} processed successfully.")

                with open(save_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                break  # Break retry loop after success

            except Exception as e:
                print(f"[✗] Exception in sample {sample['id']}: {type(e).__name__}: {e}")
                retry_count += 1

                if retry_count > max_retries:
                    print(f"[!] Exceeded retry limit for sample {sample['id']}")
                    error_result = {
                        "id": sample["id"],
                        "error": f"Exception after {max_retries} retries: {type(e).__name__}: {e}"
                    }
                    with open(save_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                    break
                else:
                    print(f"[↻] Retrying sample {sample['id']} (attempt {retry_count}/{max_retries})...")



if __name__ == "__main__":
    asyncio.run(main())
