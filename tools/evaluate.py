import sys
import os

# Add project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


import json
import random
from utils.global_vars import DATASET_NAME, RESULT_JSON_PATH, RESULT_DIR




def evaluate_performance(data, dataset_name):
    if dataset_name in ['LogicalDeduction', 'AR-LSAT']:
        label_counts = {}
        total_records = 0
        valid_options = ['a', 'b', 'c', 'd', 'e'] if dataset_name == "AR-LSAT" else ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        correct_counts = {opt: 0 for opt in valid_options}

        for record in data:
            total_records += 1
            predicted_choice = record.get('predicted_choice', '').strip().lower()
            actual_answer = record.get('answer', '').strip().lower()

            if predicted_choice == "no final answer found in the text.":
                fallback = str(record.get('predicted_answer', '')).strip().lower()
                if fallback in valid_options:
                    predicted_choice = fallback
                else:
                    original = record.get('original_answer', '').strip().lower()
                    if original in valid_options:
                        predicted_choice = original
                    else:
                        predicted_choice = random.choice(valid_options)

            label_counts[actual_answer] = label_counts.get(actual_answer, 0) + 1
            if predicted_choice == actual_answer:
                if actual_answer in valid_options:
                    correct_counts[actual_answer] += 1

        correct_total = sum(correct_counts.values())
        accuracy = (correct_total / total_records) * 100 if total_records > 0 else 0

        return {
            "total": total_records,
            "label_counts": label_counts,
            "correct_counts": correct_counts,
            "correct_total": correct_total,
            "accuracy": accuracy
        }

    else:
        correct_true = 0
        correct_false = 0
        correct_uncertain = 0
        total_true = 0
        total_false = 0
        total_uncertain = 0
        total_records = 0

        for record in data:
            total_records += 1
            predicted_answer = record.get('predicted_choice', '').strip().lower()
            actual_answer = record.get('answer', '').strip().lower()

            # Map "True"/"False"/"uncertain" → "a"/"b"/"c"
            if predicted_answer in ["true", "false", "uncertain"]:
                predicted_answer = {"true": "a", "false": "b", "uncertain": "c"}[predicted_answer]

            # Fallback: for prediction failure cases ("No final answer found...")
            if predicted_answer == "" or predicted_answer == "no final answer found in the text.":
                fallback = str(record.get('predicted_answer', '')).strip().lower()

                if fallback in ["true", "false", "uncertain"]:
                    predicted_answer = {"true": "a", "false": "b", "uncertain": "c"}[fallback]
                else:
                    predicted_answer = record.get('original_answer', '').strip().lower()
                    if predicted_answer in ["true", "false", "uncertain"]:
                        predicted_answer = {"true": "a", "false": "b", "uncertain": "c"}[predicted_answer]


            if actual_answer == 'a':
                total_true += 1
            elif actual_answer == 'b':
                total_false += 1
            elif actual_answer == 'c':
                total_uncertain += 1

            if predicted_answer == actual_answer:
                if actual_answer == 'a':
                    correct_true += 1
                elif actual_answer == 'b':
                    correct_false += 1
                elif actual_answer == 'c':
                    correct_uncertain += 1

        correct_total = correct_true + correct_false + correct_uncertain
        accuracy = (correct_total / total_records) * 100 if total_records > 0 else 0

        return {
            "total": total_records,
            "gt_true": total_true,
            "gt_false": total_false,
            "gt_uncertain": total_uncertain,
            "correct_true": correct_true,
            "correct_false": correct_false,
            "correct_uncertain": correct_uncertain,
            "correct_total": correct_total,
            "accuracy": accuracy
        }

def evaluate(result_path):
    print("📄 Result file:", result_path)
    with open(result_path, 'r', encoding='utf-8') as f:
        # Read .jsonl file line by line
        data = [json.loads(line) for line in f if line.strip()]

    stats = evaluate_performance(data, DATASET_NAME)

    print(f"\n🔢 Total records: {stats['total']}\n")

    if DATASET_NAME in ['LogicalDeduction', 'AR-LSAT']:
        print("🧾 Ground Truth Distribution:")
        for label, count in sorted(stats['label_counts'].items()):
            print(f"   - {label.upper()}: {count}")

        print("\n✅ Correct Predictions:")
        for label, count in sorted(stats['correct_counts'].items()):
            print(f"   - {label.upper()}: {count}")
    else:
        print("🧾 Ground Truth Distribution:")
        print(f"   - TRUE (a): {stats['gt_true']}")
        print(f"   - FALSE (b): {stats['gt_false']}")
        if stats['gt_uncertain'] > 0:
            print(f"   - uncertain (c): {stats['gt_uncertain']}")

        print("\n✅ Correct Predictions:")
        print(f"   - TRUE (a): {stats['correct_true']}")
        print(f"   - FALSE (b): {stats['correct_false']}")
        if stats['correct_uncertain'] > 0:
            print(f"   - uncertain (c): {stats['correct_uncertain']}")

    print(f"\n🎯 Total Correct Predictions: {stats['correct_total']}")
    print(f"📊 Accuracy: {stats['accuracy']:.2f}%")

    # Save evaluation results
    os.makedirs(RESULT_DIR, exist_ok=True)
    eval_output_path = os.path.join(RESULT_DIR, f"{DATASET_NAME}_evaluate_result.jsonl")
    with open(eval_output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n📝 Evaluation result written to: {eval_output_path}")

if __name__ == "__main__":
    evaluate(RESULT_JSON_PATH)
