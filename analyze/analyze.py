#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze.py: Analyze complexity and semantic diversity for dev.json in each dataset under the data directory.
Usage:
    python analyze.py [--data_dir path/to/data]
Defaults to './data'.
"""
import os
import json
import statistics
import math
import re
import argparse
from collections import Counter

from textstat import flesch_kincaid_grade, avg_sentence_length

# List of logical connectives
LOGICAL_CONNECTORS = ['every', 'each', 'all', 'some', 'not', 'are', 'is']


def calculate_ttr(text):
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def calculate_mtld(text, threshold=0.72):
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 50:
        return calculate_ttr(text) * 50
    factors = 0
    start = 0
    for i in range(len(words)):
        segment = words[start:i+1]
        if not segment:
            continue
        ttr = len(set(segment)) / len(segment)
        if ttr <= threshold:
            factors += 1
            start = i + 1
    if factors == 0:
        return float(len(words))
    return len(words) / factors


def calculate_entropy(text):
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    counts = Counter(words)
    total = len(words)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def calculate_simpson_index(text):
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    counts = Counter(words)
    total = len(words)
    simpson = sum((c / total) ** 2 for c in counts.values())
    return 1.0 - simpson


def calculate_unique_bigrams_ratio(text):
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 2:
        return 0.0
    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    unique = set(bigrams)
    return len(unique) / len(bigrams)


def extract_fields(item):
    """Try to extract context and question fields from item"""
    context = item.get('context') or item.get('passage') or item.get('premise') or ''
    question = item.get('question') or item.get('query') or item.get('question_text') or ''
    return context, question


def analyze_dataset(dev_path):
    with open(dev_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"\nDataset: {os.path.basename(os.path.dirname(dev_path))}")
    print(f"Loaded dev.json with {len(data)} samples.")

    contexts, questions = [], []
    for item in data:
        ctx, qry = extract_fields(item)
        contexts.append(ctx)
        questions.append(qry)

    # Complexity metrics
    context_fk = []
    question_fk = []
    context_sl = []
    question_sl = []
    context_conn = []
    for ctx in contexts:
        try:
            context_fk.append(flesch_kincaid_grade(ctx))
        except:
            context_fk.append(0.0)
        try:
            context_sl.append(avg_sentence_length(ctx))
        except:
            context_sl.append(0.0)
        words = ctx.lower().split()
        context_conn.append(sum(1 for w in words if w in LOGICAL_CONNECTORS))
    for qry in questions:
        try:
            question_fk.append(flesch_kincaid_grade(qry))
        except:
            question_fk.append(0.0)
        try:
            question_sl.append(avg_sentence_length(qry))
        except:
            question_sl.append(0.0)

    # Semantic diversity metrics
    ctx_ttr = [calculate_ttr(c) for c in contexts]
    ctx_mtld = [calculate_mtld(c) for c in contexts]
    ctx_ent = [calculate_entropy(c) for c in contexts]
    ctx_simp = [calculate_simpson_index(c) for c in contexts]
    ctx_big = [calculate_unique_bigrams_ratio(c) for c in contexts]
    qry_ttr = [calculate_ttr(q) for q in questions]
    qry_ent = [calculate_entropy(q) for q in questions]

    # Output
    print("\n=== COMPLEXITY ANALYSIS ===")
    print(f"Average Context Flesch-Kincaid Grade: {statistics.mean(context_fk):.2f}")
    print(f"Average Question Flesch-Kincaid Grade: {statistics.mean(question_fk):.2f}")
    print(f"Average Context Sentence Length: {statistics.mean(context_sl):.2f}")
    print(f"Average Question Sentence Length: {statistics.mean(question_sl):.2f}")
    print(f"Average Logical Connectors per Context: {statistics.mean(context_conn):.2f}")

    print("\n=== SEMANTIC DIVERSITY ANALYSIS ===")
    print("Context Semantic Diversity:")
    print(f"  Average TTR: {statistics.mean(ctx_ttr):.3f}")
    print(f"  Average MTLD: {statistics.mean(ctx_mtld):.2f}")
    print(f"  Average Entropy: {statistics.mean(ctx_ent):.3f}")
    print(f"  Average Simpson Index: {statistics.mean(ctx_simp):.3f}")
    print(f"  Average Unique Bigrams Ratio: {statistics.mean(ctx_big):.3f}")

    print("\nQuestion Semantic Diversity:")
    print(f"  Average TTR: {statistics.mean(qry_ttr):.3f}")
    print(f"  Average Entropy: {statistics.mean(qry_ent):.3f}")

    # Additional statistics
    all_words = []
    for ctx in contexts:
        all_words.extend(re.findall(r"\b\w+\b", ctx.lower()))
    vocab = len(set(all_words))
    total_tok = len(all_words)
    overall_ttr = vocab / total_tok if total_tok > 0 else 0
    print("\n=== ADDITIONAL STATISTICS ===")
    print(f"Total vocabulary size in contexts: {vocab}")
    print(f"Total tokens in contexts: {total_tok}")
    print(f"Overall TTR across all contexts: {overall_ttr:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data', help='Path to data directory containing subfolders with dev.json')
    args = parser.parse_args()

    base = args.data_dir
    if not os.path.isdir(base):
        print(f"Data directory not found: {base}")
        return
    for sub in sorted(os.listdir(base)):
        dev_file = os.path.join(base, sub, 'dev.json')
        if os.path.isfile(dev_file):
            analyze_dataset(dev_file)
    print("\nAll datasets processed.")

if __name__ == '__main__':
    main()
