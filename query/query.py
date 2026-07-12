"""Query a RAG space/strategy collection from the command line.

Usage:
    python query/query.py --space kerala_finance --strategy s1_overlap --query "What is the fiscal deficit?"
"""

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from retriever import search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, help="Logical RAG space name, e.g. kerala_finance")
    parser.add_argument("--strategy", required=True, help="e.g. s1_overlap, s2_parent_child")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    hits = search(args.space, args.strategy, args.query, top_k=args.top_k)
    if not hits:
        print("No results.")
        return

    for i, hit in enumerate(hits, 1):
        print(f"\n--- [{i}] score={hit['score']:.4f}  doc={hit['source_doc']} ---")
        print(hit["text"])


if __name__ == "__main__":
    main()
