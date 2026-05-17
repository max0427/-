\
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_core import RAGIndex, answer_question


DEFAULT_QUESTIONS = [
    {
        "question": "班機延誤5小時可以理賠嗎？",
        "expect_keywords": ["延誤", "小時", "旅程", "班機"],
    },
    {
        "question": "海外突發疾病是否可以理賠？",
        "expect_keywords": ["海外", "突發疾病", "醫療"],
    },
    {
        "question": "未滿十五歲身故保險金如何給付？",
        "expect_keywords": ["未滿十五", "喪葬費用", "身故"],
    },
    {
        "question": "保險期間因交通工具延遲抵達會自動延長多久？",
        "expect_keywords": ["延長", "二十四小時"],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="index")
    parser.add_argument("--embedding", default="BAAI/bge-m3")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    index = RAGIndex(Path(args.index), embedding_name=args.embedding)

    rows = []
    for item in DEFAULT_QUESTIONS:
        q = item["question"]
        result = answer_question(q, index, model=args.model, top_k=args.top_k, rerank=args.rerank)

        retrieved_text = " ".join(c["preview"] for c in result["citations"])
        hit_keywords = [kw for kw in item["expect_keywords"] if kw in retrieved_text or kw in result["answer"]]
        hit_rate = len(hit_keywords) / max(1, len(item["expect_keywords"]))

        rows.append({
            "question": q,
            "answer": result["answer"],
            "hit_keywords": hit_keywords,
            "keyword_hit_rate": round(hit_rate, 3),
            "citations": result["citations"],
        })

        print("=" * 80)
        print(q)
        print("keyword_hit_rate:", round(hit_rate, 3), "hit:", hit_keywords)
        print(result["answer"])

    Path(args.output).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已輸出：{args.output}")


if __name__ == "__main__":
    main()
