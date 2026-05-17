\
from __future__ import annotations

import argparse
from pathlib import Path

from rag_core import RAGIndex, answer_question, build_index, print_result


def main() -> None:
    parser = argparse.ArgumentParser(description="AI 旅平險專員：本地 RAG 系統")
    parser.add_argument("--build", action="store_true", help="建立 PDF 向量索引")
    parser.add_argument("--data", type=str, default="data", help="PDF 資料夾")
    parser.add_argument("--index", type=str, default="index", help="索引資料夾")
    parser.add_argument("--embedding", type=str, default="BAAI/bge-m3", help="Embedding 模型")
    parser.add_argument("--device", type=str, default=None, help="cuda 或 cpu；不填則自動判斷")
    parser.add_argument("--batch-size", type=int, default=8, help="Embedding batch size")
    parser.add_argument("--ask", type=str, default=None, help="單次提問")
    parser.add_argument("--interactive", action="store_true", help="互動問答模式")
    parser.add_argument("--model", type=str, default="llama3.1:8b", help="Ollama 模型")
    parser.add_argument("--top-k", type=int, default=6, help="送入 LLM 的檢索片段數")
    parser.add_argument("--rerank", action="store_true", help="啟用 BGE reranker，較準但較慢")
    args = parser.parse_args()

    data_dir = Path(args.data)
    index_dir = Path(args.index)

    if args.build:
        build_index(
            data_dir=data_dir,
            index_dir=index_dir,
            embedding_name=args.embedding,
            batch_size=args.batch_size,
            device=args.device,
        )

    if args.ask:
        index = RAGIndex(index_dir=index_dir, embedding_name=args.embedding, device=args.device)
        result = answer_question(
            question=args.ask,
            index=index,
            model=args.model,
            top_k=args.top_k,
            rerank=args.rerank,
        )
        print_result(result)

    if args.interactive:
        index = RAGIndex(index_dir=index_dir, embedding_name=args.embedding, device=args.device)
        print("AI 旅平險專員互動模式。輸入 exit 離開。")
        while True:
            q = input("\n問題> ").strip()
            if q.lower() in {"exit", "quit", "q"}:
                break
            if not q:
                continue
            result = answer_question(
                question=q,
                index=index,
                model=args.model,
                top_k=args.top_k,
                rerank=args.rerank,
            )
            print_result(result)


if __name__ == "__main__":
    main()
