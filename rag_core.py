\
from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF
import jieba
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None


CJK_VARIANT_MAP = str.maketrans({
    "旅": "旅", "行": "行", "金": "金", "契": "契", "理": "理", "便": "便",
    "不": "不", "利": "利", "狀": "狀", "若": "若", "量": "量", "年": "年",
    "臺": "台",
})

ARTICLE_RE = re.compile(r"(第[一二三四五六七八九十百零〇\d\-]+條(?:之[一二三四五六七八九十百零〇\d]+)?)[　\s]*(.+?)?(?=\n|$)")


@dataclass
class Chunk:
    chunk_id: str
    file: str
    page: int
    article: str
    text: str


def normalize_text(text: str) -> str:
    text = text.translate(CJK_VARIANT_MAP)
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_article(text: str) -> str:
    matches = ARTICLE_RE.findall(text)
    if not matches:
        return ""
    art, title = matches[0]
    title = title or ""
    return f"{art} {title}".strip()


def split_text(text: str, max_chars: int = 700, overlap: int = 120) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    # 優先依條號切，再對過長片段做滑動切分
    starts = [m.start() for m in ARTICLE_RE.finditer(text)]
    segments: List[str] = []

    if len(starts) >= 2:
        starts.append(len(text))
        for i in range(len(starts) - 1):
            seg = text[starts[i]:starts[i + 1]].strip()
            if seg:
                segments.append(seg)
    else:
        # 以段落為單位累積
        paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
        buf = ""
        for p in paragraphs:
            if len(buf) + len(p) + 1 <= max_chars:
                buf = f"{buf}\n{p}".strip()
            else:
                if buf:
                    segments.append(buf)
                buf = p
        if buf:
            segments.append(buf)

    chunks: List[str] = []
    for seg in segments:
        if len(seg) <= max_chars:
            chunks.append(seg)
        else:
            start = 0
            while start < len(seg):
                end = min(start + max_chars, len(seg))
                part = seg[start:end].strip()
                if part:
                    chunks.append(part)
                if end >= len(seg):
                    break
                start = max(0, end - overlap)
    return chunks


def extract_pdf_chunks(data_dir: Path) -> List[Chunk]:
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"找不到 PDF：{data_dir}")

    chunks: List[Chunk] = []
    for pdf_path in pdfs:
        doc = fitz.open(pdf_path)
        for page_index in range(len(doc)):
            page = doc[page_index]
            text = normalize_text(page.get_text("text"))
            page_no = page_index + 1
            page_chunks = split_text(text)

            for idx, chunk_text in enumerate(page_chunks):
                article = detect_article(chunk_text)
                chunks.append(
                    Chunk(
                        chunk_id=f"{pdf_path.stem}_p{page_no}_c{idx}",
                        file=pdf_path.name,
                        page=page_no,
                        article=article,
                        text=chunk_text,
                    )
                )
        doc.close()
    return chunks


def load_embedding_model(name: str, device: Optional[str] = None) -> SentenceTransformer:
    if device:
        return SentenceTransformer(name, device=device)
    return SentenceTransformer(name)


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 16) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(emb, dtype=np.float32)


def save_chunks(chunks: List[Chunk], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks.append(Chunk(**obj))
    return chunks


def build_index(
    data_dir: Path,
    index_dir: Path,
    embedding_name: str = "BAAI/bge-m3",
    batch_size: int = 8,
    device: Optional[str] = None,
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] 解析 PDF 並分塊...")
    chunks = extract_pdf_chunks(data_dir)
    if not chunks:
        raise RuntimeError("沒有產生任何 chunk，請確認 PDF 可抽取文字。")

    print(f"共產生 {len(chunks)} 個 chunks")

    print("[2/4] 載入中文 Embedding 模型...")
    model = load_embedding_model(embedding_name, device=device)

    print("[3/4] 建立向量...")
    texts = [format_chunk_for_embedding(ch) for ch in chunks]
    embeddings = embed_texts(model, texts, batch_size=batch_size)

    print("[4/4] 儲存索引...")
    save_chunks(chunks, index_dir / "chunks.jsonl")
    np.save(index_dir / "embeddings.npy", embeddings)

    meta = {
        "embedding_model": embedding_name,
        "chunk_count": len(chunks),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": sorted(list({ch.file for ch in chunks})),
    }
    (index_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"索引完成：{index_dir}")


def format_chunk_for_embedding(ch: Chunk) -> str:
    article = f" 條款：{ch.article}" if ch.article else ""
    return f"文件：{ch.file} 第{ch.page}頁{article}\n{ch.text}"


def tokenize_zh(text: str) -> List[str]:
    text = normalize_text(text)
    tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
    return tokens


class RAGIndex:
    def __init__(self, index_dir: Path, embedding_name: Optional[str] = None, device: Optional[str] = None):
        self.index_dir = index_dir
        self.chunks = load_chunks(index_dir / "chunks.jsonl")
        self.embeddings = np.load(index_dir / "embeddings.npy").astype(np.float32)

        meta_path = index_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.embedding_name = embedding_name or meta.get("embedding_model", "BAAI/bge-m3")
        self.model = load_embedding_model(self.embedding_name, device=device)

        self.bm25 = None
        if BM25Okapi is not None:
            corpus = [tokenize_zh(format_chunk_for_embedding(ch)) for ch in self.chunks]
            self.bm25 = BM25Okapi(corpus)

    def search(
        self,
        query: str,
        top_k: int = 6,
        candidates: int = 40,
        dense_weight: float = 0.75,
        use_bm25: bool = True,
        rerank: bool = False,
        reranker_name: str = "BAAI/bge-reranker-v2-m3",
    ) -> List[Tuple[Chunk, float]]:
        q_emb = embed_texts(self.model, [query], batch_size=1)[0]
        dense_scores = self.embeddings @ q_emb

        final_scores = minmax(dense_scores) * dense_weight

        if use_bm25 and self.bm25 is not None:
            bm25_scores = np.asarray(self.bm25.get_scores(tokenize_zh(query)), dtype=np.float32)
            final_scores += minmax(bm25_scores) * (1.0 - dense_weight)

        candidate_count = min(candidates, len(self.chunks))
        cand_idx = np.argsort(-final_scores)[:candidate_count]

        if rerank:
            cand_idx = rerank_candidates(query, self.chunks, cand_idx, reranker_name)

        results = [(self.chunks[i], float(final_scores[i])) for i in cand_idx[:top_k]]
        return results


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    if math.isclose(hi, lo):
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def rerank_candidates(
    query: str,
    chunks: List[Chunk],
    cand_idx: Iterable[int],
    reranker_name: str,
) -> np.ndarray:
    from sentence_transformers import CrossEncoder

    idx_list = list(cand_idx)
    pairs = [(query, format_chunk_for_embedding(chunks[i])) for i in idx_list]
    reranker = CrossEncoder(reranker_name)
    scores = reranker.predict(pairs)
    order = np.argsort(-np.asarray(scores))
    return np.asarray([idx_list[i] for i in order])


def build_context(results: List[Tuple[Chunk, float]], max_chars: int = 4200) -> str:
    parts: List[str] = []
    total = 0
    for n, (ch, score) in enumerate(results, start=1):
        header = f"[{n}] 文件：{ch.file}｜第 {ch.page} 頁"
        if ch.article:
            header += f"｜{ch.article}"
        body = ch.text
        block = f"{header}\n{body}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n---\n".join(parts)

SYSTEM_PROMPT = """
你是一位專業的保險理賠專員。請僅根據提供的文件內容（Context）來回答客戶問題，如果遇到數字，可以適度轉成中文數字去做對比。
如果文件中沒有相關資訊，請禮貌地告知你目前無法提供該資訊，並建議客戶聯繫人工客服，不要嘗試自行猜測或引用外部知識。請使用繁體中文回答。
數字判斷規則：
- 「以上」表示 >=
- 「以下」表示 <=
- 「以內」表示 <=
- 「不得超過」表示 <=
- 「未滿」表示 <
- 「超過」表示 >
- 「為限」表示 <=
- 例如：條款寫「四小時以上」，問題是「5小時」，則 5 >= 4，所以符合。
- 例如：條款寫「二十四小時以內」，問題是「25小時」，則 25 > 24，所以不符合。
"""

def call_ollama(
    question: str,
    context: str,
    model: str = "llama3.1:8b",
    temperature: float = 0.1,
    num_ctx: int = 8192,
    host: str = "http://localhost:11434",
) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}",
            },
        ],
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }

    try:
        resp = requests.post(f"{host}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()
    except requests.exceptions.ConnectionError:
        return "錯誤：無法連線到 Ollama。請先執行 `ollama serve`，並確認已下載 `llama3.1:8b`。"
    except Exception as exc:
        return f"錯誤：Ollama 呼叫失敗：{exc}"


def answer_question(
    question: str,
    index: RAGIndex,
    model: str = "llama3.1:8b",
    top_k: int = 6,
    rerank: bool = False,
) -> Dict[str, Any]:
    results = index.search(question, top_k=top_k, rerank=rerank)
    context = build_context(results)
    answer = call_ollama(question, context, model=model)

    citations = []
    for n, (ch, score) in enumerate(results, start=1):
        citations.append({
            "id": n,
            "file": ch.file,
            "page": ch.page,
            "article": ch.article,
            "score": round(score, 4),
            "chunk_id": ch.chunk_id,
            "preview": ch.text[:160].replace("\n", " "),
        })

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
    }


def print_result(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("RAG 回答")
    print("=" * 80)
    print(result["answer"])

    print("\n" + "=" * 80)
    print("檢索來源 Citations")
    print("=" * 80)
    for c in result["citations"]:
        article = f"｜{c['article']}" if c["article"] else ""
        print(f"[{c['id']}] {c['file']}｜第 {c['page']} 頁{article}｜score={c['score']}")
        print(f"    {c['preview']}")
