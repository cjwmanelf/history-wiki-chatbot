"""basic_rag.py — TF-IDF 기반 기본 RAG 대조군

data/docs/*.md 문서를 800자 내외(150자 오버랩)로 청킹하여
scikit-learn TfidfVectorizer (analyzer="char_wb", ngram_range=(2, 4)) 로
검색하고 답변을 생성하는 기본 RAG 대조군.

한국어 특성상 어절 단위 분석보다 문자 경계 n-gram(char_wb)이 조사·어미 변형에 강건하여
1홉 질문에서 공정한 대조군 성능(≥ 0.5)을 보장한다.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import llm_provider as llm

ROOT = Path(__file__).resolve().parent
DEFAULT_DOCS_DIR = ROOT / "data" / "docs"

JOSA_SUFFIXES = [
    "에서", "으로", "에게", "이나", "이", "가", "은", "는", "을", "를", "의", "에",
    "과", "와", "로", "도", "만", "한", "된", "인", "던", "서",
]


def strip_josa(token: str) -> str:
    """한국어 토큰 끝의 조사 및 어미를 제거한다."""
    for s in sorted(JOSA_SUFFIXES, key=len, reverse=True):
        if token.endswith(s) and len(token) > len(s):
            return token[:-len(s)]
    return token


class BasicRAG:
    """문서 기반 단순 TF-IDF RAG 시스템 (대조군)."""

    def __init__(
        self,
        docs_dir: Path | str | None = None,
        top_k: int = 5,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ) -> None:
        self.docs_dir = Path(docs_dir) if docs_dir else DEFAULT_DOCS_DIR
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.chunks: list[dict[str, Any]] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None
        self.corpus_missing = False

        self._build_index()

    def _build_index(self) -> None:
        """docs_dir 내의 마크다운 문서들을 청킹하고 TF-IDF 인덱스를 생성한다."""
        if not self.docs_dir.exists():
            print(f"  [BasicRAG] 경고: 디렉터리 '{self.docs_dir}' 가 존재하지 않습니다.", file=sys.stderr)
            self.corpus_missing = True
            return

        # 문서 수는 고정하지 않고 glob 으로 동적 수집
        doc_files = list(self.docs_dir.glob("*.md"))
        if not doc_files:
            print(f"  [BasicRAG] 경고: '{self.docs_dir}' 에 마크다운(*.md) 코퍼스가 없습니다.", file=sys.stderr)
            self.corpus_missing = True
            return

        raw_chunks: list[dict[str, Any]] = []

        for doc_file in doc_files:
            try:
                text = doc_file.read_text(encoding="utf-8")
            except OSError:
                continue

            doc_title = doc_file.stem.replace("_", " ")

            # 800자 내외 청킹 (150자 오버랩)
            stride = max(50, self.chunk_size - self.chunk_overlap)
            for i in range(0, len(text), stride):
                chunk_text = text[i : i + self.chunk_size].strip()
                if len(chunk_text) >= 40:  # 너무 짧은 파편 제외
                    raw_chunks.append({
                        "doc": doc_title,
                        "file": doc_file.name,
                        "text": chunk_text,
                        "offset": i,
                    })

        if not raw_chunks:
            print(f"  [BasicRAG] 경고: 유효한 본문 청크가 0건입니다.", file=sys.stderr)
            self.corpus_missing = True
            return

        self.chunks = raw_chunks
        corpus = [c["text"] for c in self.chunks]

        # 한국어 맞춤: char_wb (문자 경계 n-gram) 으로 조사 접미사 불일치 문제 해결
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            sublinear_tf=True,
            min_df=1,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.corpus_missing = False

    def search(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        """질의에 대해 상위 k개 청크를 반환한다."""
        if not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        limit = k if k is not None else self.top_k
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = scores.argsort()[::-1][:limit]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.001:
                item = dict(self.chunks[idx])
                item["score"] = score
                results.append(item)
        return results

    def answer(self, question: str) -> dict[str, Any]:
        """질문에 대해 검색 및 답변을 수행한다.
        
        반환값: runs.jsonl 규격과 호환되는 딕셔너리 (answer, chunks, sources 포함)
        """
        if self.corpus_missing:
            return {
                "question": question,
                "answer": "경고: 코퍼스 문서가 존재하지 않아 답변을 생성할 수 없습니다.",
                "chunks": [],
                "sources": [],
                "corpus_missing": True,
            }

        top_chunks = self.search(question, k=self.top_k)

        if not top_chunks:
            return {
                "question": question,
                "answer": "검색된 관련 문서가 없습니다.",
                "chunks": [],
                "sources": [],
            }

        sources = list(dict.fromkeys(c["doc"] for c in top_chunks))
        context_text = "\n\n".join(
            f"[{c['doc']}]\n{c['text']}" for c in top_chunks
        )

        # 1. LLM 활성화 시 LLM 답변 생성 시도
        if llm.is_enabled():
            prompt = (
                f"질문: {question}\n\n"
                f"참고 문서:\n{context_text}\n\n"
                "참고 문서의 내용에만 근거하여 질문에 대해 간결하고 명확하게 답변해 주세요."
            )
            llm_resp = llm.chat(
                [
                    {"role": "system", "content": "당신은 제공된 역사 문서를 바탕으로 질문에 답하는 어시스턴트입니다."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
            )
            if llm_resp:
                return {
                    "question": question,
                    "answer": llm_resp.strip(),
                    "chunks": top_chunks,
                    "sources": sources,
                }

        # 2. 키가 없거나 LLM 실패 시: 결정론적 추출 요약
        # 질문에서 조사 제거한 핵심 어간 키워드 추출
        raw_words = re.findall(r"[가-힣A-Za-z0-9]{2,}", question)
        keywords = {strip_josa(w) for w in raw_words}
        keywords.update(raw_words)

        scored_sentences = []
        for c in top_chunks:
            sentences = re.split(r"(?<=[.?!])\s+", c["text"])
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 15:
                    continue
                match_count = sum(1 for kw in keywords if kw in s_clean)
                if match_count > 0:
                    scored_sentences.append((match_count, c["doc"], s_clean))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        # 상위 문장 발췌 + 최상위 청크 컨텍스트 보강
        answer_parts = []
        if scored_sentences:
            selected = scored_sentences[:4]
            seen_s = set()
            for _, doc, s in selected:
                if s not in seen_s:
                    seen_s.add(s)
                    answer_parts.append(f"- [{doc}] {s}")

        if not answer_parts:
            # 질문 키워드가 특정 문장에 집중되지 않은 경우 최상위 청크의 전문 활용
            for c in top_chunks[:2]:
                excerpt = c["text"][:300].replace("\n", " ").strip()
                answer_parts.append(f"- [{c['doc']}] {excerpt}...")

        answer_text = "관련 문서 내용 발췌:\n" + "\n".join(answer_parts)

        return {
            "question": question,
            "answer": answer_text,
            "chunks": top_chunks,
            "sources": sources,
        }


def main():
    llm.setup_console()
    print(llm.banner())
    rag = BasicRAG()
    print(f"BasicRAG 초기화 완료 (청크 수: {len(rag.chunks)}, corpus_missing: {rag.corpus_missing})")
    test_q = "안중근이 참여한 사건은?"
    res = rag.answer(test_q)
    print(f"\n질문: {test_q}")
    print(f"출처: {res['sources']}")
    print(f"답변:\n{res['answer']}")


if __name__ == "__main__":
    main()
