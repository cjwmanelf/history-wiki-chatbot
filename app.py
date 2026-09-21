"""app.py — Streamlit 기반 GraphRAG 웹 데모 (탭 3개)

CONTRACT.md §8 에 정의된 3탭 구조를 구현:
- 탭1 💬 질문하기: 질문 입력, 답변, 탐색 경로, 근거 삼중항 표, 출처 문서 expander, 예시 질문 버튼
- 탭2 📊 평가 결과: output/eval.json 읽어 홉 수별 대조표, 거절 정확도, 실패 층 집계 표시
- 탭3 ⚙️ 설정: OpenAI API 키 입력(password), 세션적용/파일저장/삭제/연결테스트, 마스킹 상태 표시
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from agent import GraphRAGAgent
import llm_provider as llm

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "data" / "docs"
EVAL_JSON_PATH = ROOT / "output" / "eval.json"
DEFAULT_GRAPH_PATH = ROOT / "output" / "graph.json"
FIXTURE_GRAPH_PATH = ROOT / "tests" / "fixture_graph.json"

st.set_page_config(
    page_title="한국 독립운동사 GraphRAG QA",
    page_icon="📜",
    layout="wide",
)


def load_config() -> dict[str, Any]:
    """config.json 설정을 안전하게 읽는다."""
    cfg_path = ROOT / "config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


@st.cache_resource
def get_agent() -> GraphRAGAgent | None:
    """GraphRAGAgent 를 캐싱하여 재사용한다. 그래프가 없으면 None."""
    if DEFAULT_GRAPH_PATH.exists():
        return GraphRAGAgent(graph_path=DEFAULT_GRAPH_PATH)
    if FIXTURE_GRAPH_PATH.exists():
        return GraphRAGAgent(graph_path=FIXTURE_GRAPH_PATH)
    return None


def render_source_doc(source_name: str, evidence_list: list[str]) -> None:
    """출처 문서 원문 또는 근거 문장을 expander 로 렌더링한다."""
    doc_path = DOCS_DIR / f"{source_name}.md"
    if not doc_path.exists():
        # 언더스코어 치환 시도
        doc_path = DOCS_DIR / f"{source_name.replace(' ', '_')}.md"

    with st.expander(f"📄 출처 문서: {source_name}"):
        if evidence_list:
            st.markdown("**수집된 핵심 근거 문장:**")
            for ev in evidence_list:
                st.info(f"💬 \"{ev}\"")

        if doc_path.exists():
            try:
                full_text = doc_path.read_text(encoding="utf-8")
                st.markdown("**문서 본문 미리보기:**")
                st.text_area(
                    label="문서 본문",
                    value=full_text[:1200] + ("..." if len(full_text) > 1200 else ""),
                    height=200,
                    disabled=True,
                    key=f"doc_view_{source_name}",
                )
            except Exception as e:
                st.caption(f"문서 로드 실패: {e}")
        else:
            st.caption("※ 로컬 원문 마크다운 파일이 존재하지 않아 추출된 근거 문장만 표시합니다.")


# ---------------------------------------------------------------------------
# 메인 헤더 및 URL 쿼리 파라미터 처리 (헤드리스 데모 캡처 지원)
# ---------------------------------------------------------------------------

st.title("📜 한국 독립운동사 지식 그래프 QA 시스템")
st.caption("LangGraph 기반 Multi-hop GraphRAG 에이전트 + TF-IDF BasicRAG 대조 평가")

# URL 쿼리 파라미터 읽기: ?q=<질문>&auto=1 지원
params = st.query_params
url_q = params.get("q", "")
url_auto = params.get("auto", "0") in ("1", "true", "True")

if "query_input" not in st.session_state:
    st.session_state["query_input"] = url_q if url_q else "안중근이 참여한 사건은?"
elif url_q and st.session_state.get("_last_url_q") != url_q:
    st.session_state["query_input"] = url_q
    st.session_state["_last_url_q"] = url_q

# 무한 루프 방지용 1회 자동 실행 플래그
auto_search = False
if url_auto and url_q and not st.session_state.get("_auto_searched"):
    st.session_state["_auto_searched"] = True
    auto_search = True

tab1, tab2, tab3 = st.tabs(["💬 질문하기", "📊 평가 결과", "⚙️ 설정"])


# ---------------------------------------------------------------------------
# 탭 1 — 질문하기
# ---------------------------------------------------------------------------

with tab1:
    st.subheader("지식 그래프 기반 멀티홉 질의응답")

    agent = get_agent()
    if agent is None:
        st.error("⚠️ 지식 그래프 파일이 없습니다. 먼저 `python build_graph.py` 를 실행하세요.")
    else:
        # 그래프 상태 요약 뱃지
        graph_kind = "실제 수집 그래프" if agent.graph_path == DEFAULT_GRAPH_PATH else "테스트 픽스처 그래프"
        st.info(f"💡 연결된 그래프: **{graph_kind}** ({len(agent.nodes_by_id)} 노드, {len(agent.graph_data.get('edges', []))} 엣지)")

        st.markdown("**예시 질문 클릭:**")
        ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)

        if ex_col1.button("1홉: 안중근 참여 사건", use_container_width=True):
            st.session_state["query_input"] = "안중근이 참여한 사건은?"
        if ex_col2.button("2홉: 함께 참여한 인물이 세운 조직", use_container_width=True):
            st.session_state["query_input"] = "안중근이 참여한 사건에 같이 있던 인물이 세운 조직은?"
        if ex_col3.button("3홉: 의거 단체 설립자의 참여 사건", use_container_width=True):
            st.session_state["query_input"] = "훙커우 공원 의거에 관련된 단체를 설립한 인물이 참여한 사건은?"
        if ex_col4.button("거절 예시: 안중근 설립 방송국", use_container_width=True):
            st.session_state["query_input"] = "안중근이 설립한 방송국은?"

        query_text = st.text_input(
            "질문을 입력하세요:",
            value=st.session_state["query_input"],
            key="main_query_field",
        )

        search_clicked = st.button("🔎 검색 및 답변 생성", type="primary") or auto_search

        if search_clicked and query_text.strip():
            with st.spinner("지식 그래프 경로 탐색 및 답변 생성 중..."):
                res = agent.answer(query_text.strip())

            # 1. 실행 메타데이터 배너
            meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
            meta_col1.metric("탐색 라우트", res.get("route", "-"))
            meta_col2.metric("사용 홉 수", f"{res.get('hops_used', 1)} 홉")
            meta_col3.metric("Deepen 확장", "발동됨" if res.get("deepened") else "미발동")
            meta_col4.metric("거절 여부", "거절(Refused)" if res.get("refused") else "정상 답변")

            if res.get("skipped_hubs"):
                skipped_parts = []
                for h in res["skipped_hubs"]:
                    if "kept" in h:
                        skipped_parts.append(f"**{h['node']}** (차수: {h['degree']}, 확장: {h.get('kept')}개, 절삭: {h.get('truncated')}개)")
                    else:
                        skipped_parts.append(f"**{h['node']}** (차수: {h['degree']})")
                st.warning(f"⚠️ 허브 노드 이웃 제한 적용: {', '.join(skipped_parts)}")

            # 2. 최종 답변 표시
            st.markdown("### 💬 답변 결과")
            if res.get("refused"):
                st.error(res["answer"])
            else:
                st.success(res["answer"])

            # 3. 탄 경로 시각화
            st.markdown("### 🧭 탐색 및 추론 경로 (Traversed Path)")
            traversed = res.get("traversed_path", [])
            if traversed:
                path_md = []
                for step in traversed:
                    if step.get("direction") == "reverse":
                        path_md.append(
                            f"**[Hop {step['hop']}]** `{step['from_node']}` ⇦(`{step['relation']}`)— `{step['to_node']}` *(역방향 탐색)* "
                            f"(origin: `{step['origin']}`, 출처: {', '.join(step.get('sources', []))})"
                        )
                    else:
                        path_md.append(
                            f"**[Hop {step['hop']}]** `{step['from_node']}` —(`{step['relation']}`)➔ `{step['to_node']}` "
                            f"(origin: `{step['origin']}`, 출처: {', '.join(step.get('sources', []))})"
                        )
                st.markdown("\n\n".join(path_md))
            else:
                st.write(res.get("path_explanation", "(탐색된 경로 없음)"))

            # 4. 근거 삼중항 표
            st.markdown("### 📑 수집된 근거 삼중항 표 (Context Triples)")
            triples = res.get("context_triples", [])
            if triples:
                table_rows = []
                for t in triples:
                    table_rows.append({
                        "Head (주어)": t.get("head"),
                        "Relation (관계)": t.get("relation"),
                        "Tail (목적어)": t.get("tail"),
                        "Origin": t.get("origin"),
                        "Confidence": t.get("confidence"),
                        "Sources": ", ".join(t.get("sources", [])),
                    })
                df_triples = pd.DataFrame(table_rows)
                st.dataframe(df_triples, use_container_width=True)
            else:
                st.info("수집된 근거 삼중항이 없습니다.")

            # 5. 출처 문서 원문 펼쳐보기
            st.markdown("### 📚 출처 문서 및 원문 근거 (Evidence Expander)")
            sources = res.get("sources", [])
            if sources:
                # 출처별 근거 문장 매핑
                src_evidence: dict[str, list[str]] = {}
                for t in triples:
                    for s in t.get("sources", []):
                        src_evidence.setdefault(s, []).extend(t.get("evidence", []))

                for s in sources:
                    ev_list = list(dict.fromkeys(src_evidence.get(s, [])))
                    render_source_doc(s, ev_list)
            else:
                st.caption("참조된 출처 문서가 없습니다.")


# ---------------------------------------------------------------------------
# 탭 2 — 평가 결과
# ---------------------------------------------------------------------------

with tab2:
    st.subheader("GraphRAG vs BasicRAG 홉 수별 대조 평가")

    if not EVAL_JSON_PATH.exists():
        st.warning("⚠️ 아직 평가 결과(`output/eval.json`)가 없습니다. 먼저 터미널에서 `python evaluate.py` 를 실행하세요.")
    else:
        try:
            eval_data = json.loads(EVAL_JSON_PATH.read_text(encoding="utf-8"))

            st.caption(f"평가 실행 시각: {eval_data.get('run_at')} (총 {eval_data.get('n_items')}문항)")

            # 상단 핵심 메트릭
            overall = eval_data.get("overall", {})
            unans = eval_data.get("unanswerable", {})

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("GraphRAG 전체 정답률", f"{overall.get('graphrag_answer_acc', 0.0):.3f}")
            col_m2.metric("BasicRAG 전체 정답률", f"{overall.get('basic_rag_answer_acc', 0.0):.3f}")
            col_m3.metric("거절 가드레일 정확도", f"{unans.get('refusal_accuracy', 0.0) * 100:.1f}%")
            col_m4.metric("환각 발생 건수", f"{unans.get('hallucinated', 0)} 건")

            # 1. 홉 수별 대조표
            st.markdown("### 1. 홉 수별 정답률 및 경로 재현율 대조표")
            by_hop = eval_data.get("by_hop", {})
            hop_rows = []
            for hop, s in by_hop.items():
                gr = s.get("graphrag", {})
                br = s.get("basic_rag", {})
                gr_acc = gr.get("answer_acc", 0.0)
                br_acc = br.get("answer_acc", 0.0)
                winner = "GraphRAG 우세" if gr_acc > br_acc else ("동률" if gr_acc == br_acc else "BasicRAG 우세")
                hop_rows.append({
                    "홉 수": f"{hop}홉",
                    "문항 수(n)": s.get("n", 0),
                    "GraphRAG 정답률": f"{gr_acc:.3f}",
                    "GraphRAG 경로 재현율": f"{gr.get('path_recall', 0.0):.3f}",
                    "BasicRAG 정답률": f"{br_acc:.3f}",
                    "비교 우위": winner,
                })
            st.dataframe(pd.DataFrame(hop_rows), use_container_width=True)

            # 2. 실패 층 자동 분류
            st.markdown("### 2. 실패 층 자동 분류 집계 (Failure Layer Analysis)")
            fc = eval_data.get("failure_layer_counts", {})
            col_f1, col_f2, col_f3 = st.columns(3)
            col_f1.metric("색인 층 (Index)", f"{fc.get('index', 0)} 건", help="기대 삼중항이 그래프에 없음")
            col_f2.metric("검색 층 (Retrieval)", f"{fc.get('retrieval', 0)} 건", help="그래프엔 있으나 컨텍스트 수집 시 누락")
            col_f3.metric("생성 층 (Generation)", f"{fc.get('generation', 0)} 건", help="컨텍스트엔 있으나 최종 답변 실패 또는 환각")

            # 3. 실패 사례별 구체적 진단
            st.markdown("### 3. 실패 사례별 구체적 진단 및 개선안")
            failures = eval_data.get("failures", [])
            if not failures:
                st.success("🎉 모든 평가 문항을 100% 통과했습니다. (실패 사례 없음)")
            else:
                for f in failures:
                    with st.expander(f"문항 {f.get('id')} ({f.get('hops')}홉) — [{f.get('layer', '').upper()} 실패]"):
                        st.markdown(f"**진단(Diagnosis):** {f.get('diagnosis')}")
                        st.markdown(f"**개선안(Fix):** `{f.get('fix')}`")
                        st.text(f"답변 발췌: {f.get('answer_excerpt')}")

            # 4. 전체 리포트 마크다운 펼쳐보기
            eval_report_file = ROOT / "output" / "eval_report.md"
            if eval_report_file.exists():
                with st.expander("📄 eval_report.md 원문 보기"):
                    st.markdown(eval_report_file.read_text(encoding="utf-8"))

        except Exception as e:
            st.error(f"평가 결과 파일 파싱 실패: {e}")


# ---------------------------------------------------------------------------
# 탭 3 — 설정 (사용자 요청 기능)
# ---------------------------------------------------------------------------

with tab3:
    st.subheader("⚙️ 시스템 설정 및 OpenAI API 키 관리")

    # 상단 고정 필수 안내 문구
    st.info(
        "💡 **안내:** 키가 없어도 규칙 기반으로 전체 파이프라인이 동작합니다. "
        "키는 LLM 증강(추출 보강 · 답변 합성 · LLM 심판)에만 쓰입니다."
    )

    # 1. 현재 키 상태 표시 (마스킹된 키만 노출, 원문 키 절대 노출 금지)
    status = llm.key_status()

    st.markdown("### 현재 LLM 상태")
    st_col1, st_col2, st_col3 = st.columns(3)
    st_col1.metric("동작 모드", status.get("mode", "-"))
    st_col2.metric("키 출처", status.get("source") or "(없음)")
    st_col3.metric("등록된 키", status.get("masked", "(없음)"))

    st.divider()

    # 2. 키 입력 위젯 (type="password")
    st.markdown("### OpenAI API 키 등록 / 변경")
    input_key = st.text_input(
        "OpenAI API 키 입력",
        type="password",
        placeholder="sk-...",
        help="입력한 키는 화면이나 로그에 원문으로 노출되지 않습니다.",
    )

    # 버튼 3개 + 1개
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

    with btn_col1:
        if st.button("🔒 [이 세션에만 적용]", use_container_width=True):
            if not input_key.strip():
                st.warning("키를 입력해 주세요.")
            else:
                llm.set_runtime_key(input_key.strip())
                st.success("현재 세션 메모리에 키가 적용되었습니다. 브라우저를 닫으면 사라집니다.")
                st.rerun()

    with btn_col2:
        if st.button("💾 [.env에 저장]", use_container_width=True):
            raw_key = input_key.strip()
            if not raw_key:
                st.warning("키를 입력해 주세요.")
            elif not llm.looks_like_openai_key(raw_key):
                st.error("올바른 OpenAI API 키 형식이 아닙니다 (sk-... 형식 필요).")
            else:
                try:
                    llm.save_key_to_env_file(raw_key)
                    st.success("`.env` 파일에 안전하게 저장되었습니다. `.env` 는 `.gitignore` 로 차단되어 깃에 올라가지 않습니다.")
                    st.rerun()
                except Exception as e:
                    st.error(f"저장 실패: {e}")

    with btn_col3:
        if st.button("🗑️ [키 삭제]", use_container_width=True):
            llm.clear_runtime_key()
            deleted = llm.delete_saved_key()
            if deleted:
                st.success("세션 및 `.env` 파일의 API 키가 성공적으로 삭제되었습니다.")
            else:
                st.info("세션 키가 초기화되었습니다.")
            st.rerun()

    with btn_col4:
        if st.button("🔌 [연결 테스트]", use_container_width=True):
            if not llm.is_enabled():
                st.warning("등록된 키가 없습니다.")
            else:
                with st.spinner("OpenAI API 호출 테스트 중..."):
                    cfg = load_config()
                    synth_model = cfg.get("llm", {}).get("synth_model", "gpt-4.1-mini")
                    test_resp = llm.chat(
                        [{"role": "user", "content": "ping"}],
                        model=synth_model,
                        max_tokens=10,
                        use_cache=False,
                    )
                if test_resp:
                    st.success(f"✅ 연결 성공! 응답 수신 확인")
                else:
                    st.error("❌ 연결 실패: API 키 또는 네트워크 상태를 확인하세요. 앱은 오프라인 규칙 모드로 유지됩니다.")

    st.divider()

    # 캐시 통계
    st.markdown("### LLM 응답 캐시 통계")
    cstats = llm.cache_stats()
    c_col1, c_col2, c_col3, c_col4 = st.columns(4)
    c_col1.metric("캐시 적중 (Hit)", cstats.get("hit", 0))
    c_col2.metric("새 호출 (Call)", cstats.get("call", 0))
    c_col3.metric("캐시 파일 항목 수", cstats.get("cached_entries", 0))
    c_col4.metric("호출 에러", cstats.get("error", 0))
