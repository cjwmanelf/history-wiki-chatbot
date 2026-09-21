"""evaluate.py — 홉 수별 측정 및 실패 층 자동 분류 평가 스크립트

goldenset.json(또는 tests/fixture_goldenset.json)을 읽어
GraphRAG 와 BasicRAG 를 문항별로 평가하고,
결과를 홉 수별(1/2/3홉) 대조표 및 실패 층(index/retrieval/generation)으로 자동 분류하여
output/eval.json 과 output/eval_report.md 를 생성한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent import GraphRAGAgent
from basic_rag import BasicRAG
import llm_provider as llm

ROOT = Path(__file__).resolve().parent
DEFAULT_GOLDENSET_PATH = ROOT / "data" / "goldenset.json"
FIXTURE_GOLDENSET_PATH = ROOT / "tests" / "fixture_goldenset.json"
DEFAULT_GRAPH_PATH = ROOT / "output" / "graph.json"
FIXTURE_GRAPH_PATH = ROOT / "tests" / "fixture_graph.json"
EVAL_JSON_PATH = ROOT / "output" / "eval.json"
EVAL_REPORT_PATH = ROOT / "output" / "eval_report.md"


def score_answer_acc(answer_text: str, expected_answers: list[str]) -> float:
    """답변 정확도 채점: 완전 포함 1.0, 부분 포함 0.5, 없음 0.0."""
    if not expected_answers:
        return 0.0

    ans_norm = answer_text.replace(" ", "")
    for exp in expected_answers:
        exp_norm = exp.replace(" ", "")
        if exp_norm in ans_norm:
            return 1.0

    # 부분 일치 검사 (3음절 이상 중 2음절 이상 포함)
    for exp in expected_answers:
        exp_clean = exp.strip()
        if len(exp_clean) >= 3:
            for i in range(len(exp_clean) - 1):
                sub = exp_clean[i : i + 2]
                if sub in answer_text:
                    return 0.5
    return 0.0


def score_path_recall(
    expected_path: list[dict[str, Any]],
    context_triples: list[dict[str, Any]],
) -> float:
    """기대 삼중항이 컨텍스트에 포함된 비율 (Path Recall)."""
    if not expected_path:
        return 1.0

    ctx_set = {
        (t["head"].strip(), t["relation"].strip(), t["tail"].strip())
        for t in context_triples
    }

    matched = 0
    for ep in expected_path:
        key = (ep["head"].strip(), ep["relation"].strip(), ep["tail"].strip())
        if key in ctx_set:
            matched += 1

    return round(matched / len(expected_path), 3)


def check_triples_in_graph(
    triples: list[dict[str, Any]],
    graph_edges: list[dict[str, Any]],
) -> tuple[bool, list[tuple[str, str, str]]]:
    """삼중항들이 그래프 엣지에 모두 존재하는지 확인."""
    graph_set = {
        (e["head"].strip(), e["relation"].strip(), e["tail"].strip())
        for e in graph_edges
    }
    missing = []
    for t in triples:
        key = (t["head"].strip(), t["relation"].strip(), t["tail"].strip())
        if key not in graph_set:
            missing.append(key)
    return (len(missing) == 0, missing)


def check_triples_in_context(
    triples: list[dict[str, Any]],
    context_triples: list[dict[str, Any]],
) -> tuple[bool, list[tuple[str, str, str]]]:
    """삼중항들이 수집된 컨텍스트 삼중항에 모두 존재하는지 확인."""
    ctx_set = {
        (t["head"].strip(), t["relation"].strip(), t["tail"].strip())
        for t in context_triples
    }
    missing = []
    for t in triples:
        key = (t["head"].strip(), t["relation"].strip(), t["tail"].strip())
        if key not in ctx_set:
            missing.append(key)
    return (len(missing) == 0, missing)


def run_evaluation(
    goldenset_path: Path | str | None = None,
    graph_path: Path | str | None = None,
    docs_dir: Path | str | None = None,
) -> tuple[dict[str, Any], str]:
    """평가를 수행하고 eval.json 딕셔너리와 eval_report.md 문자열을 반환한다."""
    # 1. 경로 결정
    if goldenset_path:
        gset_p = Path(goldenset_path)
    elif DEFAULT_GOLDENSET_PATH.exists():
        gset_p = DEFAULT_GOLDENSET_PATH
    else:
        gset_p = FIXTURE_GOLDENSET_PATH

    if graph_path:
        grp_p = Path(graph_path)
    elif DEFAULT_GRAPH_PATH.exists():
        grp_p = DEFAULT_GRAPH_PATH
    else:
        grp_p = FIXTURE_GRAPH_PATH

    goldenset_data = json.loads(gset_p.read_text(encoding="utf-8"))
    items = goldenset_data.get("items", [])
    criteria = goldenset_data.get("scoring_criteria", {})

    graph_data = json.loads(grp_p.read_text(encoding="utf-8"))
    graph_edges = graph_data.get("edges", [])

    print(f"[Eval] 골든셋: {gset_p} ({len(items)}문항)")
    print(f"[Eval] 그래프: {grp_p} (노드 {len(graph_data.get('nodes', []))}개, 엣지 {len(graph_edges)}개)")

    # 2. 에이전트 및 대조군 초기화
    agent = GraphRAGAgent(graph_path=grp_p)
    basic_rag = BasicRAG(docs_dir=docs_dir)

    # 3. 문항별 평가 수행
    results_by_item = []
    by_hop_acc: dict[str, list[dict[str, float]]] = {}
    unanswerable_items = []
    failures = []
    failure_counts = {"index": 0, "retrieval": 0, "generation": 0}

    for item in items:
        q_id = item["id"]
        q_text = item["question"]
        hops = str(item.get("hops", 1))
        is_answerable = item.get("answerable", True)
        expected_ans = item.get("expected_answer", [])
        expected_path = item.get("expected_path", [])

        # GraphRAG 실행
        gr_res = agent.answer(q_text)

        # BasicRAG 실행
        br_res = basic_rag.answer(q_text)

        if not is_answerable:
            # 거절 문항 평가
            refusal_ok = 1.0 if gr_res["refused"] else 0.0
            hallucinated = 0 if gr_res["refused"] else 1
            unanswerable_items.append({
                "id": q_id,
                "question": q_text,
                "refusal_ok": refusal_ok,
                "hallucinated": hallucinated,
                "answer": gr_res["answer"],
            })

            if refusal_ok < 1.0:
                failure_counts["generation"] += 1
                failures.append({
                    "id": q_id,
                    "hops": item.get("hops", 2),
                    "score": 0.0,
                    "layer": "generation",
                    "diagnosis": f"거절 대상 질문('{q_text}')임에도 불구하고 거절하지 않고 환각 답변을 생성함",
                    "expected_path_present_in_graph": False,
                    "expected_path_present_in_context": False,
                    "answer_excerpt": gr_res["answer"][:120].replace("\n", " "),
                    "fix": "require_relation_match 가드레일 활성화 및 refusal_text 출력 규칙 강화",
                })
            continue

        # 답변 가능 문항 평가
        gr_acc = score_answer_acc(gr_res["answer"], expected_ans)
        gr_path_recall = score_path_recall(expected_path, gr_res["context_triples"])
        br_acc = score_answer_acc(br_res["answer"], expected_ans)

        # LLM-as-judge 보조 평가 (선택적)
        judge_score = None
        if llm.is_enabled() and expected_ans:
            judge_prompt = (
                f"질문: {q_text}\n"
                f"기대 정답: {', '.join(expected_ans)}\n"
                f"모델 답변:\n{gr_res['answer']}\n\n"
                "모델 답변이 기대 정답의 사실을 정확히 담고 있으면 1.0, 부분적으로 맞으면 0.5, 틀리거나 없으면 0.0 으로 숫자만 답하세요."
            )
            judge_resp = llm.chat(
                [{"role": "system", "content": "You are an objective evaluation judge."},
                 {"role": "user", "content": judge_prompt}],
                model="gpt-4.1",
                max_tokens=10,
            )
            if judge_resp:
                try:
                    judge_score = float(judge_resp.strip())
                except ValueError:
                    judge_score = None

        item_eval = {
            "id": q_id,
            "hops": item.get("hops", 1),
            "question": q_text,
            "graphrag_acc": gr_acc,
            "graphrag_path_recall": gr_path_recall,
            "basic_rag_acc": br_acc,
            "judge_score": judge_score,
        }
        results_by_item.append(item_eval)

        by_hop_acc.setdefault(hops, []).append({
            "graphrag_acc": gr_acc,
            "path_recall": gr_path_recall,
            "basic_rag_acc": br_acc,
        })

        # 실패 층 자동 분류 (score < 1.0 인 경우)
        if gr_acc < 1.0 or gr_path_recall < 1.0:
            in_graph, missing_in_graph = check_triples_in_graph(expected_path, graph_edges)
            in_ctx, missing_in_ctx = check_triples_in_context(expected_path, gr_res["context_triples"])

            if not in_graph:
                layer = "index"
                missing_str = "; ".join(f"({h},{r},{t})" for h, r, t in missing_in_graph)
                diagnosis = f"골든셋 기대 삼중항이 지식 그래프에 없음: {missing_str}"
                fix = "build_graph.py 관계 추출 규칙(regex) 추가 또는 LLM 추출 프롬프트 보강"
            elif not in_ctx:
                layer = "retrieval"
                missing_str = "; ".join(f"({h},{r},{t})" for h, r, t in missing_in_ctx)
                diagnosis = f"그래프에는 삼중항이 존재하나 BFS 탐색 시 누락됨: {missing_str}"
                fix = "retrieval.start_hops / max_hops 상향, 또는 per_relation/max_triples 예산 상한 확대"
            else:
                layer = "generation"
                diagnosis = f"근거 삼중항은 컨텍스트에 수집되었으나 최종 답변에 정답 키워드 도출 실패"
                fix = "synthesize 프롬프트 개선 및 질의 대상 개체 추출 템플릿 정교화"

            failure_counts[layer] += 1
            failures.append({
                "id": q_id,
                "hops": item.get("hops", 1),
                "score": gr_acc,
                "layer": layer,
                "diagnosis": diagnosis,
                "expected_path_present_in_graph": in_graph,
                "expected_path_present_in_context": in_ctx,
                "answer_excerpt": gr_res["answer"][:120].replace("\n", " "),
                "fix": fix,
            })

    # 4. 집계 및 eval.json 구조 생성
    by_hop_summary: dict[str, Any] = {}
    all_gr_acc = []
    all_br_acc = []

    for hop_str in sorted(by_hop_acc.keys(), key=lambda x: int(x)):
        records = by_hop_acc[hop_str]
        n = len(records)
        avg_gr_acc = round(sum(r["graphrag_acc"] for r in records) / n, 3)
        avg_path_recall = round(sum(r["path_recall"] for r in records) / n, 3)
        avg_br_acc = round(sum(r["basic_rag_acc"] for r in records) / n, 3)

        all_gr_acc.extend(r["graphrag_acc"] for r in records)
        all_br_acc.extend(r["basic_rag_acc"] for r in records)

        by_hop_summary[hop_str] = {
            "n": n,
            "graphrag": {
                "answer_acc": avg_gr_acc,
                "path_recall": avg_path_recall,
                "refusal_ok": None,
            },
            "basic_rag": {
                "answer_acc": avg_br_acc,
            },
        }

    n_unans = len(unanswerable_items)
    refusal_acc = (
        round(sum(u["refusal_ok"] for u in unanswerable_items) / n_unans, 3)
        if n_unans > 0
        else 1.0
    )
    hallucinated_cnt = sum(u["hallucinated"] for u in unanswerable_items)

    overall_gr = round(sum(all_gr_acc) / len(all_gr_acc), 3) if all_gr_acc else 0.0
    overall_br = round(sum(all_br_acc) / len(all_br_acc), 3) if all_br_acc else 0.0

    eval_json_data = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "graph_source": grp_p.name,
        "goldenset_source": gset_p.name,
        "corpus_docs": len(list(Path(docs_dir or ROOT / "data" / "docs").glob("*.md"))),
        "n_items": len(items),
        "criteria": criteria,
        "by_hop": by_hop_summary,
        "unanswerable": {
            "n": n_unans,
            "refusal_accuracy": refusal_acc,
            "hallucinated": hallucinated_cnt,
        },
        "overall": {
            "graphrag_answer_acc": overall_gr,
            "basic_rag_answer_acc": overall_br,
        },
        "failures": failures,
        "failure_layer_counts": failure_counts,
    }

    if basic_rag.corpus_missing:
        eval_json_data["basic_rag_corpus_missing"] = True

    # 5. 사람이 읽는 eval_report.md 생성
    report_lines = [
        "# GraphRAG vs BasicRAG 평가 리포트",
        "",
        f"- **실행 시각**: {eval_json_data['run_at']}",
        f"- **그래프 출처**: `{grp_p.name}`",
        f"- **골든셋 출처**: `{gset_p.name}`",
        f"- **코퍼스 문서 수**: {eval_json_data['corpus_docs']}건" + (" ⚠️ (코퍼스 없음)" if basic_rag.corpus_missing else ""),
        f"- **평가 문항 수**: 총 {len(items)}문항 (답변 가능 {len(results_by_item)}문항, 거절 대상 {n_unans}문항)",
        "",
        "---",
        "",
        "## 1. 홉 수별 대조표 (Hop-by-Hop Comparison)",
        "",
        "| 홉 수 | 문항 수(n) | GraphRAG 정답률 | GraphRAG 경로재현율 | BasicRAG 정답률 | 비교 우위 |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for hop_str, stats in by_hop_summary.items():
        gr_acc = stats["graphrag"]["answer_acc"]
        gr_recall = stats["graphrag"]["path_recall"]
        br_acc = stats["basic_rag"]["answer_acc"]
        advantage = "GraphRAG 우세" if gr_acc > br_acc else ("동률" if gr_acc == br_acc else "BasicRAG 우세")
        report_lines.append(
            f"| {hop_str}홉 | {stats['n']} | {gr_acc:.3f} | {gr_recall:.3f} | {br_acc:.3f} | **{advantage}** |"
        )

    report_lines.extend([
        "",
        f"**전체 정답률 요약**: GraphRAG **{overall_gr:.3f}** vs BasicRAG **{overall_br:.3f}**",
        "",
        "---",
        "",
        "## 2. 거절 가드레일 정확도 (Unanswerable Guardrails)",
        "",
        f"- **거절 문항 수**: {n_unans}건",
        f"- **거절 정확도 (Refusal Accuracy)**: **{refusal_acc * 100:.1f}%**",
        f"- **환각 발생 건수 (Hallucinated)**: **{hallucinated_cnt}건**",
        "",
        "---",
        "",
        "## 3. 실패 층 집계 (Failure Layer Analysis)",
        "",
        "| 실패 층 | 건수 | 설명 |",
        "| :--- | :---: | :--- |",
        f"| **색인 층 (Index)** | {failure_counts['index']}건 | 기대 삼중항이 지식 그래프에 없음 |",
        f"| **검색 층 (Retrieval)** | {failure_counts['retrieval']}건 | 그래프엔 있으나 컨텍스트 수집에서 누락됨 |",
        f"| **생성 층 (Generation)** | {failure_counts['generation']}건 | 컨텍스트엔 있으나 답변 생성 실패 또는 환각 |",
        "",
        "---",
        "",
        "## 4. 실패 사례별 구체적 진단 및 개선안 (Diagnosis & Fixes)",
        "",
    ])

    if not failures:
        report_lines.append("모든 문항을 성공적으로 통과하였습니다. (실패 사례 없음)")
    else:
        for f in failures:
            report_lines.extend([
                f"### 문항 {f['id']} ({f['hops']}홉) — [{f['layer'].upper()} 실패]",
                f"- **진단**: {f['diagnosis']}",
                f"- **조치 방안**: {f['fix']}",
                f"- **답변 발췌**: `{f['answer_excerpt']}`",
                "",
            ])

    report_content = "\n".join(report_lines) + "\n"

    # 파일 저장
    EVAL_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_JSON_PATH.write_text(
        json.dumps(eval_json_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    EVAL_REPORT_PATH.write_text(report_content, encoding="utf-8")

    print(f"[Eval] 저장 완료: {EVAL_JSON_PATH}")
    print(f"[Eval] 리포트 저장 완료: {EVAL_REPORT_PATH}")

    return eval_json_data, report_content


def main():
    llm.setup_console()
    print(llm.banner())

    parser = argparse.ArgumentParser(description="GraphRAG vs BasicRAG Evaluation")
    parser.add_argument("--goldenset", type=str, default=None, help="Path to goldenset JSON")
    parser.add_argument("--graph", type=str, default=None, help="Path to graph JSON")
    parser.add_argument("--docs", type=str, default=None, help="Path to docs directory")
    args = parser.parse_args()

    data, report = run_evaluation(
        goldenset_path=args.goldenset,
        graph_path=args.graph,
        docs_dir=args.docs,
    )

    print("\n" + "=" * 50)
    print(report)
    print("=" * 50)


if __name__ == "__main__":
    main()
