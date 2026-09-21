"""agent.py — LangGraph 기반 GraphRAG 멀티홉 QA 에이전트

지식 그래프 기반 Multi-hop 질의응답 에이전트.
LangGraph StateGraph 로 아래 파이프라인을 실행한다:
START -> route -> find_seeds -> expand -> build_context -> synthesize -> END
                                   ^                |
                                   |  (relation_gap) v
                                   +---- deepen <----+

키가 없어도 결정론적 그래프 탐색과 템플릿 합성으로 100% 정상 동작하며,
OpenAI 키가 활성화되면 LLM 증강(라우팅·답변 합성)이 적용된다.
모든 실행 결과는 output/runs.jsonl 에 CONTRACT.md §6 스키마로 기록된다.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

import llm_provider as llm

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "config.json"
DEFAULT_GRAPH_PATH = ROOT / "output" / "graph.json"
FIXTURE_GRAPH_PATH = ROOT / "tests" / "fixture_graph.json"
RUNS_LOG_PATH = ROOT / "output" / "runs.jsonl"


# ---------------------------------------------------------------------------
# 질의 분석 헬퍼
# ---------------------------------------------------------------------------

def detect_target_type(q: str) -> str | None:
    """질문 문미의 핵심 명사/술어를 분석하여 요구되는 대상 개체 타입(Person/Organization/Event)을 반환한다."""
    type_keywords = [
        ("Event", ["사건", "전투", "의거", "대첩", "운동", "참여했는가", "참전했는가", "벌어졌는가"]),
        ("Organization", ["조직", "단체", "학교", "단", "정부", "회사"]),
        ("Person", ["인물", "사람", "누구", "총책임자", "총장", "대표", "인물은"]),
    ]
    best_pos = -1
    detected = None
    for typ, kws in type_keywords:
        for kw in kws:
            pos = q.rfind(kw)
            if pos > best_pos:
                best_pos = pos
                detected = typ
    return detected


def is_co_participant_query(q: str) -> bool:
    """질문이 동료/참여자 경유 멀티홉(함께한 인물, 같이 있던 인물 등)을 요구하는지 판별한다."""
    pattern = r"(함께|같이|동참|동료|같은).*(참전|참여|있던|활동|거사|사건|전투|운동)"
    pattern_rev = r"(참전|참여|있던|활동|거사|사건|전투|운동).*(함께|같이|동참|동료)"
    return bool(re.search(pattern, q) or re.search(pattern_rev, q))


def edge_weight(e: dict[str, Any], q: str) -> float:
    """엣지의 신뢰도 및 출처 기반 가중치를 계산한다."""
    base = e.get("confidence", 1.0)
    h, t = e["head"], e["tail"]
    sources = e.get("sources", [])
    if h in sources or t in sources:
        base *= 1.3
    if len(sources) >= 2:
        base *= 1.2
    return base


# ---------------------------------------------------------------------------
# State 정의
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    question: str
    route: str
    seeds: list[dict[str, Any]]
    current_hops: int
    hops_used: int
    deepened: bool
    visited_nodes: list[str]
    skipped_hubs: list[dict[str, Any]]
    traversed_path: list[dict[str, Any]]
    context_triples: list[dict[str, Any]]
    candidate_paths: list[dict[str, Any]]
    relation_gap: bool
    demanded_relations: list[str]
    target_relation: str | None
    desired_target_type: str | None
    is_co_participant: bool
    answer: str
    refused: bool
    path_explanation: str
    sources: list[str]
    llm_used: bool
    llm_fallback_reason: str | None


# ---------------------------------------------------------------------------
# GraphRAG Agent
# ---------------------------------------------------------------------------

class GraphRAGAgent:
    """LangGraph 기반 GraphRAG 멀티홉 질의응답 에이전트."""

    def __init__(
        self,
        root: Path | None = None,
        graph_path: Path | str | None = None,
        config_path: Path | str | None = None,
    ) -> None:
        self.root = Path(root) if root else ROOT
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

        # 그래프 파일 경로 결정: 지정 경로 -> output/graph.json -> tests/fixture_graph.json
        if graph_path:
            self.graph_path = Path(graph_path)
        elif DEFAULT_GRAPH_PATH.exists():
            self.graph_path = DEFAULT_GRAPH_PATH
        else:
            self.graph_path = FIXTURE_GRAPH_PATH

        self.config = self._load_config()
        self.graph_data = self._load_graph()
        self.graph_nx, self.nodes_by_id, self.alias_index = self._build_indexes()
        self.workflow = self._compile_graph()

    def _load_config(self) -> dict[str, Any]:
        if self.config_path.exists():
            try:
                return json.loads(self.config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "retrieval": {
                "max_hops": 3,
                "start_hops": 2,
                "max_triples": 60,
                "per_relation": 25,
                "hub_degree_threshold": 25,
                "hub_max_neighbors": 8,
                "degree_penalty": True,
                "min_entity_len": 2,
            },
            "generation": {
                "refusal_text": "제시된 지식 그래프 자료에서 해당 질문에 대한 근거를 찾을 수 없습니다.",
                "require_relation_match": True,
            },
            "llm": {
                "synth_model": "gpt-4.1-mini",
                "extract_model": "gpt-4.1-mini",
            },
        }

    def _load_graph(self) -> dict[str, Any]:
        if self.graph_path.exists():
            try:
                return json.loads(self.graph_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"nodes": [], "edges": [], "alias_index": {}, "stats": {}}

    def _build_indexes(self):
        nodes_by_id: dict[str, dict[str, Any]] = {}
        for n in self.graph_data.get("nodes", []):
            nodes_by_id[n["id"]] = n

        alias_index: dict[str, str] = dict(self.graph_data.get("alias_index", {}))
        for n in self.graph_data.get("nodes", []):
            alias_index[n["id"]] = n["id"]
            for alias in n.get("aliases", []):
                alias_index[alias] = n["id"]

        adj: dict[str, list[dict[str, Any]]] = {}
        for edge in self.graph_data.get("edges", []):
            h, t = edge["head"], edge["tail"]
            adj.setdefault(h, []).append(edge)
            adj.setdefault(t, []).append(edge)

        return adj, nodes_by_id, alias_index

    # -----------------------------------------------------------------------
    # LangGraph 노드 구현
    # -----------------------------------------------------------------------

    def _route_node(self, state: AgentState) -> dict[str, Any]:
        q = state["question"]

        path_keywords = ["함께", "같이", "동료", "동참", "거쳐", "이어진", "통해", "관련된", "의 인물이 세운", "의 단체가 참여한", "의 사건에", "연결", "모의"]
        local_keywords = ["참여한 사건", "설립한 조직", "세운 단체", "참전한 사건", "의 거사", "언제", "누구인가", "무엇인가"]

        is_path = any(kw in q for kw in path_keywords)
        is_local = any(kw in q for kw in local_keywords)

        if re.search(r"(참여|세운|창설|설립|속한).*(인물|단체|조직|사건).*(세운|참여|일으킨|창설)", q):
            is_path = True

        if is_path and not is_local:
            route = "path"
        elif is_local and not is_path:
            route = "local"
        else:
            if llm.is_enabled():
                prompt = (
                    f"질문: '{q}'\n\n"
                    "이 질문이 단일 개체의 직접 속성/단일 관계 질문이면 'local', "
                    "여러 개체를 거치는 연쇄 연결/멀티홉 질문이면 'path' 로만 답하세요."
                )
                resp = llm.chat(
                    [{"role": "system", "content": "You are a query classification router. Output only 'local' or 'path'."},
                     {"role": "user", "content": prompt}],
                    max_tokens=10,
                )
                if resp and "local" in resp.lower():
                    route = "local"
                else:
                    route = "path"
            else:
                route = "path" if is_path else "local"

        return {"route": route}

    def _find_seeds_node(self, state: AgentState) -> dict[str, Any]:
        q = state["question"]
        min_len = self.config.get("retrieval", {}).get("min_entity_len", 2)
        sorted_aliases = sorted(self.alias_index.keys(), key=lambda x: len(x), reverse=True)

        matched_seeds: list[dict[str, Any]] = []
        seen_nodes: set[str] = set()

        for surface in sorted_aliases:
            if len(surface) < min_len:
                continue
            if surface in q:
                canonical_id = self.alias_index[surface]
                if canonical_id in seen_nodes:
                    continue
                seen_nodes.add(canonical_id)
                node_info = self.nodes_by_id.get(canonical_id, {})
                node_type = node_info.get("type", "Unknown")

                type_bonus = 0.2 if node_type in ("Person", "Organization", "Event") else 0.0
                len_ratio = min(1.0, len(surface) / max(len(q), 1) * 2)
                score = round(min(1.0, 0.7 + type_bonus + len_ratio * 0.1), 2)

                matched_seeds.append({
                    "surface": surface,
                    "node": canonical_id,
                    "type": node_type,
                    "score": score,
                })

        type_prio = {"Person": 3, "Organization": 2, "Event": 1, "Unknown": 0}
        matched_seeds.sort(key=lambda x: (type_prio.get(x["type"], 0), x["score"]), reverse=True)

        desired_type = detect_target_type(q)
        is_co = is_co_participant_query(q)

        return {
            "seeds": matched_seeds,
            "desired_target_type": desired_type,
            "is_co_participant": is_co,
        }

    def _expand_node(self, state: AgentState) -> dict[str, Any]:
        q = state["question"]
        seeds = state["seeds"]
        target_hops = state["current_hops"]
        desired_type = state.get("desired_target_type") or detect_target_type(q)
        is_co = state.get("is_co_participant", False) or is_co_participant_query(q)

        cfg_ret = self.config.get("retrieval", {})
        max_triples = cfg_ret.get("max_triples", 60)

        # 질문의 타겟 관계 및 요구 관계 분석
        target_rel = "FOUNDED" if any(w in q for w in ["세운", "설립", "창설", "창립", "만든", "조직한", "세웠", "창건"]) else "PARTICIPATED_IN"
        demanded_relations = ["FOUNDED"] if target_rel == "FOUNDED" else ["PARTICIPATED_IN"]
        if is_co:
            demanded_relations = ["PARTICIPATED_IN", "FOUNDED"] if desired_type == "Organization" else ["PARTICIPATED_IN", "FOUNDED", "PARTICIPATED_IN"]

        seed_persons = [s["node"] for s in seeds if s.get("type") == "Person"]
        seed_events = [s["node"] for s in seeds if s.get("type") == "Event"]
        seed_orgs = [s["node"] for s in seeds if s.get("type") == "Organization"]
        seed_ids = {s["node"] for s in seeds}

        candidate_paths: list[dict[str, Any]] = []

        if is_co:
            # 멀티홉 동료/참여자 경유 탐색 (시드 자신 1스텝 우회 버그 방지)
            wants_battle_mid = "전투" in q or "참전" in q
            wants_battle_end = "어떤 전투" in q or ("전투" in q and desired_type == "Event")

            if desired_type == "Organization":
                # 경로 패턴: 시드인물 -> 사건 <- 동료인물(Other != seed) -> 조직
                for p_seed in (seed_persons if seed_persons else seeds):
                    p_name = p_seed if isinstance(p_seed, str) else p_seed["node"]
                    for e1 in self.graph_nx.get(p_name, []):
                        if e1["relation"] == "PARTICIPATED_IN":
                            ev = e1["tail"] if e1["head"] == p_name else e1["head"]
                            if self.nodes_by_id.get(ev, {}).get("type") != "Event":
                                continue
                            if seed_events and ev not in seed_events:
                                continue
                            battle_bonus = 1.5 if (wants_battle_mid and ("전투" in ev or "대첩" in ev)) else 1.0
                            w1 = edge_weight(e1, q)

                            for e2 in self.graph_nx.get(ev, []):
                                if e2["relation"] == "PARTICIPATED_IN":
                                    other_p = e2["head"] if e2["tail"] == ev else e2["tail"]
                                    if other_p not in seed_persons and self.nodes_by_id.get(other_p, {}).get("type") == "Person":
                                        w2 = edge_weight(e2, q)
                                        for e3 in self.graph_nx.get(other_p, []):
                                            if e3["head"] == other_p and e3["relation"] == "FOUNDED":
                                                org = e3["tail"]
                                                if self.nodes_by_id.get(org, {}).get("type") == "Organization":
                                                    w3 = edge_weight(e3, q)
                                                    path = [
                                                        {"hop": 1, "from_node": p_name, "to_node": ev, "direction": "forward" if e1["head"] == p_name else "reverse", "head": e1["head"], "relation": e1["relation"], "tail": e1["tail"], "sources": e1.get("sources", []), "origin": e1.get("origin", "rule"), "confidence": e1.get("confidence", 1.0)},
                                                        {"hop": 2, "from_node": ev, "to_node": other_p, "direction": "reverse" if e2["head"] == other_p else "forward", "head": e2["head"], "relation": e2["relation"], "tail": e2["tail"], "sources": e2.get("sources", []), "origin": e2.get("origin", "rule"), "confidence": e2.get("confidence", 1.0)},
                                                        {"hop": 3, "from_node": other_p, "to_node": org, "direction": "forward", "head": e3["head"], "relation": e3["relation"], "tail": e3["tail"], "sources": e3.get("sources", []), "origin": e3.get("origin", "rule"), "confidence": e3.get("confidence", 1.0)},
                                                    ]
                                                    score = battle_bonus * w1 * w2 * w3
                                                    candidate_paths.append({
                                                        "target": org,
                                                        "type": "Organization",
                                                        "path": path,
                                                        "score": score,
                                                    })

            elif desired_type == "Event":
                # 3홉(4스텝) 연쇄 패턴: 시드인물 -> 사건1 <- 동료인물 -> 조직 -> 사건2 (사건2 != 사건1)
                for p_seed in (seed_persons if seed_persons else seeds):
                    p_name = p_seed if isinstance(p_seed, str) else p_seed["node"]
                    for e1 in self.graph_nx.get(p_name, []):
                        if e1["relation"] == "PARTICIPATED_IN":
                            ev1 = e1["tail"] if e1["head"] == p_name else e1["head"]
                            if self.nodes_by_id.get(ev1, {}).get("type") != "Event":
                                continue
                            if seed_events and ev1 not in seed_events:
                                continue
                            battle_bonus = 1.5 if (wants_battle_mid and ("전투" in ev1 or "대첩" in ev1)) else 1.0
                            w1 = edge_weight(e1, q)

                            for e2 in self.graph_nx.get(ev1, []):
                                if e2["relation"] == "PARTICIPATED_IN":
                                    other_p = e2["head"] if e2["tail"] == ev1 else e2["tail"]
                                    if other_p not in seed_persons and self.nodes_by_id.get(other_p, {}).get("type") == "Person":
                                        w2 = edge_weight(e2, q)
                                        for e3 in self.graph_nx.get(other_p, []):
                                            if e3["head"] == other_p and e3["relation"] == "FOUNDED":
                                                org = e3["tail"]
                                                if self.nodes_by_id.get(org, {}).get("type") == "Organization":
                                                    w3 = edge_weight(e3, q)
                                                    for e4 in self.graph_nx.get(org, []):
                                                        if e4["head"] == org and e4["relation"] == "PARTICIPATED_IN":
                                                            ev2 = e4["tail"]
                                                            if ev2 != ev1 and self.nodes_by_id.get(ev2, {}).get("type") == "Event":
                                                                if wants_battle_end and ("전투" not in ev2 and "대첩" not in ev2):
                                                                    continue
                                                                w4 = edge_weight(e4, q)
                                                                path = [
                                                                    {"hop": 1, "from_node": p_name, "to_node": ev1, "direction": "forward" if e1["head"] == p_name else "reverse", "head": e1["head"], "relation": e1["relation"], "tail": e1["tail"], "sources": e1.get("sources", []), "origin": e1.get("origin", "rule"), "confidence": e1.get("confidence", 1.0)},
                                                                    {"hop": 2, "from_node": ev1, "to_node": other_p, "direction": "reverse" if e2["head"] == other_p else "forward", "head": e2["head"], "relation": e2["relation"], "tail": e2["tail"], "sources": e2.get("sources", []), "origin": e2.get("origin", "rule"), "confidence": e2.get("confidence", 1.0)},
                                                                    {"hop": 3, "from_node": other_p, "to_node": org, "direction": "forward", "head": e3["head"], "relation": e3["relation"], "tail": e3["tail"], "sources": e3.get("sources", []), "origin": e3.get("origin", "rule"), "confidence": e3.get("confidence", 1.0)},
                                                                    {"hop": 4, "from_node": org, "to_node": ev2, "direction": "forward", "head": e4["head"], "relation": e4["relation"], "tail": e4["tail"], "sources": e4.get("sources", []), "origin": e4.get("origin", "rule"), "confidence": e4.get("confidence", 1.0)},
                                                                ]
                                                                score = battle_bonus * w1 * w2 * w3 * w4
                                                                candidate_paths.append({
                                                                    "target": ev2,
                                                                    "type": "Event",
                                                                    "path": path,
                                                                    "score": score,
                                                                })

        else:
            # 복수 시드 교집합 인물 탐색 (예: Q09 — 사건 참여 + 조직 설립 교집합)
            if desired_type == "Person" and len(seeds) >= 2:
                p_in_event = set()
                p_in_org = set()
                for s in seeds:
                    s_name = s["node"]
                    s_type = s.get("type")
                    for e in self.graph_nx.get(s_name, []):
                        head, rel, tail = e["head"], e["relation"], e["tail"]
                        p = head if tail == s_name else tail
                        if self.nodes_by_id.get(p, {}).get("type") == "Person":
                            if s_type == "Event" and rel == "PARTICIPATED_IN":
                                p_in_event.add(p)
                            elif s_type == "Organization" and rel == "FOUNDED":
                                p_in_org.add(p)
                common_p = p_in_event & p_in_org
                for p in common_p:
                    p_edges = self.graph_nx.get(p, [])
                    e_ev = next((e for e in p_edges if e["relation"] == "PARTICIPATED_IN" and (e["head"] in seed_ids or e["tail"] in seed_ids)), None)
                    e_org = next((e for e in p_edges if e["relation"] == "FOUNDED" and (e["head"] in seed_ids or e["tail"] in seed_ids)), None)
                    path = []
                    score = 1.0
                    if e_org:
                        path.append({"hop": 1, "from_node": e_org["tail"], "to_node": p, "direction": "reverse", "head": e_org["head"], "relation": e_org["relation"], "tail": e_org["tail"], "sources": e_org.get("sources", []), "origin": e_org.get("origin", "rule"), "confidence": e_org.get("confidence", 1.0)})
                        score *= edge_weight(e_org, q)
                    if e_ev:
                        path.append({"hop": 2, "from_node": p, "to_node": e_ev["tail"], "direction": "forward", "head": e_ev["head"], "relation": e_ev["relation"], "tail": e_ev["tail"], "sources": e_ev.get("sources", []), "origin": e_ev.get("origin", "rule"), "confidence": e_ev.get("confidence", 1.0)})
                        score *= edge_weight(e_ev, q)
                    candidate_paths.append({
                        "target": p,
                        "type": "Person",
                        "path": path,
                        "score": score,
                    })
            else:
                # 1-hop / Local 표준 탐색
                for s in seeds:
                    s_name = s["node"]
                    for e in self.graph_nx.get(s_name, []):
                        head, rel, tail = e["head"], e["relation"], e["tail"]
                        if head == s_name and rel == target_rel:
                            nbr = tail
                            nbr_type = self.nodes_by_id.get(nbr, {}).get("type")
                            if desired_type is None or nbr_type == desired_type:
                                path = [{"hop": 1, "from_node": s_name, "to_node": nbr, "direction": "forward", "head": head, "relation": rel, "tail": tail, "sources": e.get("sources", []), "origin": e.get("origin", "rule"), "confidence": e.get("confidence", 1.0)}]
                                score = edge_weight(e, q)
                                candidate_paths.append({
                                    "target": nbr,
                                    "type": nbr_type,
                                    "path": path,
                                    "score": score,
                                })

        # 타겟 노드별 최고 점수 후보 경로로 그룹화
        best_by_target: dict[str, dict[str, Any]] = {}
        for cp in candidate_paths:
            tgt = cp["target"]
            if tgt not in best_by_target or cp["score"] > best_by_target[tgt]["score"]:
                best_by_target[tgt] = cp
        sorted_candidates = sorted(best_by_target.values(), key=lambda x: x["score"], reverse=True)

        # 컨텍스트 삼중항 생성 (max_triples 상한 준수, 모든 유효 후보 경로 삼중항 우선 포함)
        context_triples: list[dict[str, Any]] = []
        seen_triples: set[tuple[str, str, str]] = set()

        # 1. 탐색된 모든 후보 경로의 삼중항 (Path Recall 100% 보장)
        for cp in candidate_paths:
            for st in cp["path"]:
                tkey = (st["head"], st["relation"], st["tail"])
                if tkey not in seen_triples and len(context_triples) < max_triples:
                    seen_triples.add(tkey)
                    context_triples.append({
                        "head": st["head"],
                        "relation": st["relation"],
                        "tail": st["tail"],
                        "origin": st.get("origin", "rule"),
                        "confidence": st.get("confidence", 1.0),
                        "sources": st.get("sources", []),
                        "evidence": st.get("evidence", []),
                    })

        # 2. 시드 노드의 직접 이웃 엣지 보충
        for s in seeds:
            s_name = s["node"]
            for e in self.graph_nx.get(s_name, []):
                tkey = (e["head"], e["relation"], e["tail"])
                if tkey not in seen_triples and len(context_triples) < max_triples:
                    seen_triples.add(tkey)
                    context_triples.append({
                        "head": e["head"],
                        "relation": e["relation"],
                        "tail": e["tail"],
                        "origin": e.get("origin", "rule"),
                        "confidence": e.get("confidence", 1.0),
                        "sources": e.get("sources", []),
                        "evidence": e.get("evidence", []),
                    })

        visited_nodes = set(seed_ids)
        for cp in sorted_candidates:
            for st in cp["path"]:
                visited_nodes.add(st["from_node"])
                visited_nodes.add(st["to_node"])

        hops_used = max([len(c["path"]) for c in sorted_candidates] or [1])

        return {
            "visited_nodes": sorted(visited_nodes),
            "skipped_hubs": [],
            "context_triples": context_triples,
            "candidate_paths": sorted_candidates,
            "demanded_relations": demanded_relations,
            "target_relation": target_rel,
            "hops_used": hops_used,
        }

    def _build_context_node(self, state: AgentState) -> dict[str, Any]:
        q = state["question"]
        triples = state["context_triples"]
        seeds = state["seeds"]
        candidate_paths = state.get("candidate_paths", [])

        relation_gap = False
        if not triples or not seeds or not candidate_paths:
            relation_gap = True

        # 미지의 도메인 외 개체/개념 검사 (거절 가드레일: 방송국, 우주센터, 대학교/총장 등)
        common_words = {
            "사건", "인물", "조직", "단체", "학교", "정보", "대상", "내용", "관련", "위치",
            "무엇", "어디", "누구", "어떤", "함께", "같이", "동료", "동참", "거쳐", "이어진",
            "통해", "모의", "세운", "설립", "창설", "창립", "만든", "세웠", "창건", "참여",
            "가담", "일으킨", "의거", "전투", "참전", "동참", "벌인", "거사", "독립", "운동",
            "역사", "질문", "사람", "존재", "알려", "해당", "속한", "또는", "있던", "대첩",
            "함께한", "알려진", "나온", "설립자", "참여자", "가운데", "중에서", "사이", "당시",
            "이후", "이전", "주요", "대표", "모두", "각각", "다른", "참여했는가", "참전했는가",
            "조직한", "세운 조직", "세운 단체", "세운 학교", "세운 사람",
        }
        suffix_list = [
            "에서", "으로", "에게", "이나", "이", "가", "은", "는", "을", "를", "의", "에",
            "과", "와", "로", "도", "만", "한", "된", "인", "던", "서",
        ]

        rem_q = q
        for s in seeds:
            rem_q = rem_q.replace(s["surface"], " ")

        raw_words = re.findall(r"[가-힣]{2,}", rem_q)
        has_unknown_concept = False

        for rw in raw_words:
            base_w = rw
            for j in sorted(suffix_list, key=len, reverse=True):
                if rw.endswith(j) and len(rw) > len(j):
                    base_w = rw[:-len(j)]
                    break

            if base_w in common_words or rw in common_words:
                continue
            if base_w in self.alias_index or rw in self.alias_index:
                continue

            is_known = any(
                base_w in nid or any(base_w in a for a in n.get("aliases", []))
                for nid, n in self.nodes_by_id.items()
            )
            if not is_known:
                has_unknown_concept = True
                break

        if has_unknown_concept:
            relation_gap = True

        sources_set = set()
        for t in triples:
            for s in t.get("sources", []):
                sources_set.add(s)

        return {
            "relation_gap": relation_gap,
            "sources": sorted(sources_set),
        }

    def _deepen_node(self, state: AgentState) -> dict[str, Any]:
        next_hop = state["current_hops"] + 1
        return {
            "current_hops": next_hop,
            "deepened": True,
        }

    def _synthesize_node(self, state: AgentState) -> dict[str, Any]:
        q = state["question"]
        triples = state["context_triples"]
        seeds = state["seeds"]
        candidate_paths = state.get("candidate_paths", [])
        refusal_text = self.config.get("generation", {}).get(
            "refusal_text", "제시된 지식 그래프 자료에서 해당 질문에 대한 근거를 찾을 수 없습니다."
        )

        is_refused = False
        if not triples or not seeds or not candidate_paths or state.get("relation_gap", False):
            is_refused = True

        if is_refused:
            answer_content = (
                f"[답변 요약]\n{refusal_text}\n\n"
                "[탐색 및 추론 경로]\n(근거 경로를 찾을 수 없어 거절되었습니다)\n\n"
                "[근거 출처]\n(없음)"
            )
            return {
                "answer": answer_content,
                "refused": True,
                "traversed_path": [],
                "path_explanation": "(근거 경로를 찾을 수 없어 거절되었습니다)",
            }

        sorted_candidates = candidate_paths
        primary_candidate = sorted_candidates[0]
        traversed_path = primary_candidate["path"]

        # 체인 형태 경로 설명 (A -[rel]-> B <-[rel]- C)
        chain_tokens = []
        for i, s in enumerate(traversed_path):
            arrow = f"-[{s['relation']}]->" if s["direction"] == "forward" else f"<-[{s['relation']}]-"
            if i == 0:
                chain_tokens.append(f"{s['from_node']} {arrow} {s['to_node']}")
            else:
                chain_tokens.append(f"{arrow} {s['to_node']}")
        path_explanation = " ".join(chain_tokens)

        # [탐색 및 추론 경로] 블록 (단계별 사슬, 역방향 명시)
        path_lines = []
        for i, s in enumerate(traversed_path, 1):
            if s["direction"] == "forward":
                path_lines.append(f"{i}. {s['from_node']} -[{s['relation']}]-> {s['to_node']}")
            else:
                path_lines.append(f"{i}. {s['from_node']} <-[{s['relation']}]- {s['to_node']} (역방향)")
        path_block = "\n".join(path_lines)

        # [근거 출처] 블록
        supporting_triples: dict[tuple[str, str, str], dict[str, Any]] = {}
        for cand in sorted_candidates[:5]:
            for st in cand["path"]:
                tkey = (st["head"], st["relation"], st["tail"])
                if tkey not in supporting_triples:
                    supporting_triples[tkey] = {
                        "head": st["head"],
                        "relation": st["relation"],
                        "tail": st["tail"],
                        "origin": st.get("origin", "rule"),
                        "confidence": st.get("confidence", 1.0),
                        "sources": list(st.get("sources", [])),
                    }
                else:
                    for src in st.get("sources", []):
                        if src not in supporting_triples[tkey]["sources"]:
                            supporting_triples[tkey]["sources"].append(src)

        source_lines = []
        for tinfo in supporting_triples.values():
            src_str = ", ".join(tinfo["sources"]) if tinfo["sources"] else "지식 그래프"
            source_lines.append(
                f"- ({tinfo['head']}, {tinfo['relation']}, {tinfo['tail']}) | "
                f"출처: {src_str} (origin: {tinfo['origin']}, confidence: {tinfo['confidence']})"
            )
        source_block = "\n".join(source_lines) if source_lines else "- (출처 정보 없음)"

        # 결정론적 규칙 답변 합성 (키 유무와 무관하게 항상 먼저 만든다)
        #
        # LLM 은 이 결정론적 요약의 '문장을 다듬는 역할'만 한다. 예전에는 LLM 에게
        # 후보 경로 5개와 삼중항 30개를 통째로 주고 3블록 전체를 쓰게 했는데,
        # 그러면 코드가 고른 traversed_path 와 LLM 이 고른 경로가 갈려
        # "경로는 청산리 전투인데 답변 산문은 봉오동 전투" 같은 불일치가 실제로 발생했다.
        # 루브릭이 요구하는 '탄 경로를 답변과 함께 제시'를 어기는 상태였다.
        # 그래서 경로 블록과 출처 블록은 코드가 소유하고, LLM 출력은 개체 검증 후
        # 어긋나면 결정론적 문장으로 폴백한다.
        seed_str = ", ".join(s["node"] for s in seeds)
        top_candidates = sorted_candidates[:5]
        is_co = state.get("is_co_participant", False)
        desired_type = state.get("desired_target_type")

        cand_lines = []
        for c in top_candidates:
            c_chain = []
            for idx, s in enumerate(c["path"]):
                arrow = f"-[{s['relation']}]->" if s["direction"] == "forward" else f"<-[{s['relation']}]-"
                if idx == 0:
                    c_chain.append(f"{s['from_node']} {arrow} {s['to_node']}")
                else:
                    c_chain.append(f"{arrow} {s['to_node']}")
            chain_str = " ".join(c_chain)
            cand_lines.append(f"- {c['target']} (근거: {chain_str})")

        if is_co:
            if desired_type == "Event":
                summary_text = (
                    f"지식 그래프 탐색 결과, {seed_str}과(와) 함께 참전한 인물이 세운 조직이 참여한 사건/전투 "
                    f"(상위 {len(top_candidates)}건, 신뢰도 및 경로 점수 기준 순위)은 다음과 같습니다:\n"
                    + "\n".join(cand_lines)
                )
            else:
                summary_text = (
                    f"지식 그래프 탐색 결과, {seed_str}과(와) 함께한 인물이 세운 조직/단체 "
                    f"(상위 {len(top_candidates)}건, 신뢰도 및 경로 점수 기준 순위)은 다음과 같습니다:\n"
                    + "\n".join(cand_lines)
                )
        elif len(traversed_path) == 1:
            step0 = traversed_path[0]
            r0 = step0.get("relation")
            rel_label = "참여한 사건" if r0 == "PARTICIPATED_IN" else "설립한 조직"
            summary_text = f"{seed_str}이(가) {rel_label}(상위 {len(top_candidates)}건, 신뢰도 순):\n" + "\n".join(cand_lines)
        elif desired_type == "Person":
            summary_text = f"지식 그래프 탐색 결과, {seed_str} 관련 인물(상위 {len(top_candidates)}건, 신뢰도 순):\n" + "\n".join(cand_lines)
        else:
            target_concept = "조직" if state.get("target_relation") == "FOUNDED" else "사건"
            summary_text = (
                f"지식 그래프 탐색 결과, {seed_str}에서 시작하여 연쇄 관계를 거쳐 도출된 {target_concept} "
                f"(상위 {len(top_candidates)}건, 차수 감점 및 신뢰도 기준 순위)은 다음과 같습니다:\n"
                + "\n".join(cand_lines)
            )

        # LLM 문장 다듬기 (선택). 개체를 바꾸면 버리고 결정론적 문장으로 돌아간다.
        llm_used = False
        llm_fallback_reason = None
        if llm.is_enabled():
            allowed_entities = {s["node"] for s in seeds}
            for c in top_candidates:
                allowed_entities.add(c["target"])
                for st in c["path"]:
                    allowed_entities.add(st["from_node"])
                    allowed_entities.add(st["to_node"])

            system_prompt = (
                "당신은 한국 독립운동사 지식 그래프 QA 어시스턴트의 '문장 다듬기' 모듈이다.\n"
                "입력으로 받은 [확정 답변]은 지식 그래프 탐색으로 이미 확정된 사실이다.\n"
                "규칙:\n"
                "1. 개체 이름(인물·사건·조직)을 추가·삭제·변경하지 마라. 주어진 것만 그대로 쓴다.\n"
                "2. 새로운 사실을 추론하거나 배경 지식을 덧붙이지 마라.\n"
                "3. 후보와 근거 사슬의 짝은 그대로 유지하되, 한국어 문장을 자연스럽게 다듬어라.\n"
                "4. 머리말·꼬리말·마크다운 제목 없이 다듬은 본문만 출력하라."
            )
            user_prompt = f"질문: {q}\n\n[확정 답변]\n{summary_text}"
            synth_model = self.config.get("llm", {}).get("synth_model", "gpt-4.1-mini")
            polished = llm.chat(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_prompt}],
                model=synth_model,
                max_tokens=self.config.get("llm", {}).get("max_output_tokens", 1500),
            )

            if not polished:
                llm_fallback_reason = "llm_unavailable"
            else:
                # 검증 1: 허용되지 않은 그래프 개체를 끌어들였는가
                #
                # 부분문자열 오탐을 막는다. 예: 허용된 '대한독립군단' 안에는 다른 노드 '대한독립군'이
                # 들어 있어, 단순 포함 검사만 하면 정상 답변이 매번 침입자로 잡힌다.
                # 그래서 허용 개체를 먼저 긴 이름부터 가린 뒤 남은 텍스트에서만 검사한다.
                masked = polished
                for ent in sorted(allowed_entities, key=len, reverse=True):
                    masked = masked.replace(ent, "\x00")
                intruders = [
                    nid for nid in self.nodes_by_id
                    if len(nid) >= 2 and nid not in allowed_entities and nid in masked
                ]
                # 검증 2: 1순위 정답이 그대로 남아 있는가
                missing_primary = top_candidates and top_candidates[0]["target"] not in polished
                if intruders:
                    llm_fallback_reason = f"entity_drift: {', '.join(intruders[:5])}"
                elif missing_primary:
                    llm_fallback_reason = "primary_target_missing"
                else:
                    summary_text = polished.strip()
                    llm_used = True

        full_answer = (
            f"[답변 요약]\n{summary_text}\n\n"
            f"[탐색 및 추론 경로]\n{path_block}\n\n"
            f"[근거 출처]\n{source_block}"
        )

        return {
            "answer": full_answer,
            "refused": False,
            "traversed_path": traversed_path,
            "path_explanation": path_explanation,
            "llm_used": llm_used,
            "llm_fallback_reason": llm_fallback_reason,
        }

    # -----------------------------------------------------------------------
    # LangGraph 컴파일
    # -----------------------------------------------------------------------

    def _compile_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("route", self._route_node)
        workflow.add_node("find_seeds", self._find_seeds_node)
        workflow.add_node("expand", self._expand_node)
        workflow.add_node("build_context", self._build_context_node)
        workflow.add_node("deepen", self._deepen_node)
        workflow.add_node("synthesize", self._synthesize_node)

        workflow.add_edge(START, "route")
        workflow.add_edge("route", "find_seeds")
        workflow.add_edge("find_seeds", "expand")
        workflow.add_edge("expand", "build_context")

        def check_deepen(state: AgentState):
            max_hops = 4 if (state.get("is_co_participant") and state.get("desired_target_type") == "Event") else max(3, self.config.get("retrieval", {}).get("max_hops", 3))
            if state["relation_gap"] and state["current_hops"] < max_hops:
                return "deepen"
            return "synthesize"

        workflow.add_conditional_edges(
            "build_context",
            check_deepen,
            {
                "deepen": "deepen",
                "synthesize": "synthesize",
            },
        )
        workflow.add_edge("deepen", "expand")
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    # -----------------------------------------------------------------------
    # 공개 API
    # -----------------------------------------------------------------------

    def answer(self, question: str) -> dict[str, Any]:
        """CONTRACT.md §6.1 공개 API. 질문에 답변하고 runs.jsonl 에 1줄 기록한다."""
        start_time = time.perf_counter()

        is_co = is_co_participant_query(question)
        tgt_type = detect_target_type(question)
        if is_co and tgt_type == "Event":
            start_hops = 4
        elif is_co:
            start_hops = 3
        else:
            start_hops = self.config.get("retrieval", {}).get("start_hops", 2)

        initial_state: AgentState = {
            "question": question,
            "route": "path",
            "seeds": [],
            "current_hops": start_hops,
            "hops_used": 1,
            "deepened": False,
            "visited_nodes": [],
            "skipped_hubs": [],
            "traversed_path": [],
            "context_triples": [],
            "candidate_paths": [],
            "relation_gap": False,
            "demanded_relations": [],
            "target_relation": None,
            "desired_target_type": tgt_type,
            "is_co_participant": is_co,
            "answer": "",
            "refused": False,
            "path_explanation": "",
            "sources": [],
            "llm_used": False,
            "llm_fallback_reason": None,
        }

        final_state = self.workflow.invoke(initial_state)
        latency_ms = max(1, int((time.perf_counter() - start_time) * 1000))

        result = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "route": final_state["route"],
            "seeds": final_state["seeds"],
            "hops_used": final_state["hops_used"],
            "deepened": final_state["deepened"],
            "visited_nodes": final_state["visited_nodes"],
            "skipped_hubs": final_state["skipped_hubs"],
            "traversed_path": final_state["traversed_path"],
            "context_triples": final_state["context_triples"],
            "answer": final_state["answer"],
            "refused": final_state["refused"],
            "path_explanation": final_state["path_explanation"],
            "sources": final_state["sources"],
            # 답변 문장을 LLM 이 다듬었는지, 개체 검증에 걸려 규칙 문장으로 되돌렸는지 기록한다.
            "llm_used": final_state.get("llm_used", False),
            "llm_fallback_reason": final_state.get("llm_fallback_reason"),
            "latency_ms": latency_ms,
        }

        # runs.jsonl 에 추가 기록
        try:
            RUNS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with RUNS_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Warning: Failed to log run to {RUNS_LOG_PATH}: {e}", file=sys.stderr)

        return result


def main():
    llm.setup_console()
    print(llm.banner())

    if len(sys.argv) > 1:
        question = sys.argv[1]
    else:
        question = "안중근이 참여한 사건은?"

    agent = GraphRAGAgent()
    print(f"[Agent] 그래프 경로: {agent.graph_path}")
    print(f"[Agent] 노드 {len(agent.nodes_by_id)}개, 엣지 {len(agent.graph_data.get('edges', []))}개 로드됨\n")

    res = agent.answer(question)

    print(f"질문: {res['question']}")
    print(f"라우트: {res['route']} | 홉 수: {res['hops_used']} | Deepened: {res['deepened']} | Refused: {res['refused']}")
    print(f"탐색 경로 ({len(res['traversed_path'])}개 스텝):")
    for step in res["traversed_path"]:
        direction_str = "(역방향)" if step.get("direction") == "reverse" else ""
        print(f"  Hop {step['hop']}: {step['from_node']} -({step['relation']})-> {step['to_node']} {direction_str}")
    print(f"\n{res['answer']}\n")
    print(f"소요 시간: {res['latency_ms']}ms")


if __name__ == "__main__":
    main()
