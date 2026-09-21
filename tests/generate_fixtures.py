"""tests/generate_fixtures.py — Generate fixture_graph.json and fixture_goldenset.json

Follows CONTRACT.md §4.3 and §5 precisely.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = ROOT / "tests"

# ---------------------------------------------------------------------------
# Fixture Graph Data
# ---------------------------------------------------------------------------

RAW_NODES = [
    # Person
    {"id": "안중근", "type": "Person", "aliases": ["안중근", "安重根", "도마 안중근"], "sources": ["안중근", "하얼빈 의거"], "doc_count": 2},
    {"id": "우덕순", "type": "Person", "aliases": ["우덕순", "禹德淳"], "sources": ["우덕순", "하얼빈 의거"], "doc_count": 2},
    {"id": "김구", "type": "Person", "aliases": ["김구", "金九", "백범 김구"], "sources": ["김구", "대한민국 임시정부", "한인애국단"], "doc_count": 3},
    {"id": "윤봉길", "type": "Person", "aliases": ["윤봉길", "尹奉吉", "매헌 윤봉길"], "sources": ["윤봉길", "훙커우 공원 의거"], "doc_count": 2},
    {"id": "이봉창", "type": "Person", "aliases": ["이봉창", "李奉昌"], "sources": ["이봉창", "도쿄 의거"], "doc_count": 2},
    {"id": "신채호", "type": "Person", "aliases": ["신채호", "申采浩", "단재 신채호"], "sources": ["신채호", "신민회"], "doc_count": 2},
    {"id": "안창호", "type": "Person", "aliases": ["안창호", "安昌浩", "도산 안창호"], "sources": ["안창호", "신민회"], "doc_count": 2},
    {"id": "김좌진", "type": "Person", "aliases": ["김좌진", "金佐鎭", "백야 김좌진"], "sources": ["김좌진", "청산리 전투", "북로군정서"], "doc_count": 3},
    {"id": "홍범도", "type": "Person", "aliases": ["홍범도", "洪範圖"], "sources": ["홍범도", "봉오동 전투", "청산리 전투"], "doc_count": 3},
    {"id": "이동휘", "type": "Person", "aliases": ["이동휘", "李東輝"], "sources": ["이동휘", "대한민국 임시정부"], "doc_count": 2},
    {"id": "이회영", "type": "Person", "aliases": ["이회영", "李會榮", "우당 이회영"], "sources": ["이회영", "신흥무관학교", "신민회"], "doc_count": 3},

    # Event
    {"id": "하얼빈 의거", "type": "Event", "aliases": ["하얼빈 의거", "하얼빈역 의거", "이토 히로부미 저격 사건"], "sources": ["안중근", "우덕순", "하얼빈 의거"], "doc_count": 3},
    {"id": "훙커우 공원 의거", "type": "Event", "aliases": ["훙커우 공원 의거", "상하이 의거", "훙커우 공원 폭탄 투척"], "sources": ["윤봉길", "한인애국단"], "doc_count": 2},
    {"id": "도쿄 의거", "type": "Event", "aliases": ["도쿄 의거", "사쿠라다문 의거"], "sources": ["이봉창", "한인애국단"], "doc_count": 2},
    {"id": "3·1 운동", "type": "Event", "aliases": ["3·1 운동", "3.1 운동", "삼일운동", "기미독립운동"], "sources": ["김구", "신채호", "대한민국 임시정부"], "doc_count": 3},
    {"id": "청산리 전투", "type": "Event", "aliases": ["청산리 전투", "청산리 대첩"], "sources": ["김좌진", "홍범도", "북로군정서"], "doc_count": 3},
    {"id": "봉오동 전투", "type": "Event", "aliases": ["봉오동 전투", "봉오동 대첩"], "sources": ["홍범도", "대한독립군"], "doc_count": 2},

    # Organization
    {"id": "한인애국단", "type": "Organization", "aliases": ["한인애국단", "韓人愛國團"], "sources": ["김구", "윤봉길", "이봉창"], "doc_count": 3},
    {"id": "대한민국 임시정부", "type": "Organization", "aliases": ["대한민국 임시정부", "임시정부", "상해 임시정부"], "sources": ["김구", "이동휘", "대한민국 임시정부"], "doc_count": 3},
    {"id": "신민회", "type": "Organization", "aliases": ["신민회", "新民會"], "sources": ["안창호", "신채호", "이회영"], "doc_count": 3},
    {"id": "대한국민회", "type": "Organization", "aliases": ["대한국민회", "大韓國民會"], "sources": ["우덕순"], "doc_count": 1},
    {"id": "북로군정서", "type": "Organization", "aliases": ["북로군정서", "北路軍政署"], "sources": ["김좌진", "청산리 전투"], "doc_count": 2},
    {"id": "신흥무관학교", "type": "Organization", "aliases": ["신흥무관학교", "新興武官學校"], "sources": ["이회영"], "doc_count": 1},
    {"id": "대한독립군", "type": "Organization", "aliases": ["대한독립군", "大韓獨立軍"], "sources": ["홍범도", "봉오동 전투"], "doc_count": 2},
]

RAW_EDGES = [
    # 1. 안중근 -PARTICIPATED_IN-> 하얼빈 의거
    {
        "head": "안중근",
        "relation": "PARTICIPATED_IN",
        "tail": "하얼빈 의거",
        "origin": "rule",
        "confidence": 0.95,
        "sources": ["안중근"],
        "evidence": ["1909년 10월 26일 안중근은 하얼빈 역에서 이토 히로부미를 사살하였다."],
        "extractor": "rule_participated_v1",
    },
    # 2. 우덕순 -PARTICIPATED_IN-> 하얼빈 의거
    {
        "head": "우덕순",
        "relation": "PARTICIPATED_IN",
        "tail": "하얼빈 의거",
        "origin": "rule",
        "confidence": 0.95,
        "sources": ["우덕순"],
        "evidence": ["우덕순은 안중근과 함께 하얼빈 의거를 모의하고 거사에 동참하였다."],
        "extractor": "rule_participated_v1",
    },
    # 3. 우덕순 -FOUNDED-> 대한국민회
    {
        "head": "우덕순",
        "relation": "FOUNDED",
        "tail": "대한국민회",
        "origin": "rule",
        "confidence": 0.90,
        "sources": ["우덕순"],
        "evidence": ["우덕순은 블라디보스토크에서 대한국민회를 창립하고 독립운동을 전개하였다."],
        "extractor": "rule_founded_v1",
    },
    # 4. 김구 -PARTICIPATED_IN-> 3·1 운동
    {
        "head": "김구",
        "relation": "PARTICIPATED_IN",
        "tail": "3·1 운동",
        "origin": "rule",
        "confidence": 0.90,
        "sources": ["김구"],
        "evidence": ["김구는 1919년 3·1 운동 이후 상하이로 망명하여 독립운동에 투신하였다."],
        "extractor": "rule_participated_v2",
    },
    # 5. 김구 -FOUNDED-> 한인애국단
    {
        "head": "김구",
        "relation": "FOUNDED",
        "tail": "한인애국단",
        "origin": "rule",
        "confidence": 0.95,
        "sources": ["김구", "한인애국단"],
        "evidence": ["김구는 1931년 임시정부의 침체를 극복하기 위해 한인애국단을 조직하였다."],
        "extractor": "rule_founded_v1",
    },
    # 6. 김구 -FOUNDED-> 대한민국 임시정부
    {
        "head": "김구",
        "relation": "FOUNDED",
        "tail": "대한민국 임시정부",
        "origin": "llm",
        "confidence": 0.85,
        "sources": ["김구"],
        "evidence": ["김구는 상하이에서 대한민국 임시정부의 기틀을 마련하고 정부를 이끌었다."],
        "extractor": "llm_relation_extractor",
    },
    # 7. 윤봉길 -PARTICIPATED_IN-> 훙커우 공원 의거
    {
        "head": "윤봉길",
        "relation": "PARTICIPATED_IN",
        "tail": "훙커우 공원 의거",
        "origin": "rule",
        "confidence": 0.95,
        "sources": ["윤봉길"],
        "evidence": ["1932년 윤봉길은 훙커우 공원 의거를 감행하여 일본 군관민 수뇌부를 처단하였다."],
        "extractor": "rule_participated_v1",
    },
    # 8. 이봉창 -PARTICIPATED_IN-> 도쿄 의거
    {
        "head": "이봉창",
        "relation": "PARTICIPATED_IN",
        "tail": "도쿄 의거",
        "origin": "rule",
        "confidence": 0.95,
        "sources": ["이봉창"],
        "evidence": ["1932년 1월 이봉창은 도쿄 사쿠라다문 앞에서 일왕에게 폭탄을 투척하는 도쿄 의거를 벌였다."],
        "extractor": "rule_participated_v1",
    },
    # 9. 한인애국단 -PARTICIPATED_IN-> 훙커우 공원 의거
    {
        "head": "한인애국단",
        "relation": "PARTICIPATED_IN",
        "tail": "훙커우 공원 의거",
        "origin": "rule",
        "confidence": 0.92,
        "sources": ["한인애국단", "윤봉길"],
        "evidence": ["한인애국단은 윤봉길을 파견하여 상하이 훙커우 공원 의거를 성공적으로 결행하였다."],
        "extractor": "rule_participated_v3",
    },
    # 10. 한인애국단 -PARTICIPATED_IN-> 도쿄 의거
    {
        "head": "한인애국단",
        "relation": "PARTICIPATED_IN",
        "tail": "도쿄 의거",
        "origin": "rule",
        "confidence": 0.92,
        "sources": ["한인애국단", "이봉창"],
        "evidence": ["한인애국단 소속 이봉창 의사가 도쿄 의거를 일으켰다."],
        "extractor": "rule_participated_v3",
    },
    # 11. 안창호 -FOUNDED-> 신민회
    {
        "head": "안창호",
        "relation": "FOUNDED",
        "tail": "신민회",
        "origin": "rule",
        "confidence": 0.95,
        "sources": ["안창호", "신민회"],
        "evidence": ["안창호는 1907년 양기탁, 신채호 등과 함께 비밀결사 신민회를 창립하였다."],
        "extractor": "rule_founded_v1",
    },
    # 12. 신채호 -FOUNDED-> 신민회
    {
        "head": "신채호",
        "relation": "FOUNDED",
        "tail": "신민회",
        "origin": "rule",
        "confidence": 0.95,
        "sources": ["신채호", "신민회"],
        "evidence": ["신채호는 신민회에 발기인으로 참여하여 항일 구국 운동을 이끌었다."],
        "extractor": "rule_founded_v1",
    },
    # 13. 신채호 -PARTICIPATED_IN-> 3·1 운동
    {
        "head": "신채호",
        "relation": "PARTICIPATED_IN",
        "tail": "3·1 운동",
        "origin": "rule",
        "confidence": 0.88,
        "sources": ["신채호"],
        "evidence": ["신채호는 1919년 3·1 운동의 정신을 계승하여 무장 독립 투쟁론을 주창하였다."],
        "extractor": "rule_participated_v2",
    },
    # 14. 이회영 -FOUNDED-> 신흥무관학교
    {
        "head": "이회영",
        "relation": "FOUNDED",
        "tail": "신흥무관학교",
        "origin": "rule",
        "confidence": 0.95,
        "sources": ["이회영", "신흥무관학교"],
        "evidence": ["이회영 6형제는 만주로 망명하여 신흥무관학교를 설립하고 독립군을 양성하였다."],
        "extractor": "rule_founded_v1",
    },
    # 15. 이회영 -FOUNDED-> 신민회
    {
        "head": "이회영",
        "relation": "FOUNDED",
        "tail": "신민회",
        "origin": "rule",
        "confidence": 0.95,
        "sources": ["이회영", "신민회"],
        "evidence": ["이회영은 1907년 안창호 등과 함께 신민회를 조직하였다."],
        "extractor": "rule_founded_v1",
    },
    # 16. 김좌진 -FOUNDED-> 북로군정서
    {
        "head": "김좌진",
        "relation": "FOUNDED",
        "tail": "북로군정서",
        "origin": "rule",
        "confidence": 0.95,
        "sources": ["김좌진", "북로군정서"],
        "evidence": ["김좌진은 북로군정서 총사령관으로 독립군을 훈련시켰다."],
        "extractor": "rule_founded_v1",
    },
    # 17. 김좌진 -PARTICIPATED_IN-> 청산리 전투
    {
        "head": "김좌진",
        "relation": "PARTICIPATED_IN",
        "tail": "청산리 전투",
        "origin": "rule",
        "confidence": 0.95,
        "sources": ["김좌진", "청산리 전투"],
        "evidence": ["김좌진은 북로군정서를 이끌고 1920년 10월 청산리 전투에서 일본군을 대파하였다."],
        "extractor": "rule_participated_v1",
    },
    # 18. 북로군정서 -PARTICIPATED_IN-> 청산리 전투
    {
        "head": "북로군정서",
        "relation": "PARTICIPATED_IN",
        "tail": "청산리 전투",
        "origin": "rule",
        "confidence": 0.95,
        "sources": ["북로군정서", "청산리 전투"],
        "evidence": ["북로군정서는 청산리 전투에서 주력 부대로 참전하여 큰 전과를 올렸다."],
        "extractor": "rule_participated_v3",
    },
    # 19. 홍범도 -PARTICIPATED_IN-> 봉오동 전투
    {
        "head": "홍범도",
        "relation": "PARTICIPATED_IN",
        "tail": "봉오동 전투",
        "origin": "rule",
        "confidence": 0.95,
        "sources": ["홍범도", "봉오동 전투"],
        "evidence": ["홍범도는 1920년 6월 대한독립군을 지휘하여 봉오동 전투에서 대승을 거두었다."],
        "extractor": "rule_participated_v1",
    },
    # 20. 홍범도 -PARTICIPATED_IN-> 청산리 전투
    {
        "head": "홍범도",
        "relation": "PARTICIPATED_IN",
        "tail": "청산리 전투",
        "origin": "rule",
        "confidence": 0.95,
        "sources": ["홍범도", "청산리 전투"],
        "evidence": ["홍범도의 연합부대는 김좌진의 북로군정서와 함께 청산리 전투에 참전하였다."],
        "extractor": "rule_participated_v1",
    },
    # 21. 대한독립군 -PARTICIPATED_IN-> 봉오동 전투
    {
        "head": "대한독립군",
        "relation": "PARTICIPATED_IN",
        "tail": "봉오동 전투",
        "origin": "rule",
        "confidence": 0.95,
        "sources": ["대한독립군", "봉오동 전투"],
        "evidence": ["대한독립군은 봉오동 전투에서 일본 정규군을 상대로 독립전쟁 첫 승리를 거두었다."],
        "extractor": "rule_participated_v3",
    },
    # 22. 이동휘 -PARTICIPATED_IN-> 3·1 운동
    {
        "head": "이동휘",
        "relation": "PARTICIPATED_IN",
        "tail": "3·1 운동",
        "origin": "rule",
        "confidence": 0.88,
        "sources": ["이동휘"],
        "evidence": ["이동휘는 3·1 운동 이후 해외 무장 독립투쟁을 지원하였다."],
        "extractor": "rule_participated_v2",
    },
    # 23. 이동휘 -FOUNDED-> 대한민국 임시정부
    {
        "head": "이동휘",
        "relation": "FOUNDED",
        "tail": "대한민국 임시정부",
        "origin": "llm",
        "confidence": 0.85,
        "sources": ["이동휘"],
        "evidence": ["이동휘는 대한민국 임시정부의 초대 국무총리로 취임하여 통합정부 수립에 기여하였다."],
        "extractor": "llm_relation_extractor",
    },
    # 24. 대한민국 임시정부 -PARTICIPATED_IN-> 3·1 운동
    {
        "head": "대한민국 임시정부",
        "relation": "PARTICIPATED_IN",
        "tail": "3·1 운동",
        "origin": "rule",
        "confidence": 0.90,
        "sources": ["대한민국 임시정부"],
        "evidence": ["대한민국 임시정부는 3·1 운동의 정신을 계승하여 1919년 수립되었다."],
        "extractor": "rule_participated_v3",
    },
]

def build_fixture_graph():
    # Calculate degree for each node from edges
    deg_map: dict[str, int] = {}
    for edge in RAW_EDGES:
        deg_map[edge["head"]] = deg_map.get(edge["head"], 0) + 1
        deg_map[edge["tail"]] = deg_map.get(edge["tail"], 0) + 1

    nodes = []
    alias_index: dict[str, str] = {}

    for n in RAW_NODES:
        node_id = n["id"]
        degree = deg_map.get(node_id, 0)
        # Note: Set "대한민국 임시정부" degree to 26 in metadata to test hub avoidance threshold if needed
        # But let's check its natural degree: deg_map.get("대한민국 임시정부", 0)
        # For testing hub threshold (config is 25), we can set a specific hub attribute or natural degree
        node_dict = {
            "id": node_id,
            "type": n["type"],
            "aliases": n["aliases"],
            "sources": n["sources"],
            "doc_count": n["doc_count"],
            "degree": degree,
        }
        nodes.append(node_dict)

        for alias in n["aliases"]:
            alias_index[alias] = node_id

    # Compute stats
    by_type: dict[str, int] = {}
    for n in nodes:
        by_type[n["type"]] = by_type.get(n["type"], 0) + 1

    by_relation: dict[str, int] = {}
    by_origin: dict[str, int] = {}
    for e in RAW_EDGES:
        by_relation[e["relation"]] = by_relation.get(e["relation"], 0) + 1
        by_origin[e["origin"]] = by_origin.get(e["origin"], 0) + 1

    graph = {
        "nodes": nodes,
        "edges": RAW_EDGES,
        "alias_index": alias_index,
        "stats": {
            "nodes": len(nodes),
            "edges": len(RAW_EDGES),
            "by_type": by_type,
            "by_relation": by_relation,
            "by_origin": by_origin,
        },
    }
    return graph

# ---------------------------------------------------------------------------
# Fixture Golden Set Data
# ---------------------------------------------------------------------------

GOLDENSET = {
    "version": "1.0",
    "scoring_criteria": {
        "answer_correct": "기대 정답 문자열(또는 aliases 중 하나)이 답변에 포함되면 1.0, 부분 포함 0.5, 없으면 0.0",
        "path_recall": "expected_path의 삼중항 중 실제 수집된 근거 삼중항에 포함된 비율",
        "refusal_correct": "unanswerable 문항에서 config.generation.refusal_text 를 출력하면 1.0",
    },
    "items": [
        # 1-hop (4 items)
        {
            "id": "Q01",
            "hops": 1,
            "type": "local",
            "question": "안중근이 참여한 사건은?",
            "expected_answer": ["하얼빈 의거"],
            "expected_path": [
                {"head": "안중근", "relation": "PARTICIPATED_IN", "tail": "하얼빈 의거"}
            ],
            "evidence": [
                {"doc": "안중근", "quote": "1909년 10월 26일 안중근은 하얼빈 역에서 이토 히로부미를 사살하였다."}
            ],
            "answerable": True,
        },
        {
            "id": "Q02",
            "hops": 1,
            "type": "local",
            "question": "김구가 설립한 조직은?",
            "expected_answer": ["한인애국단", "대한민국 임시정부"],
            "expected_path": [
                {"head": "김구", "relation": "FOUNDED", "tail": "한인애국단"}
            ],
            "evidence": [
                {"doc": "김구", "quote": "김구는 1931년 임시정부의 침체를 극복하기 위해 한인애국단을 조직하였다."}
            ],
            "answerable": True,
        },
        {
            "id": "Q03",
            "hops": 1,
            "type": "local",
            "question": "김좌진이 참전한 사건은?",
            "expected_answer": ["청산리 전투"],
            "expected_path": [
                {"head": "김좌진", "relation": "PARTICIPATED_IN", "tail": "청산리 전투"}
            ],
            "evidence": [
                {"doc": "김좌진", "quote": "김좌진은 북로군정서를 이끌고 1920년 10월 청산리 전투에서 일본군을 대파하였다."}
            ],
            "answerable": True,
        },
        {
            "id": "Q04",
            "hops": 1,
            "type": "local",
            "question": "이회영이 설립한 학교 또는 조직은?",
            "expected_answer": ["신흥무관학교", "신민회"],
            "expected_path": [
                {"head": "이회영", "relation": "FOUNDED", "tail": "신흥무관학교"}
            ],
            "evidence": [
                {"doc": "이회영", "quote": "이회영 6형제는 만주로 망명하여 신흥무관학교를 설립하고 독립군을 양성하였다."}
            ],
            "answerable": True,
        },

        # 2-hop (5 items)
        {
            "id": "Q05",
            "hops": 2,
            "type": "path",
            "question": "안중근이 참여한 사건에 함께 참여한 인물은?",
            "expected_answer": ["우덕순"],
            "expected_path": [
                {"head": "안중근", "relation": "PARTICIPATED_IN", "tail": "하얼빈 의거"},
                {"head": "우덕순", "relation": "PARTICIPATED_IN", "tail": "하얼빈 의거"}
            ],
            "evidence": [
                {"doc": "우덕순", "quote": "우덕순은 안중근과 함께 하얼빈 의거를 모의하고 거사에 동참하였다."}
            ],
            "answerable": True,
        },
        {
            "id": "Q06",
            "hops": 2,
            "type": "path",
            "question": "김구가 설립한 조직이 참여한 사건은?",
            "expected_answer": ["훙커우 공원 의거", "도쿄 의거"],
            "expected_path": [
                {"head": "김구", "relation": "FOUNDED", "tail": "한인애국단"},
                {"head": "한인애국단", "relation": "PARTICIPATED_IN", "tail": "훙커우 공원 의거"}
            ],
            "evidence": [
                {"doc": "한인애국단", "quote": "한인애국단은 윤봉길을 파견하여 상하이 훙커우 공원 의거를 성공적으로 결행하였다."}
            ],
            "answerable": True,
        },
        {
            "id": "Q07",
            "hops": 2,
            "type": "path",
            "question": "안중근이 참여한 사건에 같이 있던 인물이 세운 조직은?",
            "expected_answer": ["대한국민회"],
            "expected_path": [
                {"head": "안중근", "relation": "PARTICIPATED_IN", "tail": "하얼빈 의거"},
                {"head": "우덕순", "relation": "PARTICIPATED_IN", "tail": "하얼빈 의거"},
                {"head": "우덕순", "relation": "FOUNDED", "tail": "대한국민회"}
            ],
            "evidence": [
                {"doc": "우덕순", "quote": "우덕순은 블라디보스토크에서 대한국민회를 창립하고 독립운동을 전개하였다."}
            ],
            "answerable": True,
        },
        {
            "id": "Q08",
            "hops": 2,
            "type": "path",
            "question": "윤봉길이 참여한 사건에 관련된 조직을 창설한 인물은?",
            "expected_answer": ["김구"],
            "expected_path": [
                {"head": "윤봉길", "relation": "PARTICIPATED_IN", "tail": "훙커우 공원 의거"},
                {"head": "한인애국단", "relation": "PARTICIPATED_IN", "tail": "훙커우 공원 의거"},
                {"head": "김구", "relation": "FOUNDED", "tail": "한인애국단"}
            ],
            "evidence": [
                {"doc": "김구", "quote": "김구는 1931년 임시정부의 침체를 극복하기 위해 한인애국단을 조직하였다."}
            ],
            "answerable": True,
        },
        {
            "id": "Q09",
            "hops": 2,
            "type": "path",
            "question": "3·1 운동에 참여한 신채호가 설립에 참여한 조직은?",
            "expected_answer": ["신민회"],
            "expected_path": [
                {"head": "신채호", "relation": "PARTICIPATED_IN", "tail": "3·1 운동"},
                {"head": "신채호", "relation": "FOUNDED", "tail": "신민회"}
            ],
            "evidence": [
                {"doc": "신민회", "quote": "신채호는 신민회에 발기인으로 참여하여 항일 구국 운동을 이끌었다."}
            ],
            "answerable": True,
        },

        # 3-hop (1 item)
        {
            "id": "Q10",
            "hops": 3,
            "type": "path",
            "question": "훙커우 공원 의거에 관련된 단체를 설립한 인물이 참여한 사건은?",
            "expected_answer": ["3·1 운동"],
            "expected_path": [
                {"head": "한인애국단", "relation": "PARTICIPATED_IN", "tail": "훙커우 공원 의거"},
                {"head": "김구", "relation": "FOUNDED", "tail": "한인애국단"},
                {"head": "김구", "relation": "PARTICIPATED_IN", "tail": "3·1 운동"}
            ],
            "evidence": [
                {"doc": "김구", "quote": "김구는 1919년 3·1 운동 이후 상하이로 망명하여 독립운동에 투신하였다."}
            ],
            "answerable": True,
        },

        # Unanswerable (2 items)
        {
            "id": "Q11",
            "hops": 2,
            "type": "unanswerable",
            "question": "안중근이 설립한 방송국은?",
            "expected_answer": [],
            "expected_path": [],
            "evidence": [],
            "answerable": False,
        },
        {
            "id": "Q12",
            "hops": 2,
            "type": "unanswerable",
            "question": "윤봉길이 세운 우주센터는?",
            "expected_answer": [],
            "expected_path": [],
            "evidence": [],
            "answerable": False,
        },
    ],
}

def main():
    graph = build_fixture_graph()
    graph_path = TESTS_DIR / "fixture_graph.json"
    goldenset_path = TESTS_DIR / "fixture_goldenset.json"

    graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    goldenset_path.write_text(json.dumps(GOLDENSET, ensure_ascii=False, indent=2), encoding="utf-8")

    # Validate that every expected_path triple in goldenset exists in graph
    edge_set = {(e["head"], e["relation"], e["tail"]) for e in graph["edges"]}
    for item in GOLDENSET["items"]:
        for ep in item["expected_path"]:
            triple = (ep["head"], ep["relation"], ep["tail"])
            assert triple in edge_set, f"Triple {triple} in {item['id']} not found in graph!"

    print(f"Generated {graph_path} with {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
    print(f"Generated {goldenset_path} with {len(GOLDENSET['items'])} items")

if __name__ == "__main__":
    main()
