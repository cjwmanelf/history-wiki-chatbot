# CONTRACT.md — 워커 간 인터페이스 계약 (변경 금지)

이 문서는 Worker A(데이터·그래프 층)와 Worker B(에이전트·평가·데모 층)가
**병렬로 작업하면서도 최종에 무결하게 결합**되도록 하는 고정 계약이다.
여기 정의된 파일 경로·키 이름·값 도메인은 **어느 워커도 단독으로 바꿀 수 없다.**
변경이 필요하면 코디네이터에게 `ask`로 질문한다.

---

## 0. 프로젝트 개요

- **과제**: 모두의연구소 「[실습 프로젝트] 지식 그래프 에이전트 만들기」
- **주제**: 역사·인물 (한국 독립운동사, 1890~1945)
- **스키마**: `Person` · `Event` · `Organization` / `PARTICIPATED_IN` · `FOUNDED`
- **대표 2홉 질문**: "A가 참여한 사건에 같이 있던 인물이 세운 조직은?"
- **커뮤니티 요약(전역 검색)은 구현 범위 밖** (과제 명시)

### 저장소 루트
```
C:\Users\cjwma\OneDrive\바탕 화면\HISTORY WIKI CHATBOT\
```
모든 상대 경로는 이 루트 기준이다.

### 최종 디렉터리 구조 (과제 지정 구조를 그대로 따른다)
```
HISTORY WIKI CHATBOT/
├── data/
│   ├── docs/              # 원본 문서 *.md  (Worker A)
│   ├── manifest.json      # 수집 기록       (Worker A)
│   └── goldenset.json     # 평가셋          (Worker A)
├── config.json            # 도메인 고정값: 노드·관계·반경·상한 (Worker A)
├── collect_corpus.py      # 위키백과 수집   (Worker A)
├── build_graph.py         # 추출 + 정제     (Worker A)
├── agent.py               # 시작개체→n홉→답변(경로기록) (Worker B)
├── basic_rag.py           # 대조군 RAG      (Worker B)
├── evaluate.py            # 홉수별 측정 + basic RAG 대조 (Worker B)
├── app.py                 # Streamlit 데모  (Worker B)
├── output/
│   ├── graph.graphml      # (Worker A)
│   ├── graph.json         # (Worker A)
│   ├── extraction_report.json # (Worker A)
│   ├── runs.jsonl         # (Worker B)
│   ├── eval.json          # (Worker B)
│   └── eval_report.md     # (Worker B)
├── README.md              # (Worker B)
└── REPORT.md              # (코디네이터)
```

---

## 1. 절대 규칙 (양쪽 공통)

1. **키가 있든 없든 무오류 실행이 최우선이다.**
   사용자는 OpenAI API 키를 **가지고 있지만 저장소에 넣지 않는다.** 실행 환경에는 키가
   설정되어 있지 않을 수 있다. 따라서
   `collect_corpus.py` / `build_graph.py` / `agent.py` / `evaluate.py` / `app.py` 는
   **키 없이도 끝까지 정상 종료**해야 한다 (루브릭 1: "전체 워크플로우가 오류 없이 실행되는가").
   - 기본 경로 = **결정론적 규칙 기반**. `origin: "rule"`.
   - LLM은 **선택적 증강**이다. 있으면 품질이 올라가고, 없으면 조용히 규칙으로 폴백한다.
   - **§1-A의 `llm_provider` 모듈만을 통해** LLM에 접근한다. 아래 §1-A 참조.
2. **인코딩**: 모든 파일 입출력은 `encoding="utf-8"` 명시. JSON 저장은
   `json.dump(..., ensure_ascii=False, indent=2)`.
   실행 스크립트는 `main()` 첫 줄에서 `llm_provider.setup_console()` 를 호출한다
   (Windows cp949 콘솔에서 한글 출력이 깨지거나 `UnicodeEncodeError` 로 죽는 것 방지).
3. **경로**: `pathlib.Path(__file__).resolve().parent` 기준 상대 경로. 하드코딩된 절대경로 금지.
4. **재현성**: 난수 사용 시 `random.seed(42)`. 정렬 없는 `set` 순회 결과를 그대로 출력하지 않는다.
5. **API 키를 저장소에 커밋하지 않는다.**
6. Python 3.14 / Windows. 설치된 패키지: `networkx`, `streamlit`, `gradio`, `requests`,
   `langgraph`, `openai`, `python-dotenv`, `scikit-learn`, `numpy`, `pymupdf`.
   `rank_bm25`는 **없다** → BM25가 필요하면 직접 구현하거나 `sklearn`의 TF-IDF를 쓴다.
   **새 패키지 설치 금지.**

---

## 1-A. LLM 접근 규약 — `llm_provider.py` (코디네이터 소유, **수정 금지**)

사용자는 OpenAI 키를 보유하지만 **유출을 원치 않는다.** 그래서 키를 만지는 지점을
`llm_provider.py` **하나로 통일**했다. 이 파일은 이미 작성되어 있다.

### 절대 금지
- `os.environ["OPENAI_API_KEY"]` 를 워커 코드에서 **직접 읽지 않는다.**
- 키를 변수·설정파일·JSON 산출물·로그·예외 메시지에 **절대 남기지 않는다.**
- `.env`, `.streamlit/secrets.toml` 을 저장소에 커밋하지 않는다 (`.gitignore` 로 이미 차단됨).
- 키 값을 화면에 표시해야 하면 반드시 `llm_provider.mask(key)` 를 쓴다.

### 사용법
```python
import llm_provider as llm

llm.setup_console()
print(llm.banner())          # [LLM] 규칙 전용 모드 (오프라인) | 키: (없음) | 출처: -

if llm.is_enabled():         # 키가 있을 때만 LLM 경로
    data = llm.chat_json(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": prompt}],
        model=cfg["llm"]["extract_model"],
    )
    if data is None:         # 호출 실패해도 예외는 안 난다 → 반드시 폴백
        data = rule_based_fallback(...)
else:
    data = rule_based_fallback(...)
```

| 함수 | 용도 |
|---|---|
| `setup_console()` | 콘솔 UTF-8 고정 |
| `is_enabled()` | 키 존재 여부 |
| `key_status()` | `{available, source, masked, mode}` — 설정 탭·리포트용 (키 원문 없음) |
| `banner()` | 시작 시 1줄 출력 |
| `chat(messages, model=, temperature=, max_tokens=)` | 문자열 반환, 키 없거나 실패 시 **`None`** |
| `chat_json(...)` | JSON 파싱 반환, 실패 시 **`None`** |
| `set_runtime_key(k)` / `clear_runtime_key()` | 설정 탭에서 주입 (메모리 전용, 디스크 기록 없음) |
| `save_key_to_env_file(k)` / `delete_saved_key()` | 사용자가 **명시적으로 원할 때만** `.env` 저장/삭제 |
| `looks_like_openai_key(k)` | 형식 사전 검증 |
| `mask(k)` | `sk-********abcd` |
| `cache_stats()` | 캐시/호출 통계 |

모든 응답은 `output/llm_cache.json` 에 캐시된다 → 재실행 비용 0, 채점자 재현성 확보.
캐시 파일에 키는 들어가지 않는다.

### `config.json` 의 `llm` 섹션 (Worker A가 작성)
```jsonc
"llm": {
  "provider": "openai",
  "extract_model": "gpt-4.1-mini",   // build_graph.py 관계 추출 증강
  "synth_model":   "gpt-4.1-mini",   // agent.py 답변 합성 (선택)
  "judge_model":   "gpt-4.1",        // evaluate.py LLM-as-judge (선택)
  "temperature": 0,
  "max_output_tokens": 1500
}
```
Worker B는 모델명을 하드코딩하지 말고 `config.json` 에서 읽는다.

---

## 2. `config.json` — 도메인 고정값 (Worker A가 작성, Worker B가 읽기 전용으로 소비)

```jsonc
{
  "domain": "korean-independence-history",
  "schema": {
    "node_types": ["Person", "Event", "Organization"],
    "relations": [
      {"head": "Person", "relation": "PARTICIPATED_IN", "tail": "Event"},
      {"head": "Person", "relation": "FOUNDED",         "tail": "Organization"},
      {"head": "Organization", "relation": "PARTICIPATED_IN", "tail": "Event"}
    ]
  },
  "retrieval": {
    "max_hops": 3,              // 기본 반경
    "start_hops": 2,            // 최초 확장 반경
    "max_triples": 60,          // 컨텍스트 삼중항 총 상한
    "per_relation": 25,         // 관계별 예산
    "hub_degree_threshold": 25, // 이 차수를 넘으면 허브로 간주
    "degree_penalty": true,     // 1/(1+sqrt(degree)) 가중
    "min_entity_len": 2
  },
  "generation": {
    "refusal_text": "제시된 지식 그래프 자료에서 해당 질문에 대한 근거를 찾을 수 없습니다.",
    "require_relation_match": true
  }
}
```
- `relations`는 **방향 고정 튜플**이다. `(head_type, relation, tail_type)`에 맞지 않는 엣지는 버린다.
- Worker B는 이 값들을 **하드코딩하지 말고 반드시 `config.json`에서 읽는다.**

---

## 3. 코퍼스 — `data/docs/*.md`, `data/manifest.json` (Worker A)

### 3.1 문서 파일 형식 (과제 지정 형식 그대로)
파일명: `문서제목의 공백을 _ 로 바꾼 것 + .md`

```markdown
# 문서 제목

분류: 쉼표로 구분한 분류 목록

본문 ...
```

### 3.2 `data/manifest.json`
```jsonc
{
  "collected_at": "2026-09-21T00:00:00Z",
  "source": "ko.wikipedia.org MediaWiki API",
  "seeds": ["안중근", "..."],
  "stats": {"seed_count": 10, "hop2_candidates": 0, "category_shared": 0, "saved": 60},
  "docs": [
    {"title": "안중근", "file": "안중근.md", "categories": ["..."],
     "chars": 12345, "hop": 0, "reason": "seed"}
  ],
  "rejected": [
    {"title": "1909년", "reason": "year/list/template/category page"}
  ]
}
```
`rejected`에 **채택·탈락 기준을 데이터로 남기는 것**이 채점 항목이다 (과제 5단계-1).

---

## 4. 그래프 — `output/graph.graphml` + `output/graph.json` (Worker A)

`networkx.DiGraph`를 `nx.write_graphml`로 저장한다.
**GraphML은 list/dict 속성을 지원하지 않으므로 모든 다중값 속성은 `;`(세미콜론)으로 join한 문자열이다.**

### 4.1 노드 속성 (키 이름 고정)
| 키 | 타입 | 설명 |
|---|---|---|
| `id`(노드 키) | str | **정규화된 표준 이름**. 예: `안중근` |
| `type` | str | `Person` \| `Event` \| `Organization` |
| `aliases` | str | `;` join. 병합된 모든 표기. 예: `안중근;安重根;안 중근` |
| `sources` | str | `;` join된 문서 제목 |
| `doc_count` | int | 등장 문서 수 |
| `degree` | int | 전체 차수 (빌드 시 계산해 저장) |

### 4.2 엣지 속성 (키 이름 고정)
| 키 | 타입 | 설명 |
|---|---|---|
| `relation` | str | `PARTICIPATED_IN` \| `FOUNDED` |
| `origin` | str | `rule` \| `llm` \| `rule+llm` |
| `confidence` | float | 0.0~1.0 |
| `sources` | str | `;` join된 출처 문서 제목 |
| `evidence` | str | **원문 근거 문장 1개** (200자 이내, `;`로 여러 개 가능) |
| `extractor` | str | 발동한 규칙 패턴 id. 예: `rule_participated_v3` |

### 4.3 `output/graph.json` (Worker B가 실제로 읽는 1차 소스)
GraphML 파싱 차이를 없애기 위해 **동일 내용을 JSON으로도 저장한다.**
```jsonc
{
  "nodes": [
    {"id": "안중근", "type": "Person", "aliases": ["안중근", "安重根"],
     "sources": ["안중근", "하얼빈 의거"], "doc_count": 2, "degree": 7}
  ],
  "edges": [
    {"head": "안중근", "relation": "PARTICIPATED_IN", "tail": "하얼빈 의거",
     "origin": "rule", "confidence": 0.95,
     "sources": ["안중근"],
     "evidence": ["1909년 10월 26일 안중근은 하얼빈 역에서 이토 히로부미를 사살하였다."],
     "extractor": "rule_participated_v3"}
  ],
  "alias_index": {"安重根": "안중근", "도마 안중근": "안중근"},
  "stats": {"nodes": 0, "edges": 0, "by_type": {}, "by_relation": {}, "by_origin": {}}
}
```
`alias_index`는 **표기 → 표준 노드 id** 사전이다. Worker B의 개체 연결(Entity Linking)이 이것을 쓴다.

### 4.4 `output/extraction_report.json` (Worker A)
정규화·병합이 **무엇을 어떤 기준으로 합쳤는지** 남긴다 (루브릭 2: "같은 개체를 하나로 합치는 기준").
```jsonc
{
  "merge_rules": [
    {"rule": "괄호 한정어 제거", "example": "안창호 (독립운동가) → 안창호", "applied": 12}
  ],
  "merges": [
    {"canonical": "대한민국 임시정부",
     "merged": ["대한민국임시정부", "상해 임시정부", "임시정부"],
     "rule": "공백정규화+별칭사전", "confidence": 1.0}
  ],
  "rejected_edges": [
    {"triple": ["김구", "FOUNDED", "3·1 운동"], "reason": "schema_type_mismatch: FOUNDED tail must be Organization"}
  ],
  "not_extracted_relations": [
    {"relation": "INFLUENCED", "reason": "...", "cost": "..."}
  ]
}
```

---

## 5. 평가셋 — `data/goldenset.json` (Worker A)

**12문항 이상.** 홉 수별로 갈라 채점해야 하므로 `hops` 필드가 필수다.
`unanswerable` 문항을 **최소 2건** 포함한다 (루브릭 2: "근거가 없을 때 지어내지 않고 거절하는가").

```jsonc
{
  "version": "1.0",
  "scoring_criteria": {
    "answer_correct": "기대 정답 문자열(또는 aliases 중 하나)이 답변에 포함되면 1.0, 부분 포함 0.5, 없으면 0.0",
    "path_recall": "expected_path의 삼중항 중 실제 수집된 근거 삼중항에 포함된 비율",
    "refusal_correct": "unanswerable 문항에서 config.generation.refusal_text 를 출력하면 1.0"
  },
  "items": [
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
        {"doc": "안중근", "quote": "1909년 10월 26일 ... 하얼빈 역에서 ..."}
      ],
      "answerable": true
    },
    {
      "id": "Q07",
      "hops": 2,
      "type": "path",
      "question": "안중근이 참여한 사건에 같이 있던 인물이 세운 조직은?",
      "expected_answer": ["..."],
      "expected_path": [
        {"head": "안중근", "relation": "PARTICIPATED_IN", "tail": "..."},
        {"head": "...",   "relation": "PARTICIPATED_IN", "tail": "..."},
        {"head": "...",   "relation": "FOUNDED",         "tail": "..."}
      ],
      "evidence": [{"doc": "...", "quote": "..."}],
      "answerable": true
    },
    {
      "id": "Q12",
      "hops": 2,
      "type": "unanswerable",
      "question": "안중근이 설립한 방송국은?",
      "expected_answer": [],
      "expected_path": [],
      "evidence": [],
      "answerable": false
    }
  ]
}
```
**중요**: `expected_path`의 모든 삼중항은 **실제로 `graph.json`에 존재해야 한다.**
Worker A는 goldenset 저장 직전에 그래프와 대조 검증하고, 불일치는
`output/extraction_report.json`의 `goldenset_validation`에 기록한다.
`hops` 분포: 1홉 ≥ 4건, 2홉 ≥ 5건, 3홉 ≥ 1건, unanswerable ≥ 2건.

---

## 6. 에이전트 실행 기록 — `output/runs.jsonl` (Worker B)

`agent.answer(question)` 1회 = JSONL 1줄. **실제로 탄 경로를 남기는 것이 채점 항목이다.**

```jsonc
{
  "ts": "2026-09-21T00:00:00Z",
  "question": "...",
  "route": "local",                  // local | path
  "seeds": [{"surface": "안중근", "node": "안중근", "type": "Person", "score": 0.8}],
  "hops_used": 2,
  "deepened": false,                 // relation_gap 때문에 반경을 넓혔는가
  "visited_nodes": ["안중근", "하얼빈 의거"],
  "skipped_hubs": [{"node": "대한민국 임시정부", "degree": 41, "reason": "hub_degree_threshold"}],
  "traversed_path": [                // 실제로 탄 경로 (순서대로)
    {"hop": 1, "head": "안중근", "relation": "PARTICIPATED_IN", "tail": "하얼빈 의거",
     "origin": "rule", "sources": ["안중근"]}
  ],
  "context_triples": [ /* 위와 같은 형태, 컨텍스트에 실제로 들어간 것 전부 */ ],
  "answer": "...",
  "refused": false,
  "path_explanation": "안중근 -(PARTICIPATED_IN)-> 하얼빈 의거",
  "sources": ["안중근"],
  "latency_ms": 123
}
```

### 6.1 `agent.py` 공개 API (Worker B, Worker A는 호출하지 않음)
```python
class GraphRAGAgent:
    def __init__(self, root: Path | None = None): ...
    def answer(self, question: str) -> dict:  # runs.jsonl 1줄과 동일한 dict 반환
        ...
```
`evaluate.py`와 `app.py`는 **반드시 이 API만** 사용한다.

### 6.2 답변 포맷 (`answer` 필드 안의 문자열)
```
[답변 요약]
...

[탐색 및 추론 경로]
1. 안중근 -(PARTICIPATED_IN)-> 하얼빈 의거
2. ...

[근거 출처]
- 안중근 (origin: rule, confidence: 0.95)
```
근거가 없으면 **오직** `config.generation.refusal_text` 문자열만 `[답변 요약]`에 넣는다.

---

## 7. 평가 결과 — `output/eval.json` (Worker B)

**평균 하나로 뭉개지 말고 홉 수별로 갈라 보고한다** (루브릭 3).

```jsonc
{
  "run_at": "...",
  "n_items": 12,
  "criteria": { /* goldenset.scoring_criteria 를 그대로 복사 */ },
  "by_hop": {
    "1": {"n": 4, "graphrag": {"answer_acc": 1.0, "path_recall": 1.0, "refusal_ok": null},
          "basic_rag": {"answer_acc": 0.75}},
    "2": {"n": 5, "graphrag": {"answer_acc": 0.8,  "path_recall": 0.86},
          "basic_rag": {"answer_acc": 0.2}},
    "3": {"n": 1, "graphrag": {"answer_acc": 0.0,  "path_recall": 0.33},
          "basic_rag": {"answer_acc": 0.0}}
  },
  "unanswerable": {"n": 2, "refusal_accuracy": 1.0, "hallucinated": 0},
  "overall": {"graphrag_answer_acc": 0.0, "basic_rag_answer_acc": 0.0},
  "failures": [
    {"id": "Q09", "hops": 2, "score": 0.0,
     "layer": "retrieval",            // index | retrieval | generation
     "diagnosis": "KG에는 (김구,FOUNDED,한인애국단)이 있으나 per_relation 예산 초과로 컨텍스트에서 잘림",
     "expected_path_present_in_graph": true,
     "expected_path_present_in_context": false,
     "answer_excerpt": "...",
     "fix": "per_relation 25 → 40 또는 rule 엣지 예산 면제"}
  ],
  "failure_layer_counts": {"index": 1, "retrieval": 1, "generation": 0}
}
```

### 7.1 실패 층 판정 규칙 (고정)
| 조건 | layer |
|---|---|
| `expected_path` 삼중항이 **그래프에 없다** | `index` |
| 그래프엔 있으나 **`context_triples`에 없다** | `retrieval` |
| 컨텍스트엔 있으나 **답변이 틀렸다** | `generation` |

이 판정을 코드로 자동 수행하고, `diagnosis`는 **실제 실패 사례를 읽고 쓴 문장**이어야 한다.

---

## 8. 데모 — `app.py` (Worker B)

`streamlit run app.py` 로 로컬 실행. **탭 3개** 구조로 만든다.

### 탭 1 — 💬 질문하기
- 질문 입력 → 답변 표시.
- **답변 + 탄 경로 + 근거 삼중항 + 출처 문서**를 모두 표시한다 (과제 5단계-5 필수).
  - 탄 경로: `안중근 -(PARTICIPATED_IN)-> 하얼빈 의거 -(PARTICIPATED_IN)<- 우덕순` 형태로 홉 순서대로.
  - 근거 삼중항: 표로. 컬럼 = head / relation / tail / origin / confidence / sources.
  - 출처 문서: 문서 제목 + `data/docs/*.md` 원문 근거 문장 `st.expander` 로 펼쳐보기.
  - 확장 반경(`hops_used`), `deepened` 여부, 건너뛴 허브 노드도 함께 노출한다.
- 예시 질문 버튼(1홉 / 2홉 / 3홉 / **거절 케이스**)을 넣어 채점자가 바로 눌러볼 수 있게 한다.
- 그래프 파일이 없으면 "먼저 `python build_graph.py` 를 실행하세요" 안내만 띄우고 죽지 않는다.

### 탭 2 — 📊 평가 결과
- `output/eval.json` 이 있으면 **홉 수별 표**(GraphRAG vs basic RAG), 거절 정확도,
  실패 층 분류(`index`/`retrieval`/`generation`) 집계를 표시한다. 없으면 안내만.

### 탭 3 — ⚙️ 설정  ← **사용자 요청 기능**
사용자가 OpenAI API 키를 **직접 입력해서 쓰는 탭**이다. 다음을 정확히 지킨다.

1. 현재 상태를 `llm_provider.key_status()` 로 표시한다:
   `모드`, `출처`(runtime/env/dotenv), `마스킹된 키`. **원문 키는 절대 표시하지 않는다.**
2. 입력 위젯은 반드시 `st.text_input("OpenAI API 키", type="password")` 로 가린다.
3. 버튼 3개:
   - **[이 세션에만 적용]** → `llm_provider.set_runtime_key(key)`.
     디스크에 쓰지 않는다. "브라우저를 닫으면 사라집니다" 안내.
   - **[.env에 저장]** → `llm_provider.save_key_to_env_file(key)`.
     저장 전 `looks_like_openai_key()` 로 형식 검증하고, 실패 시 저장하지 않는다.
     저장 후 "`.env` 는 `.gitignore` 로 차단되어 깃에 올라가지 않습니다" 안내.
   - **[키 삭제]** → `clear_runtime_key()` + `delete_saved_key()`.
4. **[연결 테스트]** 버튼: `llm.chat([...], use_cache=False)` 로 짧게 1회 호출해
   성공/실패만 표시한다. 실패해도 앱이 죽지 않아야 한다.
5. 키 없이도 앱은 **완전히 동작**해야 한다. 설정 탭 상단에 이 문구를 고정 표시한다:
   > 키가 없어도 규칙 기반으로 전체 파이프라인이 동작합니다. 키는 LLM 증강(추출 보강 · 답변 합성 · LLM 심판)에만 쓰입니다.
6. `st.session_state` 에 키를 넣더라도 **로그·`runs.jsonl`·화면 어디에도 원문을 남기지 않는다.**

---

## 9. 통합 검증 (양쪽 완료 후 코디네이터가 실행)

```bash
python collect_corpus.py      # 또는 이미 수집된 data/docs 사용
python build_graph.py
python evaluate.py
streamlit run app.py          # 수동 확인
```
네 단계가 **예외 없이 종료**해야 한다.

---

## 10. 병렬 작업을 위한 임시 픽스처

Worker B는 Worker A의 실제 그래프를 기다리지 않는다.
`tests/fixture_graph.json`(+ `tests/fixture_goldenset.json`)을 **§4.3/§5 스키마 그대로** 직접 만들어
개발·테스트하고, 실제 `output/graph.json`이 나오면 경로만 바꿔 재검증한다.
픽스처는 최소 12노드/15엣지로 2홉·3홉이 성립해야 한다.
