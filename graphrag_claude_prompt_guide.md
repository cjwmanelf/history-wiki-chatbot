# GraphRAG 시스템 설계 및 운영 지침서 (Claude 프롬프트 & 지식 참조 가이드)

본 문서는 시네필 위키 지식 그래프 구축 및 GraphRAG 검증 자료를 바탕으로 작성된 **GraphRAG 원리, 아키텍처, 구축 및 운영 파이프라인, 프롬프트용 지식 가이드**입니다. Claude 등 대화형 AI 모델의 시스템 프롬프트(System Prompt) 또는 컨텍스트 참조 자료로 활용할 수 있도록 정밀하게 구조화되었습니다.

---

## 1. GraphRAG 개요 및 핵심 원리 (Overview & Concepts)

### 1.1. Vector RAG (Basic RAG)의 한계
1. **연쇄적 정보 추적(Multi-hop Reasoning)의 한계**:
   - Vector RAG는 질문 문장과 텍스트 조각(Chunk) 사이의 단어/임베딩 유사도에 의존함.
   - 징검다리를 건너듯 꼬리를 무는 질문(예: "봉준호 감독 영화에 출연한 배우가 나온 또 다른 작품은?")은 질문 어휘와 정답 문서 간 직접 겹침이 없어 검색 후보(Top-k)에 들어오지 못함.
   - 하이브리드 검색(BM25 + Vector) 및 리랭킹(Re-ranking) 기법은 후보군 내 정답이 있을 때 정밀도(Precision)를 올릴 뿐, 재현율(Recall)이 0인 경우 구제 불가능함.
2. **전역적 맥락 파악(Global Sense-making)의 한계**:
   - "이 데이터셋 전체에 등장하는 한국 영화의 주요 흐름과 집단은 무엇인가?" 같은 넓은 질문(Global Query)에 대해 상위 k개 청크만으로는 코퍼스 전체를 요약할 수 없음.

### 1.2. GraphRAG의 핵심 해결책
- **색인 시점 사전 연결 (Pre-indexed Relationships)**:
  - 질문이 들어온 뒤 연결을 찾는 것이 아니라, **색인(Indexing) 시점에 정보 간 관계를 지식 그래프(Knowledge Graph)로 미리 저장**.
  - 질문 시점에는 검색이 아니라 **그래프 위 노드/엣지 이동(Graph Traversal)** 및 **사전 생성된 커뮤니티 보고서(Community Reports)**를 통해 답변 생성.

### 1.3. 지식 그래프의 최소 정보 단위
- **노드 (Node / Entity)**: 정보의 대상이 되는 명사 (예: `봉준호` [Person], `기생충` [Film], `스릴러` [Genre]).
- **엣지 (Edge / Relation)**: 노드 간을 잇는 관계 동사 (예: `DIRECTED`, `ACTED_IN`, `HAS_GENRE`, `WON_AWARD`).
- **삼중항 (Triple)**: `[주어(Head)] -(관계(Relation))-> [목적어(Tail)]` 형태의 최소 사실 단위.
  - 삼중항은 이어 붙일 수 있어(`[봉준호] -> DIRECTED -> [기생충] <- ACTED_IN <- [송강호]`) 문서 조각과 달리 복잡한 추론 경로를 복원 가능함.

---

## 2. 지식 그래프 구축 및 정규화 파이프라인 (KG Construction & Normalization)

### 2.1. 스키마 제어 추출 (Schema-Controlled Extraction)
- **무제약 추출의 문제점**:
  - 오염된 관계(예: 동일 문단 등장만으로 생성된 어색한 엣지), 미세한 표기 파편화, 문맥 내 일시적 관계(등장인물 간 사적 관계 등)로 인한 그래프 비대화.
- **역산 스키마 설계 (Reverse-engineered Schema)**:
  - 풀고자 하는 목표 질의/골든셋 요구사항으로부터 필요한 관계 유형을 역산하여 스키마 정의.
  - 방향 고정 튜플 스키마: 단순 관계명이 아닌 `("Person", "ACTED_IN", "Film")` 형태로 출발/도착 노드 타입을 제약하여 방향 역전 방지.
- **추출 지침 (Prompt Instructions)**:
  - 본문 최상단에 `[문서 제목]`, `[문서 유형]` 헤더를 삽입하여 배역 목록 등에서 문맥 왜곡 방지.
  - 시상식 부문(남우주연상 등)은 별도 노드가 아닌 엣지 속성(`properties: {category: ...}`)으로 수집.

### 2.2. 정규화 및 규칙-LLM 하이브리드 병합 (Normalization & Hybrid Merger)
- **표기 변형 및 한정어 정규화**:
  - 조사 떼기, 작품명 기호 제거, 동음이의어/한정어 괄호(`(영화)`, `(2006년)`) 처리.
  - **고유명사(Entity: Film/Person/Award/Org)** 계열과 **보통명사/속성(Attr: Genre/Theme/Location)** 계열은 이름이 같아도 노드를 합치지 않음 (`괴물`[Film] vs `괴물`[Genre] 분리).
- **규칙(Rule)과 LLM의 상호보완적 결합**:
  - **위키 분류/정규식 규칙**: 장르, 배경, 소재, 시상식 등 형식이 고정된 정보 추출 (정확도 100%, 환각 없음, 비용 0).
  - **LLM 본문 추출**: 배역/출연(`ACTED_IN`) 등 본문 자유 서술에 존재하는 관계 수집.
- **출처 및 신뢰도 태깅**:
  - 모든 엣지에 `origin` (`rule`, `llm`, `rule+llm`), `confidence`, `sources` 명시.

---

## 3. 커뮤니티 탐지 및 전역 보고서 (Community Detection & Global Reports)

### 3.1. Louvain 커뮤니티 탐지 (Community Detection)
- 모듈러리티(Modularity)를 극대화하는 방향으로 촘촘하게 연결된 노드 집단(Community)을 탐지.
- **속성 노드 가중치 감점 (`ATTR_WEIGHT = 0.3`)**:
  - 장르, 소재, 장소 노드는 수많은 영화와 연결되어 그래프 전체를 하나의 거대한 덩어리로 뭉개버리는 "가짜 다리" 역할을 함. 가중치를 낮추어 밀접한 인물-작품 중심 커뮤니티 형성.

### 3.2. 전역 보고서 (Global Community Reports)
- 커뮤니티별 핵심 인물, 대표 작품, 주요 장르/소재, 수상 경향을 프로필로 집계.
- LLM이 해당 프로필만을 근거로 커뮤니티 요약 보고서(JSON) 사전 작성.
- 전역 질문(Global Query) 입력 시 미리 작성된 커뮤니티 보고서를 Map-Reduce 방식으로 조합하여 답변.

---

## 4. LangGraph 에이전트 아키텍처 및 실행 (Agent Architecture)

```
START ──> [route (2단계 라우터)] ──┬──> (global) ──> [global_map] ──> [global_reduce] ──┐
                                  └──> (local/path) ──> [find_seeds] ──> [expand] ───┼─> [build_context] ──> [synthesize] ──> END
                                                                              │      ▲
                                                                     (근거 부족)│      │
                                                                              └─> [deepen]
```

### 4.1. 2단계 라우팅 (Routing)
- **Local**: 특정 개체의 직접 속성 질의 (예: "기생충 감독은?").
- **Path**: 개체 간 연결 및 추천 질의 (예: "기생충 감독의 다른 작품 추천해줘").
- **Global**: 전체 자료의 경향/그룹 질의 (예: "한국 영화계의 주요 집단 특징은?").
- **2단 구조**: 키워드 규칙 라우터(공짜, 고속) 1차 판정 후, 판단 불명(None) 시에만 LLM 라우터 2차 호출.

### 4.2. 개체 연결 및 조회 계층 (Entity Linking & Retrieval)
- **개체 연결 (Entity Linking)**: 질문에서 정규화된 노드 식별 (`MIN_ENTITY_LEN >= 2`). 고유명사를 속성보다 우선 지정.
- **차수 감점 (Degree Penalty)**:
  - 과도한 허브 노드 탐색 억제: $1 / (1 + \sqrt{	ext{degree}})$ 점수 가중치 적용.
- **관계별 예산 할당 (Relation-budget Triple Collection)**:
  - `ACTED_IN` 등 다수 관계의 예산 독점 방지 (`per_relation` 제한).
  - **규칙 계열(`rule`) 엣지 할당량 면제**: 사람 분류 태그 기반 엣지는 환각이 없으므로 잘리지 않도록 보호.

### 4.3. 이중 가드레일 (Multi-layered Guardrails)
- **코드 층 가드레일 (`relation_gap`)**:
  - 질문이 요구한 관계(예: `DIRECTED`, `WON_AWARD`)가 수집된 삼중항에 없으면 탐색 반경을 확장(`deepen`).
  - 최대 반경 도달 시 근거 미달 판정.
- **프롬프트 층 가드레일**:
  - 근거 텍스트에 포함되지 않은 사실에 대해 "제시된 자료에서 근거를 찾을 수 없습니다"라고 명시적 답변 거부.

---

## 5. 평가 및 디버깅 프레임워크 (Evaluation & Debugging)

### 5.1. 3단계 평가 체계 (3-Tier Evaluation Framework)
1. **색인층 (Index)**: 기준 삼중항 커버리지 (Reference Triple Coverage) — 지식 그래프에 필요한 사실 포함 여부.
2. **검색층 (Retrieval)**: 컨텍스트 재현율 (Context Recall) — 수집된 삼중항에 필수 근거 포함 여부.
3. **생성층 (Generation)**: LLM-as-judge — 독립 심판 모델(`gpt-4.1` 등)을 사용하여 원문 근거 대비 최종 답변 정확도 평가 (1.0 / 0.5 / 0.0, 실패 시 `None`).

### 5.2. 실패 원인 자동 분류 (Failure Cause Diagnosis)
- **① 색인 문제**: KG에 사실 자체가 없음 $ightarrow$ 추출 스키마 및 규칙 보강 필요.
- **② 검색 문제**: KG엔 있으나 삼중항 수집 시 누락됨 $ightarrow$ 탐색 반경($r$) 및 예산(`max_triples`, `per_relation`) 조정.
- **③ 생성 문제**: 근거 삼중항은 정확하나 답변 생성 실패 $ightarrow$ 합성 프롬프트 개선.

---

## 6. GraphRAG 챗봇용 Claude 시스템 프롬프트 템플릿 (Claude System Prompt Template)

Claude를 GraphRAG 기반 챗봇으로 동작시키거나 GraphRAG 파이프라인의 에이전트로 사용할 때 아래 프롬프트를 적용하십시오.

```text
[System Prompt for Claude GraphRAG Engine]

You are "Cinephile GraphRAG Assistant", an AI expert specializing in Knowledge Graph-based Retrieval-Augmented Generation for film databases.
Your job is to answer user queries strictly using the provided Knowledge Graph triples and Community Reports, while maintaining complete transparency and path explainability.

### Instructions & Operating Rules:

1. Grounding & Factuality:
   - Base every factual claim strictly on the provided context (Triples: `(Head, Relation, Tail)` or Community Reports).
   - Never extrapolate outside the provided context. If the necessary triples/facts are missing, state clearly: "제시된 지식 그래프 자료에서 해당 질문에 대한 근거를 찾을 수 없습니다."

2. Multi-hop Reasoning & Path Transparency:
   - When answering path/recommendation queries, explicitly present the multi-hop reasoning path.
   - Example format: "[추천 작품: 살인의 추억] (이유: 기생충 -(DIRECTED)-> 봉준호 -(DIRECTED)-> 살인의 추억)"

3. Source Attribution & Confidence:
   - If metadata contains origins (`origin: rule` or `origin: llm`), indicate high confidence for rule-verified relationships.

4. Query Response Structure:
   - [답변 요약]: Direct answer to the user query.
   - [탐색 및 추론 경로]: Step-by-step Knowledge Graph traversal path used to derive the answer.
   - [근거 출처]: Referenced document titles and origin confidence.
```

---

## 7. 핵심 비교 요약 (Basic RAG vs GraphRAG)

| 평가 항목 | Basic RAG (BM25 / Vector) | GraphRAG (Knowledge Graph) |
| :--- | :--- | :--- |
| **단순 질의 (1-hop)** | **우수 (1.00)** (원문 부가 서술 활용) | 보통 (0.83) (삼중항 요약으로 상세 서술 손실 가능) |
| **추천/연결 질의 (Multi-hop)**| 취약 (0.35) (어휘 불일치 시 검색 실패) | **우수 (0.80)** (그래프 선 타고 탐색) |
| **전역 질의 (Global Summary)**| 불가능 (0.00) (Top-k 조각으로 전체 파악 불가) | **우수 (0.50~)** (사전 생성 커뮤니티 보고서 활용) |
| **설명 가능성 (Explainability)**| 낮은 편 (문단 블록 제시) | **매우 높음** (구체적 노드-관계 추적 경로 제공) |
| **구축 비용 (Indexing Cost)** | 낮음 (단순 청킹 및 임베딩) | 높음 (LLM 추출, 정규화, 커뮤니티 탐지) |

> **실무 결론**: GraphRAG로의 단순 전환보다는, **질문 유형 라우터(Router)**를 두어 단순/서술형 질문은 Basic/Vector RAG로, 멀티홉/추천/전역 질문은 GraphRAG로 분기 처리하는 **하이브리드 에이전틱 RAG(Hybrid Agentic RAG)** 구조가 최적의 솔루션입니다.
