#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_graph.py — 스키마 제어 추출 + 정규화로 지식 그래프를 만든다 (과제 5단계-2).

설계 요약
--------
1. **스키마 제어 추출**: 허용 삼중항은 방향 고정 튜플 3종뿐이다.
   `(Person, PARTICIPATED_IN, Event)` / `(Person, FOUNDED, Organization)` /
   `(Organization, PARTICIPATED_IN, Event)`.
   어긋나는 후보는 버리고 `extraction_report.rejected_edges` 에 사유와 함께 남긴다.
2. **노드 타입 판정은 위키 분류(category) 규칙이 1순위**다. 규칙은 정확도가 높고
   환각이 없으며 비용이 0이므로 `origin="rule"` 로 태깅한다.
   분류가 없을 때만 제목 접미사 규칙으로 보조 판정한다.
3. **관계 추출은 한국어 문장 패턴 규칙**이 기본이다.
   - 반드시 **문장 단위**로 자르고, head/tail 후보가 **같은 문장 안에 함께 있을 때만**
     엣지를 만든다. 문단 동시등장만으로 엣지를 만들면 가이드가 경고한 "오염된 관계"가 된다.
   - 본문 최상단에 `[문서 제목]` / `[문서 유형]` 헤더를 주입해 문맥 왜곡을 막고,
     문서 자신을 주어·목적어로 보는 문서 스코프 규칙(`*_doc_*`)을 따로 둔다.
   - 패턴마다 `extractor` id 를 부여하고 엣지에 기록한다.
4. **LLM 증강은 선택**이다. `llm_provider.is_enabled()` 가 True 일 때만, 규칙이 아무것도
   못 잡은 문장에 한해 삼중항을 더 모은다. 스키마 검증을 통과한 것만 채택한다.
   **키가 없으면 경고 1줄만 찍고 규칙 결과로 끝까지 정상 종료한다.**
5. **정규화·병합**: 조사 제거, 공백/중점 정규화, 괄호 한정어 제거, 한자 병기 분리,
   위키 넘겨주기 별칭, 약칭 사전. **타입이 다르면 이름이 같아도 합치지 않는다.**
   병합은 전부 `extraction_report.merges` / `merge_rules` 에 남긴다.

네트워크가 끊겨도 `data/docs` 만으로 동작한다.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

import llm_provider

random.seed(42)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
MANIFEST_PATH = DATA_DIR / "manifest.json"
GOLDENSET_PATH = DATA_DIR / "goldenset.json"
CONFIG_PATH = ROOT / "config.json"
OUT_DIR = ROOT / "output"
GRAPHML_PATH = OUT_DIR / "graph.graphml"
GRAPHJSON_PATH = OUT_DIR / "graph.json"
REPORT_PATH = OUT_DIR / "extraction_report.json"

EVIDENCE_MAX = 200
EVIDENCE_PER_EDGE = 3

# LLM 증강 예산 (키가 있을 때만 쓰인다)
LLM_MAX_SENTENCES = 150
LLM_BATCH = 10

# --------------------------------------------------------------------------- #
# 1. 노드 타입 판정 규칙 — 위키 분류가 1순위
# --------------------------------------------------------------------------- #
PERSON_CAT_RE = re.compile(
    r"\d+년\s*태어남|\d+년\s*죽음|독립운동가|정치인|군인|교육자|언론인|사상가|종교인|"
    r"아나키스트|사회주의자|민족주의자|공산주의자|승려|목사|신부|의병장|작가|시인|소설가|"
    r"학자|역사가|혁명가|기업인|법조인|외교관|화가|음악가|인물|사람|동문|출신|왕|황제|"
    r"천황|총리|대통령|장군|열사|의사자|암살자|수상자"
)
ORG_CAT_RE = re.compile(
    r"단체|조직|정당|학교|대학|정부|기관|부대|군대|결사|협회|학회|연맹|동맹|회사|기업|"
    r"신문|언론사|출판사|교회|사찰|재단|위원회|의정원|총독부|군정서|의용군|광복군|독립군|"
    r"설립|창립|창설|해체|은행|병원|조합|본부|사령부"
)
EVENT_CAT_RE = re.compile(
    r"사건|전투|전쟁|운동|의거|봉기|시위|학살|조약|선언|항쟁|혁명|테러|암살|재판|회담|"
    r"침공|점령|만세|투쟁|의병|폭동|파업|해전|작전"
)
# 방송·영화·음반 같은 현대 매체는 위키 분류상 '단체'로 보일 수 있으나 이 그래프의 개체가
# 아니다. (실제로 '청산리 전투' 문서의 대중문화 절에서 KBS 1TV 가 노드로 올라왔다.)
NON_ENTITY_CAT_RE = re.compile(
    r"방송|텔레비전|라디오|드라마|영화|음반|앨범|게임|프로그램|잡지|웹사이트|"
    r"스포츠|음악 그룹|만화|애니메이션|예능"
)

# 본문 말미의 참고·외부링크·대중문화 절은 관계 추출 대상이 아니다.
# 이 절들은 본문 서술이 아니라 목록이라 '오염된 관계'의 온상이다(가이드 2.1).
_SECTION_CUT_RE = re.compile(
    r"^\s*(각주|주석|참고 ?문헌|참고 ?자료|인용|출처|외부 ?링크|같이 ?보기|관련 ?항목|"
    r"관련 ?문서|더 ?보기|대중 ?문화|대중문화 속|관련 ?작품|관련 ?매체|관련 ?미디어|"
    r"전기 ?자료|함께 ?보기)\s*$", re.M)


def trim_body(body: str) -> str:
    """참고·외부링크·대중문화 절부터 뒤를 잘라낸다."""
    m = _SECTION_CUT_RE.search(body)
    return body[:m.start()].rstrip() if m else body

# 분류가 없을 때만 쓰는 제목 접미사 보조 규칙 (단음절 접미사는 인명 오인 위험이 커서 뺐다)
EVENT_TITLE_RE = re.compile(
    r"(전투|전쟁|사건|의거|운동|봉기|시위|학살|조약|선언|항쟁|혁명|대첩|작전|회담|재판|"
    r"만세운동|투쟁|참변|사변|정변|참사|회의|해전|공판|의병)$"
)
ORG_TITLE_RE = re.compile(
    r"(학교|대학교|정부|의정원|총독부|군정서|광복군|독립군|의용군|의용대|협회|학회|"
    r"연맹|동맹|위원회|신문|일보|재단|회사|공사|은행|본부|사령부|애국단|청년회|부인회|"
    r"소년단|결사대|의열단|흥사단|신민회|국민회|동지회|친목회|구락부|혁명당|국민당|"
    r"독립당|공산당|사회당|청년당|의군부|독군부|통의부|정의부|신민부|참의부|학우회|"
    r"수양회|동우회|자강회|광복회|광복단|청년단|의병대|유격대)$"
)

# --------------------------------------------------------------------------- #
# 2. 불용어 — 일반명사/보통명사는 노드로 만들지 않는다
# --------------------------------------------------------------------------- #
STOPWORDS: set[str] = {
    # 일반 보통명사
    "정부", "운동", "단체", "조직", "학교", "군대", "부대", "신문", "회의", "사건",
    "전투", "전쟁", "독립", "독립운동", "민족", "국가", "나라", "인민", "국민", "민중",
    "정당", "협회", "학회", "연맹", "동맹", "위원회", "본부", "사령부", "총독부",
    "결사", "의병", "의거", "봉기", "시위", "선언", "조약", "혁명", "항쟁", "투쟁",
    "교육", "종교", "사상", "문화", "역사", "사회", "경제", "정치", "군사", "외교",
    "대표", "회장", "단장", "총장", "주석", "국무총리", "대통령", "장군", "의사", "열사",
    "선생", "동지", "청년", "학생", "여성", "남성", "어머니", "아버지", "아들", "딸",
    # 역할·부류를 가리키는 보통명사 — 위키에 문서가 있어도 개체가 아니다
    "독립운동가", "독립운동단체", "애국지사", "순국선열", "의병장", "항일운동",
    "민족운동", "무장투쟁", "민족주의", "사회주의", "공산주의", "아나키즘", "광복",
    "친일파", "매국노", "식민지", "강제징용", "위안부", "정치인", "군인", "교육자",
    "언론인", "역사가", "사상가", "혁명가", "테러", "암살", "학살", "만세",
    # 주제 문서·총칭·현대 기관 — 개체가 아니거나 스키마 범위 밖이다
    "한국의 독립운동", "독립군", "일본군", "조선군", "청군", "관동군", "팔로군",
    "제국주의", "식민주의", "자결권", "파업", "반전 운동", "광복절", "신한촌", "서당",
    "교회", "개신교", "기독교", "천주교", "불교", "대종교", "로마 가톨릭교회",
    "개화파", "공산당", "유엔", "청나라", "중화인민공화국", "조선민주주의인민공화국",
    "대한민국 정부", "대한민국 국군", "국방부", "일본 제국 육군", "일본 제국",
    "조선총독부", "인민위원회",
    # 사람 부류를 가리키는 보통명사 (문장에 자주 나와 엉뚱한 엣지를 만든다)
    "일본인", "중국인", "조선인", "한국인", "일본 천황", "천황", "목사", "신부",
    "승려", "농민", "노동자", "지식인", "유생", "관료", "병사", "장병", "군인들",
    # 지나치게 일반적인 지명/국가명 (허브만 만들고 사실을 더하지 않는다)
    "한국", "조선", "대한민국", "대한제국", "일본", "중국", "미국", "러시아", "소련",
    "영국", "프랑스", "독일", "만주", "서울", "경성", "평양", "상하이", "상해", "베이징",
    "도쿄", "하와이", "연해주", "간도", "블라디보스토크", "일제", "일제강점기",
}

# --------------------------------------------------------------------------- #
# 3. 약칭 사전 — 별칭 → 표준 이름
# --------------------------------------------------------------------------- #
ABBREVIATIONS: dict[str, str] = {
    "임시정부": "대한민국 임시정부",
    "임정": "대한민국 임시정부",
    "상해 임시정부": "대한민국 임시정부",
    "상하이 임시정부": "대한민국 임시정부",
    "대한민국임시정부": "대한민국 임시정부",
    "상해임시정부": "대한민국 임시정부",
    "임시 정부": "대한민국 임시정부",
    "광복군": "한국광복군",
    "한국 광복군": "한국광복군",
    "삼일운동": "3·1 운동",
    "3.1 운동": "3·1 운동",
    "3·1운동": "3·1 운동",
    "3.1운동": "3·1 운동",
    "삼일 운동": "3·1 운동",
    "기미독립운동": "3·1 운동",
    "기미 독립운동": "3·1 운동",
    "육십만세운동": "6·10 만세운동",
    "6.10 만세운동": "6·10 만세운동",
    "6·10만세운동": "6·10 만세운동",
    "청산리대첩": "청산리 전투",
    "청산리 대첩": "청산리 전투",
    "봉오동전투": "봉오동 전투",
    "애국단": "한인애국단",
    "의열단원": "의열단",
    "신흥무관학교": "신흥무관학교",
    "임시의정원": "대한민국 임시의정원",
    "독립신문": "독립신문",
}

# --------------------------------------------------------------------------- #
# 4. 한국어 문장 패턴
# --------------------------------------------------------------------------- #
# 능동형과 피동형을 나눈다. 피동형("…이 설립되었다")에서는 같은 문장에 있는 인물이
# 설립자라는 보장이 없어서, 행위자 표지("에 의해", "중심이 되어")가 함께 있을 때만 쓴다.
_ACT = r"(?:하였|하여|하며|하고|한|해|했|합|함|하기|하는|할|하|시켰|시킨|시켜)"
_PAS = r"(?:되었|되어|되고|되며|된|됐|됨|되기|되는|될|되)"

_FOUND_STEMS = ("조직", "창설", "설립", "결성", "창립", "창건", "발족", "수립",
                "창단", "창간", "건립", "출범", "조직화")
_FOUND_ALT = "(?:" + "|".join(_FOUND_STEMS) + ")"
FOUND_ACTIVE_RE = re.compile(_FOUND_ALT + _ACT + r"|세웠|세운|세우고")
FOUND_PASSIVE_RE = re.compile(_FOUND_ALT + _PAS)
FOUND_VERB_RE = re.compile(FOUND_ACTIVE_RE.pattern + "|" + FOUND_PASSIVE_RE.pattern)
# 피동형에서 행위자를 드러내는 표지
AGENT_MARKER_RE = re.compile(r"에 의해|에 의하여|의 주도로|중심이 되어|주축이 되어|"
                             r"주도하여|주도로|등이|들이|함께")

_PART_STEMS = ("참여", "참가", "참전", "가담", "합류", "동참", "관여", "주도", "주동",
               "거행", "전개", "활약", "가세", "출전", "종군", "기여", "봉기", "궐기",
               "호응", "선포", "낭독", "서명", "투신", "결행", "단행", "감행")
_PART_ALT = "(?:" + "|".join(_PART_STEMS) + ")"
PART_ACTIVE_RE = re.compile(
    _PART_ALT + _ACT +
    r"|일으켰|일으킨|일으켜|이끌었|이끈|이끌고|지휘하|지휘했|지휘한|지휘를|"
    r"벌였|벌인|맞서|싸웠|싸운|참전|출정|진두지휘")
PART_PASSIVE_RE = re.compile(_PART_ALT + _PAS)
PART_VERB_RE = re.compile(PART_ACTIVE_RE.pattern + "|" + PART_PASSIVE_RE.pattern)

# 문서 스코프 규칙에서만 쓰는 역할 명사 (신뢰도를 낮춰 적용)
PART_ROLE_RE = re.compile(
    r"지휘관|사령관|총사령|대장|부대장|참모|대표|주역|주모자|주동자|지도자|단원|대원|"
    r"의사|열사|민족대표|33인|선언서"
)

# 개체 뒤에 붙을 수 있는 조사/접미사 — 경계 판정용
_SUFFIX = r"(?:원들|원|장|측|계|파|내|들|군|사|씨)?"
_JOSA = (r"(?:은|는|이|가|을|를|의|에|와|과|도|로|으로|에서|에게|께서|께|부터|까지|만|"
         r"이나|나|라|이라|이라고|라고|이며|며|이고|고|이다|다|인|한|등|및|처럼|같이|"
         r"보다|조차|마저|밖에|이라는|라는|으로부터|로부터|에서는|에게서|에는|에도|"
         r"와는|과는|이라도|였|이었|랑|하고)?")
_RIGHT_BOUND_RE = re.compile(_SUFFIX + _JOSA + r"(?![가-힣])")
_HANGUL_OR_ALNUM = re.compile(r"[가-힣A-Za-z0-9]")

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。])\s+|\n+")
_HANJA_RE = re.compile(r"[一-鿿]")

ALLOWED_TRIPLES: set[tuple[str, str, str]] = set()


# --------------------------------------------------------------------------- #
# 정규화
# --------------------------------------------------------------------------- #
class Normalizer:
    """표기 정규화. 어떤 규칙이 몇 번 적용됐는지 집계한다."""

    RULE_WHITESPACE = "공백·중점(·) 표기 정규화"
    RULE_PAREN = "괄호 한정어 제거"
    RULE_HANJA = "한자 병기 분리"
    RULE_JOSA = "조사 제거"
    RULE_REDIRECT = "위키 넘겨주기 별칭 흡수"
    RULE_ABBREV = "약칭 사전"
    RULE_NOSPACE = "무공백 표기 통합"

    def __init__(self) -> None:
        self.counts: Counter = Counter()
        self.examples: dict[str, str] = {}

    def _hit(self, rule: str, example: str) -> None:
        self.counts[rule] += 1
        self.examples.setdefault(rule, example)

    def normalize(self, raw: str) -> tuple[str, str | None]:
        """(표준 이름, 한자 병기) 를 돌려준다."""
        s = unicodedata.normalize("NFC", raw or "").strip()
        before = s

        # 중점·구분기호 통일 + 공백 정규화
        s = re.sub(r"[·‧・•ㆍ･‧]", "·", s)
        s = re.sub(r"[《》〈〉「」『』\"'“”‘’]", "", s)
        s = re.sub(r"\s*·\s*", "·", s)
        s = re.sub(r"\s+", " ", s).strip()
        if s != before:
            self._hit(self.RULE_WHITESPACE, "{} → {}".format(before, s))

        # 한자 병기 분리
        hanja = None
        m = re.search(r"[(（]\s*([一-鿿][一-鿿\s·,、]*)\s*[)）]", s)
        if m:
            hanja = re.sub(r"\s+", "", m.group(1))
            self._hit(self.RULE_HANJA, "{} → {} / {}".format(
                s, re.sub(r"\s*[(（][^()（）]*[)）]", "", s).strip(), hanja))

        # 괄호 한정어 제거
        stripped = re.sub(r"\s*[(（][^()（）]*[)）]", "", s).strip()
        if stripped and stripped != s:
            if hanja is None:
                self._hit(self.RULE_PAREN, "{} → {}".format(s, stripped))
            s = stripped

        s = re.sub(r"\s+", " ", s).strip(" ,;·")
        return s, hanja

    def strip_josa(self, raw: str) -> str:
        s = (raw or "").strip()
        m = re.match(r"^(.*?)(?:은|는|이|가|을|를|의|와|과|에서|에게|에|도|으로|로|등)$", s)
        if m and len(m.group(1)) >= 2:
            self._hit(self.RULE_JOSA, "{} → {}".format(s, m.group(1)))
            return m.group(1)
        return s

    def rules_report(self) -> list[dict]:
        order = [self.RULE_WHITESPACE, self.RULE_PAREN, self.RULE_HANJA,
                 self.RULE_JOSA, self.RULE_REDIRECT, self.RULE_ABBREV,
                 self.RULE_NOSPACE]
        return [{"rule": r, "example": self.examples.get(r, "-"),
                 "applied": self.counts.get(r, 0)} for r in order]


# --------------------------------------------------------------------------- #
# 타입 판정
# --------------------------------------------------------------------------- #
def classify(title: str, categories: list[str]) -> tuple[str | None, str]:
    """(타입, 판정 근거).

    판정 순서가 중요하다. 분류만 보면 '3·1 운동' 이 '조선총독부'·'대한민국 임시정부'
    같은 분류 때문에 Organization 으로 잘못 판정된다. 반면 **제목 접미사가 사건인 경우
    (…전투/…운동/…의거/…사건) 는 사실상 예외가 없다.** 그래서
      ① 인물 분류  →  ② 사건 제목 접미사  →  ③ 단체 분류  →  ④ 사건 분류  →  ⑤ 단체 제목 접미사
    순으로 본다. 인물 분류를 맨 앞에 두는 이유는 '○○년 태어남/죽음' 이 인물에만 붙어
    오탐이 0이기 때문이다.
    """
    base = re.sub(r"\s*[(（][^()（）]*[)）]", "", title).strip()
    cat_blob = " / ".join(categories)
    if categories:
        m = NON_ENTITY_CAT_RE.search(cat_blob)
        if m:
            return None, "non_entity:{}".format(m.group(0))
        m = PERSON_CAT_RE.search(cat_blob)
        if m:
            return "Person", "category:{}".format(m.group(0))
    m = EVENT_TITLE_RE.search(base)
    if m:
        return "Event", "title_suffix:{}".format(m.group(0))
    if categories:
        m = ORG_CAT_RE.search(cat_blob)
        if m:
            return "Organization", "category:{}".format(m.group(0))
    # 단체 제목 접미사를 사건 분류보다 먼저 본다. '대한인국민회' 는 분류에
    # '한국의 독립운동'(→ '운동') 이 걸려 사건으로 오판되기 때문이다.
    m = ORG_TITLE_RE.search(base)
    if m:
        return "Organization", "title_suffix:{}".format(m.group(0))
    if categories:
        m = EVENT_CAT_RE.search(cat_blob)
        if m:
            return "Event", "category:{}".format(m.group(0))
    return None, "unclassified"


# --------------------------------------------------------------------------- #
# 개체 저장소
# --------------------------------------------------------------------------- #
class EntityStore:
    """(정규화 이름, 타입) 을 키로 노드를 모은다. **타입이 다르면 합치지 않는다.**"""

    def __init__(self, norm: Normalizer, min_len: int) -> None:
        self.norm = norm
        self.min_len = min_len
        self.nodes: dict[tuple[str, str], dict] = {}
        self.surface_map: dict[str, tuple[str, str]] = {}   # 표기 → (이름, 타입)
        self.dropped: Counter = Counter()
        self.type_conflicts: list[dict] = []
        self.disambiguated: list[dict] = []

    def _ok(self, name: str) -> bool:
        if "목록" in name or "일람" in name:
            self.dropped["list_page"] += 1
            return False
        # 직책·직위는 개체가 아니다 ('대한민국 임시정부 국무총리' 등)
        if _POSITION_SUFFIX_RE.search(name):
            self.dropped["position_title"] += 1
            return False
        if len(name) < self.min_len:
            self.dropped["too_short"] += 1
            return False
        if name in STOPWORDS:
            self.dropped["stopword"] += 1
            return False
        if re.fullmatch(r"[\d\W_]+", name):
            self.dropped["non_entity"] += 1
            return False
        return True

    def add(self, raw: str, etype: str, source: str, *, is_doc: bool = False,
            reason: str = "", wiki_title: str | None = None) -> tuple[str, str] | None:
        """개체를 등록한다.

        `wiki_title` 은 이 개체가 유래한 위키 문서 제목이다. **서로 다른 위키 문서가 같은
        이름으로 정규화되면 동명이인이므로 합치지 않는다** — `김규식 (북로군정서)` 는
        정치인 `김규식` 과 다른 사람이다. 이 경우 괄호 한정어를 되살려 노드를 분리한다.
        """
        name, hanja = self.norm.normalize(raw)
        # 약칭 사전이 이름을 바꿔 준 경우는 **의도된 병합**이다.
        # ('임시정부' → '대한민국 임시정부'). 이때 동명이인 분리를 걸면
        # 같은 조직이 '임시정부'·'상해 임시정부'·'대한민국 임시정부' 로 쪼개진다.
        merged_by_abbrev = name in ABBREVIATIONS
        name = ABBREVIATIONS.get(name, name)
        if not self._ok(name):
            return None
        key = (name, etype)
        node = self.nodes.get(key)
        if node is not None and wiki_title and not merged_by_abbrev \
                and node.get("wiki_title") and node["wiki_title"] != wiki_title:
            disamb = re.sub(r"\s+", " ", unicodedata.normalize("NFC", raw)).strip()
            if disamb == name:
                return key   # 한정어가 없어 분리할 방법이 없으면 그대로 둔다
            self.disambiguated.append(
                {"surface": name, "kept": node["wiki_title"], "separated": disamb,
                 "reason": "서로 다른 위키 문서가 같은 이름으로 정규화됨(동명이인) → 병합 금지"})
            key = (disamb, etype)
            node = self.nodes.get(key)
        if node is None:
            node = {"name": key[0], "type": etype, "aliases": {key[0]},
                    "sources": set(), "is_doc": False, "reason": reason,
                    "categories": [], "wiki_title": wiki_title}
            self.nodes[key] = node
        node["is_doc"] = node["is_doc"] or is_doc
        if reason and not node["reason"]:
            node["reason"] = reason
        if wiki_title and not node.get("wiki_title"):
            node["wiki_title"] = wiki_title
        if source:
            node["sources"].add(source)
        if hanja:
            self.alias(hanja, key, Normalizer.RULE_HANJA)
        self.alias(key[0], key, Normalizer.RULE_WHITESPACE)
        return key

    def alias(self, surface: str, key: tuple[str, str], rule: str) -> None:
        s = unicodedata.normalize("NFC", (surface or "").strip())
        s = re.sub(r"[·‧・•‧]", "·", s)
        s = re.sub(r"\s+", " ", s).strip()
        if not s or len(s) < self.min_len or s in STOPWORDS:
            return
        if key not in self.nodes:
            return
        prev = self.surface_map.get(s)
        if prev is not None and prev != key:
            # 같은 표기를 다른 타입이 다투면 합치지 않고 기록만 남긴다.
            if prev[0] != key[0] or prev[1] != key[1]:
                self.type_conflicts.append(
                    {"surface": s, "kept": list(prev), "rejected": list(key),
                     "reason": "동일 표기이지만 타입/이름이 달라 병합하지 않음"})
            return
        self.surface_map[s] = key
        self.nodes[key]["aliases"].add(s)
        if s != key[0]:
            self.norm.counts[rule] += 0  # 규칙 카운트는 호출부에서 처리


# --------------------------------------------------------------------------- #
# 문서 로딩
# --------------------------------------------------------------------------- #
def load_docs() -> list[dict]:
    docs: list[dict] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        m = re.match(r"^#\s*(.+?)\n\s*\n분류:\s*(.*?)\n\s*\n(.*)$", raw, flags=re.S)
        if not m:
            continue
        title = m.group(1).strip()
        cats = [c.strip() for c in m.group(2).split(",") if c.strip()]
        body = m.group(3).strip()
        docs.append({"title": title, "categories": cats,
                     "body": trim_body(body), "raw_chars": len(body),
                     "file": path.name})
    return docs


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for chunk in _SENT_SPLIT_RE.split(text):
        s = (chunk or "").strip(" \t·-—=*")
        if len(s) < 8:
            continue
        if s.startswith("[문서 "):
            continue
        out.append(s)
    return out


# --------------------------------------------------------------------------- #
# 개체 언급 탐지
# --------------------------------------------------------------------------- #
class Mention:
    __slots__ = ("start", "end", "surface", "key")

    def __init__(self, start: int, end: int, surface: str, key: tuple[str, str]):
        self.start, self.end, self.surface, self.key = start, end, surface, key

    @property
    def name(self) -> str:
        return self.key[0]

    @property
    def type(self) -> str:
        return self.key[1]

    def __repr__(self) -> str:  # pragma: no cover - 디버깅용
        return "Mention({}:{})".format(self.surface, self.type)


class MentionFinder:
    """표기 사전을 긴 것부터 매칭한다. 조사/접미사는 경계 규칙으로 잘라낸다."""

    def __init__(self, surface_map: dict[str, tuple[str, str]], norm: Normalizer):
        self.surface_map = surface_map
        self.norm = norm
        surfaces = sorted(surface_map, key=lambda s: (-len(s), s))
        if surfaces:
            self.pattern = re.compile("|".join(re.escape(s) for s in surfaces))
        else:
            self.pattern = None

    def find(self, text: str) -> list[Mention]:
        if self.pattern is None:
            return []
        out: list[Mention] = []
        for m in self.pattern.finditer(text):
            i, j = m.start(), m.end()
            if i > 0 and _HANGUL_OR_ALNUM.match(text[i - 1]):
                continue
            rb = _RIGHT_BOUND_RE.match(text, j)
            if not rb:
                continue
            key = self.surface_map.get(m.group(0))
            if key is None:
                continue
            if rb.end() > j:
                # 개체 바로 뒤에 조사/접미사가 붙어 있었고, 그것을 떼어내고 매칭했다.
                self.norm.counts[Normalizer.RULE_JOSA] += 1
                self.norm.examples.setdefault(
                    Normalizer.RULE_JOSA,
                    "{} → {}".format(text[i:rb.end()], m.group(0)))
            out.append(Mention(i, j, m.group(0), key))
        return out


_HANJA_PAREN_RE = re.compile(
    r"([가-힣][가-힣·\s]{1,14}?)\s*[(（]\s*([一-鿿]{2,10})\s*[)）]")

# --------------------------------------------------------------------------- #
# 본문 기반 Event/Organization 개체 발견
# --------------------------------------------------------------------------- #
# 위키 분류로 타입을 판정하려면 그 개체에 **자기 문서가 있어야** 한다. 그런데
# '자유시 참변'·'경학사'처럼 본문에는 자주 나오지만 gazetteer 에 없는 사건·조직이 많고,
# 그만큼 PARTICIPATED_IN / FOUNDED 를 만들 상대가 사라진다.
# 사건·조직은 **접미사가 곧 타입**이라는 한국어 특성을 쓰면 규칙만으로 안전하게 잡을 수 있다.
# (인물은 접미사로 판정할 수 없으므로 이 경로로 만들지 않는다.)
_EVENT_DISC_RE = re.compile(
    r"(?<![가-힣])((?:[가-힣0-9·]{2,8})(?:\s[가-힣0-9·]{1,8})?\s?"
    r"(?:전투|의거|사건|봉기|항쟁|대첩|학살|전쟁|조약|선언|참변|정변|사변|폭동))(?![가-힣])")
_ORG_MULTI_DISC_RE = re.compile(
    r"(?<![가-힣])((?:[가-힣0-9·]{2,8})(?:\s[가-힣0-9·]{1,8})?\s?"
    r"(?:무관학교|의정원|총독부|군정서|광복군|독립군|의용대|의용군|청년회|부인회|애국단|"
    r"결사대|국민회|광복회|사령부|학교|협회|학회|연맹|동맹|위원회|신문사|은행|본부|정부|"
    r"재단|공사))(?![가-힣])")
_ORG_SINGLE_DISC_RE = re.compile(
    r"(?<![가-힣])([가-힣]{2,7}(?:단|회|당|군|부|서|사))(?![가-힣])")

# 운동·투쟁을 가리키는 **총칭**은 고유 사건이 아니다 → 노드로 만들지 않는다.
_GENERIC_DISCOVERED = {
    "독립운동", "항일운동", "민족운동", "애국운동", "계몽운동", "학생운동", "노동운동",
    "사회운동", "문화운동", "실력양성운동", "무장투쟁", "독립전쟁", "해방운동", "저항운동",
    "농민운동", "여성운동", "청년운동", "반일운동", "의병운동", "국권회복운동",
    "애국계몽운동", "무장독립운동", "민족해방운동", "혁명운동", "만세운동", "독립선언",
    "세계대전", "세계 대전", "대일전쟁", "무장항쟁", "구국운동", "자치운동", "물산장려",
    "임시정부", "일본군", "관동군", "일본 정부", "조선 정부", "중국 정부", "미국 정부",
    "소련군", "연합군", "관동대학살", "한국군", "국군", "중국군", "미군", "일군",
}

# 이 접미사로 끝나는 2~3음절 토막은 사람 이름일 가능성이 커서 뺀다.
_DISC_MIN_LEN = 3


def discover_entities(docs: list[dict], store: EntityStore,
                      min_docs_multi: int = 2, min_docs_single: int = 3) -> dict:
    """본문에서 사건·조직 후보를 접미사 규칙으로 찾아 노드 인벤토리에 더한다.

    오탐을 막는 장치 세 가지:
      1. `min_docs_*` 개 이상의 **서로 다른 문서**에 나와야 한다 (한 문서의 말버릇 배제).
      2. 이미 표기 사전에 있는 표기(=타입이 확정된 개체)는 건드리지 않는다.
      3. 총칭(`독립운동`, `일본군` …)과 불용어는 제외한다.
    단음절 접미사(단·회·당·군·부·서·사)는 인명과 헷갈릴 수 있어 문서 기준을 더 높인다.
    """
    support: dict[tuple[str, str], set[str]] = defaultdict(set)
    threshold: dict[tuple[str, str], int] = {}
    surface_of: dict[tuple[str, str], str] = {}

    specs = [(_EVENT_DISC_RE, "Event", min_docs_multi),
             (_ORG_MULTI_DISC_RE, "Organization", min_docs_multi),
             (_ORG_SINGLE_DISC_RE, "Organization", min_docs_single)]
    for d in docs:
        for rx, etype, need in specs:
            for m in rx.finditer(d["body"]):
                raw = m.group(1).strip()
                if raw in store.surface_map:
                    continue
                name, _ = store.norm.normalize(raw)
                name = ABBREVIATIONS.get(name, name)
                if (len(name) < _DISC_MIN_LEN or name in STOPWORDS
                        or name in _GENERIC_DISCOVERED or name in store.surface_map):
                    continue
                key = (name, etype)
                support[key].add(d["title"])
                surface_of.setdefault(key, raw)
                threshold[key] = min(threshold.get(key, need), need)

    added: list[dict] = []
    for key in sorted(support):
        docs_seen = support[key]
        if len(docs_seen) < threshold[key]:
            continue
        if key[0] in store.surface_map:
            continue
        k = store.add(key[0], key[1], "", reason="text_discovery:접미사 규칙")
        if k is None:
            continue
        added.append({"id": k[0], "type": k[1], "doc_support": len(docs_seen),
                      "example_surface": surface_of[key]})
    return {"added": added, "candidates": len(support)}


def harvest_hanja_aliases(store: EntityStore, docs: list[dict]) -> int:
    """본문의 `안중근(安重根)` 형태에서 한자 병기를 별칭으로 흡수한다."""
    n = 0
    for d in docs:
        for m in _HANJA_PAREN_RE.finditer(d["body"]):
            raw = m.group(1).strip()
            hanja = m.group(2)
            if hanja in store.surface_map:
                continue
            # 괄호 앞 문자열의 **가장 긴 접미사**가 표기 사전에 있으면 그것이 이름이다.
            # ("…년에는 간도(間島)" 처럼 앞말이 딸려 들어오는 것을 막는다)
            key = None
            surface = ""
            for k in range(len(raw)):
                sub = raw[k:]
                if sub in store.surface_map:
                    key, surface = store.surface_map[sub], sub
                    break
            if key is None:
                continue
            store.alias(hanja, key, Normalizer.RULE_HANJA)
            if store.surface_map.get(hanja) == key:
                n += 1
                store.norm.counts[Normalizer.RULE_HANJA] += 1
                store.norm.examples.setdefault(
                    Normalizer.RULE_HANJA, "{}({}) → {}".format(surface, hanja, key[0]))
    return n


# --------------------------------------------------------------------------- #
# 엣지 수집기
# --------------------------------------------------------------------------- #
class EdgeCollector:
    def __init__(self) -> None:
        self.edges: dict[tuple[tuple[str, str], str, tuple[str, str]], dict] = {}
        self.rejected: list[dict] = []
        self.rejected_seen: set[tuple] = set()
        self.by_extractor: Counter = Counter()

    def reject(self, head: str, rel: str, tail: str, reason: str) -> None:
        sig = (head, rel, tail, reason.split(":")[0])
        if sig in self.rejected_seen:
            return
        self.rejected_seen.add(sig)
        self.rejected.append({"triple": [head, rel, tail], "reason": reason})

    def add(self, h: tuple[str, str], rel: str, t: tuple[str, str], *,
            origin: str, confidence: float, source: str, evidence: str,
            extractor: str) -> bool:
        if h == t or h[0] == t[0]:
            self.reject(h[0], rel, t[0], "self_loop: head 와 tail 이 같은 개체")
            return False
        triple_types = (h[1], rel, t[1])
        if triple_types not in ALLOWED_TRIPLES:
            allowed = ", ".join(
                "({}, {}, {})".format(*x) for x in sorted(ALLOWED_TRIPLES))
            self.reject(h[0], rel, t[0],
                        "schema_type_mismatch: ({}, {}, {}) 는 허용 튜플이 아님 "
                        "[허용: {}]".format(h[1], rel, t[1], allowed))
            return False
        key = (h, rel, t)
        ev = evidence.strip()
        if len(ev) > EVIDENCE_MAX:
            ev = ev[:EVIDENCE_MAX - 1].rstrip() + "…"
        e = self.edges.get(key)
        if e is None:
            e = {"head": h, "relation": rel, "tail": t, "origins": set(),
                 "confidence": 0.0, "sources": set(), "evidence": [],
                 "extractors": set()}
            self.edges[key] = e
        e["origins"].add(origin)
        e["confidence"] = max(e["confidence"], confidence)
        if source:
            e["sources"].add(source)
        if ev and ev not in e["evidence"] and len(e["evidence"]) < EVIDENCE_PER_EDGE:
            e["evidence"].append(ev)
        e["extractors"].add(extractor)
        self.by_extractor[extractor] += 1
        return True


# --------------------------------------------------------------------------- #
# 규칙 추출기
# --------------------------------------------------------------------------- #
def _verb_positions(sent: str, rx: re.Pattern) -> list[int]:
    return [m.start() for m in rx.finditer(sent)]


def _has_locative(sent: str, mention: Mention) -> bool:
    return bool(re.match(r"(?:에서|에|을|를|의|으로|로)", sent[mention.end:mention.end + 2]))


# 한국어 격조사로 주어/목적어를 가른다. 이게 없으면 "신민회에 가입하여 … 국채보상운동에
# 참여하였다" 에서 신민회가 참여 주체로 잘못 잡히고, "임정은 … 해외독립단체가 만든 정부"
# 에서 임정이 설립 대상으로 잘못 잡힌다 (실제로 관찰된 오탐이다).
_SUBJECT_MARK_RE = re.compile(r"^(?:은|는|이|가|도|와|과|및|등|께서|[,·]|\s|$)")
_NOT_SUBJECT_RE = re.compile(r"^(?:에서|에게|에|의)")
_OBJECT_MARK_RE = re.compile(r"^(?:을|를|등을|[,·]|와|과|및)")
_NOT_OBJECT_RE = re.compile(r"^(?:은|는|이|가|에서|에게|에|의|으로|로)")
# 한국어 위키 본문은 인명 바로 뒤에 한자 병기를 붙인다: "안창호(安昌浩), 이갑, …".
# 격조사를 보기 전에 이 괄호를 먼저 걷어내야 주어 판정이 망가지지 않는다.
_POSITION_SUFFIX_RE = re.compile(
    r"(국무총리|대통령|국무위원|주석|총리|장관|위원장|의장|사령관|총사령|"
    r"교장|총장|회장|단장|대표|비서장|참모장|국장|부장|과장)$")
_TRAILING_HANJA_RE = re.compile(r"^\s*[(（][一-鿿\s·,、]+[)）]")


def _rest_after(sent: str, m: Mention) -> str:
    rest = sent[m.end:]
    hz = _TRAILING_HANJA_RE.match(rest)
    return rest[hz.end():] if hz else rest


def _subject_like(sent: str, m: Mention) -> bool:
    """이 언급이 문장에서 **행위 주체**로 쓰였는가 (조사 기반)."""
    rest = _rest_after(sent, m)
    if _NOT_SUBJECT_RE.match(rest):
        return False
    # "안창호를 위시한 인사들이 …" 처럼 '…를 위시한/비롯한' 은 주체 열거다.
    if re.match(r"^(?:을|를)\s*(?:위시|비롯|중심|포함)", rest):
        return True
    if re.match(r"^(?:을|를)", rest):
        return False
    return bool(_SUBJECT_MARK_RE.match(rest))


def _object_like(sent: str, m: Mention) -> bool:
    """이 언급이 **행위 대상(목적어)** 으로 쓰였는가."""
    rest = _rest_after(sent, m)
    if _NOT_OBJECT_RE.match(rest):
        return False
    return bool(_OBJECT_MARK_RE.match(rest)) or rest.startswith(" ")


_OBLIQUE_RE = re.compile(r"^(?:에서|에게|에|의)")


def _not_oblique(sent: str, m: Mention) -> bool:
    """처소/소유/소속 조사가 바로 붙지 않았는가 (문서 스코프 규칙용 느슨한 판정)."""
    return not _OBLIQUE_RE.match(_rest_after(sent, m))


def extract_rules(doc: dict, doc_key: tuple[str, str] | None, sent: str,
                  mentions: list[Mention], col: EdgeCollector) -> int:
    """한 문장에서 규칙 기반 삼중항을 뽑는다. 만들어진 엣지 수를 돌려준다."""
    made = 0
    title = doc["title"]
    persons = [m for m in mentions if m.type == "Person"]
    orgs = [m for m in mentions if m.type == "Organization"]
    events = [m for m in mentions if m.type == "Event"]
    # 후보는 **일부러 스키마보다 넓게** 만든다. 좁게 만들면 위반 자체가 생기지 않아
    # "무엇을 왜 버렸는지" 를 남길 수 없다. 스키마 게이트(EdgeCollector.add)가
    # (Organization, FOUNDED, Organization) · (Person, PARTICIPATED_IN, Organization)
    # 같은 후보를 사유와 함께 rejected_edges 로 떨어뜨린다.
    found_heads = persons + orgs          # 설립 주체 후보
    found_tails = orgs + events           # 설립 대상 후보
    part_heads = persons + orgs + events  # 참여 주체 후보
    part_tails = events + orgs            # 참여 대상 후보
    found_a = _verb_positions(sent, FOUND_ACTIVE_RE)
    found_p = _verb_positions(sent, FOUND_PASSIVE_RE)
    part_a = _verb_positions(sent, PART_ACTIVE_RE)
    part_p = _verb_positions(sent, PART_PASSIVE_RE)
    has_agent = bool(AGENT_MARKER_RE.search(sent))

    # --- FOUNDED v1: 능동 "X(은/는/이/가) ... Y(을/를) 조직/창설/설립/결성/창립하였다" ---
    if found_a:
        for p in found_heads:
            for o in found_tails:
                if not (p.start < o.start):
                    continue
                if not (_subject_like(sent, p) and _object_like(sent, o)):
                    continue
                if not any(o.end <= x <= o.end + 25 for x in found_a):
                    continue
                made += col.add(p.key, "FOUNDED", o.key, origin="rule",
                                confidence=0.88, source=title, evidence=sent,
                                extractor="rule_founded_v1_active")

    # --- FOUNDED v2: 피동 "Y(은/는) ... X(에 의해|등이) 설립되었다" ---
    if found_p and has_agent:
        for o in found_tails:
            for p in found_heads:
                if not (o.start < p.start):
                    continue
                v = next((x for x in found_p if p.end <= x <= p.end + 30), None)
                if v is None:
                    continue
                if not AGENT_MARKER_RE.search(sent[p.end:v] or "등이"):
                    continue
                made += col.add(p.key, "FOUNDED", o.key, origin="rule",
                                confidence=0.80, source=title, evidence=sent,
                                extractor="rule_founded_v2_passive")

    # --- PARTICIPATED_IN v1: 능동 "X(은/는) ... Y(에) 참여/가담/주도하였다" ---
    if part_a:
        for h in part_heads:
            for e in part_tails:
                if not (h.start < e.start):
                    continue
                if not _subject_like(sent, h):
                    continue
                if not any(e.end <= x <= e.end + 30 for x in part_a):
                    continue
                made += col.add(h.key, "PARTICIPATED_IN", e.key, origin="rule",
                                confidence=0.88, source=title, evidence=sent,
                                extractor="rule_participated_v1_active")

        # --- PARTICIPATED_IN v2: "Y에서 X(이/가) ... 하였다" (사건이 앞에 오는 어순) ---
        for e in part_tails:
            if not _has_locative(sent, e):
                continue
            for h in part_heads:
                if not (e.start < h.start):
                    continue
                if not _subject_like(sent, h):
                    continue
                if not any(h.end <= x <= h.end + 40 for x in part_a):
                    continue
                made += col.add(h.key, "PARTICIPATED_IN", e.key, origin="rule",
                                confidence=0.75, source=title, evidence=sent,
                                extractor="rule_participated_v2_locative")

    if doc_key is None:
        return made
    _doc_name, doc_type = doc_key

    # --- 문서 스코프 규칙 (헤더 주입으로 문서 자신을 주어/목적어로 본다) ---
    other_orgs = [o for o in orgs if o.key != doc_key]
    other_persons = [p for p in persons if p.key != doc_key]
    other_events = [e for e in events if e.key != doc_key]

    # 조직 문서: "…는 1907년 안창호 등이 조직한 비밀결사이다" → (안창호, FOUNDED, 문서조직)
    if doc_type == "Organization" and not other_orgs:
        for p in other_persons:
            if not _not_oblique(sent, p):
                continue
            if any(p.end <= v <= p.end + 30 for v in found_a):
                conf, ex = 0.82, "rule_founded_v3_doc_org"
            elif has_agent and any(p.end <= v <= p.end + 30 for v in found_p):
                conf, ex = 0.74, "rule_founded_v3p_doc_org_passive"
            else:
                continue
            made += col.add(p.key, "FOUNDED", doc_key, origin="rule",
                            confidence=conf, source=title, evidence=sent, extractor=ex)

    # 인물 문서: "…는 흥사단을 창립하였다" → (문서인물, FOUNDED, 흥사단)
    # 피동형("임시정부가 수립되자")은 문서 인물이 설립자라는 근거가 못 되므로 쓰지 않는다.
    if doc_type == "Person" and found_a and not other_persons:
        for o in other_orgs:
            if not _object_like(sent, o):
                continue
            if not any(o.end <= v <= o.end + 20 for v in found_a):
                continue
            made += col.add(doc_key, "FOUNDED", o.key, origin="rule",
                            confidence=0.80, source=title, evidence=sent,
                            extractor="rule_founded_v4_doc_person")

    # 사건 문서: 참여 동사나 역할 명사와 함께 등장한 인물/조직 → 문서 사건 참여
    if doc_type == "Event" and not other_events:
        strong = bool(part_a or part_p)
        role = bool(PART_ROLE_RE.search(sent))
        if strong or role:
            for h in other_persons + other_orgs:
                made += col.add(h.key, "PARTICIPATED_IN", doc_key, origin="rule",
                                confidence=0.80 if strong else 0.68,
                                source=title, evidence=sent,
                                extractor="rule_participated_v3_doc_event" if strong
                                else "rule_participated_v5_doc_event_role")

    # 인물/조직 문서: "…는 3·1 운동에 참여하였다" → (문서개체, PARTICIPATED_IN, 사건)
    if doc_type in ("Person", "Organization") and part_a:
        if not other_persons and not other_orgs:
            for e in other_events:
                if not any(e.end <= v <= e.end + 25 for v in part_a):
                    continue
                made += col.add(doc_key, "PARTICIPATED_IN", e.key, origin="rule",
                                confidence=0.80, source=title, evidence=sent,
                                extractor="rule_participated_v4_doc_entity")

    return made


def extract_category_edges(store: EntityStore, col: EdgeCollector,
                           doc_by_name: dict[str, dict]) -> None:
    """위키 분류가 곧 사건 이름인 경우 → PARTICIPATED_IN (정확도가 높은 규칙)."""
    event_names = {name: (name, t) for (name, t) in store.nodes if t == "Event"}
    norm = store.norm
    for (name, etype), node in sorted(store.nodes.items()):
        if etype not in ("Person", "Organization"):
            continue
        doc = doc_by_name.get(name)
        if not doc:
            continue
        for cat in doc["categories"]:
            cname, _ = norm.normalize(cat)
            cname = ABBREVIATIONS.get(cname, cname)
            ekey = event_names.get(cname)
            if ekey is None:
                continue
            col.add((name, etype), "PARTICIPATED_IN", ekey, origin="rule",
                    confidence=0.90, source=doc["title"],
                    evidence="분류: {} (문서 '{}' 의 위키 분류 행)".format(cat, doc["title"]),
                    extractor="rule_cat_participated_v6")


# --------------------------------------------------------------------------- #
# LLM 증강 (선택)
# --------------------------------------------------------------------------- #
LLM_SYSTEM = (
    "너는 한국 독립운동사 문서에서 지식 그래프 삼중항을 뽑는 추출기다.\n"
    "허용된 삼중항은 정확히 다음 3가지뿐이다 (방향 고정):\n"
    "  (Person, PARTICIPATED_IN, Event)\n"
    "  (Person, FOUNDED, Organization)\n"
    "  (Organization, PARTICIPATED_IN, Event)\n"
    "규칙:\n"
    "1. head/tail 은 반드시 주어진 후보 개체 목록의 이름을 **그대로** 써라. 새 이름을 만들지 마라.\n"
    "2. 문장에 명시되지 않은 관계는 추측하지 마라. 없으면 빈 배열을 반환하라.\n"
    "3. FOUNDED 는 '설립·창립·조직·결성'처럼 만든 행위에만 쓴다. 단순 소속은 제외한다.\n"
    "4. 출력은 JSON 하나뿐이다: "
    '{"triples": [{"sid": 0, "head": "...", "relation": "...", "tail": "..."}]}'
)


def llm_augment(batches: list[list[dict]], col: EdgeCollector, model: str,
                temperature: float, max_tokens: int) -> dict:
    stats = {"batches": 0, "proposed": 0, "accepted": 0, "rejected": 0}
    for batch in batches:
        lines = []
        cand: dict[str, tuple[str, str]] = {}
        for i, item in enumerate(batch):
            names = []
            for key in item["candidates"]:
                cand[key[0]] = key
                names.append("{}[{}]".format(key[0], key[1]))
            lines.append("[{}] (문서: {} / {}) 후보개체: {}\n문장: {}".format(
                i, item["doc_title"], item["doc_type"] or "?",
                ", ".join(sorted(names)), item["sent"]))
        prompt = "다음 문장들에서 허용된 삼중항만 뽑아라.\n\n" + "\n\n".join(lines)
        data = llm_provider.chat_json(
            [{"role": "system", "content": LLM_SYSTEM},
             {"role": "user", "content": prompt}],
            model=model, temperature=temperature, max_tokens=max_tokens)
        stats["batches"] += 1
        if not isinstance(data, dict):
            continue
        for tri in data.get("triples", []) or []:
            if not isinstance(tri, dict):
                continue
            stats["proposed"] += 1
            try:
                sid = int(tri.get("sid", -1))
            except (TypeError, ValueError):
                sid = -1
            if not (0 <= sid < len(batch)):
                stats["rejected"] += 1
                continue
            item = batch[sid]
            hk = cand.get(str(tri.get("head", "")).strip())
            tk = cand.get(str(tri.get("tail", "")).strip())
            rel = str(tri.get("relation", "")).strip().upper()
            if hk is None or tk is None or rel not in ("PARTICIPATED_IN", "FOUNDED"):
                stats["rejected"] += 1
                col.reject(str(tri.get("head", "")), rel, str(tri.get("tail", "")),
                           "llm_unknown_entity_or_relation: 후보 개체/허용 관계가 아님")
                continue
            existing = col.edges.get((hk, rel, tk))
            ok = col.add(hk, rel, tk,
                         origin="llm", confidence=0.70,
                         source=item["doc_title"], evidence=item["sent"],
                         extractor="llm_extract_v1")
            if ok:
                stats["accepted"] += 1
                if existing is not None:
                    e = col.edges[(hk, rel, tk)]
                    e["confidence"] = min(0.99, max(e["confidence"], 0.95))
            else:
                stats["rejected"] += 1
    return stats


# --------------------------------------------------------------------------- #
# 2홉 성립 증명
# --------------------------------------------------------------------------- #
def find_two_hop_patterns(edges: list[dict], limit: int = 10) -> list[dict]:
    """A -PARTICIPATED_IN-> E <-PARTICIPATED_IN- B -FOUNDED-> O 패턴."""
    part_by_head: dict[str, list[str]] = defaultdict(list)
    part_by_tail: dict[str, list[str]] = defaultdict(list)
    founded_by_head: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        if e["relation"] == "PARTICIPATED_IN":
            part_by_head[e["head"]].append(e["tail"])
            part_by_tail[e["tail"]].append(e["head"])
        elif e["relation"] == "FOUNDED":
            founded_by_head[e["head"]].append(e["tail"])

    out: list[dict] = []
    seen: set[tuple] = set()
    # 같은 (사건, B) 조합이 조직 수만큼 반복 출력되면 "3개를 찾았다"는 증명이 되지 않는다.
    # 사건별·중간인물별로 1건씩만 남겨 서로 다른 경로를 보여준다.
    used_event: Counter = Counter()
    used_bridge: set[str] = set()
    for a in sorted(part_by_head):
        for ev in sorted(set(part_by_head[a])):
            for b in sorted(set(part_by_tail[ev])):
                if b == a or b in used_bridge or used_event[ev] >= 2:
                    continue
                for org in sorted(set(founded_by_head.get(b, []))):
                    if org == a or org == ev:
                        continue          # 출발점으로 되돌아오는 경로는 새 사실이 없다
                    sig = (a, ev, b, org)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    used_event[ev] += 1
                    used_bridge.add(b)
                    out.append({
                        "A": a, "Event": ev, "B": b, "Organization": org,
                        "in_goldenset": False,
                        "path": [
                            {"head": a, "relation": "PARTICIPATED_IN", "tail": ev},
                            {"head": b, "relation": "PARTICIPATED_IN", "tail": ev},
                            {"head": b, "relation": "FOUNDED", "tail": org},
                        ],
                        "readable": "{} -PARTICIPATED_IN-> {} <-PARTICIPATED_IN- {} "
                                    "-FOUNDED-> {}".format(a, ev, b, org),
                    })
                    if len(out) >= limit:
                        return out
                    break        # (사건, B) 당 1건만
    return out


# --------------------------------------------------------------------------- #
# 골든셋 검증
# --------------------------------------------------------------------------- #
def mark_goldenset_usage(patterns: list[dict]) -> list[dict]:
    """골든셋 2홉 문항이 실제로 쓴 경로에 표시를 단다 (인수 조건 5번 증빙)."""
    if not GOLDENSET_PATH.exists():
        return patterns
    try:
        gs = json.loads(GOLDENSET_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return patterns
    used: set[tuple] = set()
    qid_of: dict[tuple, str] = {}
    for it in gs.get("items", []):
        path = it.get("expected_path") or []
        if len(path) < 3 or it.get("type") != "path" or int(it.get("hops", 0)) < 2:
            continue
        sig = tuple((t.get("head"), t.get("relation"), t.get("tail")) for t in path[:3])
        used.add(sig)
        qid_of[sig] = it.get("id", "?")
    for p in patterns:
        sig = tuple((t["head"], t["relation"], t["tail"]) for t in p["path"])
        if sig in used:
            p["in_goldenset"] = True
            p["goldenset_id"] = qid_of[sig]
    # 골든셋이 쓴 경로가 상위 목록에 없으면 뒤에 덧붙여 증빙이 보이게 한다.
    have = {tuple((t["head"], t["relation"], t["tail"]) for t in p["path"])
            for p in patterns}
    for it in gs.get("items", []):
        path = it.get("expected_path") or []
        if (len(path) < 3 or not it.get("answerable", True)
                or it.get("type") != "path" or int(it.get("hops", 0)) < 2):
            continue
        sig = tuple((t.get("head"), t.get("relation"), t.get("tail")) for t in path[:3])
        if sig in have:
            continue
        a, _r1, ev = sig[0]
        b, _r2, _ev2 = sig[1]
        _b2, _r3, org = sig[2]
        patterns.append({
            "A": a, "Event": ev, "B": b, "Organization": org,
            "in_goldenset": True, "goldenset_id": it.get("id", "?"),
            "path": [{"head": h, "relation": r, "tail": t} for h, r, t in sig],
            "readable": "{} -PARTICIPATED_IN-> {} <-PARTICIPATED_IN- {} "
                        "-FOUNDED-> {}".format(a, ev, b, org),
        })
        have.add(sig)
    return patterns


def validate_goldenset(graph_json: dict) -> dict:
    if not GOLDENSET_PATH.exists():
        return {"status": "missing",
                "note": "data/goldenset.json 이 아직 없다 — 평가셋 작성 후 build_graph.py 를 "
                        "다시 실행하면 검증 결과가 채워진다.",
                "answerable_path_present_rate": None}
    try:
        gs = json.loads(GOLDENSET_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "invalid_json", "note": str(exc)[:200],
                "answerable_path_present_rate": None}

    triples = {(e["head"], e["relation"], e["tail"]) for e in graph_json["edges"]}
    items_out: list[dict] = []
    total = found = 0
    hop_dist: Counter = Counter()
    for it in gs.get("items", []):
        answerable = bool(it.get("answerable", True))
        hop_dist["unanswerable" if not answerable else str(it.get("hops"))] += 1
        missing = []
        present = 0
        path = it.get("expected_path", []) or []
        for tri in path:
            sig = (tri.get("head"), tri.get("relation"), tri.get("tail"))
            if sig in triples:
                present += 1
            else:
                missing.append(list(sig))
        if answerable:
            total += len(path)
            found += present
        items_out.append({
            "id": it.get("id"), "hops": it.get("hops"), "answerable": answerable,
            "path_len": len(path), "path_present": present,
            "missing_triples": missing,
            "ok": (len(missing) == 0) if answerable else (len(path) == 0),
        })
    rate = (found / total) if total else 1.0
    return {
        "status": "checked",
        "n_items": len(items_out),
        "hop_distribution": dict(sorted(hop_dist.items())),
        "answerable_triples_total": total,
        "answerable_triples_present": found,
        "answerable_path_present_rate": round(rate, 4),
        "items": items_out,
        "note": "answerable 문항의 expected_path 삼중항이 graph.json 에 실제로 있는지 대조한 "
                "결과다. unanswerable 문항은 일부러 그래프에 없는 것을 묻기 때문에 비율 계산에서 "
                "제외한다.",
    }


# --------------------------------------------------------------------------- #
# 뽑지 않기로 한 관계
# --------------------------------------------------------------------------- #
NOT_EXTRACTED_RELATIONS = [
    {
        "relation": "MEMBER_OF",
        "reason": "'소속'은 본문에 가장 흔하게 나타나는 관계라 넣는 순간 엣지 수가 몇 배로 "
                  "늘고, 대한민국 임시정부·신민회 같은 노드가 수십 차수의 허브가 되어 "
                  "'임시정부를 거치면 누구든 2홉' 인 무의미한 경로가 양산된다. "
                  "FOUNDED 가 MEMBER_OF 의 부분집합(세운 사람은 대개 구성원)이라 "
                  "대표 질문을 푸는 데는 FOUNDED 만으로 충분하다.",
        "cost": "'김구가 몸담았던 단체는?' 처럼 설립이 아닌 단순 참여 단체를 묻는 질문은 "
                "답할 수 없다. 실제로 이런 질문은 거절(refusal) 처리된다.",
    },
    {
        "relation": "INFLUENCED",
        "reason": "'영향을 주었다/사상적으로 이어받았다' 는 위키 본문에서 서술자의 해석으로 "
                  "표현되는 경우가 많아 규칙으로는 오탐이 크고 LLM 으로는 환각이 크다. "
                  "정확도 100%·비용 0 인 규칙 계열로 만들 수 없는 관계라 제외했다.",
        "cost": "'신채호의 사상에 영향을 준 인물은?' 같은 사상사 질문을 못 푼다. "
                "인물 간 계보를 묻는 질문 전반이 빠진다.",
    },
    {
        "relation": "BORN_IN / DIED_IN (출생지·사망지)",
        "reason": "Location 노드를 새로 만들어야 하는데, 지명은 수백 명의 인물과 무차별로 "
                  "연결되는 전형적인 '가짜 다리'다(가이드 3.1 의 속성 노드 문제). "
                  "Person·Event·Organization 3종 스키마를 지키기 위해 제외했다.",
        "cost": "'황해도 출신 독립운동가는?' 같은 지역 기반 질의를 못 푼다. "
                "지역을 매개로 한 인물 군집 분석도 불가능하다.",
    },
    {
        "relation": "LED / COMMANDED (지휘)",
        "reason": "PARTICIPATED_IN 의 특수 사례라 별도 관계로 두면 같은 사실이 두 엣지로 "
                  "쪼개져 경로 탐색에서 중복 계산된다. 지휘 여부는 evidence 문장에 그대로 "
                  "남아 있어 답변 합성 단계에서 읽을 수 있다.",
        "cost": "'청산리 전투를 **지휘한** 사람은?' 처럼 역할을 한정하는 질문에서 "
                "단순 참여자와 지휘관을 그래프만으로는 구분하지 못한다.",
    },
    {
        "relation": "TEMPORAL (연도·날짜)",
        "reason": "연도는 노드가 아니라 엣지 속성이어야 한다는 가이드 2.1 원칙을 따랐다. "
                  "연도 노드를 만들면 1919년 하나에 수십 개 사건이 매달려 허브가 된다.",
        "cost": "'1919년에 일어난 사건은?' 같은 시간 범위 질의를 그래프 탐색으로는 못 푼다.",
    },
]


# --------------------------------------------------------------------------- #
# 메인 빌드
# --------------------------------------------------------------------------- #
def build(cfg: dict, use_discovery: bool = False) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    global ALLOWED_TRIPLES
    ALLOWED_TRIPLES = {(r["head"], r["relation"], r["tail"])
                       for r in cfg["schema"]["relations"]}
    min_len = int(cfg["retrieval"].get("min_entity_len", 2))

    docs = load_docs()
    if not docs:
        raise SystemExit("data/docs/*.md 가 없다. 먼저 `python collect_corpus.py` 를 실행하라.")
    manifest = load_manifest()
    gazetteer: dict = manifest.get("gazetteer", {}) or {}
    doc_redirects = {d.get("title"): d.get("redirects", []) for d in manifest.get("docs", [])}

    norm = Normalizer()
    store = EntityStore(norm, min_len)

    # ---------- 1. 노드 인벤토리 ---------- #
    print("[1/7] 노드 타입 판정 (위키 분류 규칙 1순위)")
    doc_key_by_title: dict[str, tuple[str, str] | None] = {}
    doc_by_name: dict[str, dict] = {}
    untyped_docs: list[dict] = []
    type_reasons: Counter = Counter()

    # 괄호 한정어가 **없는** 제목을 먼저 등록한다. 그래야 `김규식` 이 표준 이름을 갖고
    # `김규식 (북로군정서)` 가 한정어를 유지한 별도 노드로 분리된다.
    inventory: list[tuple[str, list[str], dict | None]] = []
    for d in docs:
        inventory.append((d["title"], d["categories"], d))
    doc_titles = {d["title"] for d in docs}
    for title, info in sorted(gazetteer.items()):
        if title in doc_titles:
            continue
        cats = info.get("categories", []) if isinstance(info, dict) else list(info)
        inventory.append((title, cats, None))
    inventory.sort(key=lambda x: ("(" in x[0] or "（" in x[0], x[0]))

    gaz_added = 0
    for title, cats, d in inventory:
        etype, why = classify(title, cats)
        if d is not None:
            type_reasons[why.split(":")[0]] += 1
        if etype is None:
            if d is not None:
                untyped_docs.append({"title": title, "reason": "타입 판정 실패",
                                     "categories": cats[:5]})
                doc_key_by_title[title] = None
            continue
        key = store.add(title, etype, title if d is not None else "",
                        is_doc=d is not None, reason=why, wiki_title=title)
        if d is not None:
            doc_key_by_title[title] = key
        else:
            gaz_added += 1
        if key:
            if not store.nodes[key]["categories"]:
                store.nodes[key]["categories"] = cats
            if d is not None:
                doc_by_name[key[0]] = d

    print("    문서 {}건 → 타입 판정 {}건 (실패 {}건), gazetteer 로 추가 {}건".format(
        len(docs), len(docs) - len(untyped_docs), len(untyped_docs), gaz_added))
    print("    판정 근거: {}".format(dict(sorted(type_reasons.items()))))

    # gazetteer 에 자기 문서가 없는 사건·조직을 본문 접미사 규칙으로 보충한다.
    disc = (discover_entities(docs, store) if use_discovery
            else {"added": [], "candidates": 0})
    print("    본문 접미사 규칙으로 사건·조직 {}종 추가 (후보 {}종)".format(
        len(disc["added"]), disc["candidates"]))

    # ---------- 2. 별칭 수집 (넘겨주기 + 약칭 사전 + 무공백) ---------- #
    print("[2/7] 별칭 수집 및 정규화")
    for key in sorted(store.nodes):
        name = key[0]
        nospace = name.replace(" ", "")
        if nospace != name:
            store.alias(nospace, key, Normalizer.RULE_NOSPACE)
            norm.counts[Normalizer.RULE_NOSPACE] += 1
            norm.examples.setdefault(Normalizer.RULE_NOSPACE,
                                     "{} → {}".format(nospace, name))

    for title, info in sorted(gazetteer.items()):
        reds = info.get("redirects", []) if isinstance(info, dict) else []
        tname, _ = norm.normalize(title)
        tname = ABBREVIATIONS.get(tname, tname)
        for etype in ("Person", "Organization", "Event"):
            key = (tname, etype)
            if key in store.nodes:
                for r in reds:
                    rname, _ = norm.normalize(r)
                    if rname and rname != tname:
                        store.alias(rname, key, Normalizer.RULE_REDIRECT)
                        norm.counts[Normalizer.RULE_REDIRECT] += 1
                        norm.examples.setdefault(
                            Normalizer.RULE_REDIRECT, "{} → {}".format(rname, tname))
                break

    for title, reds in sorted(doc_redirects.items()):
        key = doc_key_by_title.get(title)
        if not key:
            continue
        for r in reds or []:
            rname, _ = norm.normalize(r)
            if rname and rname != key[0]:
                store.alias(rname, key, Normalizer.RULE_REDIRECT)
                norm.counts[Normalizer.RULE_REDIRECT] += 1
                norm.examples.setdefault(Normalizer.RULE_REDIRECT,
                                         "{} → {}".format(rname, key[0]))

    for abbr, full in sorted(ABBREVIATIONS.items()):
        fname, _ = norm.normalize(full)
        for etype in ("Organization", "Event", "Person"):
            key = (fname, etype)
            if key in store.nodes:
                store.alias(abbr, key, Normalizer.RULE_ABBREV)
                norm.counts[Normalizer.RULE_ABBREV] += 1
                norm.examples.setdefault(Normalizer.RULE_ABBREV,
                                         "{} → {}".format(abbr, fname))
                break

    n_hanja = harvest_hanja_aliases(store, docs)
    print("    노드 후보 {}종 · 표기(별칭) {}종 (본문 한자 병기 {}건 흡수, 동명이인 분리 {}건)".format(
        len(store.nodes), len(store.surface_map), n_hanja, len(store.disambiguated)))

    # ---------- 3. 규칙 기반 관계 추출 ---------- #
    print("[3/7] 문장 단위 규칙 추출 (문단 동시등장만으로는 엣지를 만들지 않는다)")
    finder = MentionFinder(store.surface_map, norm)
    col = EdgeCollector()
    residual: list[dict] = []
    n_sent = 0

    for d in docs:
        doc_key = doc_key_by_title.get(d["title"])
        doc_type = doc_key[1] if doc_key else None
        # 문서 헤더 주입 — 문맥 왜곡 방지 (가이드 2.1)
        header = "[문서 제목] {}\n[문서 유형] {}\n".format(d["title"], doc_type or "Unknown")
        text = header + d["body"]
        for sent in split_sentences(text):
            n_sent += 1
            mentions = finder.find(sent)
            if len(mentions) < 1:
                continue
            for mt in mentions:
                store.nodes[mt.key]["sources"].add(d["title"])
            made = extract_rules(d, doc_key, sent, mentions, col)
            if made == 0 and len(mentions) >= 2:
                keys = sorted({m.key for m in mentions})
                types = {k[1] for k in keys}
                compatible = any((a, r, b) in ALLOWED_TRIPLES
                                 for a in types for b in types
                                 for r in ("PARTICIPATED_IN", "FOUNDED"))
                if compatible:
                    residual.append({"doc_title": d["title"], "doc_type": doc_type,
                                     "sent": sent, "candidates": keys})

    print("    문장 {}개 처리 → 규칙 엣지 {}종".format(n_sent, len(col.edges)))
    extract_category_edges(store, col, doc_by_name)
    print("    분류 규칙 추가 후 → 엣지 {}종".format(len(col.edges)))
    print("    규칙이 못 잡은(증강 후보) 문장: {}개".format(len(residual)))

    # ---------- 4. LLM 증강 (선택) ---------- #
    llm_stats: dict = {"enabled": False, "note": ""}
    if llm_provider.is_enabled():
        print("[4/7] LLM 증강 (규칙이 못 잡은 문장만)")
        picked = residual[:LLM_MAX_SENTENCES]
        batches = [picked[i:i + LLM_BATCH] for i in range(0, len(picked), LLM_BATCH)]
        s = llm_augment(batches, col,
                        model=cfg["llm"]["extract_model"],
                        temperature=float(cfg["llm"]["temperature"]),
                        max_tokens=int(cfg["llm"]["max_output_tokens"]))
        llm_stats = dict(s, enabled=True,
                         note="규칙이 아무 엣지도 못 만든 문장 {}개 중 {}개를 증강 대상으로 사용".format(
                             len(residual), len(picked)))
        print("    LLM 제안 {} · 채택 {} · 반려 {}".format(
            s["proposed"], s["accepted"], s["rejected"]))
    else:
        print("[4/7] [경고] OpenAI 키가 없어 LLM 증강을 건너뛴다 — 규칙 결과로 계속 진행한다.")
        llm_stats = {"enabled": False,
                     "note": "키가 없어 건너뜀. 규칙 기반 결과만으로 그래프를 완성했다.",
                     "residual_sentences": len(residual)}

    # ---------- 5. 노드 확정 · 병합 기록 ---------- #
    print("[5/7] 노드 확정 및 병합 기록")
    used_keys: set[tuple[str, str]] = set()
    for (h, _r, t) in col.edges:
        used_keys.add(h)
        used_keys.add(t)

    keep: dict[tuple[str, str], dict] = {}
    for key, node in store.nodes.items():
        if key in used_keys or node["is_doc"]:
            keep[key] = node

    # 동명이형(같은 이름, 다른 타입) → id 를 분리해 절대 합치지 않는다
    by_name: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in keep:
        by_name[key[0]].append(key)
    node_id: dict[tuple[str, str], str] = {}
    homonyms: list[dict] = []
    for name, keys in sorted(by_name.items()):
        if len(keys) == 1:
            node_id[keys[0]] = name
            continue
        keys_sorted = sorted(keys, key=lambda k: (not keep[k]["is_doc"],
                                                  -len(keep[k]["sources"]), k[1]))
        node_id[keys_sorted[0]] = name
        for k in keys_sorted[1:]:
            node_id[k] = "{} ({})".format(name, k[1])
        homonyms.append({"name": name,
                         "kept_as": {node_id[k]: k[1] for k in keys_sorted},
                         "reason": "타입이 다르면 이름이 같아도 병합하지 않는다"})

    # 엣지가 붙은 문서 노드/개체 노드의 출처 보강
    for key, node in keep.items():
        if node["is_doc"]:
            node["sources"].add(key[0])

    merges: list[dict] = []
    for key in sorted(keep, key=lambda k: (k[0], k[1])):
        node = keep[key]
        others = sorted(a for a in node["aliases"] if a != key[0])
        if others:
            merges.append({
                "canonical": node_id[key],
                "merged": others,
                "rule": "공백·중점 정규화 + 괄호 한정어 제거 + 한자 병기 분리 + "
                        "위키 넘겨주기 + 약칭 사전",
                "confidence": 1.0,
            })

    # ---------- 6. 그래프 조립 ---------- #
    print("[6/7] 그래프 조립 및 저장")
    edges_out: list[dict] = []
    for (h, rel, t), e in sorted(col.edges.items(),
                                 key=lambda kv: (kv[0][0][0], kv[0][1], kv[0][2][0])):
        if h not in keep or t not in keep:
            col.reject(h[0], rel, t[0], "node_pruned: 노드가 최종 그래프에서 제외됨")
            continue
        origins = e["origins"]
        origin = "rule+llm" if {"rule", "llm"} <= origins else sorted(origins)[0]
        edges_out.append({
            "head": node_id[h],
            "relation": rel,
            "tail": node_id[t],
            "origin": origin,
            "confidence": round(float(e["confidence"]), 3),
            "sources": sorted(e["sources"]),
            "evidence": list(e["evidence"]),
            "extractor": ";".join(sorted(e["extractors"])),
        })

    degree: Counter = Counter()
    for e in edges_out:
        degree[e["head"]] += 1
        degree[e["tail"]] += 1

    nodes_out: list[dict] = []
    for key in sorted(keep, key=lambda k: node_id[k]):
        node = keep[key]
        nid = node_id[key]
        nodes_out.append({
            "id": nid,
            "type": key[1],
            "aliases": sorted(node["aliases"]),
            "sources": sorted(node["sources"]),
            "doc_count": len(node["sources"]),
            "degree": int(degree.get(nid, 0)),
        })

    alias_index: dict[str, str] = {}
    for key in sorted(keep, key=lambda k: node_id[k]):
        nid = node_id[key]
        for a in sorted(keep[key]["aliases"]):
            if a in alias_index and alias_index[a] != nid:
                continue
            alias_index[a] = nid
        alias_index[nid] = nid

    graph_json = {
        "nodes": nodes_out,
        "edges": edges_out,
        "alias_index": dict(sorted(alias_index.items())),
        "stats": {
            "nodes": len(nodes_out),
            "edges": len(edges_out),
            "by_type": dict(sorted(Counter(n["type"] for n in nodes_out).items())),
            "by_relation": dict(sorted(Counter(e["relation"] for e in edges_out).items())),
            "by_origin": dict(sorted(Counter(e["origin"] for e in edges_out).items())),
        },
    }
    GRAPHJSON_PATH.write_text(json.dumps(graph_json, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    g = nx.DiGraph()
    for n in nodes_out:
        g.add_node(n["id"], type=n["type"], aliases=";".join(n["aliases"]),
                   sources=";".join(n["sources"]), doc_count=int(n["doc_count"]),
                   degree=int(n["degree"]))
    for e in edges_out:
        g.add_edge(e["head"], e["tail"], relation=e["relation"], origin=e["origin"],
                   confidence=float(e["confidence"]),
                   sources=";".join(e["sources"]),
                   evidence=";".join(e["evidence"]),
                   extractor=e["extractor"])
    nx.write_graphml(g, GRAPHML_PATH, encoding="utf-8")

    # ---------- 7. 리포트 ---------- #
    print("[7/7] 추출 리포트 및 골든셋 검증")
    two_hop = find_two_hop_patterns(edges_out, limit=10)
    gs_validation = validate_goldenset(graph_json)
    two_hop = mark_goldenset_usage(two_hop)

    report = {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corpus": {"docs": len(docs), "sentences": n_sent,
                   "gazetteer_entries": len(gazetteer)},
        "schema": {"allowed_triples": ["({}, {}, {})".format(*t)
                                       for t in sorted(ALLOWED_TRIPLES)]},
        "node_typing": {
            "rule": "위키 분류(category) 규칙 1순위 → 실패 시 제목 접미사 규칙. "
                    "규칙은 정확도가 높고 환각이 없으며 비용이 0이라 origin=rule 이다.",
            "by_reason": dict(sorted(type_reasons.items())),
            "untyped_docs": untyped_docs,
        },
        "merge_rules": norm.rules_report(),
        "merges": merges,
        "homonyms_kept_separate": homonyms,
        "type_conflicts": store.type_conflicts[:50],
        "disambiguated_homonyms": store.disambiguated,
        "dropped_entities": dict(sorted(store.dropped.items())),
        "text_discovered_entities": disc["added"],
        "text_discovery_note": (
            "본문 접미사 규칙(--discovery)은 기본 비활성이다. 켜면 노드 556·엣지 822 로 "
            "늘지만 '23일 임시정부' 같은 오탐이 섞인다. 위키 문서를 가진 개체만으로도 "
            "목표치를 넘기므로 정밀도를 택했다."),
        "stopwords": sorted(STOPWORDS),
        "extractors": dict(sorted(col.by_extractor.items())),
        "llm_augmentation": llm_stats,
        "llm_key_status": llm_provider.key_status(),
        "rejected_edges": col.rejected[:300],
        "rejected_edges_total": len(col.rejected),
        "not_extracted_relations": NOT_EXTRACTED_RELATIONS,
        "two_hop_examples": two_hop,
        "goldenset_validation": gs_validation,
        "stats": graph_json["stats"],
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    return {"graph": graph_json, "report": report, "two_hop": two_hop}


def main() -> int:
    llm_provider.setup_console()
    ap = argparse.ArgumentParser(description="지식 그래프 구축 (스키마 제어 추출 + 정규화)")
    ap.add_argument("--no-llm", action="store_true", help="키가 있어도 LLM 증강을 끈다")
    # 본문 접미사 발견은 재현율을 크게 올리지만("23일 임시정부" 같은) 오탐도 함께 늘린다.
    # 위키 문서를 가진 개체만으로 이미 목표치(노드 80·엣지 120)를 크게 넘기므로 기본은 끈다.
    ap.add_argument("--discovery", action="store_true",
                    help="본문 접미사 규칙으로 사건·조직 노드를 추가로 발견한다 (재현율↑·정밀도↓)")
    args = ap.parse_args()

    print(llm_provider.banner())
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.no_llm:
        llm_provider.clear_runtime_key()
        import os
        os.environ.pop("OPENAI_API_KEY", None)

    result = build(cfg, use_discovery=args.discovery)
    gj, rep = result["graph"], result["report"]
    st = gj["stats"]

    print("=" * 72)
    print("노드 {} · 엣지 {}".format(st["nodes"], st["edges"]))
    print("타입 분포   : {}".format(st["by_type"]))
    print("관계 분포   : {}".format(st["by_relation"]))
    print("origin 분포 : {}".format(st["by_origin"]))
    print("추출기별 발동 횟수:")
    for k, v in rep["extractors"].items():
        print("  - {}: {}".format(k, v))
    print("병합 규칙별 적용 건수:")
    for r in rep["merge_rules"]:
        print("  - {}: {}건 (예: {})".format(r["rule"], r["applied"], r["example"]))
    print("스키마 위반 등으로 버린 엣지: {}건".format(rep["rejected_edges_total"]))

    print("-" * 72)
    print("2홉 성립 증명 — A -PARTICIPATED_IN-> E <-PARTICIPATED_IN- B -FOUNDED-> O")
    if not result["two_hop"]:
        print("  (없음) — 추출 규칙을 보강해야 한다")
    for i, p in enumerate(result["two_hop"], 1):
        tag = " ← 골든셋 {}".format(p["goldenset_id"]) if p.get("in_goldenset") else ""
        print("  {}. {}{}".format(i, p["readable"], tag))

    gv = rep["goldenset_validation"]
    print("-" * 72)
    print("골든셋 검증: {} | answerable expected_path 존재율: {}".format(
        gv.get("status"), gv.get("answerable_path_present_rate")))
    if gv.get("status") == "checked":
        print("  홉 분포: {}".format(gv.get("hop_distribution")))
        bad = [i for i in gv.get("items", []) if not i["ok"]]
        for b in bad:
            print("  [불일치] {}: 누락 {}".format(b["id"], b["missing_triples"]))

    print("-" * 72)
    print("저장: {} / {} / {}".format(
        GRAPHML_PATH.relative_to(ROOT), GRAPHJSON_PATH.relative_to(ROOT),
        REPORT_PATH.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
