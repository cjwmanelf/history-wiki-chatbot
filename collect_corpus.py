#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""collect_corpus.py — 한국어 위키백과 코퍼스 수집 (주제: 한국 독립운동사, 1890~1945).

수집 전략 (과제 5단계-1 "자료 수집과 평가셋 생성")
--------------------------------------------------
1. **시드**: 인물·사건·조직을 섞은 `SEEDS` 16건. MediaWiki API 로 실재 여부를 확인하고,
   없는 제목은 `rejected` 에 사유와 함께 남긴다.
2. **2홉 후보**: 각 시드 본문의 내부 링크(`prop=links&plnamespace=0`) 를 모아,
   **여러 시드가 함께 가리키는 문서**(기본 2개 이상)를 우선 후보로 삼는다.
   → 한 시드에만 걸린 링크는 주제와의 관련이 약하므로 탈락시킨다.
3. **분류 공유 필터**: 후보 중 **시드와 위키 분류(category)를 하나 이상 공유하는 것만** 남긴다.
   단, 공유로 인정하는 분류는 **시드 2개 이상이 함께 가진 주제 분류**뿐이다.
   '20세기 한국 사람' 같은 구조적 분류나 시드 한 개에만 붙은 분류(예: '일제강점기의 소설가')를
   인정하면 조선시대 문인·현대 대통령까지 딸려 들어온다(1차 수집에서 실제로 발생).
   연도·목록·틀·분류·동음이의 문서는 제목 규칙으로 먼저 걷어낸다.
4. **본문**: `prop=extracts&explaintext=1`. 800자 미만 토막글은 저장하지 않는다.
5. 목표 수집량(기본 160건)을 못 채우면 `EXTRA_SEEDS` 로 시드를 넓혀 재시도하고,
   그 사실을 manifest 의 `seed_expansion` 에 남긴다.

채택/탈락 기준을 **데이터로** 남기는 것이 채점 항목이므로 모든 탈락은
`data/manifest.json` 의 `rejected` 에 {title, reason} 으로 기록된다.

재실행 안전: 이미 받은 문서는 건너뛴다. `--force` 로 전체 재수집.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

import llm_provider

random.seed(42)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
MANIFEST_PATH = DATA_DIR / "manifest.json"
# 응답 캐시는 저장소가 아니라 OS 임시 디렉터리에 둔다 (지워도 무해한 순수 캐시).
CACHE_PATH = Path(tempfile.gettempdir()) / "history_wiki_chatbot" / "wiki_cache.json"

API = "https://ko.wikipedia.org/w/api.php"
# User-Agent 는 반드시 아스키만 (한글을 넣으면 헤더 인코딩 오류가 난다).
# Wikimedia User-Agent 정책: "제품명/버전 (연락 가능한 URL 또는 연락처)" 형식이어야 한다.
# 포괄적인 UA("MyBot/1.0 (educational project)") 는 정책 미달로 강하게 스로틀(429)된다.
# 실측: 포괄적 UA → 429, 아래 형식 → 200. **아스키만** 쓴다(한글은 헤더 인코딩 오류).
USER_AGENT = ("HistoryWikiChatbotCoursework/0.1 "
              "(https://ko.wikipedia.org/wiki/User:Coursework; educational use)")
SLEEP = 1.2              # 정책 준수 UA 기준 1.0~1.5초면 429 없이 안정적이다
MAX_ATTEMPTS = 6         # 429/5xx 는 Retry-After 를 존중하며 재시도

MIN_CHARS = 800          # 토막글 기준
TARGET_DOCS = 160        # 목표 수집량 (주제 필터 통과분을 넉넉히 담는다)
MIN_DOCS = 50            # 과제 하한
MIN_SEED_SUPPORT = 2     # 2홉 후보로 인정할 최소 "시드 링크 수"
MIN_CATEGORY_SEEDS = 2   # 공유 분류로 인정할 최소 "그 분류를 가진 시드 수"

# --------------------------------------------------------------------------- #
# 시드 — 인물 8 / 사건 3 / 조직 5
# --------------------------------------------------------------------------- #
SEEDS = [
    # 인물
    "안중근", "김구", "안창호", "윤봉길", "이회영", "신채호", "여운형", "김좌진",
    # 사건
    "3·1 운동", "청산리 전투", "봉오동 전투",
    # 조직
    "대한민국 임시정부", "신민회", "의열단", "한인애국단", "흥사단",
]

# 목표 수집량 미달 시에만 투입하는 예비 시드 (manifest.seed_expansion 에 기록된다).
EXTRA_SEEDS = [
    "홍범도", "이동휘", "김규식", "이봉창", "유관순", "이상설", "지청천", "이시영",
    "신간회", "한국광복군", "대한광복회", "북로군정서", "서로군정서", "조선어학회",
    "6·10 만세운동", "광주학생항일운동", "105인 사건", "대한인국민회",
]

# --------------------------------------------------------------------------- #
# 제목 수준 배제 규칙 (연도·목록·틀·분류·동음이의)
# --------------------------------------------------------------------------- #
_TITLE_BLOCK_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^\d+년(\s|$)"), "year_page: 연도 문서"),
    (re.compile(r"^\d+년대"), "year_page: 연대 문서"),
    (re.compile(r"^\d+월\s*\d+일$"), "date_page: 날짜 문서"),
    (re.compile(r"^(기원전\s*)?\d+$"), "year_page: 숫자 제목"),
    (re.compile(r"목록"), "list_page: 목록 문서"),
    (re.compile(r"일람"), "list_page: 일람 문서"),
    (re.compile(r"^(틀|분류|위키프로젝트|파일|포털):"), "namespace_page: 틀/분류/포털 문서"),
    (re.compile(r"\(동음이의\)"), "disambiguation: 동음이의 문서"),
    (re.compile(r"^(대한민국|조선민주주의인민공화국|일본|중국|미국|러시아|소련|영국)$"),
     "too_generic: 국가명 일반 문서"),
]

# 시드와의 "공유 분류" 계산에서 제외할 관리/형식 분류 조각.
_CATEGORY_NOISE = (
    "위키", "문서", "출처", "인용", "오류", "정리", "넘겨주기", "링크", "틀",
    "ISBN", "모호", "동음이의", "생몰년", "날짜", "기여", "번역",
)

# 주제와 무관한 **구조적 분류**. 이런 분류를 공유했다고 채택하면
# "20세기 한국 사람" 하나로 조선시대 문인·현대 대통령까지 딸려 들어온다.
# (1차 수집에서 실제로 김만중·이이·정철·전두환이 들어왔던 원인이라 규칙으로 못박았다.)
_GENERIC_CATEGORY_PATTERNS = [
    re.compile(r"^\d+세기"),
    re.compile(r"^\d+년\s*(출생|사망|태어남|죽음)$"),
    re.compile(r"^(한국|대한민국|대한제국|조선|일본|중국) 사람$"),
    re.compile(r"세기\s.*사람$"),
    re.compile(r"\s씨$"),                      # 본관·성씨 (고령 신씨, 경주 이씨)
    re.compile(r"동문$"),
    re.compile(r"출신$"),
    re.compile(r"(기독교|장로교|천주교|가톨릭|개신교|불교|무신론|유교)"),
    re.compile(r"(피해자|사망자|옥사한|사형된|암살된|자살한|화기에 죽은|처형된)"),
    re.compile(r"^(소설가|시인|작가|화가|음악가|배우|언론인|의사|교육인|번역가)"),
    re.compile(r"의 (소설가|시인|작가|화가|음악가|배우|서예가|국악인|축구|야구)"),
    re.compile(r"(회장|수상자|올림픽|스포츠|드라마|영화|음반)"),
]


# --------------------------------------------------------------------------- #
# 주제 범위 필터 (루브릭: "주제와 소스를 채택하거나 탈락시킨 기준을 명확히 남깁니다")
# --------------------------------------------------------------------------- #
# 채택 기준 — 아래 셋을 모두 만족해야 코퍼스에 넣는다.
#   (A) 시대: 1890~1945 와 생애/발생 시기가 겹친다.
#       인물은 '○○년 출생/사망' 분류에서 연도를 읽어 (출생 ≤ 1945) ∧ (사망 ≥ 1890) 을 본다.
#   (B) 주제: 독립운동·계몽운동·항일무장투쟁에 **직접 관여**했음을 보이는 핵심 분류가 있다.
#   (C) 정체성: 문학·예술·체육 등 운동과 무관한 활동 분류가 핵심 분류보다 많으면 제외한다.
#       ('한국의 독립운동가' 분류를 직접 가진 경우는 예외로 남긴다)
# 탈락 사유는 전부 manifest.rejected 에 데이터로 남는다.
PERIOD_START, PERIOD_END = 1890, 1945

_CORE_TOPIC_RE = re.compile(
    r"독립운동가|한국의 독립운동|독립운동단체|독립유공자|건국훈장|건국포장|"
    r"임시정부|광복군|독립군|의병|의열|애국단|애국지사|순국|민족대표|무오독립선언|"
    r"군정서|광복회|신민회|신간회|흥사단|국채보상|조선어학회|한국독립당|대한인국민회|"
    r"만세운동|항일|반일 감정|의열단|한인애국단|대한민국 임시정부"
)
_ARTS_IDENTITY_RE = re.compile(
    r"소설가|시인|문학|화가|서예가|음악가|국악|판소리|배우|영화|가수|무용|체육|축구|야구|"
    r"만화가|사진가|건축가"
)
# 아래 분류를 가지면 (C) 정체성 우위 규칙에서 면제한다.
# 건국훈장·건국포장은 독립운동 공적에만 주는 국가 서훈이라 객관적이고 문턱이 높다.
# (김창숙처럼 유학자·시인 분류가 더 많지만 독립운동이 본업인 인물을 살리기 위한 장치)
_STRONG_ACTIVIST_RE = re.compile(
    r"독립운동가|의병장|애국지사|순국선열|건국훈장|건국포장|민족대표|의열단|광복군")
# 외국 국적 인물: 한국 독립운동에 **직접 참여**한 근거(임시정부 소속·한국의 독립운동가)가
# 없으면 제외한다. 장제스처럼 서훈만 받은 외국 국가원수는 그래프에서 허브 노이즈만 만든다.
# "미국의 외교관" 처럼 **국적 정체성**만 잡는다.
# "미국에 거주한 한국인" 같은 체류 분류까지 잡으면 김순애·임병직 같은 한국인이 잘못 걸린다.
_FOREIGN_CAT_RE = re.compile(
    r"^(중국|일본|미국|대만|러시아|영국|프랑스|독일|중화민국|소련|캐나다)의 ")
_KOREAN_MARKER_RE = re.compile(r"한국인|한인|조선인|한국 사람")
# 해방 이후 대한민국 고위 공직자: 건국훈장 대한민국장은 대통령에게 직위상 수여되기도 해서
# 그것만으로는 독립운동 이력이 되지 못한다. '한국의 독립운동가' 분류가 함께 있어야 남긴다.
# (실제로 최규하가 '건국훈장 대한민국장 수훈자' 하나로 통과했다.)
# '대한민국 제N공화국' 은 단지 그 시기에 활동했다는 뜻이라 쓰지 않는다(정인보·장건상처럼
# 독립운동가이면서 해방 후에도 활동한 사람이 통째로 걸린다). **국가원수·정부수반** 직위만 본다.
_POST_LIBERATION_RE = re.compile(
    r"^대한민국의 (대통령|국무총리|대통령 권한대행)$|^대한민국 제\d+대 대통령")
_ACTIVIST_CAT_RE = re.compile(
    r"한국의 독립운동가|독립유공자|대한민국 임시정부|의열|광복군|민족대표|무오독립선언|"
    r"의병장|애국지사|순국선열")
_FOREIGN_EXEMPT_RE = re.compile(r"대한민국 임시정부 사람|한국의 독립운동가|한국광복군")
_BORN_RE = re.compile(r"^(\d{3,4})년\s*(?:출생|태어남)$")
_DIED_RE = re.compile(r"^(\d{3,4})년\s*(?:사망|죽음)$")
MIN_ARTS_FOR_PRUNE = 3   # 예술·문학 분류가 이만큼은 돼야 '정체성 우위'로 본다


def topic_scope(title: str, categories: list[str]) -> tuple[bool, str]:
    """(채택 여부, 사유). 사유는 채택일 때도 남겨 채점자가 기준을 확인할 수 있게 한다."""
    born = died = None
    for c in categories:
        m = _BORN_RE.match(c)
        if m:
            born = int(m.group(1))
        m = _DIED_RE.match(c)
        if m:
            died = int(m.group(1))

    # (A) 시대 창
    if died is not None and died < PERIOD_START:
        return False, ("out_of_period_before: {}년 사망 — 대상 시기 {}~{} 이전 인물".format(
            died, PERIOD_START, PERIOD_END))
    if born is not None and born > PERIOD_END:
        return False, ("out_of_period_after: {}년 출생 — 대상 시기 {}~{} 이후 인물".format(
            born, PERIOD_START, PERIOD_END))
    if born is not None and born < 1820:
        return False, ("out_of_period_before: {}년 출생 — 조선 전·중기 인물".format(born))

    core = [c for c in categories if _CORE_TOPIC_RE.search(c)]
    arts = [c for c in categories if _ARTS_IDENTITY_RE.search(c)]

    # (B) 주제 적합성
    if not core:
        return False, ("not_independence_topic: 독립운동·항일 계열 핵심 분류가 없음 "
                       "(분류 예: {})".format(", ".join(categories[:4]) or "없음"))

    exempt = any(_STRONG_ACTIVIST_RE.search(c) for c in categories)

    # (C) 정체성 우위 — 예술·문학 분류가 3개 이상이면서 핵심 분류보다 많을 때만 적용
    if (not exempt and len(arts) >= MIN_ARTS_FOR_PRUNE and len(arts) > len(core)):
        return False, ("arts_identity_dominant: 문학·예술 분류 {}개 > 독립운동 분류 {}개 "
                       "이고 독립운동가·건국훈장 계열 분류가 없음 "
                       "(운동 관여가 주된 이력이 아님)".format(len(arts), len(core)))

    # (D-1) 해방 이후 대한민국 고위 공직자
    post = [c for c in categories if _POST_LIBERATION_RE.match(c)]
    if post and not any(_ACTIVIST_CAT_RE.search(c) for c in categories):
        return False, ("post_liberation_office_holder: 해방 이후 대한민국 고위 공직 분류 "
                       "{}개('{}')가 있고 '한국의 독립운동가' 분류가 없음 — "
                       "건국훈장 수훈만으로는 독립운동 이력이 되지 않는다".format(
                           len(post), post[0]))

    # (D-2) 외국 국적 인물: 한국 독립운동에 직접 참여한 근거가 없으면 제외
    foreign = [c for c in categories
               if _FOREIGN_CAT_RE.match(c) and not _KOREAN_MARKER_RE.search(c)]
    if len(foreign) >= 3 and not any(_FOREIGN_EXEMPT_RE.search(c) for c in categories):
        return False, ("foreign_non_participant: 외국 국적 분류 {}개 "
                       "('대한민국 임시정부 사람'·'한국의 독립운동가' 없음) — "
                       "서훈만으로는 독립운동 직접 참여 근거가 되지 않음".format(len(foreign)))

    return True, "in_scope: 시대({}~{}) 적합 · 핵심 분류 {}개 (예: {})".format(
        PERIOD_START, PERIOD_END, len(core), core[0])


def _title_block_reason(title: str) -> str | None:
    for pat, reason in _TITLE_BLOCK_PATTERNS:
        if pat.search(title):
            return reason
    return None


def _is_noise_category(cat: str) -> bool:
    body = cat.split(":", 1)[-1]
    if any(tok in body for tok in _CATEGORY_NOISE):
        return True
    return any(p.search(body) for p in _GENERIC_CATEGORY_PATTERNS)


def safe_filename(title: str) -> str:
    """문서 제목 → 파일명. 공백은 `_`, 윈도우에서 못 쓰는 문자는 `_` 로 치환한다."""
    name = title.replace(" ", "_")
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    return name + ".md"


# --------------------------------------------------------------------------- #
# MediaWiki API
# --------------------------------------------------------------------------- #
class WikiAPI:
    """ko.wikipedia MediaWiki API 클라이언트.

    ko.wikipedia 는 익명 IP 당 **약 10요청 / 48초** 로 제한한다(429 + `Retry-After`).
    그래서 두 가지를 둔다.
    1. **적응형 페이싱**: 429 를 맞으면 호출 간격을 늘리고, 연속 성공하면 조금씩 줄인다.
    2. **영속 캐시**: 성공 응답을 임시 디렉터리에 저장해 재실행 시 재요청하지 않는다.
       (저장소를 더럽히지 않도록 OS 임시 디렉터리를 쓴다 — 지워도 무해하다.)
    """

    def __init__(self, use_cache: bool = True) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT,
                                     "Accept-Encoding": "gzip"})
        self.calls = 0
        self.failures = 0
        self.cache_hits = 0
        self.sleep = SLEEP
        self._streak = 0
        self.use_cache = use_cache
        self._cache: dict[str, dict] = {}
        self._dirty = 0
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if use_cache and CACHE_PATH.exists():
            try:
                self._cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._cache = {}

    # -- 캐시 ---------------------------------------------------------------- #
    @staticmethod
    def _ckey(params: dict) -> str:
        blob = json.dumps(params, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def flush_cache(self, force: bool = False) -> None:
        if not self.use_cache or (self._dirty < 20 and not force):
            return
        try:
            tmp = CACHE_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._cache, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(CACHE_PATH)
            self._dirty = 0
        except OSError:
            pass

    def _get(self, params: dict) -> dict:
        """429/5xx 는 `Retry-After` 를 존중하며 재시도한다. 실패하면 빈 dict."""
        ck = self._ckey(params)
        if self.use_cache and ck in self._cache:
            self.cache_hits += 1
            return self._cache[ck]

        self.calls += 1
        wait = 1.0
        for attempt in range(MAX_ATTEMPTS):
            try:
                resp = self.session.get(API, params=params, timeout=60)
                if resp.status_code in (429, 500, 502, 503, 504):
                    retry_after = resp.headers.get("Retry-After", "")
                    try:
                        wait = max(wait, float(retry_after) + 1.0)
                    except ValueError:
                        pass
                    raise requests.HTTPError("HTTP {}".format(resp.status_code))
                resp.raise_for_status()
                data = resp.json()
                self._streak += 1
                if self._streak >= 15 and self.sleep > SLEEP:
                    self.sleep = max(SLEEP, self.sleep * 0.9)
                    self._streak = 0
                if self.use_cache:
                    self._cache[ck] = data
                    self._dirty += 1
                    self.flush_cache()
                time.sleep(self.sleep)
                return data
            except (requests.RequestException, ValueError) as exc:
                self._streak = 0
                self.sleep = min(self.sleep * 1.5, 8.0)
                if attempt == MAX_ATTEMPTS - 1:
                    self.failures += 1
                    print("    [API] 포기 ({}) — 이 요청은 실패로 기록한다".format(
                        type(exc).__name__))
                    break
                print("    [API] 재시도 {}/{} ({}) — {:.0f}s 대기 "
                      "(호출 간격 {:.1f}s 로 조정)".format(
                          attempt + 1, MAX_ATTEMPTS, type(exc).__name__,
                          wait, self.sleep))
                time.sleep(wait)
                wait = min(wait * 1.6, 60.0)
        return {}

    def query(self, params: dict) -> tuple[dict[str, dict], dict[str, str]]:
        """`continue` 를 따라가며 페이지를 병합해 돌려준다.

        반환: (title -> page dict, 원제목 -> 정규화/리다이렉트된 제목)
        """
        base = dict(params)
        base.update({"action": "query", "format": "json", "formatversion": "2"})
        pages: dict[str, dict] = {}
        alias: dict[str, str] = {}
        cont: dict[str, str] = {}
        for _ in range(60):  # 안전 상한
            p = dict(base)
            p.update(cont)
            data = self._get(p)
            if not data:
                break
            q = data.get("query", {})
            for kind in ("normalized", "redirects"):
                for m in q.get(kind, []):
                    alias[m.get("from", "")] = m.get("to", "")
            for page in q.get("pages", []):
                title = page.get("title", "")
                if not title:
                    continue
                cur = pages.setdefault(title, {})
                for k, v in page.items():
                    if isinstance(v, list):
                        cur.setdefault(k, []).extend(v)
                    else:
                        cur[k] = v
            if "continue" in data:
                cont = {k: str(v) for k, v in data["continue"].items()}
            else:
                break
        return pages, alias

    # -- 개별 목적별 래퍼 ---------------------------------------------------- #
    def exists(self, titles: list[str]) -> tuple[dict[str, bool], dict[str, str]]:
        found: dict[str, bool] = {}
        alias: dict[str, str] = {}
        for chunk in _chunks(titles, 20):
            pages, al = self.query({"prop": "info", "titles": "|".join(chunk),
                                    "redirects": "1"})
            alias.update(al)
            for title, page in pages.items():
                found[title] = not page.get("missing", False)
        return found, alias

    def links(self, title: str) -> list[str]:
        pages, _ = self.query({"prop": "links", "plnamespace": "0", "pllimit": "max",
                               "titles": title, "redirects": "1"})
        out: list[str] = []
        for page in pages.values():
            out.extend(l.get("title", "") for l in page.get("links", []))
        return sorted({t for t in out if t})

    def categories(self, titles: list[str]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for chunk in _chunks(titles, 20):
            pages, _ = self.query({"prop": "categories", "cllimit": "max",
                                   "clshow": "!hidden", "titles": "|".join(chunk),
                                   "redirects": "1"})
            for title, page in pages.items():
                if page.get("missing"):
                    continue
                cats = [c.get("title", "").split(":", 1)[-1]
                        for c in page.get("categories", [])]
                result[title] = sorted({c for c in cats if c})
        return result

    def redirects_to(self, titles: list[str]) -> dict[str, list[str]]:
        """해당 문서를 가리키는 넘겨주기(=별칭) 제목들."""
        result: dict[str, list[str]] = {}
        for chunk in _chunks(titles, 20):
            pages, _ = self.query({"prop": "redirects", "rdlimit": "max",
                                   "rdnamespace": "0", "titles": "|".join(chunk)})
            for title, page in pages.items():
                if page.get("missing"):
                    continue
                names = [r.get("title", "") for r in page.get("redirects", [])]
                result[title] = sorted({n for n in names if n})
        return result

    def extract(self, title: str) -> tuple[str, str]:
        """(정규화된 제목, 본문 평문). 실패 시 ("", "")."""
        pages, _ = self.query({"prop": "extracts", "explaintext": "1",
                               "exsectionformat": "plain", "titles": title,
                               "redirects": "1"})
        for real_title, page in pages.items():
            if page.get("missing"):
                continue
            return real_title, (page.get("extract") or "").strip()
        return "", ""


def _chunks(items: list, size: int):
    items = list(items)
    for i in range(0, len(items), size):
        yield items[i:i + size]


# --------------------------------------------------------------------------- #
# 문서 파일 입출력 (CONTRACT.md §3.1)
# --------------------------------------------------------------------------- #
def write_doc(title: str, categories: list[str], body: str) -> Path:
    path = DOCS_DIR / safe_filename(title)
    text = "# {}\n\n분류: {}\n\n{}\n".format(title, ", ".join(categories), body.strip())
    path.write_text(text, encoding="utf-8")
    return path


def read_doc(path: Path) -> tuple[str, list[str], str]:
    raw = path.read_text(encoding="utf-8")
    title, cats, body = "", [], raw
    m = re.match(r"^#\s*(.+?)\n\s*\n분류:\s*(.*?)\n\s*\n(.*)$", raw, flags=re.S)
    if m:
        title = m.group(1).strip()
        cats = [c.strip() for c in m.group(2).split(",") if c.strip()]
        body = m.group(3)
    return title, cats, body


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


# --------------------------------------------------------------------------- #
# 수집 본체
# --------------------------------------------------------------------------- #
def collect(force: bool = False, target: int = TARGET_DOCS,
            no_cache: bool = False) -> dict:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    api = WikiAPI(use_cache=not no_cache)

    rejected: list[dict] = []
    seen_reject: set[str] = set()

    def reject(title: str, reason: str) -> None:
        if title in seen_reject:
            return
        seen_reject.add(title)
        rejected.append({"title": title, "reason": reason})

    # ---------- 1. 시드 검증 ---------- #
    print("=" * 72)
    print("[1/6] 시드 문서 실재 여부 확인")
    seed_titles = list(SEEDS)
    found, alias = api.exists(seed_titles)
    resolved_seeds: list[str] = []
    for s in seed_titles:
        real = alias.get(s, s)
        if found.get(real, False):
            if real not in resolved_seeds:
                resolved_seeds.append(real)
            if real != s:
                print("    · '{}' → 넘겨주기 '{}'".format(s, real))
        else:
            reject(s, "seed_missing: 위키백과에 해당 제목의 문서가 없음")
            print("    x '{}' 없음 → rejected".format(s))
    print("    시드 {}건 중 {}건 유효".format(len(seed_titles), len(resolved_seeds)))

    seed_expansion: dict = {"used": False, "reason": "", "added": []}

    def run_pipeline(seeds: list[str]) -> tuple[list[dict], dict]:
        """seeds 로부터 (저장된 문서 목록, 메타) 를 만든다."""
        # ---------- 2. 2홉 후보 ---------- #
        print("=" * 72)
        print("[2/6] 시드 본문 링크 수집 (시드 {}건)".format(len(seeds)))
        link_support: dict[str, set[str]] = defaultdict(set)
        for i, s in enumerate(seeds, 1):
            ls = api.links(s)
            for t in ls:
                link_support[t].add(s)
            print("    ({}/{}) {}: 링크 {}건".format(i, len(seeds), s, len(ls)))
        for s in seeds:
            link_support.pop(s, None)
        print("    링크 대상 총 {}종".format(len(link_support)))

        hop2 = sorted(
            (t for t, sup in link_support.items() if len(sup) >= MIN_SEED_SUPPORT),
            key=lambda t: (-len(link_support[t]), t),
        )
        for t, sup in sorted(link_support.items()):
            if len(sup) < MIN_SEED_SUPPORT:
                reject(t, "low_link_support: 시드 {}개만 링크 "
                          "(2홉 후보 임계 {} 미달)".format(len(sup), MIN_SEED_SUPPORT))
        print("    2홉 후보(시드 {}개 이상이 링크): {}건".format(MIN_SEED_SUPPORT, len(hop2)))

        # 제목 규칙 배제
        hop2_clean: list[str] = []
        for t in hop2:
            reason = _title_block_reason(t)
            if reason:
                reject(t, reason)
            else:
                hop2_clean.append(t)
        print("    제목 규칙(연도/목록/틀/분류/동음이의) 통과: {}건".format(len(hop2_clean)))

        # ---------- 3. 분류 공유 필터 ---------- #
        print("=" * 72)
        print("[3/6] 분류(category) 조회 및 시드 분류 공유 필터")
        seed_cats = api.categories(seeds)
        # 분류별로 그 분류를 가진 시드 수를 센다.
        cat_seed_count: dict[str, int] = defaultdict(int)
        for cats in seed_cats.values():
            for c in cats:
                if not _is_noise_category(c):
                    cat_seed_count[c] += 1
        # **시드 2개 이상이 공유하는 분류만** 주제 분류로 인정한다.
        # 시드 1개에만 붙은 분류(예: 신채호의 '일제강점기의 소설가')를 인정하면
        # 주제와 무관한 문인·화가가 통째로 딸려 들어온다.
        seed_cat_pool = {c for c, n in cat_seed_count.items()
                         if n >= MIN_CATEGORY_SEEDS}
        print("    시드 분류 {}종 → 시드 {}개 이상이 공유하는 주제 분류 {}종".format(
            len(cat_seed_count), MIN_CATEGORY_SEEDS, len(seed_cat_pool)))
        for c in sorted(seed_cat_pool, key=lambda x: (-cat_seed_count[x], x))[:15]:
            print("      · {} (시드 {}건)".format(c, cat_seed_count[c]))

        cand_cats = api.categories(hop2_clean)
        shared: list[tuple[int, int, str]] = []
        for t in hop2_clean:
            if t not in cand_cats:
                reject(t, "no_category_data: 분류 정보를 받지 못함 (API 실패)")
                continue
            cats = cand_cats.get(t, [])
            overlap = sorted(c for c in cats if c in seed_cat_pool)
            if overlap:
                shared.append((len(link_support[t]), len(overlap), t))
            else:
                reject(t, "not_category_shared: 시드 {}개 이상이 공유하는 주제 분류와 "
                          "겹치는 것이 없음".format(MIN_CATEGORY_SEEDS))
        shared.sort(key=lambda x: (-x[1], -x[0], x[2]))
        print("    분류 공유 후보: {}건".format(len(shared)))

        # 주제 범위 필터 — 시대(1890~1945) · 독립운동 핵심 분류 · 정체성 우위
        in_scope: list[tuple[int, int, str]] = []
        for support, overlap, t in shared:
            ok, why = topic_scope(t, cand_cats.get(t, []))
            if ok:
                in_scope.append((support, overlap, t))
            else:
                reject(t, why)
        shared = in_scope
        print("    주제 범위(1890~1945 독립운동) 통과: {}건".format(len(shared)))

        gazetteer: dict[str, list[str]] = {t: cats for t, cats in cand_cats.items() if cats}
        gazetteer.update(seed_cats)

        # ---------- 4. 본문 수집 ---------- #
        print("=" * 72)
        print("[4/6] 본문 수집 (prop=extracts, 1건씩)")
        saved: list[dict] = []
        saved_titles: set[str] = set()

        def take(title: str, hop: int, reason: str, cats_hint: list[str]) -> bool:
            if title in saved_titles:
                return False
            path = DOCS_DIR / safe_filename(title)
            if path.exists() and not force:
                d_title, d_cats, d_body = read_doc(path)
                saved_titles.add(title)
                saved.append({"title": d_title or title, "file": path.name,
                              "categories": d_cats or cats_hint,
                              "chars": len(d_body), "hop": hop,
                              "reason": reason + " (cached)"})
                print("    = {} (이미 수집됨, {}자)".format(title, len(d_body)))
                return True
            real, body = api.extract(title)
            if not body:
                reject(title, "no_extract: 본문을 가져오지 못함 (API 실패 또는 본문 없음)")
                print("    x {}: 본문 없음".format(title))
                return False
            if len(body) < MIN_CHARS:
                reject(title, "stub_too_short: 본문 {}자 < {}자 (토막글)".format(
                    len(body), MIN_CHARS))
                print("    x {}: 토막글 {}자".format(title, len(body)))
                return False
            cats = cats_hint or gazetteer.get(real or title, [])
            write_doc(real or title, cats, body)
            saved_titles.add(title)
            if real and real != title:
                saved_titles.add(real)
            saved.append({"title": real or title, "file": safe_filename(real or title),
                          "categories": cats, "chars": len(body), "hop": hop,
                          "reason": reason})
            print("    + {}: {}자".format(real or title, len(body)))
            return True

        for s in seeds:
            take(s, 0, "seed", seed_cats.get(s, []))

        for support, overlap, t in shared:
            if len(saved) >= target:
                reject(t, "over_quota: 목표 수집량 {}건 초과 "
                          "(상위 후보 우선, 시드링크 {}·공유분류 {})".format(
                              target, support, overlap))
                continue
            take(t, 2, "hop2: 시드 {}개가 링크 · 공유 분류 {}종".format(support, overlap),
                 cand_cats.get(t, []))

        stats = {
            "seed_count": len(seeds),
            "hop2_candidates": len(hop2),
            "category_shared": len(shared),
            "saved": len(saved),
        }
        return saved, {"stats": stats, "gazetteer": gazetteer, "seed_cats": seed_cats}

    saved, meta = run_pipeline(resolved_seeds)

    # ---------- 5. 목표 미달 시 시드 확장 ---------- #
    if len(saved) < MIN_DOCS:
        print("=" * 72)
        print("[5/6] 수집 {}건 < 하한 {}건 → 예비 시드 투입".format(len(saved), MIN_DOCS))
        extra_found, extra_alias = api.exists(EXTRA_SEEDS)
        added: list[str] = []
        for s in EXTRA_SEEDS:
            real = extra_alias.get(s, s)
            if extra_found.get(real, False):
                if real not in resolved_seeds:
                    resolved_seeds.append(real)
                    added.append(real)
            else:
                reject(s, "seed_missing: 위키백과에 해당 제목의 문서가 없음 (예비 시드)")
        seed_expansion = {
            "used": True,
            "reason": "1차 수집 {}건 < 하한 {}건".format(len(saved), MIN_DOCS),
            "added": added,
        }
        saved, meta = run_pipeline(resolved_seeds)
    else:
        print("=" * 72)
        print("[5/6] 수집 {}건 ≥ 하한 {}건 → 시드 확장 불필요".format(len(saved), MIN_DOCS))

    # ---------- 6. 별칭(넘겨주기) 수집 ---------- #
    print("=" * 72)
    print("[6/6] 넘겨주기(별칭) 수집")
    gazetteer = meta["gazetteer"]
    alias_targets = sorted({d["title"] for d in saved} | set(gazetteer))
    redirect_map = api.redirects_to(alias_targets)
    print("    {}건 조회 → 별칭이 있는 문서 {}건".format(
        len(alias_targets), sum(1 for v in redirect_map.values() if v)))

    for d in saved:
        d["redirects"] = redirect_map.get(d["title"], [])

    manifest = {
        "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "ko.wikipedia.org MediaWiki API",
        "topic": "한국 독립운동사 (1890~1945)",
        "seeds": resolved_seeds,
        "selection_policy": {
            "hop2_rule": "시드 {}개 이상이 함께 링크한 문서만 2홉 후보로 채택".format(
                MIN_SEED_SUPPORT),
            "category_rule": "시드 {}개 이상이 공유하는 주제 분류와 1개 이상 겹치는 후보만 채택 (시드 1개에만 붙은 분류나 '20세기 한국 사람' 같은 구조적 분류는 주제 분류로 인정하지 않는다)".format(MIN_CATEGORY_SEEDS),
            "title_block": "연도·연대·날짜·목록·일람·틀/분류/포털·동음이의 문서는 제목 규칙으로 배제",
            "length_rule": "본문 {}자 미만 토막글은 저장하지 않음".format(MIN_CHARS),
            "quota": "목표 {}건, 하한 {}건 (초과분은 over_quota 로 탈락)".format(
                target, MIN_DOCS),
        },
        "seed_expansion": seed_expansion,
        "stats": meta["stats"],
        "api_calls": api.calls,
        "api_failures": api.failures,
        "api_cache_hits": api.cache_hits,
        "docs": sorted(saved, key=lambda d: (d["hop"], d["title"])),
        # gazetteer: 본문에 등장하는 개체의 타입 판정을 위해 build_graph.py 가 읽는다.
        # 저장되지 않은 후보 문서의 분류까지 보존해 두어야 오프라인에서도 타입 판정이 된다.
        "gazetteer": {t: {"categories": c, "redirects": redirect_map.get(t, [])}
                      for t, c in sorted(gazetteer.items())},
        "rejected": sorted(rejected, key=lambda r: (r["reason"], r["title"])),
    }
    api.flush_cache(force=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    return manifest


def add_titles(titles: list[str], no_cache: bool = False) -> int:
    """특정 제목만 골라 본문을 받아 `data/docs` 에 채운다.

    채택 규칙을 고친 뒤 다시 살려야 하는 문서를 값싸게 복구하기 위한 경로다.
    (전체 재수집은 비싸고 불필요하다.)
    """
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    api = WikiAPI(use_cache=not no_cache)
    n = 0
    cats_map = api.categories(titles)
    for t in titles:
        real, body = api.extract(t)
        if not body:
            print("    x {}: 본문을 가져오지 못함".format(t))
            continue
        name = real or t
        write_doc(name, cats_map.get(name, cats_map.get(t, [])), body)
        print("    + {}: {}자".format(name, len(body)))
        n += 1
    api.flush_cache(force=True)
    return n


def prune() -> dict:
    """**네트워크 없이** 이미 받아 둔 `data/docs/*.md` 에 주제 범위 필터를 적용한다.

    수집과 필터를 분리해 둔 이유: 본문을 다시 받는 것은 비싸지만 채택 기준은 여러 번
    고쳐 가며 적용하게 된다. 탈락 문서는 파일을 지우고 **반드시** manifest.rejected 에
    사유와 함께 남긴다 (기록 없이 파일만 지우면 채점 항목을 못 채운다).
    """
    print("=" * 72)
    print("[prune] 주제 범위 필터를 기존 코퍼스에 적용 (네트워크 사용 안 함)")
    old = load_manifest()
    gazetteer: dict = old.get("gazetteer", {}) or {}
    prev_redirects = {d.get("title"): d.get("redirects", [])
                      for d in old.get("docs", []) or []}
    prev_reason = {d.get("title"): d.get("reason", "")
                   for d in old.get("docs", []) or []}

    kept: list[dict] = []
    rejected: list[dict] = [r for r in (old.get("rejected") or [])]
    rejected_titles = {r.get("title") for r in rejected}
    removed: list[tuple[str, str]] = []

    for path in sorted(DOCS_DIR.glob("*.md")):
        title, cats, body = read_doc(path)
        if not title:
            continue
        if len(body) < MIN_CHARS:
            removed.append((title, "stub_too_short: 본문 {}자 < {}자 (토막글)".format(
                len(body), MIN_CHARS)))
            path.unlink()
            continue
        ok, why = topic_scope(title, cats)
        if not ok:
            removed.append((title, why))
            path.unlink()
            continue
        gz = gazetteer.get(title) or {}
        kept.append({
            "title": title,
            "file": path.name,
            "categories": cats,
            "chars": len(body),
            "hop": 0 if title in (old.get("seeds") or []) else 2,
            "reason": prev_reason.get(title) or why,
            "redirects": prev_redirects.get(title)
                         or (gz.get("redirects", []) if isinstance(gz, dict) else []),
        })

    kept_titles = {d["title"] for d in kept}
    # 규칙이 바뀌어 다시 채택된 문서는 rejected 에서 뺀다 (기록이 현재 상태와 어긋나면 안 된다).
    rejected = [r for r in rejected if r.get("title") not in kept_titles]
    rejected_titles = {r.get("title") for r in rejected}
    for title, why in removed:
        if title not in rejected_titles:
            rejected.append({"title": title, "reason": why})
            rejected_titles.add(title)

    print("    유지 {}건 · 주제 이탈로 제거 {}건".format(len(kept), len(removed)))
    for title, why in sorted(removed):
        print("    - {} :: {}".format(title, why[:90]))

    manifest = dict(old)
    manifest.update({
        "pruned_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "topic": "한국 독립운동사 (1890~1945)",
        "docs": sorted(kept, key=lambda d: (d["hop"], d["title"])),
        "rejected": sorted(rejected, key=lambda r: (r["reason"], r["title"])),
    })
    policy = dict(manifest.get("selection_policy") or {})
    policy["topic_scope_rule"] = (
        "채택 = (A) 생애/발생 시기가 {}~{} 와 겹치고, (B) 독립운동·항일 계열 핵심 분류를 "
        "1개 이상 가지며, (C) 문학·예술 등 무관 활동 분류가 핵심 분류보다 많지 않은 문서. "
        "탈락 사유는 out_of_period_before / out_of_period_after / not_independence_topic / "
        "arts_identity_dominant / stub_too_short 로 rejected 에 기록한다."
    ).format(PERIOD_START, PERIOD_END)
    manifest["selection_policy"] = policy
    stats = dict(manifest.get("stats") or {})
    stats["saved"] = len(kept)
    stats["topic_pruned"] = len(removed)
    manifest["stats"] = stats
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    return manifest


def main() -> int:
    llm_provider.setup_console()
    ap = argparse.ArgumentParser(description="한국 독립운동사 위키백과 코퍼스 수집")
    ap.add_argument("--force", action="store_true", help="이미 받은 문서도 다시 수집")
    ap.add_argument("--target", type=int, default=TARGET_DOCS, help="목표 수집 문서 수")
    ap.add_argument("--no-cache", action="store_true", help="API 응답 캐시를 쓰지 않는다")
    ap.add_argument("--prune", action="store_true",
                    help="네트워크 없이 기존 data/docs 에 주제 범위 필터만 적용한다")
    ap.add_argument("--add", default="",
                    help="쉼표로 구분한 제목만 골라 본문을 받아 data/docs 에 채운다")
    args = ap.parse_args()

    print(llm_provider.banner())
    print("(수집 단계는 LLM 을 쓰지 않는다 — MediaWiki API 만 사용)")

    if args.add:
        wanted = [t.strip() for t in args.add.split(",") if t.strip()]
        print("[add] {}건 보충 수집".format(len(wanted)))
        print("    {}건 저장".format(add_titles(wanted, no_cache=args.no_cache)))
        if not args.prune:
            return 0

    if args.prune:
        manifest = prune()
        st = manifest["stats"]
        print("=" * 72)
        print("유지 {}건 (하한 {}건) · 탈락 기록 {}건".format(
            st["saved"], MIN_DOCS, len(manifest["rejected"])))
        reason_counts = Counter(r["reason"].split(":")[0] for r in manifest["rejected"])
        print("사유별: {}".format(dict(sorted(reason_counts.items()))))
        return 0 if st["saved"] >= MIN_DOCS else 1

    manifest = collect(force=args.force, target=args.target,
                       no_cache=args.no_cache)

    print("=" * 72)
    st = manifest["stats"]
    print("시드 {} · 2홉 후보 {} · 분류 공유 {} · 저장 {}".format(
        st["seed_count"], st["hop2_candidates"], st["category_shared"], st["saved"]))
    reason_counts = Counter(r["reason"].split(":")[0] for r in manifest["rejected"])
    print("탈락 기록: {}건 (사유별: {})".format(
        len(manifest["rejected"]), dict(sorted(reason_counts.items()))))
    print("gazetteer: {}종 (본문 내 개체 타입 판정용)".format(len(manifest["gazetteer"])))
    print("저장된 문서:")
    for d in manifest["docs"]:
        print("  - [{}홉] {} ({}자)".format(d["hop"], d["title"], d["chars"]))
    print("\nmanifest → {}".format(MANIFEST_PATH.relative_to(ROOT)))
    if st["saved"] < MIN_DOCS:
        print("[경고] 수집 {}건 < 하한 {}건".format(st["saved"], MIN_DOCS))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
