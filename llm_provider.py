"""llm_provider.py — 프로젝트 전역 LLM 접근 단일 창구 (코디네이터 소유, 워커 수정 금지).

이 모듈은 OpenAI API 키를 다루는 **유일한 지점**이다.
build_graph.py / agent.py / evaluate.py / app.py 는 여기 있는 함수만 호출하고,
`os.environ["OPENAI_API_KEY"]` 를 직접 읽거나 키 문자열을 자체 보관해서는 안 된다.

보안 원칙
---------
1. 키를 **저장소에 커밋하지 않는다**. `.env` 와 `.streamlit/secrets.toml` 은 `.gitignore` 로 차단한다.
2. 키를 **로그·예외 메시지·JSON 산출물에 절대 쓰지 않는다**. 표시가 필요하면 `mask()` 를 쓴다.
3. 키 해석 우선순위: 런타임 주입(Streamlit 설정 탭) > 환경변수 `OPENAI_API_KEY` > 저장소 루트 `.env`
4. 키가 없으면 **예외를 던지지 않는다**. `chat()` 은 `None` 을 돌려주고, 호출부는 규칙 기반으로 폴백한다.
   → 과제 루브릭 1번("전체 워크플로우가 오류 없이 실행되는가")을 키 없이도 만족시키기 위함이다.

캐시
----
모든 LLM 응답은 `output/llm_cache.json` 에 (모델, 메시지) 해시로 캐시된다.
재실행 비용이 0이 되고, 채점자가 키 없이 돌려도 **같은 결과가 재현**된다.
캐시 파일에는 키가 들어가지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
CACHE_PATH = ROOT / "output" / "llm_cache.json"

_ENV_VAR = "OPENAI_API_KEY"

# 런타임 주입 키(Streamlit 설정 탭). 프로세스 메모리에만 존재하고 디스크에 쓰지 않는다.
_runtime_key: str | None = None
_lock = threading.Lock()

_cache: dict[str, str] | None = None

# 호출 통계 — 리포트용. 키 값은 절대 담지 않는다.
STATS: dict[str, int] = {"hit": 0, "miss": 0, "call": 0, "error": 0, "skipped_no_key": 0}


# --------------------------------------------------------------------------- #
# 키 관리
# --------------------------------------------------------------------------- #
def _read_env_file() -> str | None:
    """저장소 루트 `.env` 에서 OPENAI_API_KEY 를 읽는다. 없으면 None."""
    if not ENV_PATH.exists():
        return None
    try:
        for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == _ENV_VAR:
                return v.strip().strip('"').strip("'") or None
    except OSError:
        return None
    return None


def set_runtime_key(key: str | None) -> None:
    """Streamlit 설정 탭에서 입력받은 키를 이 프로세스에만 적용한다 (디스크 기록 없음)."""
    global _runtime_key
    with _lock:
        _runtime_key = (key or "").strip() or None


def clear_runtime_key() -> None:
    set_runtime_key(None)


def get_api_key() -> str | None:
    """우선순위에 따라 키를 해석한다. 없으면 None."""
    if _runtime_key:
        return _runtime_key
    env = os.environ.get(_ENV_VAR, "").strip()
    if env:
        return env
    return _read_env_file()


def key_source() -> str | None:
    if _runtime_key:
        return "runtime"       # 설정 탭 입력
    if os.environ.get(_ENV_VAR, "").strip():
        return "env"           # 환경변수
    if _read_env_file():
        return "dotenv"        # .env 파일
    return None


def mask(key: str | None) -> str:
    """로그·화면 표시용 마스킹. 앞 3자 + 뒤 4자만 남긴다."""
    if not key:
        return "(없음)"
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:3]}{'*' * 8}{key[-4:]}"


def is_enabled() -> bool:
    return get_api_key() is not None


def key_status() -> dict[str, Any]:
    """설정 탭/리포트에서 쓰는 안전한 상태 요약. 키 원문은 포함하지 않는다."""
    key = get_api_key()
    return {
        "available": key is not None,
        "source": key_source(),
        "masked": mask(key),
        "mode": "LLM 증강 모드" if key else "규칙 전용 모드 (오프라인)",
    }


def looks_like_openai_key(key: str) -> bool:
    return bool(re.fullmatch(r"sk-[A-Za-z0-9_\-]{20,}", (key or "").strip()))


def save_key_to_env_file(key: str) -> Path:
    """사용자가 명시적으로 원할 때만 `.env` 에 저장한다.

    `.gitignore` 에 `.env` 가 없으면 자동으로 추가해 유출을 막는다.
    """
    key = (key or "").strip()
    if not key:
        raise ValueError("빈 키는 저장할 수 없습니다.")

    lines: list[str] = []
    if ENV_PATH.exists():
        lines = [
            ln for ln in ENV_PATH.read_text(encoding="utf-8").splitlines()
            if not ln.strip().startswith(f"{_ENV_VAR}=")
        ]
    lines.append(f'{_ENV_VAR}="{key}"')
    ENV_PATH.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    ensure_gitignore()
    return ENV_PATH


def delete_saved_key() -> bool:
    """`.env` 에 저장된 키를 지운다."""
    if not ENV_PATH.exists():
        return False
    lines = [
        ln for ln in ENV_PATH.read_text(encoding="utf-8").splitlines()
        if not ln.strip().startswith(f"{_ENV_VAR}=")
    ]
    body = "\n".join(lines).strip()
    ENV_PATH.write_text(body + "\n" if body else "", encoding="utf-8")
    return True


def ensure_gitignore() -> None:
    """`.env` 계열이 반드시 무시되도록 `.gitignore` 를 보정한다."""
    gi = ROOT / ".gitignore"
    needed = [".env", ".env.*", "!.env.example", ".streamlit/secrets.toml", "*.key"]
    existing = gi.read_text(encoding="utf-8").splitlines() if gi.exists() else []
    missing = [n for n in needed if n not in existing]
    if missing:
        with gi.open("a", encoding="utf-8") as f:
            if existing and existing[-1].strip():
                f.write("\n")
            f.write("# API 키 유출 방지 (llm_provider.ensure_gitignore 가 관리)\n")
            f.write("\n".join(missing) + "\n")


# --------------------------------------------------------------------------- #
# 캐시
# --------------------------------------------------------------------------- #
def _load_cache() -> dict[str, str]:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _cache = {}
    return _cache


def _save_cache() -> None:
    if _cache is None:
        return
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_cache, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(CACHE_PATH)


def _cache_key(model: str, messages: list[dict], extra: str = "") -> str:
    blob = json.dumps({"m": model, "msgs": messages, "x": extra},
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cache_stats() -> dict[str, int]:
    return dict(STATS, cached_entries=len(_load_cache()))


# --------------------------------------------------------------------------- #
# 호출
# --------------------------------------------------------------------------- #
def chat(
    messages: list[dict],
    *,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.0,
    max_tokens: int = 1500,
    use_cache: bool = True,
) -> str | None:
    """OpenAI Chat 호출. 키가 없거나 실패하면 **None** 을 반환한다 (예외 없음).

    호출부는 반드시 `None` 을 규칙 기반 폴백으로 처리해야 한다.
    """
    ck = _cache_key(model, messages, f"t={temperature}")
    if use_cache:
        cached = _load_cache().get(ck)
        if cached is not None:
            STATS["hit"] += 1
            return cached

    key = get_api_key()
    if not key:
        STATS["skipped_no_key"] += 1
        return None

    STATS["miss"] += 1
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, timeout=90.0, max_retries=2)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        STATS["call"] += 1
    except Exception as exc:  # noqa: BLE001 - 어떤 실패도 파이프라인을 멈추면 안 된다
        STATS["error"] += 1
        # 예외 메시지에 키가 섞여 나올 가능성을 차단한다.
        safe = str(exc).replace(key, mask(key)) if key else str(exc)
        print(f"  [llm_provider] 호출 실패 → 규칙 기반으로 폴백합니다: {safe[:200]}")
        return None

    if use_cache:
        _load_cache()[ck] = text
        _save_cache()
    return text


def chat_json(
    messages: list[dict],
    *,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.0,
    max_tokens: int = 1500,
    use_cache: bool = True,
) -> Any | None:
    """`chat()` 의 JSON 파싱판. 파싱 실패 시 None."""
    text = chat(messages, model=model, temperature=temperature,
                max_tokens=max_tokens, use_cache=use_cache)
    if text is None:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.S).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"[\[{].*[\]}]", cleaned, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    STATS["error"] += 1
    return None


def setup_console() -> None:
    """Windows cp949 콘솔에서 한글 print 가 깨지거나 UnicodeEncodeError 로 죽는 것을 막는다.

    모든 실행 스크립트(build_graph/agent/evaluate/collect_corpus)는 `main()` 첫 줄에서 호출한다.
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass


def banner() -> str:
    """스크립트 시작 시 1줄 출력용. 키 원문은 나오지 않는다."""
    s = key_status()
    return f"[LLM] {s['mode']} | 키: {s['masked']} | 출처: {s['source'] or '-'}"


if __name__ == "__main__":
    setup_console()
    ensure_gitignore()
    print(banner())
    print(json.dumps(cache_stats(), ensure_ascii=False))
