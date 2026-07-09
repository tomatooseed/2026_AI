"""
05. pdf_samples 문서함 Agent (RAG)
4.4 Agent_통합.ipynb 패턴을 따릅니다.

Tool 3개:
  - list_documents          : pdf_samples PDF 목록
  - get_document_catalog    : _catalog/index.json + summary txt 읽기
  - search_in_document      : 지정 PDF 내 RAG 검색
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from openai import OpenAI

# ── 1. OpenAI API 연결 (4.4 §1) ──────────────────────────────
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

# load_dotenv("../../.env") 는 실행 cwd 기준이라 IDE/다른 폴더에서 실패함 → 파일 위치에서 .env 탐색
_env_path = None
for parent in [BASE, *BASE.parents]:
    candidate = parent / ".env"
    if candidate.exists():
        load_dotenv(candidate)
        _env_path = candidate
        break
if _env_path is None:
    raise FileNotFoundError(".env 파일을 찾을 수 없습니다.")

api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(f"OPENAI_KEY가 없습니다. ({_env_path} 확인)")
client = OpenAI(api_key=api_key)

DOC_LIBRARY = BASE / "samples" / "pdf_samples"
CATALOG_DIR = DOC_LIBRARY / "_catalog"
SAMPLES_DIR = BASE / "samples"

INDEX_CACHE: dict[str, list[dict]] = {}


def ask(system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
    """4.4 스타일: system + user 분리 호출"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


# ── 2. RAG 기초 함수 (4.4 §2 / 03 노트북) ────────────────────
def split_text(text: str, chunk_size: int = 280, overlap: int = 60) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def tokenize(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[가-힣a-zA-Z0-9]+", text) if len(t) > 1}


def search_chunks(query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    q = tokenize(query)
    scored = []
    for item in chunks:
        score = len(q & tokenize(item["text"]))
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score": s, **it} for s, it in scored[:top_k]]


def read_document_text(pdf_name: str) -> str:
    """PDF 본문을 읽습니다. 없으면 동일 stem의 .txt를 찾습니다."""
    stem = Path(pdf_name).stem
    candidates = [
        DOC_LIBRARY / pdf_name,
        DOC_LIBRARY / f"{stem}.txt",
        SAMPLES_DIR / f"{stem}.txt",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix.lower() == ".pdf":
            doc = pymupdf.open(path)
            return "\n".join(page.get_text() for page in doc)
        return path.read_text(encoding="utf-8")
    return ""


def chunk_stem(pdf_name: str) -> str:
    stem = Path(pdf_name).stem
    return re.sub(r"[^\w가-힣\-]+", "_", stem)[:40].strip("_")


def build_pdf_index(pdf_name: str) -> list[dict]:
    if pdf_name in INDEX_CACHE:
        return INDEX_CACHE[pdf_name]
    text = read_document_text(pdf_name)
    if not text.strip():
        return []
    prefix = chunk_stem(pdf_name)
    index = [
        {
            "chunk_id": f"{prefix}_C{i:03d}",
            "source_pdf": pdf_name,
            "text": chunk,
        }
        for i, chunk in enumerate(split_text(text), 1)
    ]
    INDEX_CACHE[pdf_name] = index
    return index


# ── 3. Tool 함수 (4.4 §3) ─────────────────────────────────────
def list_documents() -> str:
    """pdf_samples 문서함의 PDF 파일명 목록."""
    index_path = CATALOG_DIR / "index.json"
    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        names = [doc["pdf_name"] for doc in data["documents"]]
    else:
        names = sorted(p.name for p in DOC_LIBRARY.glob("*.pdf"))
    return json.dumps({"count": len(names), "pdf_files": names}, ensure_ascii=False)


def get_document_catalog() -> str:
    """주어진 _catalog/index.json + summary txt 를 읽어 반환 (생성 X)."""
    index_path = CATALOG_DIR / "index.json"
    data = json.loads(index_path.read_text(encoding="utf-8"))
    entries = []
    for doc in data["documents"]:
        summary_path = CATALOG_DIR / doc["summary_file"]
        summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
        one_line = ""
        lines = summary_text.splitlines()
        for i, line in enumerate(lines):
            if "한 줄 요약" in line and i + 1 < len(lines):
                one_line = lines[i + 1].strip()
                break
        entries.append({
            "pdf_name": doc["pdf_name"],
            "category": doc.get("category", ""),
            "keywords": doc.get("keywords", []),
            "one_line_summary": one_line,
        })
    return json.dumps({"documents": entries}, ensure_ascii=False, indent=2)


def search_in_document(pdf_name: str, query: str, top_k: int = 3) -> str:
    """지정 PDF 안에서 RAG 검색."""
    hits = search_chunks(query, build_pdf_index(pdf_name), top_k=top_k)
    return json.dumps(
        {
            "pdf_name": pdf_name,
            "query": query,
            "chunks": hits,
            "message": "검색 결과 없음" if not hits else "ok",
        },
        ensure_ascii=False,
    )


# ── 4. Tool 등록 (4.4 §4) ─────────────────────────────────────
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "pdf_samples 폴더의 PDF 파일명 목록.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_catalog",
            "description": "문서함 카탈로그(유형·한줄요약·키워드). 어떤 PDF를 열지 정할 때 먼저 호출.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_document",
            "description": "카탈로그로 pdf_name을 고른 뒤, 해당 PDF 내에서 질문과 유사한 조항 검색.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pdf_name": {"type": "string", "description": "PDF 파일명 (예: 학칙.pdf)"},
                    "query": {"type": "string", "description": "검색할 질문 또는 키워드"},
                    "top_k": {"type": "integer", "description": "가져올 청크 개수 (기본 3)"},
                },
                "required": ["pdf_name", "query"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "list_documents": lambda **_: list_documents(),
    "get_document_catalog": lambda **_: get_document_catalog(),
    "search_in_document": search_in_document,
}


# ── 5. System Prompt (4.4 §5) ─────────────────────────────────
AGENT_SYSTEM = """
너는 pdf_samples 문서함 Q&A 도우미 Agent입니다. 제공된 도구를 적절히 사용해 답변하세요.

도구 선택 가이드:
- 문서함에 어떤 PDF가 있는지 → list_documents 또는 get_document_catalog
- 어떤 PDF를 열지 모를 때 → get_document_catalog (한줄요약·키워드 확인)
- 특정 PDF 내용 검색 → search_in_document (pdf_name + query)

규칙:
- 문서 내용이 필요하면 반드시 도구를 먼저 호출하세요.
- get_document_catalog로 pdf_name을 고른 뒤 search_in_document를 호출하세요.
- 도구 결과에 없는 내용은 추측하지 마세요.
- 답변은 한국어, bullet 3개 이내로 간결하게.
- 마지막 bullet에 근거를 적으세요. (예: 근거: [학칙.pdf] 학칙_C005)
""".strip()


# ── 6. run_agent_once (4.4 §7) — tool_calls 없을 때까지 루프 ───
def run_agent_once(
    user_question: str,
    tools: list | None = None,
    tool_functions: dict | None = None,
    system_msg: str | None = None,
    max_rounds: int = 8,
) -> str:
    tools = tools or AGENT_TOOLS
    tool_functions = tool_functions or TOOL_FUNCTIONS
    system_msg = system_msg or AGENT_SYSTEM

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_question},
    ]
    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=messages,
            tools=tools,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""
        messages.append(msg)
        for tc in msg.tool_calls:
            fn = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            result = tool_functions[fn](**args)
            print(f"  → {fn}({args})")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
    return "tool 호출 최대 횟수 초과"


# ── 7. 데모 ───────────────────────────────────────────────────
DEMO_QUESTIONS = [
    "문서함에 어떤 PDF가 있어?",
    "석사학위과정 수업연한은?",
    "LLM autonomous agent 서베이 논문의 주제는?",
    "data2vec가 다루는 modality 세 가지는?",
    "2026년 1분기 SK하이닉스 영업이익은?",
    "Vision Transformer class embedding 논문이 뭐 다루는지 한 줄로",
]

if __name__ == "__main__":
    print("✅ client 준비")
    print("등록된 tool:", list(TOOL_FUNCTIONS.keys()))
    print()

    for q in DEMO_QUESTIONS:
        print("=" * 60)
        print("Q:", q)
        print(run_agent_once(q))
        print()
