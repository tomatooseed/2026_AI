import hashlib
import hmac
import os
import ssl
import time
from collections import defaultdict
from pathlib import Path

import certifi
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from langchain_openai import ChatOpenAI
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# 프로젝트 루트 .env (OPENAI_API_KEY, MY_SLACK_*)
load_dotenv()
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

SLACK_BOT_TOKEN = os.getenv("MY_SLACK_TOKEN") or os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.getenv("MY_SLACK_SIGNING_SECRET") or os.environ["SLACK_SIGNING_SECRET"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

SYSTEM_PROMPT = "너는 나의 제일 친한 친구야. 친절하게 답을 해줘"
# 채널(또는 스레드)별로 최근 이 개수만큼 human/ai 메시지를 기억
MAX_HISTORY_MESSAGES = 20

# macOS Python SSL 이슈 대응
_ssl_context = ssl.create_default_context(cafile=certifi.where())
client = WebClient(token=SLACK_BOT_TOKEN, ssl=_ssl_context)

# 14.1 대화형 Agent 과 동일 패턴
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=OPENAI_API_KEY,
)

app = FastAPI(title="Slack LLM Friend Bot")
_seen: set[str] = set()
# key: channel 또는 channel:thread_ts → [("human"|"ai", text), ...]
_histories: dict[str, list[tuple[str, str]]] = defaultdict(list)


def verify_slack_signature(body: bytes, timestamp: str | None, signature: str | None) -> None:
    print("timestamp:", timestamp)
    print("signature:", signature)
    print("body:", body.decode())

    if not timestamp or not signature:
        print("❌ missing signature headers")
        raise HTTPException(status_code=401, detail="missing signature headers")

    if abs(time.time() - int(timestamp)) > 60 * 5:
        print("❌ stale request")
        raise HTTPException(status_code=401, detail="stale request")

    base = f"v0:{timestamp}:{body.decode('utf-8')}"
    digest = hmac.new(
        SLACK_SIGNING_SECRET.encode(),
        base.encode(),
        hashlib.sha256,
    ).hexdigest()

    expected = f"v0={digest}"

    print("expected:", expected)
    print("received:", signature)

    if not hmac.compare_digest(expected, signature):
        print("❌ invalid signature")
        raise HTTPException(status_code=401, detail="invalid signature")

    print("✅ signature ok")


def session_key(channel: str, thread_ts: str | None) -> str:
    """스레드면 스레드 단위, 아니면 채널 단위로 대화 기억."""
    return f"{channel}:{thread_ts}" if thread_ts else channel


def ask_llm(key: str, user_text: str) -> str:
    """채널/스레드 히스토리를 이어가며 LLM 응답 생성."""
    history = _histories[key]
    history.append(("human", user_text))

    messages = [("system", SYSTEM_PROMPT), *history]
    result = llm.invoke(messages)
    reply = (result.content or "").strip() or "(응답이 비어 있어요)"

    history.append(("ai", reply))
    # 최근 N개만 유지 (system 제외)
    if len(history) > MAX_HISTORY_MESSAGES:
        _histories[key] = history[-MAX_HISTORY_MESSAGES:]

    print(f"히스토리[{key}] {len(_histories[key])} turns(msgs)")
    return reply


@app.get("/health")
def health():
    return {"ok": True, "sessions": len(_histories)}


@app.post("/slack/events")
async def slack_events(
    request: Request,
    x_slack_signature: str | None = Header(default=None),
    x_slack_request_timestamp: str | None = Header(default=None),
):
    body = await request.body()
    verify_slack_signature(body, x_slack_request_timestamp, x_slack_signature)
    payload = await request.json()

    # Slack Event Subscriptions URL 검증
    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge")}

    if payload.get("type") != "event_callback":
        return {"ok": True}

    event_id = payload.get("event_id")
    if event_id:
        if event_id in _seen:
            return {"ok": True}
        _seen.add(event_id)

    event = payload.get("event") or {}
    # 봇 메시지는 무시 (안 그러면 자기 답장에 또 반응)
    if event.get("bot_id") or event.get("subtype") == "bot_message":
        return {"ok": True}

    text = (event.get("text") or "").strip()
    channel = event.get("channel")
    if not text or not channel:
        return {"ok": True}

    # app_mention 이면 "<@U...> 내용" 에서 내용만 사용
    if event.get("type") == "app_mention":
        parts = text.split(maxsplit=1)
        text = parts[1] if len(parts) > 1 else text

    # 스레드에서 멘션하면 그 스레드 맥락으로, 아니면 채널 맥락으로 기억
    thread_ts = event.get("thread_ts")
    key = session_key(channel, thread_ts)

    print("받은 메시지:", text, "| session:", key)
    try:
        reply = ask_llm(key, text)
        print("LLM 응답:", reply)
        post_kwargs = {"channel": channel, "text": reply}
        # 스레드 안이면 같은 스레드에 답장
        if thread_ts:
            post_kwargs["thread_ts"] = thread_ts
        client.chat_postMessage(**post_kwargs)
    except SlackApiError as e:
        print("postMessage 실패:", e.response.get("error"))
    except Exception as e:
        print("LLM/응답 실패:", type(e).__name__, e)
        try:
            client.chat_postMessage(
                channel=channel,
                text="잠깐 머리가 멈췄어… 다시 말 걸어줄래?",
                **({"thread_ts": thread_ts} if thread_ts else {}),
            )
        except Exception as e2:
            print("에러 메시지 전송도 실패:", type(e2).__name__, e2)

    return {"ok": True}
