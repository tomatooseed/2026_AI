"""
Streamlit 챗봇 — OpenAI API + Tool Calling

실행 방법 (3일차 폴더에서):
    pip install streamlit openai python-dotenv pytz yfinance
    streamlit run 3.5_Streamlit_Chatbot.py

프로젝트 루트(2026_AI/.env)에 OPENAI_KEY=sk-... 를 설정하세요.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pytz
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_SYSTEM = (
    "You are a helpful assistant. Use tools when needed. "
    "Answer in Korean unless the user asks otherwise."
)

CITY_TZ = {
    "서울": "Asia/Seoul",
    "seoul": "Asia/Seoul",
    "뉴욕": "America/New_York",
    "new york": "America/New_York",
    "도쿄": "Asia/Tokyo",
    "tokyo": "Asia/Tokyo",
    "런던": "Europe/London",
    "london": "Europe/London",
}


# --- Tool 함수 (3.4 Tool_Calling.ipynb) ---

def get_city_time_basic() -> str:
    """도시 현재 시간을 반환(기본 버전: 시간대 미반영)"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_city_time_tz(city: str) -> str:
    key = city.strip().lower()
    tz_name = CITY_TZ.get(key)
    if not tz_name:
        return json.dumps({"error": f"지원하지 않는 도시: {city}"}, ensure_ascii=False)

    now = datetime.now(pytz.timezone(tz_name)).strftime("%Y-%m-%d %H:%M:%S")
    return json.dumps(
        {"city": city, "timezone": tz_name, "current_time": now},
        ensure_ascii=False,
    )


def get_us_stock_price(ticker: str) -> str:
    symbol = ticker.strip().upper()
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="2d")
        if hist.empty:
            return json.dumps({"error": f"{symbol} 데이터가 없습니다."}, ensure_ascii=False)

        latest = hist.iloc[-1]
        prev_close = float(latest["Close"]) if "Close" in latest else None
        open_price = float(latest["Open"]) if "Open" in latest else None

        return json.dumps(
            {
                "ticker": symbol,
                "open": round(open_price, 2) if open_price is not None else None,
                "close": round(prev_close, 2) if prev_close is not None else None,
                "currency": "USD",
                "source": "yfinance",
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_city_time_basic",
            "description": "현재 시간을 반환합니다. (로컬 시간, 시간대 미반영)",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_city_time_tz",
            "description": "도시의 시간대를 반영해 현재 시간을 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_us_stock_price",
            "description": "미국 주식 티커 심볼로 최근 시가·종가를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "get_city_time_basic": get_city_time_basic,
    "get_city_time_tz": get_city_time_tz,
    "get_us_stock_price": get_us_stock_price,
}


def get_api_key() -> str | None:
    return os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")


@st.cache_resource
def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def assistant_to_dict(msg) -> dict:
    data = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        data["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return data


def execute_tool_call(tool_call) -> tuple[str, str]:
    fn_name = tool_call.function.name
    fn_args = json.loads(tool_call.function.arguments or "{}")
    result = TOOL_FUNCTIONS[fn_name](**fn_args)
    if not isinstance(result, str):
        result = json.dumps(result, ensure_ascii=False)
    return fn_name, result


def run_tool_loop(
    client: OpenAI,
    messages: list[dict],
    model: str,
    temperature: float,
    max_rounds: int = 5,
) -> tuple[list[dict], list[dict]]:
    """Tool 호출이 없을 때까지 반복 실행. (messages, tool_log) 반환."""
    msgs = list(messages)
    tool_log: list[dict] = []

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=msgs,
            tools=TOOLS,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msgs, tool_log

        msgs.append(assistant_to_dict(msg))
        for tc in msg.tool_calls:
            fn_name, result = execute_tool_call(tc)
            fn_args = json.loads(tc.function.arguments or "{}")
            tool_log.append({"name": fn_name, "args": fn_args, "result": result})
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return msgs, tool_log


def stream_final_response(
    client: OpenAI,
    messages: list[dict],
    model: str,
    temperature: float,
):
    stream = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
        tools=TOOLS,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta



def render_tool_log(tool_log: list[dict]):
    if not tool_log:
        return
    with st.expander(f"🔧 Tool 호출 ({len(tool_log)}건)", expanded=False):
        for item in tool_log:
            st.code(f"{item['name']}({item['args']})\n→ {item['result']}", language="text")


def main():
    st.set_page_config(page_title="Tool Calling 챗봇", page_icon="💬", layout="centered")
    st.title("💬 Tool Calling 챗봇")
    st.caption("OpenAI API + Streamlit + Tool Calling")

    api_key = get_api_key()
    if not api_key:
        st.error(
            f"`{ROOT / '.env'}` 파일에 `OPENAI_KEY=sk-...` 를 설정한 뒤 앱을 다시 실행하세요."
        )
        st.stop()

    client = get_client(api_key)

    with st.sidebar:
        st.header("설정")
        model = st.selectbox("모델", [DEFAULT_MODEL, "gpt-4o", "gpt-4.1-mini"], index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
        system_prompt = st.text_area("시스템 프롬프트", value=DEFAULT_SYSTEM, height=100)

        st.header("사용 가능한 Tool")
        st.markdown(
            "- `get_city_time_basic` — 로컬 현재 시간\n"
            "- `get_city_time_tz` — 도시별 시간 (서울·뉴욕·도쿄·런던)\n"
            "- `get_us_stock_price` — 미국 주식 시세 (yfinance)"
        )

        if st.button("대화 초기화", use_container_width=True):
            st.session_state.api_messages = []
            st.rerun()

    if "api_messages" not in st.session_state:
        st.session_state.api_messages = []

    for msg in st.session_state.api_messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        elif msg["role"] == "assistant" and msg.get("content"):
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    if prompt := st.chat_input("메시지를 입력하세요..."):
        st.session_state.api_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        working = [{"role": "system", "content": system_prompt}]
        working.extend(st.session_state.api_messages)

        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                msgs_after_tools, tool_log = run_tool_loop(
                    client, working, model, temperature
                )

            render_tool_log(tool_log)

            response = st.write_stream(
                stream_final_response(client, msgs_after_tools, model, temperature)
            )

        tool_msgs = msgs_after_tools[len(working):]
        st.session_state.api_messages.extend(tool_msgs)
        st.session_state.api_messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
