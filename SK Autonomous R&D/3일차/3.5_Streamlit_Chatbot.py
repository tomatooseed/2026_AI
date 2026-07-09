"""
간단한 Streamlit 챗봇 — OpenAI API

실행 방법 (3일차 폴더에서):
    pip install streamlit openai python-dotenv
    streamlit run 3.5_Streamlit_Chatbot.py

프로젝트 루트(2026_AI/.env)에 OPENAI_KEY=sk-... 를 설정하세요.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# 프로젝트 루트의 .env 로드 (2026_AI/.env)
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_SYSTEM = "You are a helpful assistant. Answer in Korean unless the user asks otherwise."


def get_api_key() -> str | None:
    return os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")


@st.cache_resource
def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def stream_response(client: OpenAI, messages: list[dict], model: str, temperature: float):
    """OpenAI 스트리밍 응답을 한 글자씩 yield."""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def main():
    st.set_page_config(page_title="간단한 챗봇", page_icon="💬", layout="centered")
    st.title("💬 간단한 챗봇")
    st.caption("OpenAI API + Streamlit")

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
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        system_prompt = st.text_area("시스템 프롬프트", value=DEFAULT_SYSTEM, height=100)
        if st.button("대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 이전 대화 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    if prompt := st.chat_input("메시지를 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(st.session_state.messages)

        with st.chat_message("assistant"):
            response = st.write_stream(
                stream_response(client, api_messages, model, temperature)
            )

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
