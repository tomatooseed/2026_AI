from pathlib import Path

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# "../../.env" 는 실행 cwd 기준 → IDE에서 cwd=2026_AI 이면 .env를 못 찾음
BASE = Path(__file__).resolve().parent
for parent in [BASE, *BASE.parents]:
    env_path = parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        break

api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
if not api_key:
    raise ValueError(".env에서 OPENAI_API_KEY 또는 OPENAI_KEY를 찾을 수 없습니다.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=api_key)

messages = [
    SystemMessage(content="너는 사용자를 도와주는 상담사야."),
]

while True:
    user_input = input("사용자: ")

    if user_input == "exit":
        break

    messages.append(HumanMessage(content=user_input))
    ai_response = llm.invoke(messages).content
    messages.append(AIMessage(content=ai_response))

    print("AI: " + ai_response)
