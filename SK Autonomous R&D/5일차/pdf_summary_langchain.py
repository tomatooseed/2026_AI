import os
from pathlib import Path
import pymupdf
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

BASE = Path(__file__).resolve().parent
for parent in [BASE, *BASE.parents]:
    env_path = parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        break

api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
if not api_key:
    raise ValueError(".env에서 OPENAI_API_KEY 또는 OPENAI_KEY를 찾을 수 없습니다.")




def pdf_to_text(pdf_path: str) -> str:
    """
    pdf 경로를 받아 요약한 string을 변환하는 도구 
    Args : 
    pdf_path : pdf 경로 
    """

    print("입력받은 pdf_path =", pdf_path)

    pdf_dir = os.path.dirname(pdf_path)

    doc = pymupdf.open(pdf_path)

    full_txt = ""

    for page in doc:
        full_txt += page.get_text()
        full_txt += "\n-------------------\n"

    pdf_file_name = os.path.basename(pdf_path)
    pdf_file_name = os.path.splitext(pdf_file_name)[0]

    txt_file_path = os.path.join(
        pdf_dir,
        f"{pdf_file_name}.txt"
    )

    with open(txt_file_path, "w", encoding="utf-8") as f:
        f.write(full_txt)

    return txt_file_path

@tool
def summarize_pdf(pdf_path: str) -> str:
    """
    pdf를 summary하는 함수
    Args: 
    pdf_path(str) : pdf가 들어있는 파일경로 반환
    """

    txt_file_path = pdf_to_text(pdf_path)

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

    with open(txt_file_path, "r", encoding="utf-8") as f:
        txt = f.read()


    messages = [
        SystemMessage("너는 문서 요약 전문가다."),
        HumanMessage(f"""
    너는 논문 및 문서 요약 전문가다.

    다음 형식으로 작성하라.

    # 제목

    ## 저자의 문제 인식 및 주장
    (15문장 이내)

    ## 저자 소개

    ============ 원문 ============
    {txt[:10000]}
    """)
    ]
    response=llm.invoke(messages)
    summary=response.content

    

    return summary

if __name__ == "__main__":
    pdf_path="SK Autonomous R&D/4일차/samples/Language_Models.pdf"
    summary = summarize_pdf.invoke(
    {"pdf_path": pdf_path}
)

    print(summary)