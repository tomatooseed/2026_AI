import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import pymupdf


def load_env_from_parent():
    current = Path.cwd()

    for parent in [current] + list(current.parents):
        env_path = parent / ".env"

        if env_path.exists():
            load_dotenv(env_path)
            print(f".env loaded: {env_path}")
            return env_path

    raise FileNotFoundError(".env 파일을 찾을 수 없습니다.")


def pdf_to_text(pdf_path: str) -> str:
    """
    PDF를 TXT로 변환하고 txt 파일 경로를 반환
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


def summarize_pdf(pdf_path: str) -> str:
    """
    PDF를 요약해서 요약문 문자열 반환
    """

    txt_file_path = pdf_to_text(pdf_path)

    load_env_from_parent()

    api_key = os.getenv("OPENAI_KEY")

    client = OpenAI(api_key=api_key)

    with open(txt_file_path, "r", encoding="utf-8") as f:
        txt = f.read()

    prompt = f"""
너는 논문 및 문서 요약 전문가다.

다음 형식으로 작성하라.

# 제목

## 저자의 문제 인식 및 주장
(15문장 이내)

## 저자 소개

============ 원문 ============
{txt[:10000]}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": "너는 문서 요약 전문가다."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    summary = response.choices[0].message.content

    return summary