import re
import time
from pathlib import Path
import os
from src.utils import KNOWLEDGE_BASE
from .clients import coder_clients

# MODEL_NAME = "qwen-plus"
#
# coder_client = OpenAI(
#     api_key=API_KEY,
#     base_url=BASE_URL,
# )


def extract_python_code(text):
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def get_coder_response(formula_text, prompt, MODEL_NAME='qwen-plus', temperature=0):
    coder_client =coder_clients[MODEL_NAME]
    # 调用模型处理
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": f"""                
## Description 
{formula_text}

## Python Function
        """,
        },
    ]
    response = coder_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
    )

    ans = response.choices[0].message.content
    return extract_python_code(ans)


def process_md(file_path, prompt, tar_dir: str = KNOWLEDGE_BASE):
    name = os.path.basename(file_path)
    name = name.split(".")[0]
    print(f"---- processing {name} ----")
    try:
        with open(file_path, encoding="utf-8") as f:
            formula_text = f.read()
        python_code = get_coder_response(formula_text, prompt)
        # 写入本地文件
        with open(f"{tar_dir}/{name}.py", "w", encoding="utf-8") as f:
            f.write(python_code)
        yield python_code

    except Exception as e:
        print(f"---- error ----\n{e}")
        yield str(e)


def code_chat(path, prompt, tar_dir: str | Path = KNOWLEDGE_BASE):
    if not os.path.exists(path):
        return "input image folder path not exits."

    if os.path.isfile(path) and path.endswith(".md"):
        # 如果是文件直接处理
        yield from process_md(path, prompt, tar_dir)
    elif os.path.isdir(path):
        md_files = [
            f for f in os.listdir(path) if f.lower().endswith((".md", ".markdown"))
        ]
        total_files = len(md_files)

        for index, file in enumerate(md_files):
            file_path = os.path.join(path, file)
            image_description = next(process_md(file_path, prompt, tar_dir))

            progress = (index + 1) / total_files * 100
            yield f"# Progress: {progress:.2f}% ({index + 1}/{total_files})\n\n" + image_description

        time.sleep(1)
        yield "All md processed."
