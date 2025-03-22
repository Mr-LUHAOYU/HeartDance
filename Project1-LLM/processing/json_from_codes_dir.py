import re
import time
from pathlib import Path
import os
import json_repair
from src.utils import KNOWLEDGE_BASE
from .clients import coder_clients

# MODEL_NAME = "qwen-plus"
#
# coder_client = OpenAI(
#     api_key=API_KEY,
#     base_url=BASE_URL,
# )


def extract_json_tool(text):
    pattern = r"```json\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        return json_str
    return None


def get_tool_response(formula_text, prompt, MODEL_NAME='qwen-plus', temperature=0):
    coder_client = coder_clients[MODEL_NAME]
    # 调用模型处理
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": f"""                
## Python Function 
{formula_text}

## Json Tool 
        """,
        },
    ]
    response = coder_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
    )

    ans = response.choices[0].message.content
    return extract_json_tool(ans)


def process_md(
        file_path, prompt, tar_dir: str = KNOWLEDGE_BASE,
        MODEL_NAME='qwen-plus', temperature=0
):
    name = os.path.basename(file_path)
    name = name.split(".")[0]
    print(f"---- processing {name} ----")
    try:
        with open(file_path, encoding="utf-8") as f:
            formula_text = f.read()
        json_tool = get_tool_response(formula_text, prompt, MODEL_NAME, temperature)
        # 写入本地文件
        with open(f"{tar_dir}/{name}.json", "w", encoding="utf-8") as f:
            json_object = json_repair.repair_json(json_tool, ensure_ascii=False)
            f.write(f"{json_object}")
        yield json_tool

    except Exception as e:
        print(f"---- error ----\n{e}")
        yield str(e)


def code_chat(
        path, prompt, tar_dir: str | Path = KNOWLEDGE_BASE,
        MODEL_NAME='qwen-plus', temperature=0
):
    if not os.path.exists(path):
        return "input image folder path not exits."

    if os.path.isfile(path) and path.lower().endswith((".py", ".py3")):
        # 如果是文件直接处理
        yield from process_md(path, prompt, tar_dir, MODEL_NAME, temperature)
    elif os.path.isdir(path):
        py_files = [
            f for f in os.listdir(path) if f.lower().endswith((".py", ".py3"))
        ]
        total_files = len(py_files)

        for index, file in enumerate(py_files):
            file_path = os.path.join(path, file)
            image_description = next(process_md(file_path, prompt, tar_dir, MODEL_NAME, temperature))

            progress = (index + 1) / total_files * 100
            yield f"# Progress: {progress:.2f}% ({index + 1}/{total_files})\n\n" + image_description

        time.sleep(1)
        yield "All images processed."
