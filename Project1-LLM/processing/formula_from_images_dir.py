import re
import time
from pathlib import Path
import os
from src.utils import KNOWLEDGE_BASE
from src.file2cloud import upload_to_oss
from .clients import ocr_clients


# MODEL_NAME = "qwen-vl-plus"
#
# ocr_client = OpenAI(
#     api_key=API_KEY,
#     base_url=BASE_URL,
# )

def replace_latex_delimiters(text: str):
    text = text.replace("<think>", "").replace("</think>", "").replace("&", "")
    # text = re.sub(r"<think>.*</think>", "", text, flags=re.DOTALL)

    patterns = [
        r"\\begin\{equation\}(.*?)\\end\{equation\}",  # \begin{equation} ... \end{equation}
        r"\\begin\{aligned\}(.*?)\\end\{aligned\}",  # \begin{aligned} ... \end{aligned}
        r"\\begin\{alignat\}(.*?)\\end\{alignat\}",  # \begin{alignat} ... \end{alignat}
        r"\\begin\{align\}(.*?)\\end\{align\}",  # \begin{align} ... \end{align}
        r"\\begin\{gather\}(.*?)\\end\{gather\}",  # \begin{gather} ... \end{gather}
        r"\\begin\{CD\}(.*?)\\end\{CD\}",  # \begin{CD} ... \end{CD}
    ]
    # 替换所有匹配的模式
    for pattern in patterns:
        text = re.sub(pattern, r" $$ \1 $$ ", text, flags=re.DOTALL)
    # 定义正则表达式模式
    patterns = [
        r"\\\[\n(.*?)\n\\\]",  # \[ ... \]
        r"\\\(\n(.*?)\n\\\)",  # \( ... \)
    ]
    # 替换所有匹配的模式
    for pattern in patterns:
        text = re.sub(pattern, r" $$ \1 $$ ", text, flags=re.DOTALL)
    # 定义正则表达式模式
    patterns = [
        r"\\\[(.*?)\\\]",  # \[ ... \]
        r"\\\((.*?)\\\)",  # \( ... \)
    ]
    # 替换所有匹配的模式
    for pattern in patterns:
        text = re.sub(pattern, r" $ \1 $ ", text, flags=re.DOTALL)
    return text


def get_ocr_response(image_path, MODEL_NAME='qwen-vl-plus', temperature=0):
    ocr_client = ocr_clients[MODEL_NAME]
    ext = image_path.split(".")[-1]
    download_url = upload_to_oss('temp.' + ext, image_path)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please extract the content in this image, ensuring that any LaTeX formulas are correctly transcribed, ensuring that any Tables are displayed in markdown format.",
                },
                {"type": "image_url", "image_url": {"url": f"{download_url}"}},
            ],
        },
    ]
    response = ocr_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
        stream=True,
    )

    # content = response.choices[0].message.content
    content = ''.join([chunk.choices[0].delta.content for chunk in response if chunk.choices[0].delta.content])

    content = replace_latex_delimiters(content)
    return content


def process_image(
    file_path: str, tar_dir: str = KNOWLEDGE_BASE,
    MODEL_NAME='qwen-vl-plus', temperature=0
):
    name = os.path.basename(file_path)
    name = name.split(".")[0]
    print(f"---- processing {name} ----")
    try:
        image_description = get_ocr_response(file_path, MODEL_NAME, temperature)
        # 写入本地文件
        with open(f"{tar_dir}/{name}.md", "w", encoding="utf-8") as f:
            f.write(image_description)
        yield image_description

    except Exception as e:
        print(f"---- error ----\n{e}")
        yield str(e)


def vl_chat_bot(
    path, tar_dir: str | Path = KNOWLEDGE_BASE,
    MODEL_NAME='qwen-vl-plus', temperature=0,
):
    if not os.path.exists(path):
        return "input image folder path not exits."

    if os.path.isfile(path):
        # 如果是文件直接处理
        yield from process_image(path, tar_dir, MODEL_NAME, temperature)
    elif os.path.isdir(path):
        image_files = [
            f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        total_files = len(image_files)

        for index, file in enumerate(image_files):
            file_path = os.path.join(path, file)
            image_description = next(process_image(file_path, tar_dir, MODEL_NAME, temperature))

            progress = (index + 1) / total_files * 100
            yield f"# Progress: {progress:.2f}% ({index + 1}/{total_files})\n\n" + image_description

        time.sleep(1)
        yield "All images processed."
