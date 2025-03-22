from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
coder_clients = dict()
ocr_clients = dict()
coder_models = ['qwen-plus']
ocr_models = ['qwen-vl-plus']

for MODEL_NAME in coder_models:
    coder_clients[MODEL_NAME] = OpenAI(
        api_key=os.getenv('QWEN_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

for MODEL_NAME in ocr_models:
    ocr_clients[MODEL_NAME] = OpenAI(
        api_key=os.getenv('QWEN_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )