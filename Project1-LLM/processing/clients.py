from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
coder_clients = dict()
ocr_clients = dict()
coder_models = ['qwen-plus', 'qwq-32b', 'qwq-plus']
ocr_models = ['qwen-vl-plus', 'qwen-vl-max', 'qvq-72b-preview']

for MODEL_NAME in coder_models:
    coder_clients[MODEL_NAME] = OpenAI(
        api_key=os.getenv('QWEN_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

coder_models.append('deepseek')
coder_clients['deepseek'] = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1",
)

for MODEL_NAME in ocr_models:
    ocr_clients[MODEL_NAME] = OpenAI(
        api_key=os.getenv('QWEN_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
