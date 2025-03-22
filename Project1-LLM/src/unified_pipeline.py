import os
from pathlib import Path
from src.utils import KNOWLEDGE_BASE, formulas2code, code2json, TEMP_FOLDER
from processing import (
    vl_chat_bot, code_chat_from_formulas_dir, json_chat_from_codes_dir,
    ocr_clients, coder_clients
)
import shutil
from .inject2db import inject2db


class UnifiedPipeline:
    def __init__(self, input_path: list[str], ocr_model, coder_model, ocr_temperature, coder_temperature):
        self.input_path = input_path
        self.temp_dir = Path(TEMP_FOLDER)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.ocr_name = ocr_model
        self.coder_name = coder_model
        self.ocr_temperature = ocr_temperature
        self.coder_temperature = coder_temperature
        print(f"OCR model: {ocr_model}")
        print(f"Coder model: {coder_model}")
        print(f"OCR temperature: {ocr_temperature}")
        print(f"Coder temperature: {coder_temperature}")

    def process_image(self):
        """调用formula_from_images_dir逻辑"""
        return vl_chat_bot(
            self.temp_dir, self.temp_dir,
            self.ocr_name, self.ocr_temperature
        )

    def generate_code(self):
        """调用code_from_formulas_dir逻辑"""
        return code_chat_from_formulas_dir(
            self.temp_dir, formulas2code, self.temp_dir,
            self.coder_name, self.coder_temperature
        )

    def generate_json(self):
        """调用json_from_codes_dir逻辑"""
        return json_chat_from_codes_dir(
            self.temp_dir, code2json, self.temp_dir,
            self.coder_name, self.coder_temperature
        )

    def inject_db(self):
        """调用inject2db逻辑"""
        inject2db()
        return "数据注入成功！"

    def cp_temp2kb(self):
        """将临时目录中的文件复制到知识库目录"""
        for filename in os.listdir(self.temp_dir):
            if not filename.endswith(('py', 'md', 'json')):
                continue
            if os.path.exists(os.path.join(KNOWLEDGE_BASE, filename)):
                os.remove(os.path.join(KNOWLEDGE_BASE, filename))
            shutil.move(os.path.join(self.temp_dir, filename), os.path.join(KNOWLEDGE_BASE, filename))
        yield "临时目录中的文件已复制至知识库目录！"

    def execute_pipeline(self):
        yield from self.mv2temp()
        yield "开始处理流程..."
        yield from self.process_image()
        yield from self.generate_code()
        yield from self.generate_json()
        yield from self.cp_temp2kb()
        yield from self.inject_db()
        yield "流程完成！结果已存储至知识库！"
        self.clearTemp()

    def mvFile2temp(self, file_path: str):
        file_name = os.path.basename(file_path)
        shutil.move(file_path, os.path.join(self.temp_dir, file_name))

    def mv2temp(self):
        for file_path in self.input_path:
            self.mvFile2temp(file_path)
        yield "图片读取到temp目录！"

    def clearTemp(self):
        shutil.rmtree(self.temp_dir)
