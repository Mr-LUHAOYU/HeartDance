import gradio as gr
from processing.clients import coder_models, ocr_models
import json

def settings_init():
    config = {
        "ocr_model": "qwen-vl-plus",
        "coder_model": "qwen-plus",
        "ocr_temperature": 0,
        "coder_temperature": 0,
    }
    settings = gr.Textbox(
        visible=False,
        value=str(config),
        label="Settings",
    )
    return settings


def update_settings(ocr_model, coder_model, ocr_temperature, coder_temperature):
    config = {
        "ocr_model": ocr_model,
        "coder_model": coder_model,
        "ocr_temperature": ocr_temperature,
        "coder_temperature": coder_temperature,
    }
    output_log = json.dumps(config)
    config = str(config)
    return config, "设置更新成功！\n```json\n" + output_log + "\n```\n"


def settings_interface(settings: gr.Textbox):
    config = eval(settings.value)

    with gr.Row():
        ocr_model = gr.Dropdown(
            label="OCR Model",
            choices=ocr_models,
        )
        ocr_temperature = gr.Slider(
            label="OCR Temperature",
            minimum=0,
            maximum=1,
            step=0.1,
            value=config["ocr_temperature"],
        )

    with gr.Row():
        coder_model = gr.Dropdown(
            label="Coder Model",
            choices=coder_models,
        )

        coder_temperature = gr.Slider(
            label="Coder Temperature",
            minimum=0,
            maximum=1,
            step=0.1,
            value=config["coder_temperature"],
        )

    update_btn = gr.Button("Update Settings")
    logging_btn = gr.Markdown("Show Logs")
    update_btn.click(
        fn=update_settings,
        inputs=[ocr_model, coder_model, ocr_temperature, coder_temperature],
        outputs=[settings, logging_btn],
    )
