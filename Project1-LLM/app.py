import gradio as gr
from src import file_upload, knowledge_sys, Q8A, settings_interface, settings_init

css = """
#progress-bar { margin: 20px 0; }
.file-preview { max-height: 300px; overflow-y: auto; }
.chatbot { min-height: 400px; }
"""


def create_interface():
    with gr.Blocks(css=css) as app:
        gr.Markdown("# 智能公式处理平台")
        settings = settings_init()

        with gr.Tabs():
            with gr.Tab("处理流程"):
                file_upload(settings)

            with gr.Tab("知识库"):
                knowledge_sys()

            with gr.Tab("智能问答"):
                Q8A()

            with gr.Tab("设置"):
                settings_interface(settings)

    return app


if __name__ == "__main__":
    create_interface().launch(server_port=7860, share=True)
