import gradio as gr
from .funcation_call import get_funcation_call_response
from .search import search
from .utils import latex_delimiters


def ai_chat_response(message: str, history) -> str:
    # 调用搜索函数获取最相关的文档
    search_results = search(message, top_k=1)
    if not search_results:
        return "未找到相关知识。"

    # 调用函数调用响应函数生成回答
    response = get_funcation_call_response(message)
    return response


def Q8A():
    gr.Markdown("## 智能问答系统")
    chat_interface = gr.ChatInterface(
        fn=ai_chat_response,
        chatbot=gr.Chatbot(
            label='对话历史',
            elem_classes="chatbot",
            latex_delimiters=latex_delimiters,
        ),
        textbox=gr.Textbox(placeholder="输入您的问题..."),
    )
