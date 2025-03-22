import gradio as gr
import os
from .utils import latex_delimiters, KNOWLEDGE_BASE

KEY_COUNT = 1


def view_file(f: str):
    _code = gr.Code(visible=False)
    _markdown = gr.Markdown(visible=False)
    _text = gr.Textbox(visible=False)

    try:
        file = f[-1]
    except IndexError:
        _code.visible = True
        return _code, _markdown, _text

    if file.endswith(".py"):
        _code = gr.Code(
            open(file, "r", encoding="utf-8").read(),
            language="python",
            visible=True
        )
    elif file.endswith(".md"):
        _markdown = gr.Markdown(
            open(file, "r", encoding="utf-8").read(),
            visible=True,
            latex_delimiters=latex_delimiters
        )
    elif file.endswith(".json"):
        _code = gr.JSON(
            open(file, "r", encoding="utf-8").read(),
            visible=True
        )
    else:
        _text = gr.Textbox(
            open(file, "r", encoding="utf-8").read(),
            visible=False
        )
    return _code, _markdown, _text


def delete_file(file_paths: list[str]):
    global KEY_COUNT
    for file_path in file_paths:
        os.remove(file_path)
    if file_paths:
        from .inject2db import read_all_md_files_from_knowledge_base
    KEY_COUNT += 1
    return gr.FileExplorer(root_dir=KNOWLEDGE_BASE, ignore_glob='*.db', key=KEY_COUNT)


def refresh():
    global KEY_COUNT
    KEY_COUNT += 1
    return gr.FileExplorer(root_dir=KNOWLEDGE_BASE, ignore_glob='*.db', key=KEY_COUNT)


def knowledge_sys():
    refresh_btn = gr.Button("刷新", variant='primary')
    file_browser = gr.FileExplorer(
        root_dir=KNOWLEDGE_BASE,
        ignore_glob='*.db',
        key=KEY_COUNT
    )
    delete_btn = gr.Button("删除文件")
    previewText = gr.Textbox(label="文件预览")
    previewCode = gr.Code(label="代码预览", visible=False)
    previewMarkdown = gr.Markdown(label="文件信息预览", visible=False)

    refresh_btn.click(
        fn=refresh,
        outputs=file_browser
    )

    file_browser.change(
        fn=view_file,
        inputs=file_browser,
        outputs=[previewCode, previewMarkdown, previewText]
    )

    delete_btn.click(
        fn=delete_file,
        inputs=file_browser,
        outputs=file_browser,
    )
