import gradio as gr
import os
from .utils import latex_delimiters, KNOWLEDGE_BASE


def view_file(f: str):
    _code = gr.Code(visible=False)
    _markdown = gr.Markdown(visible=False)
    _json = gr.JSON(visible=False)

    try:
        file = f[-1]
    except IndexError:
        _code.visible = True
        return _code, _markdown, _json

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
        _json = gr.JSON(
            open(file, "r", encoding="utf-8").read(),
            visible=True
        )
    else:
        _markdown = gr.Markdown(
            open(file, "r", encoding="utf-8").read(),
            visible=False
        )
    return _code, _markdown, _json


def delete_file(file_paths: list[str]):
    for file_path in file_paths:
        os.remove(file_path)
    if file_paths:
        import src.inject2db
    return gr.FileExplorer(root_dir=KNOWLEDGE_BASE, ignore_glob='*.db')


def knowledge_sys():
    refresh_btn = gr.Button("刷新", variant='primary')
    file_browser = gr.FileExplorer(
        root_dir=KNOWLEDGE_BASE,
        ignore_glob='*.db',
    )
    delete_btn = gr.Button("删除文件")
    previewJson = gr.JSON(label="文件预览", visible=False)
    previewCode = gr.Code(label="代码预览", visible=False)
    previewMarkdown = gr.Markdown(label="文件信息预览")

    refresh_btn.click(
        fn=lambda: (
            gr.FileExplorer(root_dir="Temp"),
            gr.JSON(label="文件预览", visible=False),
            gr.Code(label="代码预览", visible=False),
            gr.Markdown(label="文件信息预览"),
        ),
        outputs=[file_browser, previewJson, previewCode, previewMarkdown]
    ).then(
        fn=lambda: gr.FileExplorer(
            root_dir=KNOWLEDGE_BASE,
            ignore_glob='*.db',
        ),
        outputs=file_browser
    )

    file_browser.change(
        fn=view_file,
        inputs=file_browser,
        outputs=[previewCode, previewMarkdown, previewJson]
    )

    delete_btn.click(
        fn=delete_file,
        inputs=file_browser,
        outputs=file_browser,
    )
