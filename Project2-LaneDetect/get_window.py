import tkinter as tk
from tkinter import filedialog, messagebox
import os
import subprocess
import threading


class VideoProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("视频转图片工具")

        # 视频文件路径变量
        self.video_path = tk.StringVar()
        # 输出目录路径变量
        self.output_path = tk.StringVar()

        # 创建界面组件
        self.create_widgets()

    def create_widgets(self):
        # 视频文件选择部分
        tk.Label(self.root, text="视频文件：").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        tk.Entry(self.root, textvariable=self.video_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self.root, text="选择文件", command=self.select_video_file).grid(row=0, column=2, padx=5, pady=5)

        # 输出目录选择部分
        tk.Label(self.root, text="输出目录：").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        tk.Entry(self.root, textvariable=self.output_path, width=50).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(self.root, text="选择目录", command=self.select_output_dir).grid(row=1, column=2, padx=5, pady=5)

        # 处理按钮
        self.process_btn = tk.Button(self.root, text="开始转换", command=self.start_processing)
        self.process_btn.grid(row=2, column=1, pady=10)

    def select_video_file(self):
        filetypes = [
            ("视频文件", "*.mp4 *.avi *.mov *.mkv"),
            ("所有文件", "*.*")
        ]
        filename = filedialog.askopenfilename(title="选择视频文件", filetypes=filetypes)
        if filename:
            self.video_path.set(filename)

    def select_output_dir(self):
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.output_path.set(directory)

    def start_processing(self):
        input_path = self.video_path.get()
        output_dir = self.output_path.get()

        # 输入验证
        if not input_path or not output_dir:
            messagebox.showerror("错误", "请先选择视频文件和输出目录")
            return

        # 创建输出目录
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                messagebox.showerror("错误", f"无法创建输出目录：{str(e)}")
                return

        # 禁用处理按钮
        self.process_btn.config(state=tk.DISABLED, text="处理中...")

        # 启动处理线程
        threading.Thread(target=self.call_external_program, args=(input_path, output_dir), daemon=True).start()

    def call_external_program(self, input_path, output_dir):
        try:
            # 调用外部 Python 程序
            # 假设外部程序是 `video_to_images.py`，接受两个参数：输入视频路径和输出目录
            external_program = "python demo.py"
            command = f"{external_program} {input_path} {output_dir}"

            # 使用 subprocess 调用外部程序
            result = subprocess.run(command, shell=True, capture_output=True, text=True)

            if result.returncode == 0:
                messagebox.showinfo("完成", f"外部程序执行成功！\n输出目录：{output_dir}")
            else:
                messagebox.showerror("错误", f"外部程序执行失败：\n{result.stderr}")

        except Exception as e:
            messagebox.showerror("错误", f"调用外部程序时发生错误：\n{str(e)}")

        finally:
            # 恢复按钮状态
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL, text="开始转换"))


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessorApp(root)
    root.mainloop()