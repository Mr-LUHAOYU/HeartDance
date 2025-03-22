import subprocess
import os

def detect_plate(image_path):
    """调用车牌识别脚本并返回车牌号"""
    script_path = "detect_plate.py"
    detect_model = "weights/plate_detect.pt"
    rec_model = "weights/plate_rec_color.pth"
    
    # 确保路径兼容不同操作系统
    image_path = os.path.abspath(image_path)

    command = [
        "python", script_path,
        "--detect_model", detect_model,
        "--rec_model", rec_model,
        "--image_path", image_path,
        "--output", "image"
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        if not output:
            raise ValueError("未检测到车牌信息")
        print("车牌识别结果:", output)
        return output
    except subprocess.CalledProcessError as e:
        print("调用 detect_plate.py 失败:", e.stderr)
        return None
    except Exception as e:
        print("发生错误:", str(e))
        return None

detect_plate("../image/32010120210610092038370_orignal.jpg")
