import os
import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
import argparse

# 创建保存压线图片的目录
def create_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# 加载 YOLOv8 车辆检测模型
def load_model(model_path):
    return YOLO(model_path)

def warp_perspective(frame):
    """ 将图像转换为鸟瞰图（BEV）视角 """
    h, w = frame.shape[:2]
    src_pts = np.float32([[200, h - 100], [w - 200, h - 100], [w, h - 300], [0, h - 300]])
    dst_pts = np.float32([[300, h], [w - 300, h], [w - 300, 0], [300, 0]])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    bev_frame = cv2.warpPerspective(frame, matrix, (w, h))

    return bev_frame, matrix

def merge_similar_lines(lines, threshold=30):
    """ 合并相近且方向相同的车道线 """
    if not lines:
        return []

    merged_lines = []
    lines.sort(key=lambda line: line[1])  # 按 y1 排序

    while lines:
        x1, y1, x2, y2 = lines.pop(0)
        close_lines = [(x1, y1, x2, y2)]

        for other_line in lines[:]:
            ox1, oy1, ox2, oy2 = other_line
            if abs(y1 - oy1) < threshold and abs(y2 - oy2) < threshold:
                close_lines.append((ox1, oy1, ox2, oy2))
                lines.remove(other_line)

        avg_x1 = int(np.mean([l[0] for l in close_lines]))
        avg_y1 = int(np.mean([l[1] for l in close_lines]))
        avg_x2 = int(np.mean([l[2] for l in close_lines]))
        avg_y2 = int(np.mean([l[3] for l in close_lines]))
        merged_lines.append((avg_x1, avg_y1, avg_x2, avg_y2))

    return merged_lines

def detect_lane_lines(frame):
    """ 使用改进的 Canny 边缘检测和霍夫变换进行车道线检测 """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    mask = np.zeros_like(edges)
    height, width = edges.shape
    polygon = np.array([[(100, height), (width - 100, height), (width // 2, height // 2)]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)
    lane_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) > 30:  # 过滤掉过于水平的线
                lane_lines.append((x1, y1, x2, y2))

    return merge_similar_lines(lane_lines)

def detect_vehicles(frame, model):
    """ 使用 YOLOv8 进行车辆检测 """
    results = model(frame)
    vehicles = []
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            vehicles.append((x1, y1, x2, y2))
    return vehicles

def is_vehicle_over_lane(vehicle, lane_lines, frame_height, horizontal_threshold=30):
    """ 判断车辆是否压线，只有相互接近的多个车道线都到达车辆底部的横向中段1/3才算压线 """
    x1, y1, x2, y2 = vehicle
    bottom_center_x = (x1 + x2) // 2  # 车辆底部的横向中心
    bottom_y = y2  # 车辆底部的y坐标

    # 计算车辆底部横向区域的中段1/3
    vehicle_width = x2 - x1
    left_third_x = x1 + vehicle_width // 3
    right_third_x = x2 - vehicle_width // 3

    # 只考虑车辆底部的区域，y值接近底部的位置
    bottom_y_offset = 10  # 可以根据实际需要调整此值，表示接近底部的区域

    relevant_lines = []

    for (lx1, ly1, lx2, ly2) in lane_lines:
        if ly2 != ly1:  # 确保车道线不是水平的
            # 计算车道线与车辆底部接近区域的交点
            slope = (lx2 - lx1) / (ly2 - ly1)
            lane_x_at_bottom_y = lx1 + slope * (bottom_y + bottom_y_offset - ly1)

            # 判断车道线是否在车辆底部横向区域的中段1/3处穿过
            if left_third_x < lane_x_at_bottom_y < right_third_x:
                relevant_lines.append((lx1, ly1, lx2, ly2))

    # 只有多个接近的车道线同时穿过车辆底部的横向中段1/3处才算压线
    if len(relevant_lines) > 1:
        # 计算相互接近的车道线
        merged_lines = merge_similar_lines(relevant_lines, threshold=horizontal_threshold)
        if len(merged_lines) > 1:
            return True

    return False

def compute_ssim(img1, img2):
    """ 计算两张图像的结构相似性（SSIM） """
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim_value, _ = ssim(img1_gray, img2_gray, full=True)
    return ssim_value

def main(input_path, output_dir):
    create_output_dir(output_dir)
    model = load_model("YOLOv8/yolov8n.pt")

    cap = cv2.VideoCapture(input_path)

    frame_idx = 0
    is_in_violation = False
    best_violation_frame = None
    max_overlap = 0
    violation_count = 0  # 用于控制最大保存压线的图片数量
    previous_violation_frame = None  # 上一帧保存的压线图片

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        lane_lines = detect_lane_lines(frame)
        vehicles = detect_vehicles(frame, model)

        current_violation = False
        for vehicle in vehicles:
            if is_vehicle_over_lane(vehicle, lane_lines, frame.shape[0]):
                current_violation = True
                overlap = sum(is_vehicle_over_lane(vehicle, lane_lines, frame.shape[0]) for vehicle in vehicles)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_violation_frame = frame.copy()

        if not is_in_violation and current_violation:
            max_overlap = 0  # 重新计算

        if is_in_violation and not current_violation and best_violation_frame is not None:
            if violation_count < 3:
                if previous_violation_frame is None or compute_ssim(best_violation_frame, previous_violation_frame) < 0.9:  # SSIM阈值可以调整
                    cv2.imwrite(os.path.join(output_dir, f"violation_{frame_idx}.jpg"), best_violation_frame)
                    print(f"压线帧已保存: violation_{frame_idx}.jpg")
                    violation_count += 1
                    previous_violation_frame = best_violation_frame
            best_violation_frame = None

        is_in_violation = current_violation

        # 绘制车道线
        for (x1, y1, x2, y2) in lane_lines:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制车辆边界框
        for (x1, y1, x2, y2) in vehicles:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 显示当前帧
        cv2.imshow("Detection", frame)

        # 退出条件
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect lane violations in a video.")
    parser.add_argument("input_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the output images.")
    args = parser.parse_args()

    main(args.input_path, args.output_dir)