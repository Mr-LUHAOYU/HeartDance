from data.constant import tusimple_row_anchor
import scipy
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from model.model import parsingNet
from torchvision import transforms
from PIL import Image
from torchvision.ops import nms
import os
from ultralytics.utils import LOGGER
import subprocess

LOGGER.setLevel(50)
# 初始化 GPU 设备
device = torch.device("cpu")

# 载入 UFLD 车道线检测模型
# 修改车道线模型参数配置以匹配Tusimple预训练权重
cls_num_per_lane = 56  # 修正为Tusimple数据集正确的分割点数量

# 确保模型配置与预训练权重匹配
ufld_model = parsingNet(pretrained=False, backbone='18', 
                      cls_dim=(cls_num_per_lane + 1, 4, 2),  # 总维度为57x4x2（56个分割点+1存在性分类）
                      use_aux=False).to(device)

# 只加载匹配的权重
checkpoint = torch.load("checkpoints/tusimple_18.pth", map_location=device)
state_dict = checkpoint["model"]

# # 移除不匹配的 cls.2 层
# filtered_state_dict = {k: v for k, v in state_dict.items() if "cls.2" not in k}
# ufld_model.load_state_dict(filtered_state_dict, strict=False)

ufld_model.eval()

# YOLOv8 车辆检测模型
yolo_model = YOLO("yolov8n.pt",verbose=False)

# 图像预处理
img_transforms = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
def detect_plate(image_path):
    """调用车牌识别脚本并返回车牌号"""
    command = [
        "python", "detect_plate.py",
        "--detect_model", "../Chinese_license_plate_detection_recognition/weights/plate_detect.pt",
        "--rec_model", "../Chinese_license_plate_detection_recognition/weights/plate_rec_color.pth",
        "--image_path", image_path,
        "--output", "../image"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print("车牌识别结果:", result.stdout)
    return result.stdout.strip()
def detect_lanes(frame):
    """检测车道线并转换为像素坐标"""
    img_w, img_h = frame.shape[1], frame.shape[0]
    img = cv2.resize(frame, (800, 288))  
    img = Image.fromarray(img)  # 转换为 PIL.Image
    img = img_transforms(img).unsqueeze(0).to(device)  

    with torch.no_grad():
        output = ufld_model(img)
    
    col_sample = np.linspace(0, 800 - 1, cls_num_per_lane)  # 采样列数
    col_sample_w = col_sample[1] - col_sample[0]
    row_anchor = tusimple_row_anchor  # 预定义的行锚点
    
    out_j = output[0].cpu().numpy()
    out_j = out_j[:, ::-1, :]  # 翻转列索引
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)  # 计算softmax概率
    idx = np.arange(cls_num_per_lane) + 1  # 计算索引
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)  # 计算期望列位置
    
    out_j = np.argmax(out_j, axis=0)  # 获取最大概率索引
    loc[out_j == cls_num_per_lane] = 0  # 过滤无效点
    out_j = loc  
    
    processed_lanes = []
    for i in range(out_j.shape[1]):  # 遍历车道线
        lane = []
        if np.sum(out_j[:, i] != 0) > 2:  # 过滤无效车道
            for k in range(out_j.shape[0]):  # 遍历行坐标
                if out_j[k, i] > 0:
                    x = int(out_j[k, i] * col_sample_w * img_w / 800) - 1
                    y = int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1
                    lane.append((x, y))
        processed_lanes.append(lane)
    
    return processed_lanes

def detect_vehicles(frame):
    """检测车辆"""
    results = yolo_model(frame)
    if len(results) == 0 or results[0].boxes is None:
        print("未检测到车辆！")
        return np.array([])
    boxes = results[0].boxes.xyxy.cpu().numpy()  # 获取车辆检测框
    scores = results[0].boxes.conf.cpu().numpy()  # 获取置信度分数

    # 转换为 PyTorch Tensor
    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)

    # 运行 NMS
    keep = nms(boxes, scores, iou_threshold=0.5)

    return boxes[keep].cpu().numpy()  # 确保返回 NumPy 数组

def is_vehicle_crossing(vehicle_box, lane_lines):
    """判断车辆是否压线"""
    vx1, vy1, vx2, vy2 = vehicle_box
    cx, cy = (vx1 + vx2) // 2, vy2  # 车辆底部中心点

    for lane in lane_lines:
        lane_xs = [x for x, y in lane if abs(y - cy) < 430]  # 取 y 接近 cy 的 x 值
        if any(vx1 <= x <= vx2 for x in lane_xs):  # 车道线与车辆有交叉
            print(f"🚗 车辆压线: 车中心 {cx, cy}, 车道线 x: {lane_xs}")
            return True
    return False

def adaptive_affine_transform(src_img, dst_img):
    # 替换为ORB特征检测
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(src_img, None)
    kp2, des2 = orb.detectAndCompute(dst_img, None)

    # 使用BruteForce Hamming匹配器
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 过滤匹配点
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:int(len(matches)*0.9)]  # 取前90%的优质匹配

    if len(good_matches) < 4:
        return None, None

    # 提取匹配点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    # 增加RANSAC迭代次数和阈值调整
    affine_matrix, mask = cv2.estimateAffine2D(src_pts, dst_pts, 
                                              method=cv2.RANSAC, 
                                              ransacReprojThreshold=5.0,
                                              maxIters=2000)

    if affine_matrix is None or np.linalg.det(affine_matrix[:2,:2]) < 1e-6:
        return None, None

    # 使用双线性插值防止变形失真
    transformed_img = cv2.warpAffine(src_img, affine_matrix, 
                                    (dst_img.shape[1], dst_img.shape[0]),
                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return transformed_img, affine_matrix

def bird_view_transform(frame, reference_frame):
    """生成俯视图（鸟瞰图）"""
    transformed_img, _ = adaptive_affine_transform(frame, reference_frame)
    if transformed_img is None:
        print("仿射变换失败，使用原图")
        return frame
    return transformed_img

def draw_annotations(img, lane_lines, boxes, is_top_view=False):
    """在图像（原始帧或鸟瞰图）上标注车道线和车辆框"""
    if img is None:
        return None
    img = img.copy()

    # 画车道线
    for lane in lane_lines:
        for (x, y) in lane:
            if x > 0:
                cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)  # 绿色车道线

    # 画车辆框
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        color = (0, 0, 255) if is_vehicle_crossing(box, lane_lines) else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    return img

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, ref_frame = cap.read()
    if not ret:
        print("无法读取视频")
        return
    if not os.path.exists("image"):
        os.makedirs("image")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 生成鸟瞰图
        bird_view = bird_view_transform(frame, ref_frame)
        
        # 打印生成的鸟瞰图
        # cv2.imshow('Bird View', bird_view)
        # cv2.waitKey(3000)
        
        # 检测车道线与车辆框
        lane_lines = detect_lanes(bird_view)
        # print(lane_lines)
        vehicle_boxes = detect_vehicles(bird_view)

        # 打印标注好的鸟瞰图
        annotated_bird = draw_annotations(bird_view, lane_lines, vehicle_boxes)
        # cv2.imshow('Annotated Bird View', annotated_bird)
        # cv2.waitKey(3000)
        
        # 检测是否压线
        crossing_detected = False
        for box in vehicle_boxes:
            if is_vehicle_crossing(box, lane_lines):
                crossing_detected = True
                break
        if crossing_detected:
            print('进入')
            frame_count += 1
            timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            # 生成标注图像
            annotated_bird = draw_annotations(bird_view, lane_lines, vehicle_boxes)
            filename = os.path.splitext(os.path.basename(video_path))[0]
            # 保存图像
            cv2.imwrite(f'image/{video_path}_original.jpg', filename)
            cv2.imwrite(f'image/{video_path}_bird.jpg', filename)
            path = f'../image/{filename}_orignal.jpg'
            print(f"保存压线帧：{video_path}",filename)
            detect_plate(path)
        # 处理完第一帧后退出循环
        if frame_count == 0:
            break
    cap.release()
    cv2.destroyAllWindows()

# 运行
process_video("../车压线视频/32010120210610092038370.mp4")