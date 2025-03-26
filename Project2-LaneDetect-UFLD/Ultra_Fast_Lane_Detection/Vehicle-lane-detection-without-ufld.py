import cv2
import numpy as np
from ultralytics import YOLO

def adaptive_affine_transform(src_img, dst_img):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(src_img, None)
    kp2, des2 = sift.detectAndCompute(dst_img, None)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good_matches) < 3:
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    affine_matrix, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if affine_matrix is None:
        return None, None

    transformed_img = cv2.warpAffine(src_img, affine_matrix, (dst_img.shape[1], dst_img.shape[0]))
    return transformed_img, affine_matrix

def detect_lane(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)
    return lines

def detect_vehicle(frame, model):
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    return boxes

def draw_annotations(frame, lane_lines, vehicle_boxes):
    annotated_frame = frame.copy()
    
    # 画车道线
    if lane_lines is not None:
        for line in lane_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # 画车辆框
    for box in vehicle_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return annotated_frame

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    model = YOLO("yolov8n.pt")
    frame_id = 0
    crossing_detected = False
    pre_cross_frame, cross_frame, post_cross_frame = None, None, None
    pre_cross_top, cross_top, post_cross_top = None, None, None
    
    ret, ref_frame = cap.read()
    if not ret:
        print("无法读取视频")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        transformed_frame, affine_matrix = adaptive_affine_transform(frame, ref_frame)
        if transformed_frame is None:
            continue

        lane_lines = detect_lane(transformed_frame)
        vehicle_boxes = detect_vehicle(transformed_frame, model)
        annotated_top_view = draw_annotations(transformed_frame, lane_lines, vehicle_boxes)

        for box in vehicle_boxes:
            if lane_lines is not None and any(
                y1 < (box[3]) < y2 and x1 < (box[0] + box[2]) // 2 < x2 for line in lane_lines for x1, y1, x2, y2 in [line[0]]
            ):
                if not crossing_detected:
                    pre_cross_frame, pre_cross_top = frame.copy(), annotated_top_view.copy()
                crossing_detected = True
                cross_frame, cross_top = frame.copy(), annotated_top_view.copy()
            elif crossing_detected:
                post_cross_frame, post_cross_top = frame.copy(), annotated_top_view.copy()
                break
        
        if post_cross_frame is not None:
            break

    cap.release()
    
    # 保存关键帧及俯瞰图
    if pre_cross_frame is not None:
        cv2.imwrite("image\\pre_cross.jpg", pre_cross_frame)
        cv2.imwrite("image\\pre_cross_top.jpg", pre_cross_top)
    if cross_frame is not None:
        cv2.imwrite("image\\cross.jpg", cross_frame)
        cv2.imwrite("image\\cross_top.jpg", cross_top)
    if post_cross_frame is not None:
        cv2.imwrite("image\\post_cross.jpg", post_cross_frame)
        cv2.imwrite("image\\post_cross_top.jpg", post_cross_top)

# 运行处理
process_video("..\\车压线视频\\32010120210610092038370.mp4")
