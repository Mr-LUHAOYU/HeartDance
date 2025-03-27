from Ultra_Fast_Lane_Detection import Vehicle_lane_detection
from Chinese_license_plate_detection_recognition import test_plate_dete

if __name__ == '__main__':
    video_path = '32010120210610092038370.mp4'# 输入你的视频路径
    image_path = 'image/32010120210610092038370_orignal.jpg'
    Vehicle_lane_detection.process_video(video_path)
    test_plate_dete.detect_plate('image_path')