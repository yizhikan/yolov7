import torch
import cv2
import numpy as np
import os
import base64
from openai import OpenAI
from scipy.signal import find_peaks
from torchvision import transforms

# YOLOv7工具包
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint

# ===================== 全局参数 =====================
VIDEO_PATH = "C:\\D\\pyproject\\dongzuoshibie\\线束管路连接.mp4"
CROP_SIZE = 200
SAVE_PATH = 'C:\\D\\pyproject\\dongzuoshibie\\video_frames\\'
KP_CONF_THRESH = 0.5
MOTION_THRESH = 3.0
MIN_ACTION_DURATION = 0.5
ALI_API_KEY = "sk-1777fc7b981c4e9f90dcb10fddbd7ba8"
SAMPLE_PER_ACTION = 2
DETECT_SIZE = 640
# COCO17关键点：手腕索引
LEFT_WRIST = 9
RIGHT_WRIST = 10
# ====================================================

os.makedirs(SAVE_PATH, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载模型
weights = torch.load('yolov7-w6-pose.pt', map_location=device, weights_only=False)
pose_model = weights['model'].float().eval()
if torch.cuda.is_available():
    pose_model.half().to(device)

client = OpenAI(api_key=ALI_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"

def recognize_action(image_paths):
    if not image_paths:
        return "未检测到有效手腕"
    content = []
    for path in image_paths[:4]:
        if os.path.exists(path):
            content.append({"type": "image_url", "image_url": {"url": image_to_base64(path)}})
    content.append({"type": "text", "text": "分析手部动作，只输出动作名称"})
    messages = [{"role": "user", "content": content}]
    completion = client.chat.completions.create(model="qwen3.5-plus", messages=messages)
    return completion.choices[0].message.content

# ===================== 修复：正确解析YOLOv7-Pose关键点 =====================
def get_wrist_motion():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    frame_times = []
    wrist_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_time = frame_count / fps
        frame_times.append(current_time)

        # 预处理
        img = letterbox(frame, DETECT_SIZE, stride=64, auto=True)[0]
        img_tensor = transforms.ToTensor()(img).unsqueeze(0)
        if torch.cuda.is_available():
            img_tensor = img_tensor.half().to(device)

        # 推理
        with torch.no_grad():
            output, _ = pose_model(img_tensor)
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=pose_model.yaml['nc'], nkpt=17, kpt_label=True)
        output = output_to_keypoint(output)

        # 初始化左右手
        lx, ly, lconf = 0.0, 0.0, 0.0
        rx, ry, rconf = 0.0, 0.0, 0.0

        # ✅ 修复维度错误：output是2维 [n, 57]，17关键点×3=51个值
        if output.shape[0] > 0:
            # 左手腕 9 → 9*3=27
            lx = float(output[0, 7 + LEFT_WRIST * 3])
            ly = float(output[0, 7 + LEFT_WRIST * 3 + 1])
            lconf = float(output[0, 7 + LEFT_WRIST * 3 + 2])
            # 右手腕 10 → 10*3=30
            rx = float(output[0, 7 + RIGHT_WRIST * 3])
            ry = float(output[0, 7 + RIGHT_WRIST * 3 + 1])
            rconf = float(output[0, 7 + RIGHT_WRIST * 3 + 2])

        # 计算运动
        motion = 0
        if frame_count > 0 and len(wrist_data) > 0:
            prev_lx, prev_ly = wrist_data[-1][0], wrist_data[-1][1]
            motion = np.sqrt((lx - prev_lx) ** 2 + (ly - prev_ly) ** 2)

        wrist_data.append([lx, ly, lconf, rx, ry, rconf, motion])

        # 清显存
        del img_tensor, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        frame_count += 1

    cap.release()
    return frame_times, wrist_data, fps

# ===================== 动作分割与裁剪 =====================
def split_and_sample_frames(frame_times, wrist_data, fps):
    total_frames = len(frame_times)
    motions = np.array([d[6] for d in wrist_data])
    peaks, _ = find_peaks(np.diff(motions), height=MOTION_THRESH)
    boundaries = [0] + peaks.tolist() + [total_frames - 1]
    boundaries = [b for b in boundaries if 0 <= b < total_frames]

    actions = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if (e - s) / fps >= MIN_ACTION_DURATION:
            sample_indices = list(np.linspace(s, e, SAMPLE_PER_ACTION, dtype=int))
            actions.append({
                "start": round(frame_times[s], 2),
                "end": round(frame_times[e], 2),
                "indices": sample_indices,
                "imgs": []
            })

    cap = cv2.VideoCapture(VIDEO_PATH)
    half_size = CROP_SIZE // 2
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    for act_idx, act in enumerate(actions):
        for frame_idx in act["indices"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            lx, ly, lconf, rx, ry, rconf = wrist_data[frame_idx][:6]

            # 左手腕裁剪
            if lconf >= KP_CONF_THRESH:
                x1 = max(0, int(lx - half_size))
                y1 = max(0, int(ly - half_size))
                x2 = min(img_w, int(lx + half_size))
                y2 = min(img_h, int(ly + half_size))
                if x2 > x1 and y2 > y1:
                    save_path = os.path.join(SAVE_PATH, f"act{act_idx+1}_frame{frame_idx}_left.png")
                    cv2.imwrite(save_path, frame[y1:y2, x1:x2])
                    act["imgs"].append(save_path)

            # 右手腕裁剪
            if rconf >= KP_CONF_THRESH:
                x1 = max(0, int(rx - half_size))
                y1 = max(0, int(ry - half_size))
                x2 = min(img_w, int(rx + half_size))
                y2 = min(img_h, int(ry + half_size))
                if x2 > x1 and y2 > y1:
                    save_path = os.path.join(SAVE_PATH, f"act{act_idx+1}_frame{frame_idx}_right.png")
                    cv2.imwrite(save_path, frame[y1:y2, x1:x2])
                    act["imgs"].append(save_path)

    cap.release()
    return actions

# ===================== 主流程 =====================
if __name__ == "__main__":
    print("=" * 60)
    print("左右手腕检测 | 2帧/动作 | 最多4张图/动作 | 低显存运行")
    print("=" * 60)

    frame_times, wrist_data, fps = get_wrist_motion()
    action_list = split_and_sample_frames(frame_times, wrist_data, fps)

    for i, act in enumerate(action_list):
        action_name = recognize_action(act["imgs"])
        print(f"动作{i+1} | {act['start']}s ~ {act['end']}s | 图片数:{len(act['imgs'])} | {action_name}")

    print("=" * 60)
    print("处理完成！")