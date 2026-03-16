import torch
import cv2
import numpy as np
import os
import base64
from openai import OpenAI
from tslearn.clustering import TimeSeriesKMeans
from torchvision import transforms

# YOLOv7工具包
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint

# ===================== 连续动作专用参数 =====================
VIDEO_PATH = "C:\\D\\pyproject\\dongzuoshibie\\线束管路连接.mp4"
SAVE_PATH = 'C:\\D\\pyproject\\dongzuoshibie\\video_frames\\'
# 核心：连续动作分割（无静止帧专用）
SEGMENT_DURATION = 1.0  # 每段时长(秒)，连续动作固定切分
N_CLUSTERS = 4  # 预计动作数量（线束操作：拿/对准/插/固定，可调）
MIN_ACTION_DURATION = 0.5
ALI_API_KEY = "sk-1777fc7b981c4e9f90dcb10fddbd7ba8"
SAMPLE_PER_ACTION = 2
DETECT_SIZE = 640
# ============================================================

os.makedirs(SAVE_PATH, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载模型
weights = torch.load('yolov7-w6-pose.pt', map_location=device, weights_only=False)
pose_model = weights['model'].float().eval()
if torch.cuda.is_available():
    pose_model.half().to(device)

client = OpenAI(api_key=ALI_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


# 图片转base64
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"


# AI识别动作
def recognize_action(image_paths):
    if not image_paths:
        return "未获取到画面"
    content = []
    for path in image_paths[:2]:
        if os.path.exists(path):
            content.append({"type": "image_url", "image_url": {"url": image_to_base64(path)}})
    content.append({"type": "text", "text": "分析图片中的操作动作，只输出动作名称"})
    messages = [{"role": "user", "content": content}]
    completion = client.chat.completions.create(model="qwen3.5-plus", messages=messages)
    return completion.choices[0].message.content


# ===================== 提取全视频姿态序列（连续动作基础） =====================
def extract_pose_sequence():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    frame_times = []
    pose_sequence = []  # 保存所有帧的姿态向量

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_time = frame_count / fps
        frame_times.append(current_time)

        # YOLO姿态推理
        img = letterbox(frame, DETECT_SIZE, stride=64, auto=True)[0]
        img_tensor = transforms.ToTensor()(img).unsqueeze(0)
        if torch.cuda.is_available():
            img_tensor = img_tensor.half().to(device)

        with torch.no_grad():
            output, _ = pose_model(img_tensor)
        output = non_max_suppression_kpt(output, 0.25, 0.65,
                                         nc=pose_model.yaml['nc'],
                                         nkpt=pose_model.yaml['nkpt'],
                                         kpt_label=True)
        output = output_to_keypoint(output)

        # 提取归一化姿态向量
        if output.shape[0] > 0:
            kpts = output[0, 7:].reshape(-1, 3)[:, :2]
            kpts_norm = kpts / DETECT_SIZE
            pose_vec = kpts_norm.flatten()
        else:
            pose_vec = pose_sequence[-1] if len(pose_sequence) else np.zeros(34)
        pose_sequence.append(pose_vec)

        del img_tensor, output
        torch.cuda.empty_cache()
        frame_count += 1

    cap.release()
    return frame_times, np.array(pose_sequence), fps


# ===================== 连续动作聚类分割（无静止帧专用！） =====================
def split_continuous_actions(frame_times, pose_sequence, fps):
    total_frames = len(frame_times)
    segment_frames = int(SEGMENT_DURATION * fps)  # 每段帧数

    # 1. 把连续视频切成固定长度的小段
    segments = []
    segment_frames_list = []
    for i in range(0, total_frames - segment_frames, segment_frames):
        seg = pose_sequence[i:i + segment_frames]
        segments.append(seg)
        segment_frames_list.append((i, i + segment_frames))

    # 2. 时序聚类（相同动作归为一类，完全不依赖静止）
    segments = np.array(segments)
    kmeans = TimeSeriesKMeans(n_clusters=N_CLUSTERS, metric="dtw", verbose=0)
    labels = kmeans.fit_predict(segments)

    # 3. 找类别突变点 = 动作边界
    boundaries = [0]
    current_label = labels[0]
    for i, label in enumerate(labels):
        if label != current_label:
            boundaries.append(segment_frames_list[i][0])
            current_label = label
    boundaries.append(total_frames - 1)

    # 4. 生成动作片段
    actions = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if (e - s) / fps >= MIN_ACTION_DURATION:
            sample_indices = np.linspace(s, e, SAMPLE_PER_ACTION, dtype=int).tolist()
            actions.append({
                "start": round(frame_times[s], 2),
                "end": round(frame_times[e], 2),
                "indices": sample_indices,
                "imgs": []
            })

    # 保存帧
    cap = cv2.VideoCapture(VIDEO_PATH)
    for act_idx, act in enumerate(actions):
        for idx in act["indices"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                p = os.path.join(SAVE_PATH, f"action_{act_idx + 1}_{idx}.png")
                cv2.imwrite(p, frame)
                act["imgs"].append(p)
    cap.release()
    return actions


# ===================== 主流程 =====================
if __name__ == "__main__":
    print("=" * 60)
    print("✅ 连续无停顿动作专用 | 无静止帧 | 大幅度动作不切碎")
    print("=" * 60)

    # 1. 提取姿态序列
    frame_times, pose_seq, fps = extract_pose_sequence()
    # 2. 连续动作聚类分割
    actions = split_continuous_actions(frame_times, pose_seq, fps)
    # 3. AI识别
    for i, act in enumerate(actions):
        res = recognize_action(act["imgs"])
        print(f"动作{i + 1} | {act['start']}s ~ {act['end']}s | {res}")

    print(f"✅ 完成！文件保存在：{SAVE_PATH}")