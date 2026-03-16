import torch
import cv2
import numpy as np
import os
import base64
from openai import OpenAI
from scipy.signal import savgol_filter, find_peaks
from torchvision import transforms

# YOLOv7工具包
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint

# ===================== 全局参数 =====================
VIDEO_PATH = "C:\\D\\pyproject\\dongzuoshibie\\线束管路连接.mp4"
SAVE_PATH = 'C:\\D\\pyproject\\dongzuoshibie\\video_frames\\'
# 姿态分割核心参数
SMOOTH_WINDOW = 11          # 平滑窗口（奇数，越大越抗噪声）
SIM_GRAD_THRESH = 0.01      # 姿态突变梯度阈值（越小越灵敏）
MIN_ACTION_DURATION = 0.6   # 最小动作时长（秒）
ALI_API_KEY = "sk-1777fc7b981c4e9f90dcb10fddbd7ba8"
SAMPLE_PER_ACTION = 2
DETECT_SIZE = 640
# ====================================================

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

# ===================== 核心：提取姿态特征 + 计算相似度突变 =====================
def get_pose_change_features():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    frame_times = []
    pose_vectors = []  # 保存每帧的姿态向量

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_time = frame_count / fps
        frame_times.append(current_time)

        # YOLO Pose预处理+推理（和你原代码一致）
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

        # 提取姿态向量（归一化后展平）
        if output.shape[0] > 0:
            kpts = output[0, 7:].reshape(-1, 3)[:, :2]  # 取COCO 17个关键点的(x,y)
            kpts_norm = kpts / DETECT_SIZE  # 归一化到[0,1]，消除尺度影响
            pose_vec = kpts_norm.flatten()   # 转成一维向量：(34,)
        else:
            # 没检测到人体时，用上一帧填充（避免中断）
            pose_vec = pose_vectors[-1].copy() if len(pose_vectors) > 0 else np.zeros(34)
        pose_vectors.append(pose_vec)

        # 释放显存
        del img_tensor, output
        torch.cuda.empty_cache()
        frame_count += 1
    cap.release()

    # ---------------------- 计算姿态相似度 ----------------------
    pose_vectors = np.array(pose_vectors)
    similarity = []
    for i in range(1, len(pose_vectors)):
        vec1 = pose_vectors[i-1]
        vec2 = pose_vectors[i]
        # 余弦相似度：值越高，姿态越相似
        sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
        similarity.append(sim)
    # 第一帧用第二帧的相似度填充，保持长度一致
    similarity = [similarity[0]] + similarity

    # ---------------------- 平滑 + 计算梯度（找突变） ----------------------
    similarity_smooth = savgol_filter(similarity, window_length=SMOOTH_WINDOW, polyorder=2)
    sim_grad = np.gradient(similarity_smooth)  # 梯度：负峰值=相似度骤降=姿态突变

    return frame_times, similarity_smooth, sim_grad, fps

# ===================== 基于姿态突变的动作分割 =====================
def split_actions_by_pose_change(frame_times, similarity_smooth, sim_grad, fps):
    total_frames = len(similarity_smooth)
    # 找「相似度骤降」的点：梯度的负峰值（即sim_grad的极小值）
    peaks, _ = find_peaks(-sim_grad, height=SIM_GRAD_THRESH)  # 取负号，把极小值转成峰值
    boundaries = [0] + peaks.tolist() + [total_frames - 1]
    boundaries = sorted(list(set(boundaries)))  # 去重+排序

    actions = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        # 过滤过短的无效动作
        if (e - s) / fps >= MIN_ACTION_DURATION:
            sample_indices = np.linspace(s, e, SAMPLE_PER_ACTION, dtype=int).tolist()
            actions.append({
                "start": round(frame_times[s], 2),
                "end": round(frame_times[e], 2),
                "indices": sample_indices,
                "imgs": []
            })

    # 保存采样帧（和你原代码一致）
    cap = cv2.VideoCapture(VIDEO_PATH)
    for act_idx, act in enumerate(actions):
        for frame_idx in act["indices"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            save_path = os.path.join(SAVE_PATH, f"action_{act_idx + 1}_frame_{frame_idx}.png")
            cv2.imwrite(save_path, frame)
            act["imgs"].append(save_path)
    cap.release()

    return actions

# ===================== 主流程 =====================
if __name__ == "__main__":
    print("=" * 60)
    print("✅ 连续动作专用：基于姿态突变分割 | 无静止帧也能切分")
    print("=" * 60)

    # 1. 提取姿态特征和突变信息
    frame_times, similarity_smooth, sim_grad, fps = get_pose_change_features()
    # 2. 按姿态突变分割动作
    action_list = split_actions_by_pose_change(frame_times, similarity_smooth, sim_grad, fps)

    # 3. AI识别每个动作
    for i, act in enumerate(action_list):
        action_name = recognize_action(act["imgs"])
        print(f"动作{i + 1} | {act['start']}s ~ {act['end']}s | 识别结果：{action_name}")

    print("=" * 60)
    print(f"✅ 采样帧已保存至：{SAVE_PATH}")
    print("处理完成！")