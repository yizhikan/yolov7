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
SAVE_PATH = 'C:\\D\\pyproject\\dongzuoshibie\\video_frames\\'
MOTION_THRESH = 10.0
MIN_ACTION_DURATION = 0.5
ALI_API_KEY = "sk-1777fc7b981c4e9f90dcb10fddbd7ba8"
SAMPLE_PER_ACTION = 2
DETECT_SIZE = 640
# ===================== 【修改1】新增代码1的截图参数 =====================
CROP_SIZE = 200  # 严格沿用代码1的截取边长
KP_CONF_THRESH = 0.5  # 严格沿用代码1的关键点置信度阈值
# ====================================================================

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


# 计算运动幅度（修复运动值计算 + 变量名错误）
def get_wrist_motion():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    frame_times = []
    motion_data = []
    prev_center_x, prev_center_y = 0.0, 0.0  # 上一帧坐标

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
        output = non_max_suppression_kpt(output, 0.25, 0.65,
                                         nc=pose_model.yaml['nc'],
                                         nkpt=pose_model.yaml['nkpt'],
                                         kpt_label=True)
        output = output_to_keypoint(output)

        motion = 0.0
        if output.shape[0] > 0:
            # 取右腕坐标计算运动（简单稳定）
            cx = output[0, 7 + 3 * 10]
            cy = output[0, 8 + 3 * 10 + 1]
            if frame_count > 0:
                motion = np.sqrt((cx - prev_center_x) ** 2 + (cy - prev_center_y) ** 2)
            prev_center_x, prev_center_y = cx, cy

        motion_data.append(motion)

        # 释放显存
        del img_tensor, output
        torch.cuda.empty_cache()
        frame_count += 1

    cap.release()
    return frame_times, motion_data, fps


# ===================== 【修改2】重写函数：动作分割 + 保存全图+手腕截图（严格沿用代码1逻辑） =====================
def split_and_sample_frames(frame_times, motion_data, fps):
    total_frames = len(frame_times)
    motions = np.array(motion_data)

    # 动作分割（原有逻辑不变）
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

    # 【核心修改】保存全图 + 严格按照代码1逻辑截取手腕干净截图
    cap = cv2.VideoCapture(VIDEO_PATH)
    half_size = CROP_SIZE // 2  # 代码1固定参数

    for act_idx, act in enumerate(actions):
        for frame_idx in act["indices"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # ===================== 严格沿用代码1：手腕关键点检测 + 截图逻辑 =====================
            clean_img = frame.copy()  # 代码1：备份原始干净图像
            img_h, img_w = frame.shape[:2]

            # 代码1 identical 预处理
            img_detect = letterbox(frame, 960, stride=64, auto=True)[0]
            img_tensor = transforms.ToTensor()(img_detect).unsqueeze(0)
            if torch.cuda.is_available():
                img_tensor = img_tensor.half().to(device)

            # 代码1 identical 模型推理
            with torch.no_grad():
                output, _ = pose_model(img_tensor)
            output = non_max_suppression_kpt(output, 0.25, 0.65,
                                             nc=pose_model.yaml['nc'],
                                             nkpt=pose_model.yaml['nkpt'],
                                             kpt_label=True)
            output = output_to_keypoint(output)

            # 代码1 identical：提取左右腕坐标 + 置信度
            left_clean_roi = None
            right_clean_roi = None
            if output.shape[0] > 0:
                # 左腕9号 / 右腕10号（代码1固定关键点索引）
                left_wrist_x = output[0, 34]
                left_wrist_y = output[0, 35]
                left_wrist_conf = output[0, 36]
                right_wrist_x = output[0, 37]
                right_wrist_y = output[0, 38]
                right_wrist_conf = output[0, 39]

                # 代码1 identical：左腕截图（置信度过滤+边界限制）
                if left_wrist_conf > KP_CONF_THRESH:
                    x1 = int(left_wrist_x - half_size)
                    y1 = int(left_wrist_y - half_size)
                    x2 = int(left_wrist_x + half_size)
                    y2 = int(left_wrist_y + half_size)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_w, x2)
                    y2 = min(img_h, y2)
                    left_clean_roi = clean_img[y1:y2, x1:x2]

                # 代码1 identical：右腕截图（置信度过滤+边界限制）
                if right_wrist_conf > KP_CONF_THRESH:
                    x1_r = int(right_wrist_x - half_size)
                    y1_r = int(right_wrist_y - half_size)
                    x2_r = int(right_wrist_x + half_size)
                    y2_r = int(right_wrist_y + half_size)
                    x1_r = max(0, x1_r)
                    y1_r = max(0, y1_r)
                    x2_r = min(img_w, x2_r)
                    y2_r = min(img_h, y2_r)
                    right_clean_roi = clean_img[y1_r:y2_r, x1_r:x2_r]
            # ==================================================================================

            # 保存原始全图（原有功能保留）
            full_save_path = os.path.join(SAVE_PATH, f"action_{act_idx + 1}_frame_{frame_idx}.png")
            cv2.imwrite(full_save_path, frame)
            act["imgs"].append(full_save_path)

            # 保存手腕截图（代码1逻辑，命名避免覆盖）
            if left_clean_roi is not None:
                left_save_path = os.path.join(SAVE_PATH, f"action_{act_idx + 1}_frame_{frame_idx}_left_wrist.png")
                cv2.imwrite(left_save_path, left_clean_roi)
                act["imgs"].append(left_save_path)
            if right_clean_roi is not None:
                right_save_path = os.path.join(SAVE_PATH, f"action_{act_idx + 1}_frame_{frame_idx}_right_wrist.png")
                cv2.imwrite(right_save_path, right_clean_roi)
                act["imgs"].append(right_save_path)

            # 释放显存
            del img_tensor, output
            torch.cuda.empty_cache()

    cap.release()
    return actions


# ===================== 主流程（无修改） =====================
if __name__ == "__main__":
    print("=" * 60)
    print("✅ 视频动作分割 | 保存完整画面+手腕截图 | AI动作识别")
    print("=" * 60)

    frame_times, motion_data, fps = get_wrist_motion()
    action_list = split_and_sample_frames(frame_times, motion_data, fps)

    for i, act in enumerate(action_list):
        action_name = recognize_action(act["imgs"])
        print(f"动作{i + 1} | {act['start']}s ~ {act['end']}s | 图片数量：{len(act['imgs'])} | 识别结果：{action_name}")

    print("=" * 60)
    print(f"✅ 画面已保存至：{SAVE_PATH}")
    print("处理完成！")