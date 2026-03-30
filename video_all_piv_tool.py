import torch
import cv2
import numpy as np
import os
import base64
from openai import OpenAI
from scipy.signal import find_peaks
from torchvision import transforms
import json

# YOLOv7工具包
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint

# ===================== 全局参数（完全保留） =====================
VIDEO_PATH = "C:\\D\\pyproject\\dongzuoshibie\\线束管路连接better.mp4"
SAVE_PATH = 'C:\\D\\pyproject\\dongzuoshibie\\video_frames\\'
SAVE_PATH_action_list = 'C:\\D\\pyproject\\dongzuoshibie\\'
MOTION_THRESH = 5.0
MIN_ACTION_DURATION = 0.5
ALI_API_KEY = "sk-1777fc7b981c4e9f90dcb10fddbd7ba8"
SAMPLE_PER_ACTION = 2
DETECT_SIZE = 640
# ====================================================

os.makedirs(SAVE_PATH, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载模型（只加载一次）
weights = torch.load('yolov7-w6-pose.pt', map_location=device, weights_only=False)
pose_model = weights['model'].float().eval()
if torch.cuda.is_available():
    pose_model.half().to(device)

client = OpenAI(api_key=ALI_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

# 图片转base64（无修改）
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"

# AI识别动作（无修改）
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

# ===================== 【修改1】新增入参 video_path，取消全局硬编码 =====================
def get_wrist_motion(video_path):
    cap = cv2.VideoCapture(video_path)  # 原：VIDEO_PATH → 动态传参
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    frame_times = []
    motion_data = []
    prev_center_x, prev_center_y = 0.0, 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_time = frame_count / fps
        frame_times.append(current_time)

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

        motion = 0.0
        if output.shape[0] > 0:
            cx = output[0, 7+3*10]
            cy = output[0, 8+3*10+1]
            if frame_count > 0:
                motion = np.sqrt((cx - prev_center_x) ** 2 + (cy - prev_center_y) ** 2)
            prev_center_x, prev_center_y = cx, cy

        motion_data.append(motion)
        del img_tensor, output
        torch.cuda.empty_cache()
        frame_count += 1

    cap.release()
    return frame_times, motion_data, fps

# ===================== 【修改2】新增入参 motion_thresh，动态调整分割阈值 =====================
def split_and_sample_frames(frame_times, motion_data, fps, motion_thresh):
    total_frames = len(frame_times)
    motions = np.array(motion_data)

    # 原：height=MOTION_THRESH → 现：动态阈值
    peaks, _ = find_peaks(np.diff(motions), height=motion_thresh)
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

# 保存动作列表（无修改）
def save_action_list(action_list, save_path="action_segment_result.json"):
    processed_actions = []
    for act in action_list:
        processed_act = {
            "start": float(act["start"]),
            "end": float(act["end"]),
            "indices": [int(idx) for idx in act["indices"]],
            "imgs": act["imgs"],
            "action_name": act.get("action_name", "未识别")
        }
        processed_actions.append(processed_act)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(processed_actions, f, ensure_ascii=False, indent=4)

# 语义合并动作（无修改）
def merge_similar_actions(action_list, client, recognize_action_func):
    if not action_list:
        return []
    for act in action_list:
        if "action_name" not in act:
            act["action_name"] = recognize_action_func(act["imgs"])
    merged_actions = [action_list[0].copy()]
    for current_act in action_list[1:]:
        last_merged = merged_actions[-1]
        action1 = last_merged["action_name"]
        action2 = current_act["action_name"]
        prompt = f"""判断两个操作动作的语义是否完全一致，只输出【是】或【否】：动作1：{action1} 动作2：{action2}"""
        try:
            completion = client.chat.completions.create(model="qwen3.5-plus", messages=[{"role": "user", "content": prompt}])
            is_same = completion.choices[0].message.content.strip()
        except:
            is_same = "否"
        if is_same == "是":
            last_merged["start"] = min(last_merged["start"], current_act["start"])
            last_merged["end"] = max(last_merged["end"], current_act["end"])
            last_merged["indices"] = sorted(list(set(last_merged["indices"] + current_act["indices"])))
        else:
            merged_actions.append(current_act.copy())
    return merged_actions

# ===================== 【修改3】大模型Tool统一入口函数 =====================
def yolo_action_segment(video_path: str, motion_thresh: float):
    """
    大模型专用工具：视频动作时序分割
    阈值规则：0~20，数值越小 → 分段越多；数值越大 → 分段越少
    """
    frame_times, motion_data, fps = get_wrist_motion(video_path)
    action_list = split_and_sample_frames(frame_times, motion_data, fps, motion_thresh)
    return {
        "segment_count": len(action_list),
        "segments": [{"start": i["start"], "end": i["end"]} for i in action_list],
        "used_threshold": motion_thresh
    }

# ===================== 【修改4】Function Calling 工具定义（大模型识别格式） =====================
YOLO_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "yolo_action_segment",
            "description": "基于YOLO姿态的视频动作分割，可调节motion_thresh控制分段数量",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "视频绝对路径"},
                    "motion_thresh": {"type": "number", "description": "分割阈值0-20，越小分段越多"}
                },
                "required": ["video_path", "motion_thresh"]
            }
        }
    }
]

# 注释原有主流程，纯工具文件