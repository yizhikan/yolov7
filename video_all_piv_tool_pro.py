import torch
import cv2
import numpy as np
import os
import base64
from openai import OpenAI
from scipy.signal import find_peaks, savgol_filter  # 新增滤波函数
from torchvision import transforms
import json

# YOLOv7工具包
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint

# ===================== 全局参数（新增优化参数，原有参数完全保留） =====================
VIDEO_PATH = "C:\\D\\pyproject\\dongzuoshibie\\线束管路连接better.mp4"
SAVE_PATH = 'C:\\D\\pyproject\\dongzuoshibie\\video_frames\\'
SAVE_PATH_action_list = 'C:\\D\\pyproject\\dongzuoshibie\\'
MOTION_THRESH = 5.0
MIN_ACTION_DURATION = 0.5
ALI_API_KEY = "sk-1777fc7b981c4e9f90dcb10fddbd7ba8"
SAMPLE_PER_ACTION = 2
DETECT_SIZE = 640
# 新增高精度分割优化参数（不影响原有接口）
SMOOTH_WINDOW = 5  # 运动数据平滑窗口
STATIC_LIMIT = 0.8  # 静止状态过滤阈值
MIN_PEAK_DIST = 8  # 峰值最小间距（防止过分割）
MIN_PEAK_WIDTH = 2  # 峰值最小宽度（过滤噪声）
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


# ===================== 【优化1】多关节运动计算 + 数据平滑（接口完全不变） =====================
def get_wrist_motion(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    frame_times = []
    motion_data = []
    # 优化：存储双手+双肘关键点（替代单手腕，抗干扰）
    prev_kps = np.zeros(8)

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
            # 优化：双肘(7/8)+双手腕(9/10) 4个关键点，鲁棒性拉满
            kps = []
            for idx in [7, 8, 9, 10]:
                kps.append(output[0, 7 + 3 * idx])
                kps.append(output[0, 8 + 3 * idx + 1])
            curr_kps = np.array(kps)

            # 计算综合运动幅度
            if frame_count > 0:
                motion = np.mean(np.sqrt(np.sum((curr_kps - prev_kps) ** 2)))
            prev_kps = curr_kps

        # 过滤静止噪声
        if motion < STATIC_LIMIT:
            motion = 0.0

        motion_data.append(motion)
        del img_tensor, output
        torch.cuda.empty_cache()
        frame_count += 1

    cap.release()

    # 优化：工业级平滑滤波，去除毛刺
    if len(motion_data) > SMOOTH_WINDOW:
        motion_data = savgol_filter(motion_data, SMOOTH_WINDOW, 2)

    return frame_times, motion_data, fps


# ===================== 【优化2】增强型峰值分割（接口完全不变） =====================
def split_and_sample_frames(frame_times, motion_data, fps, motion_thresh):
    total_frames = len(frame_times)
    motions = np.array(motion_data)

    # 优化：三重峰值检测（高度+间距+宽度），精准分割
    peaks, _ = find_peaks(
        np.diff(motions),
        height=motion_thresh,
        distance=MIN_PEAK_DIST,  # 防止过分割
        width=MIN_PEAK_WIDTH  # 过滤噪声尖峰
    )

    boundaries = [0] + peaks.tolist() + [total_frames - 1]
    boundaries = sorted(list(set([b for b in boundaries if 0 <= b < total_frames])))

    actions = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        # 优化：过滤纯静止片段
        if (e - s) / fps >= MIN_ACTION_DURATION and np.max(motions[s:e]) > STATIC_LIMIT:
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
            completion = client.chat.completions.create(model="qwen3.5-plus",
                                                        messages=[{"role": "user", "content": prompt}])
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


# ===================== 大模型Tool函数（接口100%无修改） =====================
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


# ===================== Function Calling 工具定义（100%无修改） =====================
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
# ===================== 仅新增：测试主函数 =====================
if __name__ == "__main__":
    # 测试调用动作分割工具
    test_video = VIDEO_PATH
    test_thresh = 5.0
    # 执行分割
    result = yolo_action_segment(test_video, test_thresh)
    # 打印结果
    print("="*50)
    print(f"动作分割完成 | 使用阈值：{result['used_threshold']}")
    print(f"总分割段数：{result['segment_count']}")
    print("动作时间段：")
    for i, seg in enumerate(result['segments']):
        print(f"第{i+1}个动作：{seg['start']}s ~ {seg['end']}s")
    print("="*50)
