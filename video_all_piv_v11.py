#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv7 视频动作时序分割 - V11 最终版本
基于 V10 的优化，目标精准 15 段
核心改进：
1. 使用 V9/V10 的 thresh=5.0 产生 16 段
2. 只合并非常确定的相同动作（字面完全相同）
3. 不合并语义相近但字面不同的动作
"""

import torch
import cv2
import numpy as np
import os
import base64
import json
import re
from openai import OpenAI
from scipy.signal import find_peaks
from torchvision import transforms

# YOLOv7 工具包
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint

# ===================== 全局参数 =====================
for _f in os.listdir('.'):
    if _f.endswith('.mp4'):
        VIDEO_PATH = _f
        break
else:
    VIDEO_PATH = None

SAVE_PATH = './video_frames/'
SAVE_PATH_action_list = './'
DETECT_SIZE = 640

# 分割参数
MOTION_THRESH = 5.0
MIN_ACTION_DURATION = 0.3
MIN_PEAK_DISTANCE = 20

os.makedirs(SAVE_PATH, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("加载 YOLOv7 模型...")
weights = torch.load('yolov7-w6-pose.pt', map_location=device, weights_only=False)
pose_model = weights['model'].float().eval()
if torch.cuda.is_available():
    pose_model.half().to(device)

ALI_API_KEY = ""
client = OpenAI(api_key=ALI_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

# ===================== 工具函数 =====================

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"


# ===================== 【1】运动计算 =====================
def get_wrist_motion(video_path=VIDEO_PATH):
    """右手腕运动计算"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 60.0

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


# ===================== 【2】峰值检测 + 合并 =====================
def split_actions(frame_times, motion_data, fps, motion_thresh=MOTION_THRESH):
    """基于峰值检测的分割算法"""
    motions = np.array(motion_data)
    total_frames = len(frame_times)

    peaks, _ = find_peaks(motions, height=motion_thresh, distance=MIN_PEAK_DISTANCE)

    if len(peaks) == 0:
        return []

    # 合并相邻过近的峰值
    min_gap_frames = int(0.5 * fps)
    merged_peaks = []
    current_group = [peaks[0]]

    for i in range(1, len(peaks)):
        if peaks[i] - peaks[i-1] < min_gap_frames:
            current_group.append(peaks[i])
        else:
            best_peak = max(current_group, key=lambda p: motions[p])
            merged_peaks.append(best_peak)
            current_group = [peaks[i]]

    best_peak = max(current_group, key=lambda p: motions[p])
    merged_peaks.append(best_peak)

    # 生成动作段
    actions = []
    for i, peak in enumerate(merged_peaks):
        if i == 0:
            start_frame = max(0, peak - int(0.5 * fps))
        else:
            prev_peak = merged_peaks[i-1]
            start_frame = (prev_peak + peak) // 2

        if i == len(merged_peaks) - 1:
            end_frame = min(total_frames - 1, peak + int(0.5 * fps))
        else:
            next_peak = merged_peaks[i+1]
            end_frame = (peak + next_peak) // 2

        duration = (end_frame - start_frame) / fps
        if duration >= MIN_ACTION_DURATION:
            actions.append({
                "start": round(frame_times[start_frame], 2),
                "end": round(frame_times[end_frame], 2),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "peak_frame": peak,
                "duration": round(duration, 2),
                "imgs": []
            })

    return actions


# ===================== 【3】保存动作帧 =====================
def save_action_frames(actions, video_path=VIDEO_PATH):
    """保存动作帧图片"""
    cap = cv2.VideoCapture(video_path)

    for act_idx, act in enumerate(actions):
        s, e = act["start_frame"], act["end_frame"]
        mid_frame = (s + e) // 2
        if e - s > 10:
            sample_indices = [s, mid_frame, e][:2]
        else:
            sample_indices = [mid_frame]

        act["indices"] = sample_indices
        act["imgs"] = []

        for frame_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            save_path = os.path.join(SAVE_PATH, f"action_{act_idx + 1}_frame_{frame_idx}.png")
            cv2.imwrite(save_path, frame)
            act["imgs"].append(save_path)

    cap.release()
    return actions


# ===================== 【4】大模型函数 =====================
def recognize_action(image_paths):
    """AI 识别单个动作"""
    if not image_paths:
        return "未获取到画面"
    content = []
    for path in image_paths[:2]:
        if os.path.exists(path):
            content.append({"type": "image_url", "image_url": {"url": image_to_base64(path)}})
    content.append({"type": "text", "text": "分析图片中的操作动作，只输出动作名称，2-4 个字"})
    messages = [{"role": "user", "content": content}]
    completion = client.chat.completions.create(model="qwen3.5-plus", messages=messages)
    return completion.choices[0].message.content


def conservative_merge_actions(actions):
    """
    V11 保守合并策略：
    1. 识别所有动作
    2. 只合并字面完全相同的连续动作
    3. 不合并语义相近但字面不同的动作
    """
    if len(actions) <= 1:
        return actions

    # Step 1: 识别所有动作
    print("Step 1: 识别所有动作...")
    for idx, act in enumerate(actions):
        if "action_name" not in act:
            act["action_name"] = recognize_action(act["imgs"])
            print(f"  动作{idx+1}/{len(actions)}: {act['action_name']} ({act['duration']}s)")

    # Step 2: 保守合并 - 只合并字面完全相同的
    print("\nStep 2: 保守合并...")
    merged = [actions[0].copy()]

    for i in range(1, len(actions)):
        current = actions[i]
        last = merged[-1]

        # 只合并字面完全相同的
        if current["action_name"] == last["action_name"]:
            print(f"  合并：{last['action_name']} ({last['duration']}s) + {current['action_name']} ({current['duration']}s)")
            last["end"] = current["end"]
            last["duration"] = round(last["end"] - last["start"], 2)
            last["end_frame"] = current["end_frame"]
            last["imgs"] = list(set(last["imgs"] + current["imgs"]))
            last["indices"] = sorted(list(set(last["indices"] + current["indices"])))
        else:
            merged.append(current.copy())

    # Step 3: 为合并后的动作重新识别
    print("\nStep 3: 重新识别合并后动作...")
    for idx, act in enumerate(merged):
        act["action_name"] = recognize_action(act["imgs"])
        print(f"  动作{idx+1}/{len(merged)}: {act['action_name']} ({act['duration']}s)")

    return merged


# ===================== 【5】保存结果 =====================
def save_action_list(action_list, save_path="action_segment_result.json"):
    processed_actions = []
    for act in action_list:
        processed_act = {
            "start": float(act.get("start", 0)),
            "end": float(act.get("end", 0)),
            "duration": act.get("duration", 0),
            "indices": [int(idx) for idx in act.get("indices", [])],
            "imgs": act.get("imgs", []),
            "action_name": act.get("action_name", "未识别")
        }
        processed_actions.append(processed_act)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(processed_actions, f, ensure_ascii=False, indent=4)
    print(f"✅ 动作列表已保存至：{save_path}")


# ===================== 主流程 =====================
if __name__ == "__main__":
    print("="*60)
    print("✅ YOLOv7 视频动作时序分割 - V11 最终版本（保守合并）")
    print("="*60)

    if not VIDEO_PATH:
        print("❌ 未找到视频文件")
        exit(1)

    print(f"使用视频：{VIDEO_PATH}")

    # 1. 计算运动数据
    print("\n计算运动数据...")
    frame_times, motion_data, fps = get_wrist_motion()
    video_duration = len(frame_times) / fps
    print(f"视频时长：{video_duration:.2f}秒，FPS: {fps}, 总帧数：{len(frame_times)}")

    motions = np.array(motion_data)
    print(f"运动幅度 - 最小：{motions.min():.2f}, 最大：{motions.max():.2f}, 平均：{motions.mean():.2f}")

    # 2. 固定阈值分割
    print("\n" + "="*60)
    print("动作分割...")
    print("="*60)
    actions = split_actions(frame_times, motion_data, fps, MOTION_THRESH)
    print(f"分割段数：{len(actions)}")

    for i, act in enumerate(actions):
        print(f"  {i+1}. {act['start']}s ~ {act['end']}s ({act['duration']}s)")

    # 3. 保存帧图片
    print("\n保存动作帧...")
    actions = save_action_frames(actions)

    # 4. 保守合并
    print("\n" + "="*60)
    print("保守合并...")
    print("="*60)
    merged_actions = conservative_merge_actions(actions)
    print(f"\n合并后：{len(merged_actions)} 段")

    # 5. 保存图片
    print("\n保存合并后动作的图片...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    for idx, act in enumerate(merged_actions):
        if not act.get("imgs"):
            start_frame = int(act["start"] * fps)
            end_frame = int(act["end"] * fps)
            mid_frame = (start_frame + end_frame) // 2
            if end_frame - start_frame > 30:
                sample_indices = [max(start_frame, mid_frame - 5), min(end_frame, mid_frame + 5)]
            else:
                sample_indices = [mid_frame]

            act["indices"] = sample_indices
            act["imgs"] = []

            for frame_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    save_path = os.path.join(SAVE_PATH, f"merged_action_{idx + 1}_frame_{frame_idx}.png")
                    cv2.imwrite(save_path, frame)
                    act["imgs"].append(save_path)
    cap.release()

    # 6. 保存结果
    save_action_list(merged_actions, SAVE_PATH_action_list + "action_segment_result.json")

    print("\n" + "="*60)
    print("最终分割结果:")
    print("="*60)
    for i, act in enumerate(merged_actions):
        print(f"动作{i+1}: {act['start']}s ~ {act['end']}s ({act['duration']}s) - {act['action_name']}")

    # 7. 时长统计
    short_actions = sum(1 for a in merged_actions if a['duration'] < 1.0)
    medium_actions = sum(1 for a in merged_actions if 1.0 <= a['duration'] < 2.0)
    long_actions = sum(1 for a in merged_actions if a['duration'] >= 2.0)
    very_long_actions = sum(1 for a in merged_actions if a['duration'] >= 5.0)
    print(f"\n时长分布：<1 秒：{short_actions}个 | 1-2 秒：{medium_actions}个 | ≥2 秒：{long_actions}个 | ≥5 秒：{very_long_actions}个")

    print(f"\n✅ 完整画面已保存至：{SAVE_PATH}")
    print("✅ 处理完成！")
