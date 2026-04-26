#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv7 视频动作时序分割 - V14 增强版
核心特性:
1. 多关节融合运动计算 (双手 + 双肘)
2. Savitzky-Golay 数据平滑滤波
3. 三重峰值检测 (高度 + 间距 + 宽度)
4. 纯 Prompt 大模型自动工具调用
5. 增强型阈值扫描 (0.5-15.0 细粒度)
6. API 超时重试机制
"""

import torch
import cv2
import numpy as np
import os
import base64
import json
import re
import time
from openai import OpenAI
from scipy.signal import find_peaks, savgol_filter
from torchvision import transforms
import glob

# YOLOv7 工具包
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint

# ===================== 全局参数 =====================
DEFAULT_FPS = 25.0
DETECT_SIZE = 640
SAVE_PATH = './video_frames/'
SAVE_PATH_action_list = './'

# 优化参数
SMOOTH_WINDOW = 5
STATIC_LIMIT = 0.8
MIN_PEAK_DIST = 6  # 降低间距允许更密集分割
MIN_PEAK_WIDTH = 1  # 降低宽度过滤
MIN_ACTION_DURATION = 0.2  # 降低最小动作时长
MAX_ITERATIONS = 5
MAX_API_RETRIES = 3
API_TIMEOUT = 90

os.makedirs(SAVE_PATH, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("加载 YOLOv7 模型...")
weights = torch.load('yolov7-w6-pose.pt', map_location=device, weights_only=False)
pose_model = weights['model'].float().eval()
if torch.cuda.is_available():
    pose_model.half().to(device)

ALI_API_KEY = ""
client = OpenAI(api_key=ALI_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


def find_video_file():
    """自动查找视频文件"""
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        files = glob.glob(ext)
        for f in files:
            if os.path.isfile(f):
                return f
    return None


def image_to_base64(image_path):
    """图片转 base64"""
    with open(image_path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"


# ===================== 【1】运动计算优化 =====================
def get_multi_joint_motion(video_path):
    """多关节融合运动计算"""
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = DEFAULT_FPS
        print(f"⚠️ 无法读取视频 FPS，使用默认值：{DEFAULT_FPS}")

    frame_count = 0
    frame_times = []
    motion_data = []
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
        output = non_max_suppression_kpt(
            output, 0.25, 0.65,
            nc=pose_model.yaml['nc'],
            nkpt=pose_model.yaml['nkpt'],
            kpt_label=True
        )
        output = output_to_keypoint(output)

        motion = 0.0
        if output.shape[0] > 0:
            kps = []
            for idx in [7, 8, 9, 10]:
                kps.append(output[0, 7 + 3 * idx])
                kps.append(output[0, 8 + 3 * idx + 1])
            curr_kps = np.array(kps)

            if frame_count > 0:
                motion = np.mean(np.sqrt(np.sum((curr_kps - prev_kps) ** 2, axis=0)))
            prev_kps = curr_kps

        if motion < STATIC_LIMIT:
            motion = 0.0

        motion_data.append(motion)

        del img_tensor, output
        torch.cuda.empty_cache()
        frame_count += 1

    cap.release()

    if len(motion_data) > SMOOTH_WINDOW:
        window = SMOOTH_WINDOW if SMOOTH_WINDOW % 2 == 1 else SMOOTH_WINDOW + 1
        if window < len(motion_data):
            motion_data = savgol_filter(motion_data, window, 2)

    return frame_times, motion_data, fps


# ===================== 【2】时序分割算法优化 =====================
def enhanced_peak_segmentation(frame_times, motion_data, fps, motion_thresh):
    """增强型峰值检测分割"""
    motions = np.array(motion_data)
    total_frames = len(frame_times)

    # 三重峰值检测
    peaks, _ = find_peaks(
        motions,
        height=motion_thresh,
        distance=MIN_PEAK_DIST,
        width=MIN_PEAK_WIDTH
    )

    if len(peaks) == 0:
        return []

    # 合并相邻过近的峰值
    min_gap = int(0.3 * fps)
    merged_peaks = []
    current_group = [peaks[0]]

    for i in range(1, len(peaks)):
        if peaks[i] - peaks[i-1] < min_gap:
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
            start_frame = 0
        else:
            prev_peak = merged_peaks[i-1]
            valley_region = motions[prev_peak:peak+1]
            if len(valley_region) > 0:
                valley_offset = np.argmin(valley_region)
                start_frame = prev_peak + valley_offset
            else:
                start_frame = (prev_peak + peak) // 2

        if i == len(merged_peaks) - 1:
            end_frame = total_frames - 1
        else:
            next_peak = merged_peaks[i+1]
            valley_region = motions[peak:next_peak+1]
            if len(valley_region) > 0:
                valley_offset = np.argmin(valley_region)
                end_frame = peak + valley_offset
            else:
                end_frame = (peak + next_peak) // 2

        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames - 1))

        duration = (end_frame - start_frame) / fps

        if duration >= MIN_ACTION_DURATION:
            actions.append({
                "start": round(frame_times[start_frame], 2),
                "end": round(frame_times[min(end_frame, total_frames-1)], 2),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "peak_frame": int(peak),
                "duration": round(duration, 2),
                "peak_value": float(motions[peak]),
                "imgs": []
            })

    return actions


def save_action_frames(actions, video_path):
    """保存动作关键帧"""
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


# ===================== 【3】大模型函数 - 带重试机制 =====================
def recognize_action_with_retry(image_paths, max_retries=MAX_API_RETRIES):
    """AI 识别单个动作 - 带重试"""
    if not image_paths:
        return "未获取到画面"

    for attempt in range(max_retries):
        try:
            content = []
            for path in image_paths[:2]:
                if os.path.exists(path):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image_to_base64(path)}
                    })

            content.append({
                "type": "text",
                "text": "分析图片中的操作动作，只输出动作名称，2-6 个字"
            })

            messages = [{"role": "user", "content": content}]

            completion = client.chat.completions.create(
                model="qwen3.5-plus",
                messages=messages,
                timeout=API_TIMEOUT
            )
            return completion.choices[0].message.content.strip()

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  重试 {attempt + 1}/{max_retries}: {e}")
                time.sleep(2)
            else:
                print(f"⚠️ 动作识别失败：{e}")
                return "未知动作"

    return "未知动作"


def evaluate_segmentation_via_llm(actions, target_count, max_retries=MAX_API_RETRIES):
    """大模型评估分割质量 - 带重试"""
    if not actions:
        return True, False, "decrease", None, {"empty": True}

    for attempt in range(max_retries):
        try:
            # 识别所有动作
            for act in actions:
                if "action_name" not in act or not act["action_name"]:
                    act["action_name"] = recognize_action_with_retry(act["imgs"])

            action_info = "\n".join([
                f"片段{i+1}: {act['start']}s-{act['end']}s ({act['duration']}s) - {act['action_name']}"
                for i, act in enumerate(actions)
            ])

            prompt = f"""你是一个视频动作分割质量评估专家。

当前分割结果:
{action_info}

目标分割段数：{target_count}段
当前分割段数：{len(actions)}段

请评估:
1. 是否存在过分割 (一个连续动作被分成多段)?
2. 是否存在欠分割 (多个动作被合并为一段)?
3. 应该如何调整阈值来改善分割质量?

请严格返回以下 JSON 格式:
{{
    "over_segmented": true 或 false,
    "under_segmented": true 或 false,
    "adjustment": "increase" 或 "decrease" 或 "keep",
    "suggested_thresh": 数值,
    "reason": "简短说明"
}}

只输出 JSON，不要其他内容。"""

            completion = client.chat.completions.create(
                model="qwen3.5-plus",
                messages=[{"role": "user", "content": prompt}],
                timeout=API_TIMEOUT
            )
            response = completion.choices[0].message.content.strip()

            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return (
                    result.get("over_segmented", False),
                    result.get("under_segmented", False),
                    result.get("adjustment", "keep"),
                    result.get("suggested_thresh", None),
                    result
                )
            return False, False, "keep", None, {"raw": response}

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  大模型评估重试 {attempt + 1}/{max_retries}: {e}")
                time.sleep(2)
            else:
                print(f"⚠️ 大模型评估失败：{e}")
                return False, False, "keep", None, {"error": str(e)}

    return False, False, "keep", None, {"max_retries_exceeded": True}


def auto_adjust_threshold(current_thresh, over_seg, under_seg, adjustment, target_count, current_count):
    """智能自动调节阈值"""
    diff = current_count - target_count

    if adjustment == "increase" or over_seg:
        # 过分割，提高阈值减少分段
        return current_thresh * 1.2
    elif adjustment == "decrease" or under_seg:
        # 欠分割，降低阈值增加分段
        if diff < -5:
            # 严重欠分割，大幅降低
            return max(0.5, current_thresh * 0.7)
        elif diff < -2:
            return max(0.5, current_thresh * 0.8)
        else:
            return max(0.5, current_thresh * 0.9)
    return current_thresh


# ===================== 【5】保存结果 =====================
def save_action_list(action_list, save_path="action_segment_result.json"):
    """保存动作列表到 JSON"""
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


# ===================== 【6】全自动优化主循环 =====================
def auto_optimize_segmentation(video_path, target_segments=15):
    """全自动迭代优化分割参数"""
    print("="*60)
    print(f"开始全自动优化 | 目标段数：{target_segments}")
    print("="*60)

    frame_times, motion_data, fps = get_multi_joint_motion(video_path)
    motions = np.array(motion_data)

    print(f"视频时长：{len(frame_times)/fps:.2f}s | FPS: {fps}")
    print(f"运动范围：[{motions.min():.2f}, {motions.max():.2f}], 平均：{motions.mean():.2f}")

    # 增强型多阈值扫描 (细粒度)
    print("\nStep 2: 细粒度阈值扫描...")
    thresh_results = []

    # 生成细粒度阈值列表
    if target_segments > 20:
        # 需要更多分段，使用更低阈值范围
        thresh_scan = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 1.7, 2.0, 2.3, 2.5, 2.8, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]
    else:
        thresh_scan = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0]

    for thresh in thresh_scan:
        actions = enhanced_peak_segmentation(frame_times, motion_data, fps, thresh)
        diff = abs(len(actions) - target_segments)
        thresh_results.append({
            "thresh": thresh,
            "count": len(actions),
            "diff": diff,
            "actions": actions
        })
        status = "✓" if diff <= 2 else ""
        print(f"  {status} thresh={thresh:.1f}: {len(actions)}段 (差{diff})")

    # 选择最佳阈值
    best = min(thresh_results, key=lambda x: x["diff"])
    current_thresh = best["thresh"]
    best_actions = best["actions"]

    print(f"\n初始最佳阈值：{current_thresh}, 分割段数：{len(best_actions)}")

    # 如果已经超过或接近目标，直接返回
    if abs(len(best_actions) - target_segments) <= 2:
        print("✅ 初始分割已接近目标，跳过迭代")
        best_actions = save_action_frames(best_actions, video_path)
        return best_actions, current_thresh

    # 大模型迭代优化
    print("\nStep 4: 大模型迭代优化...")
    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n--- 迭代 {iteration}/{MAX_ITERATIONS} (thresh={current_thresh:.2f}) ---")

        actions = enhanced_peak_segmentation(frame_times, motion_data, fps, current_thresh)

        if not actions:
            print("⚠️ 分割结果为空，降低阈值")
            current_thresh *= 0.7
            continue

        current_count = len(actions)
        print(f"当前段数：{current_count}, 目标：{target_segments}")

        # 如果已经达标，停止迭代
        if abs(current_count - target_segments) <= 1:
            print("✅ 分割段数达标，停止迭代")
            best_actions = actions
            break

        # 大模型评估
        over_seg, under_seg, adjustment, suggested_thresh, analysis = evaluate_segmentation_via_llm(
            actions, target_segments
        )
        print(f"评估：过分割={over_seg}, 欠分割={under_seg}, 建议={adjustment}")
        if analysis.get("reason"):
            print(f"分析：{analysis['reason']}")

        # 使用建议阈值或自动调节
        if suggested_thresh is not None and suggested_thresh > 0:
            current_thresh = suggested_thresh
        else:
            current_thresh = auto_adjust_threshold(
                current_thresh, over_seg, under_seg, adjustment, target_segments, current_count
            )

        best_actions = actions

    # 保存帧图片
    if best_actions:
        best_actions = save_action_frames(best_actions, video_path)

    return best_actions, current_thresh


# ===================== 【7】主函数 =====================
def main():
    """主函数 - 全自动执行"""
    print("="*60)
    print("YOLOv7 视频动作时序分割 - V14 增强版")
    print("="*60)

    video_path = find_video_file()
    if not video_path:
        print("❌ 未找到视频文件")
        return

    print(f"检测到视频：{video_path}")

    # 根据视频名称设定目标段数
    if "线束" in video_path or "管路" in video_path:
        target_segments = 15
    elif "挡圈" in video_path or "B 平台" in video_path:
        target_segments = 25
    else:
        target_segments = 15

    print(f"目标分割段数：{target_segments}")

    actions, final_thresh = auto_optimize_segmentation(video_path, target_segments)

    if not actions:
        print("❌ 分割失败")
        return

    # 输出分割段数确认及各段详细信息
    print("\n" + "="*60)
    print(f"确定分割段数：{len(actions)} 段")
    print("="*60)
    print("\n各分段详情:")
    print(f"{'段号':<6} {'开始时间 (s)':<15} {'结束时间 (s)':<15} {'持续时长 (s)':<12}")
    print("-" * 60)
    for i, act in enumerate(actions):
        print(f"{i+1:<6} {act['start']:<15.2f} {act['end']:<15.2f} {act['duration']:<12.2f}")

    # 识别所有动作
    print("\n识别动作...")
    for act in actions:
        act["action_name"] = recognize_action_with_retry(act["imgs"])

    # 输出结果
    print("\n" + "="*60)
    print("最终分割结果:")
    print("="*60)
    for i, act in enumerate(actions):
        print(f"动作{i+1}: {act['start']}s ~ {act['end']}s ({act['duration']}s) - {act['action_name']}")

    # 时长统计
    short_actions = sum(1 for a in actions if a['duration'] < 1.0)
    medium_actions = sum(1 for a in actions if 1.0 <= a['duration'] < 2.0)
    long_actions = sum(1 for a in actions if a['duration'] >= 2.0)
    very_long_actions = sum(1 for a in actions if a['duration'] >= 5.0)

    print(f"\n时长分布：<1 秒：{short_actions}个 | 1-2 秒：{medium_actions}个 | ≥2 秒：{long_actions}个 | ≥5 秒：{very_long_actions}个")
    print(f"总段数：{len(actions)} | 目标：{target_segments}")

    save_action_list(actions, SAVE_PATH_action_list + "action_segment_result.json")
    print(f"\n✅ 帧图片已保存至：{SAVE_PATH}")
    print("✅ V14 处理完成!")


if __name__ == "__main__":
    main()
