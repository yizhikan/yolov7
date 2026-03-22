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
        # 🔥 修复1：变量名 model → pose_model（解决NameError报错）
        output = non_max_suppression_kpt(output, 0.25, 0.65,
                                         nc=pose_model.yaml['nc'],
                                         nkpt=pose_model.yaml['nkpt'],
                                         kpt_label=True)
        output = output_to_keypoint(output)

        # 🔥 修复2：恢复正确运动值计算（不再写死0，保证动作分割生效）
        motion = 0.0
        if output.shape[0] > 0:
            # 取右腕坐标计算运动（简单稳定）
            cx = output[0, 7+3*10]
            cy = output[0, 8+3*10+1]
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


# ===================== 动作分割 + 保存完整画面 =====================
def split_and_sample_frames(frame_times, motion_data, fps):
    total_frames = len(frame_times)
    motions = np.array(motion_data)

    # 动作分割
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

    # 保存完整原始画面（无任何裁剪）
    cap = cv2.VideoCapture(VIDEO_PATH)
    for act_idx, act in enumerate(actions):
        for frame_idx in act["indices"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # 直接保存全图
            save_path = os.path.join(SAVE_PATH, f"action_{act_idx + 1}_frame_{frame_idx}.png")
            cv2.imwrite(save_path, frame)
            act["imgs"].append(save_path)

    cap.release()
    return actions


import json
import numpy as np


# ===================== 新增：保存动作列表函数 =====================
def save_action_list(action_list, save_path="action_segment_result.json"):
    """
    保存动作分割结果action_list到JSON文件
    :param action_list: 分割后的动作列表（包含start/end/indices/imgs/action_name）
    :param save_path: 保存的JSON文件路径
    """
    # 转换numpy数据类型为Python原生类型（兼容JSON序列化）
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

    # 保存为JSON文件
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(processed_actions, f, ensure_ascii=False, indent=4)
    print(f"✅ 动作列表已保存至：{save_path}")


# ===================== 新增：大模型语义合并动作片段函数 =====================
def merge_similar_actions(action_list, client, recognize_action_func):
    """
    大模型语义判断：合并连续语义相同的动作片段，拼接碎片，不同语义不合并
    支持多步骤连续合并（如 A→A→A 直接合并为一个，非仅两两合并）
    :param action_list: 原始动作列表
    :param client: 通义千问OpenAI格式客户端
    :param recognize_action_func: 动作识别函数（传入图片路径返回动作名称）
    :return: 合并后的动作列表
    """
    if not action_list:
        return []

    # 1. 补全所有动作的识别名称（确保每个动作都有action_name）
    for act in action_list:
        if "action_name" not in act:
            act["action_name"] = recognize_action_func(act["imgs"])

    merged_actions = [action_list[0].copy()]  # 初始化合并列表

    # 2. 遍历后续动作，逐一对上一个合并后的动作做语义判断
    for current_act in action_list[1:]:
        last_merged = merged_actions[-1]
        action1 = last_merged["action_name"]
        action2 = current_act["action_name"]

        # 3. 大模型判断两个动作语义是否相同（仅输出是/否）
        prompt = f"""判断两个操作动作的语义是否完全一致，只输出【是】或【否】，不要其他内容：
动作1：{action1}
动作2：{action2}"""
        try:
            completion = client.chat.completions.create(
                model="qwen3.5-plus",
                messages=[{"role": "user", "content": prompt}]
            )
            is_same = completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ 语义判断失败，默认不合并：{str(e)}")
            is_same = "否"

        # 4. 语义相同 → 合并（时间、帧索引、图片全部拼接）
        if is_same == "是":
            last_merged["start"] = min(last_merged["start"], current_act["start"])
            last_merged["end"] = max(last_merged["end"], current_act["end"])
            last_merged["indices"] = sorted(list(set(last_merged["indices"] + current_act["indices"])))
            last_merged["imgs"] = list(set(last_merged["imgs"] + current_act["imgs"]))
        # 5. 语义不同 → 直接加入新动作
        else:
            merged_actions.append(current_act.copy())

    print(f"✅ 动作合并完成：原始{len(action_list)}个 → 合并后{len(merged_actions)}个")
    return merged_actions

# ===================== 主流程 =====================
# if __name__ == "__main__":
#     print("=" * 60)
#     print("✅ 视频动作分割 | 保存完整画面 | AI动作识别")
#     print("=" * 60)
#
#     frame_times, motion_data, fps = get_wrist_motion()
#     action_list = split_and_sample_frames(frame_times, motion_data, fps)
#     save_action_list(action_list, SAVE_PATH_action_list+"action_segment_result.json")
#
#     for i, act in enumerate(action_list):
#         action_name = recognize_action(act["imgs"])
#         print(f"动作{i + 1} | {act['start']}s ~ {act['end']}s | 全图数量：{len(act['imgs'])} | 识别结果：{action_name}")
#
#     print("=" * 60)
#     print(f"✅ 完整画面已保存至：{SAVE_PATH}")
#     print("处理完成！")

if __name__ == "__main__":
    print("=" * 60)
    print("✅ 视频动作分割 | 保存完整画面 | AI动作识别")
    print("=" * 60)

    frame_times, motion_data, fps = get_wrist_motion()
    action_list = split_and_sample_frames(frame_times, motion_data, fps)

    # 1. AI识别所有动作名称
    for i, act in enumerate(action_list):
        action_name = recognize_action(act["imgs"])
        act["action_name"] = action_name  # 存入动作字典
        print(f"动作{i + 1} | {act['start']}s ~ {act['end']}s | 全图数量：{len(act['imgs'])} | 识别结果：{action_name}")

    # 2. 【调用新增函数】保存原始动作列表
    save_action_list(action_list, SAVE_PATH_action_list+"action_segment_result.json")

    # 3. 【调用新增函数】大模型语义合并动作片段
    merged_action_list = merge_similar_actions(action_list, client, recognize_action)

    # 4. 保存合并后的动作列表
    save_action_list(merged_action_list, SAVE_PATH_action_list + "merged_action_result.json")

    # 打印合并结果
    print("\n📊 合并后的动作片段：")
    for i, act in enumerate(merged_action_list):
        print(f"合并动作{i + 1} | {act['start']}s ~ {act['end']}s | 动作：{act['action_name']}")

    print("=" * 60)
    print(f"✅ 完整画面已保存至：{SAVE_PATH}")
    print("处理完成！")