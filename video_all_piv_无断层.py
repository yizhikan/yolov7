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
from test_get_around_key import get_around_key

# ===================== 全局参数 =====================
VIDEO_PATH = "C:\\D\\pyproject\\dongzuoshibie\\线束管路连接better.mp4"
SAVE_PATH = 'C:\\D\\pyproject\\dongzuoshibie\\video_frames\\'
SAVE_PATH_action_list = 'C:\\D\\pyproject\\dongzuoshibie\\'
MOTION_THRESH = 5.0
MIN_ACTION_DURATION = 0.5
ALI_API_KEY = "sk-1777fc7b981c4e9f90dcb10fddbd7ba8"
SAMPLE_PER_ACTION = 3
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
    for path in image_paths[:3]:
        if os.path.exists(path):
            content.append({"type": "image_url", "image_url": {"url": image_to_base64(path)}})
    content.append({"type": "text", "text": "请仔细看图中的手中拿的物品和动作，输入的全部图都是拿的一样的工具，是同行一个动作的不同时刻，禁止输出分析语句，用中文回答"})
    messages = [{"role": "user", "content": content}]
    completion = client.chat.completions.create(model="qwen3.5-plus", messages=messages)
    return completion.choices[0].message.content


# 计算运动幅度
def get_wrist_motion():
    cap = cv2.VideoCapture(VIDEO_PATH)
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
            cx = output[0, 7 + 3 * 10]
            cy = output[0, 8 + 3 * 10 + 1]
            if frame_count > 0:
                motion = np.sqrt((cx - prev_center_x) ** 2 + (cy - prev_center_y) ** 2)
            prev_center_x, prev_center_y = cx, cy

        motion_data.append(motion)
        del img_tensor, output
        torch.cuda.empty_cache()
        frame_count += 1

    cap.release()
    return frame_times, motion_data, fps


# 动作分割 + 保存画面（你的原始代码，完全不动）
def split_and_sample_frames(frame_times, motion_data, fps):
    total_frames = len(frame_times)
    motions = np.array(motion_data)

    peaks, _ = find_peaks(np.diff(motions), height=MOTION_THRESH)
    boundaries = [0] + peaks.tolist() + [total_frames - 1]
    boundaries = [b for b in boundaries if 0 <= b < total_frames]

    actions = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if (e - s) / fps >= MIN_ACTION_DURATION:
            sample_indices = get_middle_three_frames(s, e)
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

# 🔥 辅助函数：获取区间内 纯中间三帧（排除首尾帧）
def get_middle_three_frames(start_frame, end_frame):
    total = end_frame - start_frame + 1
    if total <= 3:
        # 帧数不足3帧，直接返回所有帧
        return list(range(start_frame, end_frame+1))
    # 核心：取中间均匀三帧，排除首尾
    mid1 = start_frame + total//4
    mid2 = start_frame + total//2
    mid3 = start_frame + 3*total//4
    return [mid1, mid2, mid3]

# 保存动作列表（你的原始代码，完全不动）
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
    print(f"✅ 动作列表已保存至：{save_path}")


# 语义合并（你的原始代码，完全不动）
# def merge_similar_actions(action_list, client, recognize_action_func):
#     if not action_list:
#         return []
#
#     for act in action_list:
#         if "action_name" not in act:
#             act["action_name"] = recognize_action_func(act["imgs"])
#
#     merged_actions = [action_list[0].copy()]
#     for current_act in action_list[1:]:
#         last_merged = merged_actions[-1]
#         action1 = last_merged["action_name"]
#         action2 = current_act["action_name"]
#
#         prompt = f"""判断两个操作动作的语义是否完全一致，只输出【是】或【否】，不要其他内容：
# 动作1：{action1}
# 动作2：{action2}"""
#         try:
#             completion = client.chat.completions.create(model="qwen3.5-plus",
#                                                         messages=[{"role": "user", "content": prompt}])
#             is_same = completion.choices[0].message.content.strip()
#         except Exception as e:
#             is_same = "否"
#
#         if is_same == "是":
#             last_merged["start"] = min(last_merged["start"], current_act["start"])
#             last_merged["end"] = max(last_merged["end"], current_act["end"])
#             last_merged["indices"] = sorted(list(set(last_merged["indices"] + current_act["indices"])))
#             last_merged["imgs"] = list(set(last_merged["imgs"] + current_act["imgs"]))
#         else:
#             merged_actions.append(current_act.copy())
#
#     print(f"✅ 动作合并完成：原始{len(action_list)}个 → 合并后{len(merged_actions)}个")
#     return merged_actions
# 语义合并（升级版：对比图片 + 判断物品+动作是否一致）
def merge_similar_actions(action_list, client, recognize_action_func):
    if not action_list:
        return []

    # 先给所有动作识别名称（保留原有逻辑）
    for act in action_list:
        if "action_name" not in act:
            act["action_name"] = recognize_action_func(act["imgs"])

    merged_actions = [action_list[0].copy()]
    for current_act in action_list[1:]:
        last_merged = merged_actions[-1]

        # ===================== 🔥 核心修改：传入两个动作的图片，AI视觉对比 =====================
        try:
            # 构建多模态内容：动作1的图片 + 动作2的图片 + 判断指令
            content = []
            # 加入上一个动作的图片（取前2张，避免过多）
            for img_path in last_merged["imgs"][:2]:
                if os.path.exists(img_path):
                    result_path = get_around_key(img_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image_to_base64(result_path)}
                    })
            # 加入当前动作的图片（取前2张）
            for img_path in current_act["imgs"][:2]:
                if os.path.exists(img_path):
                    result_path = get_around_key(img_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image_to_base64(result_path)}
                    })

            # 指令：严格判断【手持物品相同 + 动作相同】才输出是
            content.append({
                "type": "text",
                "text": "前两张图片为一个动作的不同时刻截图，后两张图片为另外一个动作的不同时刻截图，请对比前后两组图片，判断这两组画面是否是同一个动作不同时刻的图片（手持相似的物品、做相似的操作动作）。只输出【是】或【否】，禁止输出任何其他文字！"
            })

            messages = [{"role": "user", "content": content}]
            completion = client.chat.completions.create(
                model="qwen3.5-plus",
                messages=messages
            )
            is_same = completion.choices[0].message.content.strip()

        except Exception as e:
            print(f"⚠️ 动作对比失败：{str(e)}")
            is_same = "否"  # 异常则不合并
        # ==================================================================================

        torch.cuda.empty_cache()
        # 一致则合并，不一致则新增
        if is_same == "是":
            last_merged["start"] = min(last_merged["start"], current_act["start"])
            last_merged["end"] = max(last_merged["end"], current_act["end"])
            last_merged["indices"] = sorted(list(set(last_merged["indices"] + current_act["indices"])))
            last_merged["imgs"] = list(set(last_merged["imgs"] + current_act["imgs"]))
        else:
            merged_actions.append(current_act.copy())

    print(f"✅ 动作合并完成：原始{len(action_list)}个 → 合并后{len(merged_actions)}个")
    return merged_actions

# ===================== 🔥 核心新增：补全连续区间（带imgs/indices，保留原始动作） =====================
def fill_continuous_segments(action_list, frame_times, fps):
    """
    补全空白时间区间，生成0开始、连续无断层的动作列表
    1. 原始动作100%保留
    2. 空白区间自动采样帧、保存图片、填充imgs/indices
    3. 时间绝对连续
    """
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = len(frame_times)
    total_time = frame_times[-1] if total_frames > 0 else 0
    sorted_actions = sorted(action_list, key=lambda x: x["start"])
    continuous_list = []
    current_time = 0.0
    empty_idx = 1  # 空白区间编号

    for act in sorted_actions:
        act_start = act["start"]
        # 填充空白区间：当前时间 -> 原始动作开始时间
        if current_time < act_start:
            # 计算空白区间的帧范围
            start_frame = int(current_time * fps)
            end_frame = int(act_start * fps)
            start_frame = max(0, start_frame)
            end_frame = min(total_frames - 1, end_frame)

            # 采样帧（和原始逻辑一致）
            sample_indices = get_middle_three_frames(start_frame, end_frame)
            imgs = []
            # 保存空白区间的帧图片
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    save_path = os.path.join(SAVE_PATH, f"empty_action_{empty_idx}_frame_{idx}.png")
                    cv2.imwrite(save_path, frame)
                    imgs.append(save_path)

            # 添加空白区间（带imgs/indices）
            continuous_list.append({
                "start": round(current_time, 2),
                "end": round(act_start, 2),
                "indices": sample_indices,
                "imgs": imgs,
                "action_name": "无动作"
            })
            empty_idx += 1

        # 添加原始动作（完全不修改！）
        continuous_list.append(act.copy())
        current_time = act["end"]

    # 填充最后一段空白：最后一个动作结束 -> 视频结束
    if current_time < total_time:
        start_frame = int(current_time * fps)
        end_frame = total_frames - 1
        start_frame = max(0, start_frame)

        sample_indices = list(np.linspace(start_frame, end_frame, SAMPLE_PER_ACTION, dtype=int))
        imgs = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                save_path = os.path.join(SAVE_PATH, f"empty_action_{empty_idx}_frame_{idx}.png")
                cv2.imwrite(save_path, frame)
                imgs.append(save_path)

        continuous_list.append({
            "start": round(current_time, 2),
            "end": round(total_time, 2),
            "indices": sample_indices,
            "imgs": imgs,
            "action_name": "无动作"
        })

    cap.release()
    return continuous_list


def load_action_list(load_path="action_segment_result.json"):
    """
    从JSON文件加载动作列表（与 save_action_list 完全对应）
    :param load_path: 要加载的JSON文件路径
    :return: 加载后的动作列表 (list[dict])，格式与保存时完全一致
    """
    try:
        # 读取JSON文件
        with open(load_path, "r", encoding="utf-8") as f:
            action_list = json.load(f)

        print(f"✅ 动作列表已加载：{load_path}")
        return action_list

    except FileNotFoundError:
        print(f"❌ 错误：文件不存在 -> {load_path}")
        return []
    except Exception as e:
        print(f"❌ 加载失败：{str(e)}")
        return []

# ===================== 主流程（仅修改最后部分） =====================
if __name__ == "__main__":
    print("=" * 60)
    print("✅ 视频动作分割 | 保留原始片段 | 0秒开始连续无断层 | 空白带图片")
    print("=" * 60)

    frame_times, motion_data, fps = get_wrist_motion()

    # action_list = split_and_sample_frames(frame_times, motion_data, fps)
    #
    # # 3. 🔥 核心：补全连续区间（带imgs/indices，0开始，无断层）
    # continuous_action_list = fill_continuous_segments(action_list, frame_times, fps)
    # save_action_list(continuous_action_list, SAVE_PATH_action_list + "continuous_action_result.json")
    # # 2. 保存原始结果
    # save_action_list(action_list, SAVE_PATH_action_list + "action_segment_result_wuduanceng.json")

    # continuous_action_list = load_action_list(SAVE_PATH_action_list + "continuous_action_result.json")
    #
    # # # 1. AI识别动作（原始逻辑）
    # for i, act in enumerate(continuous_action_list):
    #     action_name = recognize_action(act["imgs"])
    #     act["action_name"] = action_name
    #     print(f"动作{i + 1} | {act['start']}s ~ {act['end']}s | 识别结果：{action_name}")
    #
    # save_action_list(continuous_action_list, SAVE_PATH_action_list + "continuous_action_result_recognize.json")

    continuous_action_list = load_action_list(SAVE_PATH_action_list + "continuous_action_result_recognize.json")
    # 4. 合并+补全
    merged_action_list = merge_similar_actions(continuous_action_list, client, recognize_action)
    merged_continuous_list = fill_continuous_segments(merged_action_list, frame_times, fps)
    save_action_list(merged_continuous_list, SAVE_PATH_action_list + "merged_continuous_result.json")
    #
    # # 打印最终结果
    # print("\n📊 最终连续无断层动作列表：")
    # for i, act in enumerate(continuous_action_list):
    #     print(f"动作{i + 1} | {act['start']}s ~ {act['end']}s | {act['action_name']}")
    #
    # print("=" * 60)
    # print(f"✅ 原始动作100%保留 | 空白区间已填充图片 | 时间连续无断层")
    # print("处理完成！")