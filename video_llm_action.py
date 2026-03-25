from openai import OpenAI
import os
import base64
import json

# ===================== 【官方原版：视频编码函数 完全不动】 =====================
def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

# ===================== 【你原有：图片转Base64 完全不动】 =====================
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        base64_data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_data}"

# ===================== 【步骤1：从动作图提取动作列表】 =====================
def extract_actions_from_image(image_path):
    client = OpenAI(
        api_key="sk-1777fc7b981c4e9f90dcb10fddbd7ba8",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=300  # 防超时断连
    )
    content = [
        {"type": "image_url", "image_url": {"url": image_to_base64(image_path)}},
        {"type": "text", "text": "提取图片中所有操作动作，仅输出Python列表，不要任何多余文字"}
    ]
    completion = client.chat.completions.create(
        model="qwen3.5-plus",
        messages=[{"role": "user", "content": content}]
    )
    return eval(completion.choices[0].message.content.strip())

# ===================== 【核心：完全照搬官方视频调用代码 + 时序分割】 =====================
if __name__ == "__main__":
    # ========== 仅需修改这2个路径 ==========
    ACTION_IMAGE_PATH = r"C:\D\pyproject\dongzuoshibie\1774439471769(1).jpg"  # 你的动作步骤图
    VIDEO_PATH = r"C:\D\pyproject\dongzuoshibie\序列 01_25.mp4"         # 你的视频

    # 1. 提取动作列表
    action_list = extract_actions_from_image(ACTION_IMAGE_PATH)
    print(f"✅ 提取动作列表：{action_list}")

    # 2. 官方原版视频编码
    base64_video = encode_video(VIDEO_PATH)

    # 3. 官方原版客户端初始化（完全不动）
    client = OpenAI(
        api_key="sk-1777fc7b981c4e9f90dcb10fddbd7ba8",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=300  # 修复10054断连
    )

    # 4. 官方原版调用格式 + 你的时序分割指令
    completion = client.chat.completions.create(
        model="qwen3.5-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        # 官方原版：video_url 格式 完全不动
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{base64_video}"},
                        "fps": 1  # 降低帧率，彻底解决断连报错
                    },
                    {
                        "type": "text",
                        "text": f"""
                        任务：视频时序分割
                        动作列表：{action_list}
                        要求：
                        1. 将视频按时间分割为连续时间段
                        2. 每个时间段匹配动作列表中的一个动作
                        3. 仅输出标准JSON格式，无其他文字
                        格式：[{{"action":"动作名","start":开始秒数,"end":结束秒数}}]
                        """
                    },
                ],
            }
        ],
    )

    # 5. 解析并输出结果

    print("\n🎯 视频时序分割结果：")
    print(completion.choices[0].message.content.strip())