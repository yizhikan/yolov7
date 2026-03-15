import os
import base64
from openai import OpenAI

# ===================== 【核心修改】本地图片路径（换成你自己的！） =====================
# 支持单张/多张，直接填本地绝对/相对路径
LOCAL_IMAGE_PATHS = [
    "C:\\D\\pyproject\\dongzuoshibie\\video_frames\\frame_0000.jpg",  # 你的手腕截图
    "C:\\D\\pyproject\\dongzuoshibie\\video_frames\\frame_0001.jpg"   # 第二张本地图
]
# ==================================================================================

client = OpenAI(
    api_key="sk-1777fc7b981c4e9f90dcb10fddbd7ba8",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ===================== 工具函数：本地图片转Base64 =====================
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        base64_data = base64.b64encode(f.read()).decode("utf-8")
    # 返回通义API要求的格式
    return f"data:image/jpeg;base64,{base64_data}"

# 批量转换本地图片
image_contents = []
for img_path in LOCAL_IMAGE_PATHS:
    if os.path.exists(img_path):
        base64_url = image_to_base64(img_path)
        image_contents.append({"type": "image_url", "image_url": {"url": base64_url}})
    else:
        print(f"⚠️ 图片不存在：{img_path}")

# 拼接请求内容（图片 + 提问文本）
messages = [
    {
        "role": "user",
        "content": [
            *image_contents,  # 插入本地转码后的图片
            {"type": "text", "text": "分析这些手部图片，识别工人在做什么动作，只输出动作名称"}
        ]
    }
]

# 调用API
completion = client.chat.completions.create(
    model="qwen3.5-plus",
    messages=messages,
)

# 输出结果
print(completion.choices[0].message.content)