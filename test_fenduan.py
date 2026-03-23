import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ===================== 【用户自定义参数】修改这里即可 =====================
INPUT_VIDEO = "C:\\D\\pyproject\\dongzuoshibie\\线束管路连接better.mp4"
OUTPUT_VIDEO = "C:\\D\\pyproject\\dongzuoshibie\\线束管路连接better_output.mp4"

# 分段时间 + 中文描述
SEGMENTS = [
    {"time": 0, "label": "开始"},
    {"time": 5, "label": "检查线束"},
    {"time": 15, "label": "管路连接"},
    {"time": 25, "label": "完成"}  # 最后一段：25秒 ~ 视频结束
]

PROGRESS_HEIGHT = 30
BG_COLOR = (128, 128, 128)
PROGRESS_COLOR = (0, 165, 255)
LINE_COLOR = (255, 255, 255)
TEXT_COLOR = (255, 255, 255)
LINE_WIDTH = 2


# ========================================================================

# ============== 【核心：OpenCV 中文绘制函数】解决问号问题 ==============
def cv2_put_text(img, text, pos, font_size=14, text_color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("simhei.ttf", font_size, encoding="utf-8")
    draw.text(pos, text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# 打开视频
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise Exception("视频打开失败！请检查路径")

# 获取视频信息
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_time = total_frames / fps  # 视频总时长=27秒

# 下方拼接进度条
new_height = height + PROGRESS_HEIGHT
progress_y1 = height
progress_y2 = new_height

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, new_height))
SEG_TIMES = [seg["time"] for seg in SEGMENTS]

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 新建画布
    new_frame = np.zeros((new_height, width, 3), dtype=np.uint8)
    new_frame[0:height, 0:width] = frame

    # 进度计算
    current_time = frame_num / fps
    progress_width = int(width * (current_time / total_time))

    # 绘制进度条
    cv2.rectangle(new_frame, (0, progress_y1), (width, progress_y2), BG_COLOR, -1)
    cv2.rectangle(new_frame, (0, progress_y1), (progress_width, progress_y2), PROGRESS_COLOR, -1)
    for t in SEG_TIMES:
        x = int(width * (t / total_time))
        cv2.line(new_frame, (x, progress_y1), (x, progress_y2), LINE_COLOR, LINE_WIDTH)

    # ===================== 【修复核心】遍历所有分段，包含最后一段 =====================
    for i in range(len(SEGMENTS)):
        # 开始时间 = 当前分段时间
        start_time = SEGMENTS[i]["time"]
        # 结束时间：如果是最后一段，= 视频总时长；否则 = 下一个分段时间
        if i == len(SEGMENTS) - 1:
            end_time = total_time
        else:
            end_time = SEGMENTS[i + 1]["time"]

        label = SEGMENTS[i]["label"]

        # 计算精准居中位置（无偏移）
        start_x = int(width * (start_time / total_time))
        end_x = int(width * (end_time / total_time))
        center_x = (start_x + end_x) // 2 - 10  # 水平居中
        center_y = progress_y1 + 8  # 垂直居中（进度条内）

        # 绘制所有分段文字（包括最后一段25~27秒：完成）
        new_frame = cv2_put_text(new_frame, label, (center_x, center_y), 12, TEXT_COLOR)

    out.write(new_frame)
    frame_num += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"处理完成！所有分段文字正常显示：{OUTPUT_VIDEO}")