from moviepy.editor import VideoClip, CompositeVideoClip
import moviepy.editor as mp
import numpy as np

# ===================== 【你的配置】 =====================
INPUT_PATH = r"C:\D\pyproject\dongzuoshibie\线束管路连接better.mp4"
OUTPUT_PATH = r"C:\D\pyproject\dongzuoshibie\最终视频_moviepy.mp4"

# 分段时间 + 文字描述
SEGMENTS = [
    {"time": 0, "label": "开始"},
    {"time": 5, "label": "检查线束"},
    {"time": 15, "label": "管路连接"},
    {"time": 25, "label": "完成"},
]
PROGRESS_HEIGHT = 30  # 进度条高度
# ======================================================

# 加载视频
video = mp.VideoFileClip(INPUT_PATH)
W, H = video.size  # 视频宽高
DURATION = video.duration  # 总时长
TIMES = [s["time"] for s in SEGMENTS]

# 绘制进度条帧（核心函数）
def make_progress_frame(t):
    # 创建进度条画布
    frame = np.zeros((PROGRESS_HEIGHT, W, 3), dtype=np.uint8)
    frame[:] = (128, 128, 128)  # 灰色背景

    # 1. 绘制动态进度（橙色）
    progress_w = int(W * (t / DURATION))
    frame[:, :progress_w] = (255, 165, 0)

    # 2. 绘制分段竖线（白色）
    for time_point in TIMES:
        x = int(W * (time_point / DURATION))
        frame[:, x-1:x+1] = (255, 255, 255)

    return frame

# 生成进度条视频片段
progress_clip = VideoClip(make_progress_frame, duration=DURATION).set_position(("center", H))

# 绘制文字（分段标签）
def make_text_clip(label, start_time, end_time):
    x1 = int(W * (start_time / DURATION))
    x2 = int(W * (end_time / DURATION))
    cx = (x1 + x2) // 2  # 居中
    txt_clip = mp.TextClip(label, fontsize=12, color='white')
    txt_clip = txt_clip.set_position((cx - len(label)*3, H + 8))  # 进度条上居中
    txt_clip = txt_clip.set_duration(DURATION)
    return txt_clip

# 组合所有文字
text_clips = []
for i in range(len(SEGMENTS)-1):
    s = SEGMENTS[i]
    e = SEGMENTS[i+1]
    text_clips.append(make_text_clip(s["label"], s["time"], e["time"]))

# 最终合成：视频 + 进度条 + 文字
final_clip = CompositeVideoClip(
    [video, progress_clip, *text_clips],
    size=(W, H + PROGRESS_HEIGHT)  # 向下扩展画布，不遮挡视频
)

# 导出视频（自动保留音频！）
final_clip.write_videofile(
    OUTPUT_PATH,
    codec="libx264",
    audio_codec="aac",
    verbose=False,
    logger=None
)

print("✅ 处理完成！视频已保存：", OUTPUT_PATH)