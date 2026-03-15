import cv2
import os

def extract_frames(video_path, output_folder, interval_seconds):
    """
    按指定时间间隔从视频中提取帧并保存为图片
    :param video_path: 视频文件路径（支持mp4/avi/mov等常见格式）
    :param output_folder: 图片保存的文件夹路径
    :param interval_seconds: 每隔多少秒提取一帧（支持小数，如0.5）
    """
    # 1. 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建输出文件夹：{output_folder}")

    # 2. 打开视频
    cap = cv2.VideoCapture(video_path)
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("错误：无法打开视频文件，请检查路径是否正确！")
        return

    # 3. 获取视频基础信息
    fps = cap.get(cv2.CAP_PROP_FPS)  # 视频帧率（每秒多少帧）
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数
    print(f"视频信息：帧率={fps:.2f}, 总帧数={total_frames}")

    # 4. 计算【每隔多少帧】提取一次（核心公式）
    # 间隔帧数 = 间隔秒数 × 帧率
    frame_interval = int(interval_seconds * fps)
    if frame_interval < 1:
        frame_interval = 1  # 防止间隔过小，至少每1帧提取一次
    print(f"每隔 {interval_seconds} 秒 = 每隔 {frame_interval} 帧提取一帧")

    # 5. 初始化变量
    frame_count = 0  # 当前帧计数
    save_count = 0   # 保存图片计数

    # 6. 循环读取视频帧
    print("开始提取帧...")
    while True:
        # 读取一帧
        ret, frame = cap.read()
        # 如果读取失败（视频结束），退出循环
        if not ret:
            break

        # 7. 满足间隔条件就保存帧
        if frame_count % frame_interval == 0:
            # 图片命名：帧_0001.jpg （自动补零，方便排序）
            img_name = f"frame_{save_count:04d}.jpg"
            img_path = os.path.join(output_folder, img_name)
            # 保存图片
            cv2.imwrite(img_path, frame)
            save_count += 1
            print(f"已保存：{img_name}")

        frame_count += 1

    # 8. 释放视频资源
    cap.release()
    print(f"\n提取完成！共保存 {save_count} 张图片，路径：{output_folder}")

# ------------------- 【修改这里的参数即可】 -------------------
if __name__ == "__main__":
    # 1. 你的视频路径（绝对路径/相对路径都可以）
    VIDEO_PATH = "C:\\D\\pyproject\\dongzuoshibie\\线束管路连接.mp4"
    # 2. 图片保存的文件夹
    OUTPUT_FOLDER = "C:\\D\\pyproject\\dongzuoshibie\\video_frames"
    # 3. 提取间隔（单位：秒，支持小数，如0.3、0.5、1）
    INTERVAL = 0.5

    # 执行提取
    extract_frames(VIDEO_PATH, OUTPUT_FOLDER, INTERVAL)