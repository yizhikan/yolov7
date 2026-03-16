VIDEO_PATH = "C:\\D\\pyproject\\dongzuoshibie\\线束管路连接.mp4"  # 输入连续动作视频
CROP_SIZE = 200  # 手腕截图大小
SAVE_PATH = 'C:\\D\\pyproject\\dongzuoshibie\\video_frames\\'  # 临时保存手腕图片

动作1 | 3.53s ~ 4.2s | 图片数:4 | 握手
动作2 | 4.2s ~ 5.72s | 图片数:4 | 抬手
动作3 | 6.32s ~ 6.95s | 图片数:4 | 伸手
动作4 | 18.4s ~ 19.03s | 图片数:4 | 伸出手臂

[{'end': 3.83, 'imgs': ['C:\D\pyproject\dongzuoshibie\video_frames\action_1_frame_174.png', 'C:\D\pyproject\dongzuoshibie\video_frames\action_1_frame_230.png'], 'indices': [174, 230], 'start': 2.9}, {'end': 6.98, 'imgs': ['C:\D\pyproject\dongzuoshibie\video_frames\action_2_frame_272.png', 'C:\D\pyproject\dongzuoshibie\video_frames\action_2_frame_419.png'], 'indices': [272, 419], 'start': 4.53}, {'end': 7.58, 'imgs': ['C:\D\pyproject\dongzuoshibie\video_frames\action_3_frame_419.png', 'C:\D\pyproject\dongzuoshibie\video_frames\action_3_frame_455.png'], 'indices': [419, 455], 'start': 6.98}, {'end': 15.78, 'imgs': ['C:\D\pyproject\dongzuoshibie\video_frames\action_4_frame_908.png', 'C:\D\pyproject\dongzuoshibie\video_frames\action_4_frame_947.png'], 'indices': [908, 947], 'start': 15.13}, {'end': 20.98, 'imgs': ['C:\D\pyproject\dongzuoshibie\video_frames\action_5_frame_1222.png', 'C:\D\pyproject\dongzuoshibie\video_frames\action_5_frame_1259.png'], 'indices': [1222, 1259], 'start': 20.37}]

我要是很多幅度很大的动作的话根据移动幅度来分割动作边界也不太对吧有没有别的方法或者思路