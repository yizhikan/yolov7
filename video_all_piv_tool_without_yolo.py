import json

# ===================== 【测试版】工具定义：和原版完全一样 =====================
YOLO_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "yolo_action_segment",
            "description": "基于YOLO姿态的视频动作分割，可调节motion_thresh控制分段数量",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "视频绝对路径"},
                    "motion_thresh": {"type": "number", "description": "分割阈值0-20，越小分段越多"}
                },
                "required": ["video_path", "motion_thresh"]
            }
        }
    }
]

# ===================== 【测试版】工具函数：接口和原版完全一致 =====================
def yolo_action_segment(video_path: str, motion_thresh: float):
    """
    【模拟测试工具】无YOLO，纯根据阈值返回分段结果
    规则：阈值越小 → 分段越多（还原真实逻辑）
    """
    # 模拟分段数量：阈值越小，分段越多
    if motion_thresh >= 10:
        segment_count = 5
    elif 5 <= motion_thresh < 10:
        segment_count = 8
    elif 3 <= motion_thresh < 5:
        segment_count = 15  # 目标值
    elif 1 <= motion_thresh < 3:
        segment_count = 18
    else:
        segment_count = 22

    # 模拟生成时间段（格式和原版完全一致）
    segments = []
    start = 0.0
    for i in range(segment_count):
        end = start + 2.0
        segments.append({"start": round(start, 2), "end": round(end, 2)})
        start = end

    # 返回格式和原版完全一致
    return {
        "segment_count": segment_count,
        "segments": segments,
        "used_threshold": motion_thresh
    }