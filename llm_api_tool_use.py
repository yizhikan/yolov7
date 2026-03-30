import json
from openai import OpenAI
# 导入封装好的YOLO工具
from video_all_piv_tool_without_yolo import yolo_action_segment, YOLO_TOOL

# ===================== 你的原版API配置（100%无修改） =====================
client = OpenAI(
    api_key="sk-1777fc7b981c4e9f90dcb10fddbd7ba8",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ===================== 核心配置 =====================
VIDEO_PATH = "C:\\D\\pyproject\\dongzuoshibie\\线束管路连接better.mp4"
# 目标：分割出15个动作（你的标准动作列表长度）
TARGET_SEG_COUNT = 15
# 初始化对话
messages = [
    {
        "role": "user",
        "content": f"""你是动作分割智能体，需要调用yolo_action_segment工具分割视频。
规则：
1. 视频路径：{VIDEO_PATH}
2. 目标分段数：{TARGET_SEG_COUNT}
3. 工具参数motion_thresh：0~20，越小分段越多，越大分段越少
4. 自动调节阈值，直到分段数≈{TARGET_SEG_COUNT}
5. 只输出工具调用结果，无需多余文字"""
    }
]

# ===================== 大模型自主循环调用Tool =====================
if __name__ == "__main__":
    while True:
        # 1. 大模型思考：是否调用Tool/设置什么阈值
        response = client.chat.completions.create(
            model="qwen3.5-plus",
            messages=messages,
            tools=YOLO_TOOL,  # 传入工具
            tool_choice="auto"
        )
        answer = response.choices[0].message

        # 2. 如果大模型决定调用Tool
        if answer.tool_calls:
            # 解析大模型自动生成的参数
            tool_call = answer.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            print(f"\n🤖 大模型决策：调用工具 | 阈值={args['motion_thresh']}")

            # 3. 执行YOLO工具，返回结果给大模型
            tool_result = yolo_action_segment(** args)
            print(f"📊 分割结果：{tool_result['segment_count']}段")

            # 4. 把工具结果返回给大模型，让它判断是否继续调节
            messages.append(answer)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": "yolo_action_segment",
                "content": json.dumps(tool_result, ensure_ascii=False)
            })

            # 5. 终止条件：大模型认为分段合理
            if abs(tool_result['segment_count'] - TARGET_SEG_COUNT) <= 1:
                print("\n✅ 大模型判定：分段合理，任务完成！")
                print("最终分割结果：")
                for idx, seg in enumerate(tool_result['segments']):
                    print(f"动作{idx+1}：{seg['start']}s ~ {seg['end']}s")
                break

        # 大模型直接输出结果
        else:
            print("最终结果：", answer.content)
            break