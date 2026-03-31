import json
from openai import OpenAI
# 导入工具函数（仅保留函数，无需工具定义）
from video_all_piv_tool_without_yolo import yolo_action_segment

# ===================== 原版API配置（无修改） =====================
client = OpenAI(
    api_key="sk-1777fc7b981c4e9f90dcb10fddbd7ba8",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ===================== 核心配置 =====================
VIDEO_PATH = "C:\\D\\pyproject\\dongzuoshibie\\线束管路连接better.mp4"
TARGET_SEG_COUNT = 15


# 【核心】强指令Prompt：强制模型必须调用工具，输出固定JSON，全自动执行
SYSTEM_PROMPT = f"""你是专业的视频动作分割智能体，严格执行以下指令，禁止输出任何多余文字：
1. 必须调用工具：yolo_action_segment
2. 工具参数：
   - video_path：{VIDEO_PATH}
   - motion_thresh：0~20之间的数值（越小分段越多，越大分段越少）
3. 你的唯一输出格式：纯JSON，无其他内容
   {{"action":"call_tool","motion_thresh":数值}}
4. 任务目标：不断调整motion_thresh，直到分割出≈{TARGET_SEG_COUNT}个动作
5. 仅在分段达标后，输出最终结果文本"""

# 初始化对话（无用户输入，全自动启动）
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "开始执行任务，立即调用工具"}
]

# ===================== 全自动循环执行 =====================
if __name__ == "__main__":
    while True:
        # 1. 模型推理：无任何tools参数，纯文本对话
        response = client.chat.completions.create(
            model="qwen3.5-plus",
            messages=messages
        )
        model_output = response.choices[0].message.content.strip()
        print("\n🤖 模型输出：", model_output)

        try:
            # 2. 解析模型输出的工具调用指令
            tool_data = json.loads(model_output)

            # 3. 自动执行工具
            if tool_data.get("action") == "call_tool":
                thresh = tool_data["motion_thresh"]
                print(f"\n🤖 模型自动调用工具 | 阈值={thresh}")

                # 执行动作分割
                result = yolo_action_segment(
                    video_path=VIDEO_PATH,
                    motion_thresh=thresh
                )
                print(f"📊 分割结果：{result['segment_count']} 段")

                # 4. 将工具结果返回给模型，让其判断是否继续调节
                messages.append({"role": "assistant", "content": model_output})
                messages.append({
                    "role": "user",
                    "content": f"工具返回结果：{json.dumps(result, ensure_ascii=False)}，请继续执行"
                })

                # 5. 终止条件：分段数达标
                if abs(result['segment_count'] - TARGET_SEG_COUNT) <= 1:
                    print("\n✅ 任务完成！分段数量达标")
                    print("最终动作分割结果：")
                    for i, seg in enumerate(result['segments']):
                        print(f"动作{i + 1}：{seg['start']}s ~ {seg['end']}s")
                    break

        # 非JSON格式 = 模型输出最终结果
        except json.JSONDecodeError:
            print("\n📝 模型最终输出：", model_output)
            break