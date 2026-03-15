import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

# ===================== 自定义参数 =====================
CROP_SIZE = 200  # 截取正方形的边长（可自由修改：150/200/250...）
SAVE_PATH = 'C:\\D\\pyproject\\dongzuoshibie\\video_frames\\'  # 保存路径
KP_CONF_THRESH = 0.5  # 关键点置信度阈值（超过才认为识别到，可自由调整）
# ======================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 修正拼写错误 + 修复权重加载
weights = torch.load('yolov7-w6-pose.pt', map_location=device, weights_only=False)
model = weights['model']
_ = model.float().eval()

if torch.cuda.is_available():
    model.half().to(device)

# 读取图片
image = cv2.imread('C:\\D\\pyproject\\dongzuoshibie\\video_frames\\frame_0000.jpg')
image = letterbox(image, 960, stride=64, auto=True)[0]
image_ = image.copy()
image = transforms.ToTensor()(image)
image = torch.tensor(np.array([image.numpy()]))

if torch.cuda.is_available():
    image = image.half().to(device)
output, _ = model(image)

# 姿态检测后处理
output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
with torch.no_grad():
    output = output_to_keypoint(output)

# 转换图像格式
nimg = image[0].permute(1, 2, 0) * 255
nimg = nimg.cpu().numpy().astype(np.uint8)
nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

# ===================== 【核心新增】备份原始干净图像（无任何标注） =====================
clean_img = nimg.copy()
# ====================================================================================

img_h, img_w = nimg.shape[:2]  # 获取图片宽高（用于边界限制）
half_size = CROP_SIZE // 2  # 矩形半长/半宽

# 初始化截取变量（防止未识别到手腕时报错）
left_roi = None
right_roi = None
# 新增：干净无标注的截取图变量
left_clean_roi = None
right_clean_roi = None

# 遍历检测到的每一个人
for idx in range(output.shape[0]):
    # 1. 绘制骨骼关键点（原代码不变，自带置信度过滤）
    plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

    # ===================== 显示关键点编号（仅置信度达标才标注） =====================
    print(f"\n==================== 第 {idx + 1} 个人的关键点检测结果 ====================")
    kp_name = ["鼻子", "左眼", "右眼", "左耳", "右耳", "左肩", "右肩", "左肘", "右肘",
               "左腕", "右腕", "左髋", "右髋", "左膝", "右膝", "左脚踝", "右脚踝"]
    for kp_id in range(17):
        x = output[idx, 7 + kp_id * 3]
        y = output[idx, 7 + kp_id * 3 + 1]
        conf = output[idx, 7 + kp_id * 3 + 2]

        print(f"关键点 {kp_id:2d}号 | {kp_name[kp_id]:<4} | 坐标({x:6.1f}, {y:6.1f}) | 置信度: {conf:.4f}")

        # ✅ 仅置信度超过阈值，才在图像上标注编号
        if conf > KP_CONF_THRESH:
            cv2.putText(nimg, str(kp_id), (int(x) + 6, int(y) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(nimg, str(kp_id), (int(x) + 6, int(y) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    # ============================================================================

    # ===================== 提取手腕坐标 + 置信度 =====================
    # 左腕：9号关键点 → x(34), y(35), 置信度(36)
    left_wrist_x = output[idx, 34]
    left_wrist_y = output[idx, 35]
    left_wrist_conf = output[idx, 36]
    # 右腕：10号关键点 → x(37), y(38), 置信度(39)
    right_wrist_x = output[idx, 37]
    right_wrist_y = output[idx, 38]
    right_wrist_conf = output[idx, 39]

    # ===================== 左腕截取：仅置信度达标才执行 =====================
    if left_wrist_conf > KP_CONF_THRESH:
        print(f"✅ 左腕识别成功！置信度: {left_wrist_conf:.4f}")
        x1 = int(left_wrist_x - half_size)
        y1 = int(left_wrist_y - half_size)
        x2 = int(left_wrist_x + half_size)
        y2 = int(left_wrist_y + half_size)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        # 截取带标注的图（原有）
        left_roi = nimg[y1:y2, x1:x2]
        # ===================== 【新增】截取干净无标注的图 =====================
        left_clean_roi = clean_img[y1:y2, x1:x2]
        # 绘制矩形框（仅画在可视化图上，不影响干净图）
        cv2.rectangle(nimg, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        print(f"❌ 左腕未识别到！置信度: {left_wrist_conf:.4f}")

    # ===================== 右腕截取：仅置信度达标才执行 =====================
    if right_wrist_conf > KP_CONF_THRESH:
        print(f"✅ 右腕识别成功！置信度: {right_wrist_conf:.4f}")
        x1_r = int(right_wrist_x - half_size)
        y1_r = int(right_wrist_y - half_size)
        x2_r = int(right_wrist_x + half_size)
        y2_r = int(right_wrist_y + half_size)
        x1_r = max(0, x1_r)
        y1_r = max(0, y1_r)
        x2_r = min(img_w, x2_r)
        y2_r = min(img_h, y2_r)
        # 截取带标注的图（原有）
        right_roi = nimg[y1_r:y2_r, x1_r:x2_r]
        # ===================== 【新增】截取干净无标注的图 =====================
        right_clean_roi = clean_img[y1_r:y2_r, x1_r:x2_r]
        # 绘制矩形框（仅画在可视化图上，不影响干净图）
        cv2.rectangle(nimg, (x1_r, y1_r), (x2_r, y2_r), (0, 0, 255), 2)
    else:
        print(f"❌ 右腕未识别到！置信度: {right_wrist_conf:.4f}")

# ===================== 保存结果（原有截图 + 新增干净截图） =====================
cv2.imwrite(SAVE_PATH + 'pose_with_box.png', nimg)
# 原有：带标注的截取图
if left_roi is not None:
    cv2.imwrite(SAVE_PATH + 'left_wrist_crop.png', left_roi)
if right_roi is not None:
    cv2.imwrite(SAVE_PATH + 'right_wrist_crop.png', right_roi)
# 新增：完全干净、无任何标注的截取图
if left_clean_roi is not None:
    cv2.imwrite(SAVE_PATH + 'left_wrist_clean.png', left_clean_roi)
if right_clean_roi is not None:
    cv2.imwrite(SAVE_PATH + 'right_wrist_clean.png', right_clean_roi)

# 显示结果（不变）
plt.figure(figsize=(8, 8))
plt.axis('off')
plt.imshow(cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB))
plt.show()

# 单独显示手腕截取图（不变，只显示带标注的）
plt.figure(figsize=(8, 4))
if left_roi is not None:
    plt.subplot(121), plt.imshow(cv2.cvtColor(left_roi, cv2.COLOR_BGR2RGB)), plt.title('Left Wrist'), plt.axis('off')
if right_roi is not None:
    plt.subplot(122), plt.imshow(cv2.cvtColor(right_roi, cv2.COLOR_BGR2RGB)), plt.title('Right Wrist'), plt.axis('off')
plt.show()