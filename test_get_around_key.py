import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

# ===================== 自定义参数 =====================
CROP_SIZE = 600
SAVE_PATH = 'C:\\D\\pyproject\\dongzuoshibie\\video_frames\\'
KP_CONF_THRESH = 0.1
# ======================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights = torch.load('yolov7-w6-pose.pt', map_location=device, weights_only=False)
model = weights['model']
model.float().eval()

if torch.cuda.is_available():
    model.half().to(device)

def get_around_key(img_path='C:\\D\\pyproject\\dongzuoshibie\\video_frames\\frame_0000.jpg'):

    # 1. 读取原始图像（BGR，颜色纯正）
    raw_img = cv2.imread(img_path)
    img_h, img_w = raw_img.shape[:2]

    # 2. Letterbox预处理（获取缩放比例+填充，用于坐标还原）
    image_processed, ratio, pad = letterbox(raw_img, 960, stride=64, auto=True)
    image_rgb = cv2.cvtColor(image_processed, cv2.COLOR_BGR2RGB)
    image_tensor = transforms.ToTensor()(image_rgb)
    image = torch.tensor(np.array([image_tensor.numpy()]))

    # 3. 模型推理
    with torch.no_grad():
        if torch.cuda.is_available():
            image = image.half().to(device)
        output, _ = model(image)

    # 4. 后处理
    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)

    # 5. 初始化绘图（原始图，颜色100%正确）
    vis_img = raw_img.copy()
    clean_img = raw_img.copy()
    half_size = CROP_SIZE // 2


    # 6. 坐标逆变换（核心：模型坐标 → 原始图像坐标）
    def revert_coords(x, y, ratio, pad):
        x_ori = (x - pad[0]) / ratio[0]
        y_ori = (y - pad[1]) / ratio[1]
        return int(round(x_ori)), int(round(y_ori))


    # 遍历检测结果
    for idx in range(output.shape[0]):
        # 提取所有关键点并还原坐标
        kpts = output[idx, 7:].T
        kpts_ori = []
        for i in range(0, len(kpts), 3):
            x, y, conf = kpts[i], kpts[i + 1], kpts[i + 2]
            x_ori, y_ori = revert_coords(x, y, ratio, pad)
            kpts_ori.extend([x_ori, y_ori, conf])

        # 🔥 修复：在原始numpy图上绘制骨骼（无报错）
        plot_skeleton_kpts(vis_img, np.array(kpts_ori).T, 3)

        # 标注关键点编号
        kp_name = ["鼻子", "左眼", "右眼", "左耳", "右耳", "左肩", "右肩", "左肘", "右肘",
                   "左腕", "右腕", "左髋", "右髋", "左膝", "右膝", "左脚踝", "右脚踝"]
        for kp_id in range(17):
            x, y, conf = kpts_ori[kp_id * 3], kpts_ori[kp_id * 3 + 1], kpts_ori[kp_id * 3 + 2]
            if conf > KP_CONF_THRESH:
                cv2.putText(vis_img, str(kp_id), (x + 6, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(vis_img, str(kp_id), (x + 6, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 提取手腕（已还原原始坐标）
        left_wrist_x, left_wrist_y, lconf = kpts_ori[27], kpts_ori[28], kpts_ori[29]
        right_wrist_x, right_wrist_y, rconf = kpts_ori[30], kpts_ori[31], kpts_ori[32]

        # 左腕裁剪（位置精准+颜色正确）
        if lconf > KP_CONF_THRESH:
            x1 = max(0, left_wrist_x - half_size)
            y1 = max(0, left_wrist_y - half_size)
            x2 = min(img_w, left_wrist_x + half_size)
            y2 = min(img_h, left_wrist_y + half_size)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            left_roi = clean_img[y1:y2, x1:x2]
            cv2.imwrite(SAVE_PATH + 'left_wrist_crop.png', left_roi)
            cv2.imwrite(SAVE_PATH + 'left_wrist_clean.png', left_roi)

        # 右腕裁剪（位置精准+颜色正确）
        if rconf > KP_CONF_THRESH:
            x1_r = max(0, right_wrist_x - half_size)
            y1_r = max(0, right_wrist_y - half_size)
            x2_r = min(img_w, right_wrist_x + half_size)
            y2_r = min(img_h, right_wrist_y + half_size)
            cv2.rectangle(vis_img, (x1_r, y1_r), (x2_r, y2_r), (0, 0, 255), 2)
            right_roi = clean_img[y1_r:y2_r, x1_r:x2_r]
            cv2.imwrite(SAVE_PATH + 'right_wrist_crop.png', right_roi)
            cv2.imwrite(SAVE_PATH + 'right_wrist_clean.png', right_roi)

    # 保存结果
    cv2.imwrite(SAVE_PATH + 'pose_with_box.png', vis_img)
    cv2.imwrite(SAVE_PATH + 'pose_result.png', clean_img)

    del image, output, image_tensor, image_processed
    torch.cuda.empty_cache()

    return SAVE_PATH + 'right_wrist_clean.png'
    # # 显示
    # plt.figure(figsize=(8, 8))
    # plt.axis('off')
    # plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    # plt.show()

if __name__ == "__main__":
    get_around_key()