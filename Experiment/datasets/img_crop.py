import cv2
import os

# 加载大图像
image_path = "/mnt/d/exp/"
output_folder = "/mnt/d/exp/hrsc_gf2/crop/"  # 输出文件夹
os.makedirs(output_folder, exist_ok=True)  # 如果不存在则创建

# 读取图像
image = cv2.imread(image_path)
image_height, image_width, _ = image.shape

# 设置裁剪窗口的大小和步长
crop_size = 800
step_size = 200

# 开始裁剪
crop_count = 0
for y in range(0, image_height - crop_size + 1, step_size):
    for x in range(0, image_width - crop_size + 1, step_size):
        # 裁剪图像
        cropped_image = image[y : y + crop_size, x : x + crop_size]

        # 保存裁剪的图像
        crop_filename = f"crop_{crop_count}.jpg"
        crop_path = os.path.join(output_folder, crop_filename)
        cv2.imwrite(crop_path, cropped_image)
        crop_count += 1

print(f"裁剪完成，总共裁剪了 {crop_count} 张小图。")
