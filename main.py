import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from PIL import Image


def kmeans_color_quantization(hsv_image, k=2):
    """
    使用 K-means 聚类找到图像中的主要颜色。
    """
    # 重新调整图像大小以加速聚类过程
    resized_image = cv2.resize(hsv_image, (0, 0), fx=0.5, fy=0.5)

    # 将图像数据转换为二维数组，每行是一个像素的 HSV 值
    reshaped_image = resized_image.reshape(-1, 3)

    # 应用 K-means 聚类
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(reshaped_image)

    # 返回聚类中心
    return kmeans.cluster_centers_


def extract_seal_with_kmeans(image_path, output_path, k=4, hue_threshold=15):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found at {image_path}")
        return

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dominant_colors = kmeans_color_quantization(hsv_image, k=k)

    masks = []
    for color in dominant_colors:
        lower_color = np.array([color[0] - hue_threshold, 100, 100])
        upper_color = np.array([color[0] + hue_threshold, 255, 255])
        mask = cv2.inRange(hsv_image, lower_color, upper_color)
        masks.append(mask)
    combined_mask = cv2.bitwise_or(masks[0], masks[1]) if len(masks) > 1 else masks[0]

    # 膨胀掩码
    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    # 提取印章区域
    seal = cv2.bitwise_and(image, image, mask=final_mask)

    # 将OpenCV图像转换为PIL图像
    original_image = Image.open(image_path).convert('RGBA')
    seal_pil = Image.fromarray(cv2.cvtColor(seal, cv2.COLOR_BGR2RGBA)).convert('RGBA')

    # 计算新图像的宽度，原图宽度加上印章区域宽度
    original_width, original_height = original_image.size
    seal_width, seal_height = seal_pil.size
    new_width = original_width + seal_width
    new_height = max(original_height, seal_height)

    # 创建一个全透明的图像，用于左右拼接
    transparent_background = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))

    # 将原图粘贴到透明背景上
    transparent_background.paste(original_image, (0, 0))

    # 将印章区域粘贴到透明背景的右侧
    transparent_background.paste(seal_pil, (original_width, 0), seal_pil)

    # 保存结果
    transparent_background.save(output_path)

    print(f"Original image and extracted seal combined with transparent background saved to {output_path}")


def batch_extract_seals_with_dominant_color(input_dir, output_dir):
    """
    批量处理文件夹中的所有图像，使用主要颜色进行印章提取。

    参数:
    - input_dir: 包含输入图像的文件夹路径。
    - output_dir: 输出图像的保存文件夹路径。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0]+'.png')
        extract_seal_with_kmeans(image_path, output_path)


# 使用示例
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch extract seals with dominant color from images.")
    parser.add_argument("--input_dir", default='/data/projects/Matting/modnet_demo/seals',
                        help="Path to the folder containing input images.")
    parser.add_argument("--output_dir", default='/data/projects/Matting/modnet_demo/output_seals_01',
                        help="Path to the folder for saving output images.")
    args = parser.parse_args()

    batch_extract_seals_with_dominant_color(args.input_dir, args.output_dir)
