#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import json
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
from tqdm import tqdm  # 导入进度条库

# ----------------------- Hardcoded Constants -----------------------

# Dataset name
dataset_name = 'mmlongbench'  # 数据集名称
# 图片文件夹路径
IMAGE_PATH = f'images/{dataset_name}_img'  # 图片所在文件夹路径

# OCR配置
BACKEND = "gpu"  # 可选择"gpu"或"cpu"
DEVICE_ID = 0     # 如果使用GPU，可以设置设备ID

# ----------------------- Function Definitions -----------------------

def decode_image(image_path, max_size=(3136, 3136)):
    """
    解码图像并调整大小以减少内存使用。
    """
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    elif isinstance(image_path, Image.Image):
        image = image_path
    else:
        raise ValueError("无效的图像路径或图像对象。")
    
    # 调整图像大小以减少内存使用，保持宽高比
    image.thumbnail(max_size)
    
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def calculate_spaces_and_newlines(current_box, previous_box, space_threshold=45, line_threshold=15):
    """计算两个文本框之间的空格和换行数。"""
    spaces = 0
    newlines = 0
    
    # 判断文本框是否在同一行
    if abs(current_box[1] - previous_box[1]) < line_threshold:
        spaces = max(1, int(abs(current_box[0] - previous_box[0]) / space_threshold))
    else:
        newlines = max(1, int(abs(current_box[1] - previous_box[1]) / line_threshold))
    
    return spaces, newlines

def tostr_layout_preserving(result, space_threshold=45, line_threshold=15):
    """将OCR结果转换为保留布局的合并字符串 (修复版)"""
    text_boxes = []
    for res in result:
        coords = res['polygon']
        if len(coords) == 4 and all(isinstance(c, (list, np.ndarray)) and len(c) == 2 for c in coords):  
            try:
                # 计算多边形的中心点
                center_x = (coords[0][0] + coords[2][0]) / 2
                center_y = (coords[0][1] + coords[2][1]) / 2
                text_boxes.append((center_x, center_y, res['text'], coords))
            except Exception as e:
                continue

    # 1. 先进行一次粗略的从上到下排序
    text_boxes = sorted(text_boxes, key=lambda x: x[1])
    
    # 2. 【核心修复】行聚类：把 Y 坐标相近的框分到同一行，再按 X 排序
    lines = []
    current_line = []
    last_y = None
    
    for box in text_boxes:
        if last_y is None:
            current_line.append(box)
            last_y = box[1]
        elif abs(box[1] - last_y) < line_threshold: 
            # 如果 Y 坐标差距小于行容差，说明是同一行
            current_line.append(box)
            # 动态更新当前行的基准 Y，避免长表格倾斜漂移
            last_y = (last_y + box[1]) / 2 
        else: 
            # 换行了：先将上一行按 X 坐标（从左到右）排好序，装入结果
            lines.append(sorted(current_line, key=lambda x: x[0]))
            current_line = [box] # 开启新的一行
            last_y = box[1]
            
    # 把最后一行也加进去
    if current_line:
        lines.append(sorted(current_line, key=lambda x: x[0]))
        
    # 将二维的行列表展平回一维列表，此时顺序已完美修复
    text_boxes_sorted = [box for line in lines for box in line]
    
    # 3. 按修复后的顺序拼接文本和空格
    merged_text = []
    previous_box = None
    for box in text_boxes_sorted:
        if previous_box is not None:
            # 使用原有的计算逻辑，但顺序已经是正确的了
            spaces, newlines = calculate_spaces_and_newlines(box, previous_box, space_threshold, line_threshold)
            merged_text.append('\n' * newlines + ' ' * spaces)
        merged_text.append(box[2])
        previous_box = box

    return ''.join(merged_text)

# 初始化PaddleOCR实例
ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, device=BACKEND)

# ----------------------- Main Function -----------------------
def main(img_name, json_path):
    image = decode_image(img_name)
    result = ocr.predict(image)  # 运行OCR推理

    # 检查OCR结果是否包含必需的字段
    if 'rec_texts' not in result[0] or 'rec_boxes' not in result[0]:
        print(f"处理失败: {img_name} - 缺少OCR结果字段。")
        return ""  # 返回空文本

    # 将完整的结果保存到JSON文件
    result[0].save_to_json(json_path)  # 保存为JSON文件

    # 提取rec_texts和rec_boxes
    rec_texts = result[0]['rec_texts']
    rec_boxes = result[0]['rec_polys']

    # 将OCR结果处理为保留布局的字符串
    formatted_result = []
    for text, box in zip(rec_texts, rec_boxes):
        text_info = {
            'text': text,  # 提取识别的文本
            'polygon': box  # 识别文本框的多边形坐标
        }
        formatted_result.append(text_info)

    # 转换结果为保留布局的字符串
    text = tostr_layout_preserving(formatted_result)
    return text

# if __name__ == "__main__":
if __name__ == "__main__":
    # 定义输出文件夹
    results_folder = f'./texts/{dataset_name}/ppocr_results'
    json_folder = f'./texts/{dataset_name}/json_results'
    os.makedirs(results_folder, exist_ok=True)  # 确保结果文件夹存在
    os.makedirs(json_folder, exist_ok=True)  # 确保JSON文件夹存在
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith(('.jpg', '.png'))]

    # 使用tqdm显示进度条
    for img_name in tqdm(image_files, desc="处理图片", ncols=100, unit="个", position=0, leave=True):
        img_path = os.path.join(IMAGE_PATH, img_name)

        # 创建输出文件的路径
        txt_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        json_name = txt_name.replace('.txt', '.json')  # JSON文件与图片同名
        
        # 定义输出文本和JSON的路径
        txt_path = os.path.join(results_folder, os.path.basename(txt_name))
        json_path = os.path.join(json_folder, os.path.basename(json_name))
        
        # 如果结果文本文件不存在，则进行处理
        if not os.path.exists(txt_path):
            # 调用主函数进行图片处理
            text = main(img_path, json_path)
            
            # 如果没有返回文本，则跳过保存文件
            if text:
                # 将OCR结果保存为文本文件
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                print(f"OCR结果已保存到 {txt_path}")
            else:
                print(f"跳过 {img_name}: OCR处理出错。")
        else:
            print(f"{img_name} 的文本文件已存在。")
