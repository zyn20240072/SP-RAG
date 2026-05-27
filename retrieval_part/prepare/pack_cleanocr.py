#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import os
import json
import re
from tqdm import tqdm

# ==========================================
# 1. 路径配置 (请确保这里的路径与你服务器上的实际路径一致)
# ==========================================
TXT_FOLDER = "/root/nas/RAG/OURRAG/v9/texts/mmlongbench/ppocr_results"
OUTPUT_JSONL_PATH = "/root/nas/RAG/OURRAG/v9/texts/mmlongbench/mmlongbench_ocr_formatted.jsonl"

# ==========================================
# 2. 核心清洗逻辑 (解决离散换行与多栏排版问题)
# ==========================================
def clean_ocr_text(text: str) -> str:
    """
    智能清洗 OCR 文本：
    1. 保护版面鸿沟（多栏/表格），转为制表符
    2. 压缩意外的多余单词空格
    3. 压缩过多的连续换行符
    """
    if not text:
        return ""
        
    # [规则 1] 保护“版面鸿沟”：
    # 如果连续出现 4 个及以上的空格，说明这里极大概率是报纸的多栏间距或表格的列间距。
    # 我们将其替换为 " \t " (制表符)，明确告诉向量模型这里是物理分割。
    text = re.sub(r'[ \t]{4,}', ' \t ', text)
    
    # [规则 2] 清洗“普通词间距”：
    # 经过上面一步后，剩下的 2~3 个连续空格通常是 OCR 识别时的“小手抖”。
    # 我们把它们安全地压缩为 1 个正常的英文空格。
    text = re.sub(r' {2,3}', ' ', text)
    
    # [规则 3] 压缩“连续换行”：
    # 老版代码会生成很多诸如 \n\n\n\n 的无意义换行来凑高度。
    # 我们把所有包含 2 个及以上换行符的空白块，全部压缩为标准的双换行 (\n\n)，代表段落结束。
    text = re.sub(r'\n[ \t]*\n+', '\n\n', text)
    
    return text.strip()

# ==========================================
# 3. 遍历、读取、清洗与打包主程序
# ==========================================
def pack_txt_to_jsonl(input_folder: str, output_path: str):
    print(f"📂 正在扫描目录: {input_folder}")
    
    if not os.path.exists(input_folder):
        print(f"❌ 找不到文件夹: {input_folder}，请检查路径。")
        return
        
    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    print(f"🔍 共找到 {len(txt_files)} 个 TXT 文件。准备进行清洗和打包...")
    
    success_count = 0
    empty_count = 0

    # 打开目标文件，准备逐行写入 (避免一次性加载撑爆内存)
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for filename in tqdm(txt_files, desc="清洗与打包进度", ncols=100):
            txt_path = os.path.join(input_folder, filename)
            
            try:
                # 1. 读取老版 OCR 生成的生肉文本
                with open(txt_path, 'r', encoding='utf-8') as in_f:
                    raw_text = in_f.read()
                
                # 2. 调用智能清洗函数
                cleaned_text = clean_ocr_text(raw_text)
                
                # 3. 过滤并删除空文件
                if not cleaned_text:
                    empty_count += 1
                    try:
                        os.remove(txt_path)  # 🔥 物理删除这个空的 txt 文件
                        print(f"🗑️ 已删除无内容的空文件: {txt_path}") # (可选) 开启这行会在控制台提示
                    except OSError as e:
                        print(f"⚠️ 删除文件失败 {txt_path}: {e}")
                    continue
                
                # 4. 生成唯一 text_id (去掉 .txt 后缀)
                # 这个 ID 对于你后续跟 Parquet 里的 reference_text_id 做精确匹配至关重要
                text_id = os.path.splitext(filename)[0]
                
                # 5. 组装为标准字典格式
                record = {
                    "filename": filename,
                    "text_id": text_id,
                    "text": cleaned_text
                }
                
                # 6. 转为单行 JSON 字符串并写入文件
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                success_count += 1
                
            except Exception as e:
                print(f"\n❌ 读取/处理文件 {filename} 时出错: {e}")

    # ==========================================
    # 4. 打印最终统计信息
    # ==========================================
    print("\n" + "="*55)
    print(f"✅ 清洗打包任务圆满完成！")
    print(f"📄 成功写入干净数据: {success_count} 条")
    if empty_count > 0:
        print(f"⚠️ 自动跳过空文件: {empty_count} 个")
    print(f"💾 最终 JSONL 文件已保存至:\n   {output_path}")
    print("="*55)

if __name__ == "__main__":
    pack_txt_to_jsonl(TXT_FOLDER, OUTPUT_JSONL_PATH)