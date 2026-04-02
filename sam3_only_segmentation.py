#!/usr/bin/env python3
"""
SAM3 Only Segmentation Program
Without Qwen Integration
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append('.')
from core.sam3_adapter_final import SAM3AdapterFinal
from visualization.segmentation_visualizer import SegmentationVisualizer


def load_class_prompts(file_path):
    """从文件中加载类别提示"""
    if not os.path.exists(file_path):
        print(f"Warning: Class prompt file does not exist: {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # 移除重复项，保持顺序
    unique_lines = []
    seen = set()
    for line in lines:
        if line not in seen:
            unique_lines.append(line)
            seen.add(line)
    
    return unique_lines


def process_single_image(image_path, class_prompts, model_path, output_dir, score_threshold=0.2):
    """处理单张图片"""
    print(f"Processing image: {image_path}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 初始化SAM3适配器
    print("Initializing SAM3 adapter...")
    try:
        sam_adapter = SAM3AdapterFinal(checkpoint=model_path)
    except Exception as e:
        print(f"Failed to initialize SAM3 adapter: {e}")
        return False
    
    # 设置图片
    if sam_adapter.set_image(image_path) is None:
        print("Failed to set image")
        return False
    
    print(f"Image set successfully. Starting segmentation with {len(class_prompts)} class prompts...")
    
    # 执行分割
    results = sam_adapter.exhaustive_segmentation(
        object_categories=class_prompts,
        score_threshold=score_threshold
    )
    
    if not results.get('masks'):
        print("No segmentation results found")
        return False
    
    print(f"Found {len(results['masks'])} segmentation results")
    
    # 保存结果
    save_results(results, image_path, output_dir, base_name)
    
    # 创建可视化
    create_visualizations(image_path, results, output_dir, base_name)
    
    print(f"Processing completed! Results saved in: {output_dir}/{base_name}_*")
    return True


def save_results(results, image_path, output_dir, base_name):
    """保存分割结果"""
    # 保存numpy文件
    npz_path = os.path.join(output_dir, f"{base_name}_results.npz")
    try:
        np.savez_compressed(
            npz_path,
            masks=np.array(results.get('masks', [])),
            categories=results.get('categories', []),
            scores=results.get('scores', []),
            boxes=results.get('boxes', [])
        )
        print(f"  ✓ Saved segmentation data: {npz_path}")
    except Exception as e:
        print(f"  ✗ Failed to save segmentation data: {e}")
    
    # 保存文本摘要
    txt_path = os.path.join(output_dir, f"{base_name}_summary.txt")
    try:
        with open(txt_path, 'w') as f:
            f.write(f"Segmentation Results Summary\n")
            f.write(f"="*50 + "\n")
            f.write(f"Source Image: {image_path}\n")
            f.write(f"Total Regions: {len(results.get('masks', []))}\n")
            
            if results.get('categories'):
                from collections import Counter
                category_counts = Counter(results['categories'])
                f.write(f"\nCategory Distribution:\n")
                for category, count in category_counts.most_common():
                    f.write(f"  {category}: {count} regions\n")
            
            if results.get('scores'):
                scores = results['scores']
                f.write(f"\nConfidence Statistics:\n")
                f.write(f"  Mean: {np.mean(scores):.3f}\n")
                f.write(f"  Median: {np.median(scores):.3f}\n")
                f.write(f"  Min: {np.min(scores):.3f}\n")
                f.write(f"  Max: {np.max(scores):.3f}\n")
        
        print(f"  ✓ Saved result summary: {txt_path}")
    except Exception as e:
        print(f"  ✗ Failed to save summary: {e}")


def create_visualizations(image_path, results, output_dir, base_name):
    """创建可视化图像"""
    visualizer = SegmentationVisualizer()
    
    # 1. 综合可视化
    vis_path = os.path.join(output_dir, f"{base_name}_comprehensive.png")
    try:
        visualizer.create_comprehensive_visualization(
            image_path, 
            results, 
            save_path=vis_path
        )
        print(f"  ✓ Saved comprehensive visualization: {vis_path}")
    except Exception as e:
        print(f"  ✗ Failed to create comprehensive visualization: {e}")
    
    # 2. 简单叠加图
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    try:
        create_simple_overlay(image_path, results, overlay_path)
        print(f"  ✓ Saved overlay: {overlay_path}")
    except Exception as e:
        print(f"  ✗ Failed to create overlay: {e}")
    
    # 3. 类别遮罩可视化
    mask_path = os.path.join(output_dir, f"{base_name}_masks.png")
    try:
        create_mask_visualization(image_path, results, mask_path)
        print(f"  ✓ Saved mask visualization: {mask_path}")
    except Exception as e:
        print(f"  ✗ Failed to create mask visualization: {e}")


def create_simple_overlay(image_path, results, save_path):
    """创建简单叠加图"""
    # 读取图片
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    
    # 创建叠加图
    overlay = image_rgb.copy().astype(float) / 255.0
    masks = results.get('masks', [])
    categories = results.get('categories', [])
    
    if not masks:
        return
    
    # 为每个类别分配颜色
    unique_categories = list(set(categories))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_categories)))
    color_map = {cat: colors[i][:3] for i, cat in enumerate(unique_categories)}
    
    # 按区域大小排序（从大到小）
    mask_sizes = [(i, mask.sum()) for i, mask in enumerate(masks)]
    mask_sizes.sort(key=lambda x: x[1], reverse=True)
    
    for i, _ in mask_sizes:
        mask = masks[i]
        category = categories[i]
        
        if mask is None:
            continue
        
        # 调整mask尺寸
        mask_resized = cv2.resize(mask.astype(float), (w, h))
        mask_binary = mask_resized > 0.5
        
        if np.any(mask_binary):
            color = color_map.get(category, np.array([1, 0, 0]))
            
            # 创建彩色区域
            colored_region = np.zeros_like(overlay)
            colored_region[mask_binary] = color
            
            # 叠加
            alpha = 0.3
            mask_alpha = mask_binary.astype(float) * alpha
            mask_alpha_3d = np.stack([mask_alpha] * 3, axis=2)
            
            overlay = overlay * (1 - mask_alpha_3d) + colored_region * mask_alpha_3d
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    plt.imshow(np.clip(overlay, 0, 1))
    plt.title(f'Segmentation Overlay - {len(masks)} regions', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_mask_visualization(image_path, results, save_path):
    """创建mask可视化"""
    # 读取图片
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    
    masks = results.get('masks', [])
    categories = results.get('categories', [])
    
    if not masks:
        return
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 原始图片
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image', fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    # 不同类别的mask
    unique_categories = list(set(categories))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_categories)))
    
    for i, category in enumerate(unique_categories[:5]):  # 最多显示5个类别
        if i+1 >= len(axes):
            break
        
        # 合并此类别的所有mask
        category_mask = np.zeros((h, w), dtype=float)
        for mask, cat in zip(masks, categories):
            if cat == category:
                mask_resized = cv2.resize(mask.astype(float), (w, h))
                category_mask = np.maximum(category_mask, mask_resized)
        
        if category_mask.sum() > 0:
            # 创建彩色mask
            color_mask = np.zeros((h, w, 3))
            for c in range(3):
                color_mask[:, :, c] = colors[i][c] * category_mask
            
            # 混合
            blended = image_rgb.copy().astype(float) / 255.0
            alpha = category_mask * 0.6
            alpha_3d = np.stack([alpha] * 3, axis=2)
            blended = blended * (1 - alpha_3d) + color_mask * alpha_3d
            
            axes[i+1].imshow(np.clip(blended, 0, 1))
            axes[i+1].set_title(f'{category} Mask', fontsize=11, fontweight='bold')
        else:
            axes[i+1].text(0.5, 0.5, 'No regions', ha='center', va='center')
            axes[i+1].set_title(f'{category} Mask')
        
        axes[i+1].axis('off')
    
    # 隐藏额外的子图
    for j in range(i+2, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('Category Mask Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='SAM3 Only Segmentation Program (No Qwen)')
    parser.add_argument('image_path', help='Input image path')
    parser.add_argument('--class-prompts-file', 
                       help='Text file containing class prompts, one per line')
    parser.add_argument('--vaihingen', action='store_true',
                       help='Use Vaihingen class prompts')
    parser.add_argument('--potsdam', action='store_true',
                       help='Use Potsdam class prompts')
    parser.add_argument('--loveda', action='store_true',
                       help='Use LoveDA class prompts')
    parser.add_argument('--isaid', action='store_true',
                       help='Use iSAID class prompts')
    parser.add_argument('--model-path', 
                       default=r'D:\SYH\CodeReading\sam3_interactive_system\sam3.pt',
                       help='Path to SAM3 model checkpoint')
    parser.add_argument('--assets-dir', 
                       default=r'D:\SYH\CodeReading\sam3_interactive_system\assets',
                       help='Assets directory for SAM3')
    parser.add_argument('--output-dir', default='./sam3_results', 
                       help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.2,
                       help='Score threshold for segmentation')
    
    args = parser.parse_args()
    
    # 验证输入图像路径
    if not os.path.exists(args.image_path):
        print(f"Error: Image does not exist {args.image_path}")
        sys.exit(1)
    
    # 加载类别提示
    class_prompts = []
    
    if args.class_prompts_file:
        class_prompts.extend(load_class_prompts(args.class_prompts_file))
    
    if args.vaihingen:
        class_prompts.extend(load_class_prompts(r"D:\SYH\CodeReading\sam3_interactive_system\cls_vaihingen.txt"))
    
    if args.potsdam:
        class_prompts.extend(load_class_prompts(r"D:\SYH\CodeReading\sam3_interactive_system\cls_potsdam.txt"))
    
    if args.loveda:
        class_prompts.extend(load_class_prompts(r"D:\SYH\CodeReading\sam3_interactive_system\cls_loveda.txt"))
    
    if args.isaid:
        class_prompts.extend(load_class_prompts(r"D:\SYH\CodeReading\sam3_interactive_system\cls_iSAID.txt"))
    
    # 如果没有指定任何类别提示，则使用全部
    if not class_prompts:
        print("No class prompts specified, using all available class files...")
        class_prompts.extend(load_class_prompts(r"D:\SYH\CodeReading\sam3_interactive_system\cls_vaihingen.txt"))
        class_prompts.extend(load_class_prompts(r"D:\SYH\CodeReading\sam3_interactive_system\cls_potsdam.txt"))
        class_prompts.extend(load_class_prompts(r"D:\SYH\CodeReading\sam3_interactive_system\cls_loveda.txt"))
        class_prompts.extend(load_class_prompts(r"D:\SYH\CodeReading\sam3_interactive_system\cls_iSAID.txt"))
    
    if not class_prompts:
        print("Error: No class prompts available")
        sys.exit(1)
    
    print(f"Loaded {len(class_prompts)} class prompts")
    print(f"Class prompts: {class_prompts}")
    
    # 处理图像
    success = process_single_image(
        args.image_path, 
        class_prompts, 
        args.model_path, 
        args.output_dir, 
        args.threshold
    )
    
    if success:
        print("Processing completed successfully!")
    else:
        print("Processing failed!")


if __name__ == "__main__":
    main()