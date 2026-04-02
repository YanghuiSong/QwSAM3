# utils/visualization.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from PIL import Image

class Visualizer:
    """可视化分割结果"""
    
    @staticmethod
    def visualize_session(session, save_path=None):
        """可视化整个会话的结果"""
        if not session or "results" not in session:
            print("没有可可视化的结果")
            return
        
        image_path = session["image_path"]
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        results = session["results"]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # 原始图像
        axes[0].imshow(image_np)
        axes[0].set_title(f"原始图像\n{os.path.basename(image_path)}")
        axes[0].axis('off')
        
        # 分割结果
        axes[1].imshow(image_np)
        
        # 使用颜色映射
        colors = plt.cm.tab20(np.linspace(0, 1, min(len(results), 20)))
        
        for i, result in enumerate(results[:20]):  # 最多显示20个
            color = colors[i % len(colors)]
            bbox = result["bbox"]
            
            # 绘制边界框 [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            axes[1].add_patch(rect)
            
            # 显示分数
            label = f"{result.get('category', 'obj')}: {result['score']:.2f}"
            axes[1].text(
                x1, y1 - 5, label,
                color='white', fontsize=8,
                bbox=dict(facecolor=color, alpha=0.7)
            )
        
        axes[1].set_title(f"分割结果 ({len(results)}个物体)")
        axes[1].axis('off')
        
        # 添加会话信息
        info_text = (
            f"会话ID: {session['session_id']}\n"
            f"用户请求: {session['user_request'][:50]}...\n"
            f"千问3策略: {session.get('qwen_instructions', {}).get('strategy', 'N/A')[:50]}...\n"
            f"耗时: {session.get('duration', 0):.1f}秒"
        )
        
        plt.figtext(0.5, 0.01, info_text, ha="center", fontsize=9, 
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化已保存: {save_path}")
        
        plt.show()
        return fig
    
    @staticmethod
    def save_mask_overlay(image_path, results, save_path, alpha=0.5):
        """保存带掩码覆盖层的图像"""
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        overlay = image_np.copy()
        
        colors = plt.cm.tab20(np.linspace(0, 1, min(len(results), 20)))
        
        for i, result in enumerate(results[:20]):
            color = colors[i % len(colors)]
            mask = result.get("mask")
            
            if mask is not None:
                # 创建颜色覆盖层
                color_layer = np.zeros_like(image_np)
                color_layer[mask > 0.5] = (np.array(color[:3]) * 255).astype(np.uint8)
                
                # 混合覆盖层
                mask_binary = mask > 0.5
                overlay[mask_binary] = (
                    overlay[mask_binary] * (1 - alpha) + 
                    color_layer[mask_binary] * alpha
                ).astype(np.uint8)
        
        overlay_img = Image.fromarray(overlay)
        overlay_img.save(save_path)
        print(f"掩码覆盖图已保存: {save_path}")