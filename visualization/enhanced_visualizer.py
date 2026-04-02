import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from typing import Dict, List, Tuple
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba
import seaborn as sns

class EnhancedVisualizer:
    """增强型分割结果可视化"""
    
    def __init__(self, colormap='tab20'):
        self.colormap = plt.cm.get_cmap(colormap)
        self.category_colors = {}
        
    def create_comprehensive_visualization(self, image_path, results: Dict, 
                                         save_path=None, figsize=(20, 12)):
        """
        创建综合可视化图表
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print("无法读取图像")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        # 准备数据
        masks = results.get('masks', [])
        categories = results.get('categories', [])
        scores = results.get('scores', [])
        
        # 创建图形
        fig = plt.figure(figsize=figsize)
        
        # 子图布局
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. 原始图像
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image_rgb)
        ax1.set_title('原始图像', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 2. 分割叠加图
        ax2 = fig.add_subplot(gs[0, 1])
        overlay = self._create_overlay(image_rgb, masks, categories)
        ax2.imshow(overlay)
        ax2.set_title(f'分割叠加 ({len(masks)}个区域)', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # 3. 类别分布图
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_category_distribution(ax3, categories)
        
        # 4. 置信度分布
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_score_distribution(ax4, scores)
        
        # 5. 最大区域展示
        ax5 = fig.add_subplot(gs[1, 0])
        self._show_top_regions(ax5, image_rgb, masks, categories, scores, n=1)
        
        # 6. 区域大小分布
        ax6 = fig.add_subplot(gs[1, 1])
        self._plot_region_sizes(ax6, masks)
        
        # 7. 类别热图
        ax7 = fig.add_subplot(gs[1, 2:])
        self._create_category_heatmap(ax7, masks, categories, (h, w))
        
        # 8. 边界框可视化
        ax8 = fig.add_subplot(gs[2, :2])
        self._show_bbox_overlay(ax8, image_rgb, masks, scores)
        
        # 9. 区域统计表
        ax9 = fig.add_subplot(gs[2, 2:])
        self._create_statistics_table(ax9, masks, categories, scores)
        
        # 标题
        plt.suptitle(f'智能全分割结果分析 - {len(masks)}个分割区域', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 保存
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"✓ 可视化结果保存到: {save_path}")
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def _create_overlay(self, image, masks, categories, alpha=0.4):
        """创建分割叠加图"""
        overlay = image.copy().astype(float) / 255.0
        
        # 为每个类别分配颜色
        unique_categories = list(set(categories))
        for i, cat in enumerate(unique_categories):
            self.category_colors[cat] = self.colormap(i / max(1, len(unique_categories) - 1))
        
        # 绘制每个区域
        for mask, category in zip(masks, categories):
            if mask is None:
                continue
            
            # 调整mask尺寸
            mask_resized = cv2.resize(mask.astype(float), (image.shape[1], image.shape[0]))
            mask_binary = mask_resized > 0.5
            
            if np.any(mask_binary):
                # 获取颜色
                color = self.category_colors.get(category, np.array([1, 0, 0, 0.3]))
                
                # 确保颜色有alpha通道
                if len(color) == 3:
                    color = np.append(color, alpha)
                elif len(color) == 4:
                    color = list(color)
                    color[3] = alpha
                    color = np.array(color)
                
                # 创建彩色区域
                colored_region = np.zeros_like(overlay)
                colored_region[mask_binary] = color[:3]  # RGB通道
                
                # 叠加
                mask_alpha = mask_binary.astype(float) * alpha
                mask_alpha_3d = np.stack([mask_alpha] * 3, axis=2)
                
                overlay = overlay * (1 - mask_alpha_3d) + colored_region * mask_alpha_3d
        
        return np.clip(overlay, 0, 1)
    
    def _plot_category_distribution(self, ax, categories):
        """绘制类别分布"""
        if not categories:
            ax.text(0.5, 0.5, '无类别数据', ha='center', va='center')
            ax.set_title('类别分布')
            ax.axis('off')
            return
        
        from collections import Counter
        category_counts = Counter(categories)
        
        # 获取最常类别
        top_categories = category_counts.most_common(10)
        labels = [cat for cat, _ in top_categories]
        counts = [count for _, count in top_categories]
        
        # 创建颜色
        colors = [self.category_colors.get(cat, '#1f77b4') for cat in labels]
        
        # 绘制水平条形图
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, counts, color=colors, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('区域数量')
        ax.set_title('类别分布 (Top 10)', fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_score_distribution(self, ax, scores):
        """绘制置信度分布"""
        if not scores:
            ax.text(0.5, 0.5, '无分数数据', ha='center', va='center')
            ax.set_title('置信度分布')
            ax.axis('off')
            return
        
        # 直方图
        ax.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(scores), color='red', linestyle='--', 
                  label=f'平均: {np.mean(scores):.3f}')
        ax.axvline(np.median(scores), color='green', linestyle='--', 
                  label=f'中位数: {np.median(scores):.3f}')
        
        ax.set_xlabel('置信度')
        ax.set_ylabel('频次')
        ax.set_title('置信度分布', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _show_top_regions(self, ax, image, masks, categories, scores, n=3):
        """展示最高分的区域"""
        if not masks:
            ax.text(0.5, 0.5, '无分割区域', ha='center', va='center')
            ax.set_title('最高分区域')
            ax.axis('off')
            return
        
        # 按分数排序
        sorted_indices = np.argsort(scores)[::-1]
        
        # 创建子图
        n_cols = min(n, 3)
        n_rows = (n + n_cols - 1) // n_cols
        
        if n > 1:
            ax.clear()
            subfig = ax.get_figure().add_gridspec(n_rows, n_cols, hspace=0.1, wspace=0.1)
            
            for i, idx in enumerate(sorted_indices[:n]):
                row = i // n_cols
                col = i % n_cols
                sub_ax = ax.get_figure().add_subplot(subfig[row, col])
                
                mask = masks[idx]
                category = categories[idx] if idx < len(categories) else "Unknown"
                score = scores[idx] if idx < len(scores) else 0
                
                # 创建高亮显示
                highlighted = self._highlight_region(image, mask)
                sub_ax.imshow(highlighted)
                sub_ax.set_title(f'{category}\n分数: {score:.3f}', fontsize=9)
                sub_ax.axis('off')
            
            ax.axis('off')
            ax.set_title(f'最高分区域 (Top {n})', fontsize=11)
        else:
            # 只显示一个
            idx = sorted_indices[0]
            mask = masks[idx]
            category = categories[idx] if idx < len(categories) else "Unknown"
            score = scores[idx] if idx < len(scores) else 0
            
            highlighted = self._highlight_region(image, mask)
            ax.imshow(highlighted)
            ax.set_title(f'{category}\n分数: {score:.3f}', fontsize=10)
            ax.axis('off')
    
    def _highlight_region(self, image, mask, highlight_color=(1, 0, 0, 0.5)):
        """高亮显示特定区域"""
        h, w = image.shape[:2]
        mask_resized = cv2.resize(mask.astype(float), (w, h))
        mask_binary = mask_resized > 0.5
        
        highlighted = image.copy().astype(float) / 255.0
        
        if np.any(mask_binary):
            # 创建红色高亮
            highlight = np.zeros_like(highlighted)
            highlight[mask_binary] = highlight_color[:3]
            
            # 叠加
            alpha = mask_binary.astype(float) * highlight_color[3]
            alpha_3d = np.stack([alpha] * 3, axis=2)
            
            highlighted = highlighted * (1 - alpha_3d) + highlight * alpha_3d
        
        return np.clip(highlighted, 0, 1)
    
    def _plot_region_sizes(self, ax, masks):
        """绘制区域大小分布"""
        if not masks:
            ax.text(0.5, 0.5, '无区域数据', ha='center', va='center')
            ax.set_title('区域大小分布')
            ax.axis('off')
            return
        
        sizes = [mask.sum() for mask in masks]
        
        # 对数尺度
        ax.hist(np.log10(np.array(sizes) + 1), bins=20, 
               alpha=0.7, color='lightgreen', edgecolor='black')
        
        ax.set_xlabel('区域大小 (log10)')
        ax.set_ylabel('频次')
        ax.set_title('区域大小分布', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    def _create_category_heatmap(self, ax, masks, categories, image_shape):
        """创建类别热图"""
        if not masks:
            ax.text(0.5, 0.5, '无区域数据', ha='center', va='center')
            ax.set_title('类别热图')
            ax.axis('off')
            return
        
        h, w = image_shape
        heatmap = np.zeros((h, w, 3))
        
        # 为每个类别分配颜色
        unique_categories = list(set(categories))
        cat_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
        
        # 创建类别权重图
        for mask, category in zip(masks, categories):
            mask_resized = cv2.resize(mask.astype(float), (w, h))
            mask_binary = mask_resized > 0.5
            
            if np.any(mask_binary):
                cat_idx = cat_to_idx.get(category, 0)
                color = self.colormap(cat_idx / max(1, len(unique_categories) - 1))
                
                # 添加颜色贡献
                heatmap[mask_binary] += color[:3]
        
        # 归一化
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val
        
        ax.imshow(heatmap)
        ax.set_title('类别热图', fontsize=11)
        ax.axis('off')
        
        # 添加图例
        if len(unique_categories) <= 10:
            import matplotlib.patches as mpatches
            patches = []
            for cat in unique_categories[:10]:
                color = self.category_colors.get(cat, '#000000')
                patches.append(mpatches.Patch(color=color, label=cat))
            
            ax.legend(handles=patches, loc='upper right', fontsize=8)
    
    def _show_bbox_overlay(self, ax, image, masks, scores):
        """显示边界框叠加"""
        if not masks:
            ax.text(0.5, 0.5, '无区域数据', ha='center', va='center')
            ax.set_title('边界框可视化')
            ax.axis('off')
            return
        
        ax.imshow(image)
        
        # 计算边界框
        h, w = image.shape[:2]
        
        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask_resized = cv2.resize(mask.astype(float), (w, h))
            mask_binary = mask_resized > 0.5
            
            if np.any(mask_binary):
                # 找到非零像素的坐标
                rows = np.any(mask_binary, axis=1)
                cols = np.any(mask_binary, axis=0)
                
                y1, y2 = np.where(rows)[0][[0, -1]] if rows.any() else (0, h-1)
                x1, x2 = np.where(cols)[0][[0, -1]] if cols.any() else (0, w-1)
                
                # 根据分数设置颜色和线宽
                if score > 0.8:
                    color = 'lime'
                    linewidth = 2
                elif score > 0.5:
                    color = 'yellow'
                    linewidth = 1.5
                else:
                    color = 'red'
                    linewidth = 1
                
                # 绘制边界框
                rect = Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=linewidth, edgecolor=color, 
                               facecolor='none', alpha=0.7)
                ax.add_patch(rect)
        
        ax.set_title(f'边界框 ({len(masks)}个区域)', fontsize=11)
        ax.axis('off')
    
    def _create_statistics_table(self, ax, masks, categories, scores):
        """创建统计表格"""
        if not masks:
            ax.text(0.5, 0.5, '无统计数据', ha='center', va='center')
            ax.set_title('区域统计')
            ax.axis('off')
            return
        
        # 计算统计信息
        sizes = [mask.sum() for mask in masks]
        
        stats = [
            ["总区域数", len(masks)],
            ["平均置信度", f"{np.mean(scores):.3f}"],
            ["平均区域大小", f"{np.mean(sizes):.0f} px"],
            ["最小区域", f"{np.min(sizes):.0f} px"],
            ["最大区域", f"{np.max(sizes):.0f} px"],
            ["独特类别数", len(set(categories))],
            ["高置信区域(>0.8)", sum(1 for s in scores if s > 0.8)],
            ["低置信区域(<0.3)", sum(1 for s in scores if s < 0.3)]
        ]
        
        # 创建表格
        ax.axis('off')
        
        table_data = [[item[0], item[1]] for item in stats]
        
        # 使用matplotlib表格
        table = ax.table(cellText=table_data,
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # 设置样式
        for i in range(len(stats)):
            table[(i, 0)].set_facecolor('#f0f0f0')
            table[(i, 1)].set_facecolor('#e6f3ff')
        
        ax.set_title('区域统计摘要', fontsize=11, pad=20)