import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from typing import Dict, List, Tuple, Optional
import colorsys
from collections import Counter

class SegmentationVisualizer:
    """Professional Segmentation Result Visualizer"""
    
    def __init__(self, colormap='tab20'):
        # Use fixed color mapping to ensure reproducibility
        self.cmap = plt.cm.get_cmap(colormap)
        self.category_colors = {}
        self.category_names = []
        
        # Predefined colors for common categories (to ensure consistency)
        self.preset_colors = {
            'person': '#FF5252',     # Red
            'car': '#536DFE',        # Blue
            'vehicle': '#448AFF',    # Light blue
            'tree': '#4CAF50',       # Green
            'building': '#795548',   # Brown
            'road': '#607D8B',       # Gray
            'sky': '#03A9F4',        # Sky blue
            'grass': '#8BC34A',      # Light green
            'water': '#00BCD4',      # Cyan
            'sign': '#FF9800',       # Orange
            'animal': '#9C27B0',     # Purple
            'furniture': '#795548',  # Brown
            'window': '#FFEB3B',     # Yellow
            'door': '#795548',       # Brown
            'roof': '#795548',       # Brown
        }
    
    def create_comprehensive_visualization(self, image_path, results: Dict, 
                                         save_path=None, figsize=(24, 18)):
        """
        Create comprehensive visualization chart with logical layout
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print("Cannot read image")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        # Prepare data
        masks = results.get('masks', [])
        categories = results.get('categories', [])
        scores = results.get('scores', [])
        
        # Preprocess results
        processed_masks, processed_categories, processed_scores = self._preprocess_results(
            masks, categories, scores, image_shape=(h, w)
        )
        
        # Update results
        results['masks'] = processed_masks
        results['categories'] = processed_categories
        results['scores'] = processed_scores
        
        print(f"Visualization processing: {len(processed_masks)} valid regions")
        
        # Create figure with logical layout
        fig = plt.figure(figsize=figsize, facecolor='white')
        
        # Logical subplot layout: 4 rows x 3 columns for better organization
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3, 
                             height_ratios=[1.2, 1, 1, 0.8])
        
        # Row 1: Main comparison - Original vs Overlay
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image_rgb)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        overlay = self._create_smart_overlay(image_rgb, processed_masks, processed_categories)
        ax2.imshow(overlay)
        ax2.set_title(f'Segmentation Overlay ({len(processed_masks)} regions)', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Row 1: Category distribution
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_category_distribution(ax3, processed_categories)
        
        # Row 2: Confidence and region size distributions
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_score_distribution(ax4, processed_scores)
        
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_region_sizes(ax5, processed_masks)
        
        # Row 2: Category heatmap
        ax6 = fig.add_subplot(gs[1, 2])
        self._create_category_heatmap(ax6, processed_masks, processed_categories, (h, w))
        
        # Row 3: Top regions and bounding boxes
        # Create separate subplots for top regions
        top_regions_gs = gs[2, :2].subgridspec(2, 2, hspace=0.4, wspace=0.3)
        top_axes = []
        for i in range(4):
            top_axes.append(fig.add_subplot(top_regions_gs[i//2, i%2]))
        
        self._show_top_regions_simple(top_axes, image_rgb, processed_masks, processed_categories, 
                                     processed_scores, n=4)
        
        ax8 = fig.add_subplot(gs[2, 2])
        self._show_bbox_overlay(ax8, image_rgb, processed_masks, processed_scores)
        
        # Row 4: Statistical summary table
        ax9 = fig.add_subplot(gs[3, :])
        self._create_statistics_table(ax9, processed_masks, processed_categories, 
                                     processed_scores, image_path)
        
        # Title
        plt.suptitle(f'Intelligent Segmentation Analysis - {len(processed_masks)} segmented regions', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Save
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"✓ Visualization saved to: {save_path}")
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def _preprocess_results(self, masks, categories, scores, image_shape=(800, 800)):
        """Preprocess results: filter, sort, resize"""
        processed_masks = []
        processed_categories = []
        processed_scores = []
        
        h, w = image_shape
        
        # Sort by score
        if scores:
            sorted_indices = np.argsort(scores)[::-1]  # Descending
        else:
            sorted_indices = range(len(masks))
        
        seen_masks = []  # For deduplication
        for idx in sorted_indices:
            if idx >= len(masks):
                continue
            
            mask = masks[idx]
            if mask is None:
                continue
            
            # Resize mask
            mask_resized = cv2.resize(mask.astype(float), (w, h))
            
            # Ensure binary mask
            mask_binary = (mask_resized > 0.5).astype(float)
            
            # Filter too small regions
            mask_area = mask_binary.sum()
            if mask_area < 50:  # At least 50 pixels
                continue
            
            # Deduplication: check if highly overlapping with existing mask
            is_duplicate = False
            for seen_mask in seen_masks:
                iou = self._calculate_iou(mask_binary, seen_mask)
                if iou > 0.7:  # If overlap exceeds 70%, consider as duplicate
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                processed_masks.append(mask_binary)
                processed_categories.append(categories[idx] if idx < len(categories) else "unknown")
                processed_scores.append(scores[idx] if idx < len(scores) else 0.5)
                seen_masks.append(mask_binary)
        
        # Limit maximum display count
        max_regions = 50
        if len(processed_masks) > max_regions:
            print(f"Too many regions ({len(processed_masks)}), limited to first {max_regions}")
            processed_masks = processed_masks[:max_regions]
            processed_categories = processed_categories[:max_regions]
            processed_scores = processed_scores[:max_regions]
        
        return processed_masks, processed_categories, processed_scores
    
    def _create_smart_overlay(self, image, masks, categories, alpha=0.4):
        """Create smart overlay"""
        overlay = image.copy().astype(float) / 255.0
        
        # Assign fixed colors for each category
        unique_categories = list(set(categories))
        
        # Ensure color consistency
        for i, cat in enumerate(unique_categories):
            if cat in self.preset_colors:
                # Use preset color
                hex_color = self.preset_colors[cat]
                rgb_color = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (1, 3, 5))
                self.category_colors[cat] = rgb_color
            else:
                # Use colormap to assign color
                self.category_colors[cat] = self.cmap(i / max(1, len(unique_categories) - 1))[:3]
        
        # Sort by region size (largest to smallest) to avoid smaller regions being covered
        mask_sizes = [mask.sum() for mask in masks]
        sorted_indices = np.argsort(mask_sizes)[::-1]
        
        # Create overlay for each category
        for idx in sorted_indices:
            mask = masks[idx]
            category = categories[idx]
            
            if mask is None:
                continue
            
            # Get color
            color = self.category_colors.get(category, (1, 0, 0))
            
            # Create colored region
            colored_region = np.zeros_like(overlay)
            mask_indices = np.where(mask > 0.5)
            
            if len(mask_indices[0]) > 0:
                # Fill color
                for c in range(3):
                    colored_region[mask_indices[0], mask_indices[1], c] = color[c]
                
                # Smart overlay: large regions use lower transparency, small regions use higher transparency
                mask_area = len(mask_indices[0])
                if mask_area > 10000:
                    region_alpha = alpha * 0.6
                elif mask_area > 1000:
                    region_alpha = alpha * 0.8
                else:
                    region_alpha = alpha
                
                # Apply transparency
                mask_alpha = mask * region_alpha
                mask_alpha_3d = np.stack([mask_alpha] * 3, axis=2)
                
                overlay = overlay * (1 - mask_alpha_3d) + colored_region * mask_alpha_3d
        
        return np.clip(overlay, 0, 1)
    
    def _plot_category_distribution(self, ax, categories):
        """Plot category distribution"""
        if not categories:
            ax.text(0.5, 0.5, 'No category data', ha='center', va='center', fontsize=12)
            ax.set_title('Category Distribution', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        category_counts = Counter(categories)
        
        # Get most common categories (top 8)
        top_categories = category_counts.most_common(8)
        labels = [cat for cat, _ in top_categories]
        counts = [count for _, count in top_categories]
        
        # Use consistent colors
        colors = []
        for cat in labels:
            if cat in self.category_colors:
                colors.append(self.category_colors[cat])
            else:
                colors.append(self.cmap(0.5)[:3])  # Default color
        
        # Plot horizontal bar chart
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, counts, color=colors, edgecolor='black', height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Region Count', fontsize=11)
        ax.set_title('Category Distribution (Top 8)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Show count on bars
        for i, count in enumerate(counts):
            ax.text(count + max(counts)*0.01, i, str(count), 
                   va='center', fontsize=9)
    
    def _plot_score_distribution(self, ax, scores):
        """Plot confidence distribution"""
        if not scores:
            ax.text(0.5, 0.5, 'No score data', ha='center', va='center', fontsize=12)
            ax.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        # Create more refined histogram
        hist, bins = np.histogram(scores, bins=20, range=(0, 1))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Use gradient colors
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(hist)))
        
        bars = ax.bar(bin_centers, hist, width=bins[1]-bins[0], 
                     color=colors, edgecolor='black', alpha=0.8)
        
        # Add statistical lines
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        
        ax.axvline(mean_score, color='red', linestyle='-', linewidth=2, 
                  label=f'Mean: {mean_score:.3f}')
        ax.axvline(median_score, color='green', linestyle='--', linewidth=2,
                  label=f'Median: {median_score:.3f}')
        
        # Fill high confidence region
        ax.axvspan(0.8, 1.0, alpha=0.1, color='green', label='High Confidence(>0.8)')
        ax.axvspan(0.0, 0.3, alpha=0.1, color='red', label='Low Confidence(<0.3)')
        
        ax.set_xlabel('Confidence Score', fontsize=11)
        ax.set_ylabel('Region Count', fontsize=11)
        ax.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis range
        ax.set_xlim(0, 1)
    
    def _show_top_regions_simple(self, axes, image, masks, categories, scores, n=4):
        """Show top scoring regions in a simple layout"""
        if not masks:
            for ax in axes:
                ax.text(0.5, 0.5, 'No segmentation regions', ha='center', va='center', fontsize=12)
                ax.set_title('Top Scoring Regions', fontsize=12, fontweight='bold')
                ax.axis('off')
            return
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        
        for i, idx in enumerate(sorted_indices[:n]):
            if i >= len(axes):
                break
                
            mask = masks[idx]
            category = categories[idx] if idx < len(categories) else "Unknown"
            score = scores[idx] if idx < len(scores) else 0
            
            # Create highlight display
            highlighted = self._highlight_single_region(image, mask, category)
            axes[i].imshow(highlighted)
            axes[i].set_title(f'{category}\nScore: {score:.3f}', fontsize=10, fontweight='bold')
            axes[i].axis('off')
        
        # Add a main title for the group
        # We'll use the first axis to add the overall title
        if len(axes) > 0:
            axes[0].set_title(f'Top Scoring Regions (Top {min(n, len(axes))})', 
                             fontsize=12, fontweight='bold', pad=20)
    
    def _highlight_single_region(self, image, mask, category, highlight_alpha=0.6):
        """Highlight single region"""
        highlighted = image.copy().astype(float) / 255.0
        
        if mask is None or mask.sum() == 0:
            return highlighted
        
        # Get category color
        if category in self.category_colors:
            color = self.category_colors[category]
        else:
            color = (1, 0, 0)  # Red as default
        
        # Create highlight layer
        highlight = np.zeros_like(highlighted)
        mask_indices = np.where(mask > 0.5)
        
        if len(mask_indices[0]) > 0:
            # Fill color
            for c in range(3):
                highlight[mask_indices[0], mask_indices[1], c] = color[c]
            
            # Add contour
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create contour overlay
            contour_overlay = highlighted.copy()
            cv2.drawContours(contour_overlay, contours, -1, (1, 1, 1), 2)  # White contour
            
            # Blend: region interior translucent, contour clear
            mask_alpha = mask * highlight_alpha
            mask_alpha_3d = np.stack([mask_alpha] * 3, axis=2)
            
            # Apply highlight
            highlighted = highlighted * (1 - mask_alpha_3d) + highlight * mask_alpha_3d
            
            # Overlay contour
            contour_mask = np.zeros_like(mask, dtype=bool)
            for contour in contours:
                if len(contour) > 0:
                    contour_mask = contour_mask | (cv2.drawContours(
                        np.zeros_like(mask, dtype=np.uint8), 
                        [contour], -1, 1, 2) > 0)
            
            if contour_mask.any():
                highlighted[contour_mask] = contour_overlay[contour_mask]
        
        return np.clip(highlighted, 0, 1)
    
    def _plot_region_sizes(self, ax, masks):
        """Plot region size distribution"""
        if not masks:
            ax.text(0.5, 0.5, 'No region data', ha='center', va='center', fontsize=12)
            ax.set_title('Region Size Distribution', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        sizes = [mask.sum() for mask in masks]
        
        # Use logarithmic scale
        log_sizes = np.log10(np.array(sizes) + 1)
        
        # Create more beautiful histogram
        hist, bins = np.histogram(log_sizes, bins=15)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Use gradient colors
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(hist)))
        
        bars = ax.bar(bin_centers, hist, width=bins[1]-bins[0], 
                     color=colors, edgecolor='black', alpha=0.8)
        
        # Add statistics
        mean_size = np.mean(sizes)
        median_size = np.median(sizes)
        
        # Convert to log coordinates
        mean_log = np.log10(mean_size + 1)
        median_log = np.log10(median_size + 1)
        
        ax.axvline(mean_log, color='red', linestyle='-', linewidth=2, 
                  label=f'Mean: {mean_size:.0f} pixels')
        ax.axvline(median_log, color='green', linestyle='--', linewidth=2,
                  label=f'Median: {median_size:.0f} pixels')
        
        ax.set_xlabel('Region Size (log10 pixels)', fontsize=11)
        ax.set_ylabel('Region Count', fontsize=11)
        ax.set_title('Region Size Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _create_category_heatmap(self, ax, masks, categories, image_shape):
        """Create category heatmap"""
        if not masks:
            ax.text(0.5, 0.5, 'No region data', ha='center', va='center', fontsize=12)
            ax.set_title('Category Heatmap', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        h, w = image_shape
        heatmap = np.zeros((h, w, 3))
        
        # Collect category information for each pixel
        category_presence = np.zeros((h, w, len(self.category_colors)), dtype=float)
        category_names = list(self.category_colors.keys())
        
        # Add weight for each mask
        for mask, category in zip(masks, categories):
            if mask is None:
                continue
            
            mask_resized = cv2.resize(mask.astype(float), (w, h))
            mask_binary = mask_resized > 0.5
            
            if category in category_names:
                cat_idx = category_names.index(category)
                category_presence[mask_binary, cat_idx] += 1.0
        
        # Create heatmap: each pixel shows color of main category
        for cat_idx, cat_name in enumerate(category_names):
            if cat_name in self.category_colors:
                color = self.category_colors[cat_name]
                mask = category_presence[:, :, cat_idx] > 0
                if mask.any():
                    for c in range(3):
                        heatmap[mask, c] = color[c]
        
        # Blur to make heatmap smoother
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 1)
        
        ax.imshow(heatmap)
        ax.set_title('Category Spatial Distribution', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    def _show_bbox_overlay(self, ax, image, masks, scores):
        """Display bounding box overlay"""
        if not masks:
            ax.text(0.5, 0.5, 'No region data', ha='center', va='center', fontsize=12)
            ax.set_title('Bounding Box Visualization', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        ax.imshow(image)
        
        h, w = image.shape[:2]
        drawn_boxes = []  # Avoid duplicate drawing
        
        # Sort by score, draw high score boxes first
        sorted_indices = np.argsort(scores)[::-1]
        
        for i, idx in enumerate(sorted_indices[:15]):  # Show at most 15 boxes
            if idx >= len(masks):
                continue
            
            mask = masks[idx]
            score = scores[idx] if idx < len(scores) else 0
            
            if mask is None:
                continue
            
            # Find coordinates of non-zero pixels
            rows = np.any(mask > 0.5, axis=1)
            cols = np.any(mask > 0.5, axis=0)
            
            if not rows.any() or not cols.any():
                continue
            
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            
            # Check if overlapping too much with existing box
            box = (x1, y1, x2, y2)
            is_overlap = False
            for drawn_box in drawn_boxes:
                iou = self._bbox_iou(box, drawn_box)
                if iou > 0.5:  # If overlap exceeds 50%, skip
                    is_overlap = True
                    break
            
            if is_overlap:
                continue
            
            drawn_boxes.append(box)
            
            # Set color and line width based on score
            if score > 0.8:
                color = 'lime'
                linewidth = 2.5
                label = f"{score:.3f}"
            elif score > 0.6:
                color = 'yellow'
                linewidth = 2.0
                label = f"{score:.3f}"
            else:
                color = 'red'
                linewidth = 1.5
                label = f"{score:.3f}"
            
            # Draw bounding box
            import matplotlib.patches as patches
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=linewidth, edgecolor=color, 
                                   facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            
            # Add score label
            if score > 0.6:  # Only show high score labels
                ax.text(x1, y1-5, label, color=color, fontsize=8,
                       fontweight='bold', bbox=dict(boxstyle='round,pad=0.2',
                                                    facecolor='black',
                                                    alpha=0.7))
        
        ax.set_title(f'Bounding Boxes ({len(drawn_boxes)} regions)', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    def _bbox_iou(self, box1, box2):
        """Calculate bounding box IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _create_statistics_table(self, ax, masks, categories, scores, image_path):
        """Create statistics table"""
        if not masks:
            ax.text(0.5, 0.5, 'No statistics', ha='center', va='center', fontsize=12)
            ax.set_title('Statistical Summary', fontsize=12, fontweight='bold', pad=20)
            ax.axis('off')
            return
        
        # Calculate statistics
        sizes = [mask.sum() for mask in masks]
        
        # Category statistics
        category_counts = Counter(categories)
        top_categories = category_counts.most_common(5)
        
        # Build statistics table
        stats = [
            ["Total Regions", f"{len(masks)}"],
            ["Image Name", os.path.basename(image_path)[:30]],
            ["Mean Confidence", f"{np.mean(scores):.3f}"],
            ["Median Confidence", f"{np.median(scores):.3f}"],
            ["Mean Region Size", f"{np.mean(sizes):.0f} px"],
            ["Total Pixel Coverage", f"{sum(sizes)/masks[0].size*100:.1f}%"],
            ["Most Frequent Category", f"{top_categories[0][0]}({top_categories[0][1]})" if top_categories else "N/A"],
            ["Unique Categories", f"{len(category_counts)}"],
        ]
        
        # Create table
        ax.axis('off')
        
        table_data = [[item[0], item[1]] for item in stats]
        
        # Use matplotlib table
        table = ax.table(cellText=table_data,
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.4, 0.6],
                        colColours=['#f0f0f0', '#e6f3ff'])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Set style
        for i in range(len(stats)):
            table[(i, 0)].set_text_props(fontweight='bold')
        
        ax.set_title('Statistical Summary', fontsize=12, fontweight='bold', pad=20)
    
    def _calculate_iou(self, mask1, mask2):
        """Calculate IoU between two masks"""
        try:
            # Ensure binary masks
            mask1_bin = (mask1 > 0.5).astype(np.uint8)
            mask2_bin = (mask2 > 0.5).astype(np.uint8)
            
            # Resize if needed
            h1, w1 = mask1_bin.shape
            h2, w2 = mask2_bin.shape
            
            if (h1, w1) != (h2, w2):
                mask2_resized = cv2.resize(mask2_bin.astype(float), (w1, h1))
                mask2_bin = (mask2_resized > 0.5).astype(np.uint8)
            
            # Calculate IoU
            intersection = np.logical_and(mask1_bin, mask2_bin).sum()
            union = np.logical_or(mask1_bin, mask2_bin).sum()
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            return 0.0



