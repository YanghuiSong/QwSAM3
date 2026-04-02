#!/usr/bin/env python3
"""
Optimized Intelligent Full Segmentation Demo Program
"""

import os
import sys
import argparse
import numpy as np
import cv2

sys.path.append('.')
from config import Config
from core.qwen_enhanced_agent import EnhancedQwenAgent
from core.sam3_adapter_final import SAM3AdapterFinal
from core.intelligent_segmentation_coordinator import IntelligentSegmentationCoordinator
from visualization.segmentation_visualizer import SegmentationVisualizer

class OptimizedSegmentationSystem:
    """Optimized Segmentation System"""
    
    def __init__(self):
        print("Initializing optimized segmentation system...")
        
        try:
            # Initialize components
            self.qwen_agent = EnhancedQwenAgent(Config.QWEN_MODEL_PATH)
            self.sam_adapter = SAM3AdapterFinal()
            self.coordinator = IntelligentSegmentationCoordinator(self.qwen_agent, self.sam_adapter)
            self.visualizer = SegmentationVisualizer()
            
            print("✓ System initialization completed")
        except Exception as e:
            print(f"✗ System initialization failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def process_image(self, image_path, save_dir="./results", max_iterations=2):
        """Process single image"""
        if not os.path.exists(image_path):
            print(f"✗ Image does not exist: {image_path}")
            return
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        print(f"\n{'='*70}")
        print(f"Processing image: {image_path}")
        print(f"{'='*70}")
        
        try:
            # Step 1: Execute intelligent segmentation
            print("\n[Phase 1] Intelligent Segmentation...")
            results = self.coordinator.intelligent_full_segmentation(
                image_path, 
                max_iterations=max_iterations
            )
            
            if not results.get('masks'):
                print("⚠ No segmentation regions found, trying fallback method...")
                results = self._fallback_segmentation(image_path)
            
            if not results.get('masks'):
                print("✗ Segmentation failed")
                return
            
            # Step 2: Optimize results
            print("\n[Phase 2] Optimizing results...")
            optimized_results = self._optimize_results(results, image_path)
            
            # Step 3: Save results
            print("\n[Phase 3] Saving results...")
            self._save_all_results(optimized_results, save_dir, base_name)
            
            # Step 4: Create visualization
            print("\n[Phase 4] Creating visualization...")
            self._create_all_visualizations(image_path, optimized_results, save_dir, base_name)
            
            print(f"\n{'='*70}")
            print("✓ Processing completed!")
            print(f"   Results saved in: {save_dir}/{base_name}_*")
            print(f"{'='*70}")
            
        except Exception as e:
            print(f"\n✗ Error during processing: {e}")
            import traceback
            traceback.print_exc()
    
    def _optimize_results(self, results, image_path):
        """Optimize segmentation results"""
        masks = results.get('masks', [])
        categories = results.get('categories', [])
        scores = results.get('scores', [])
        
        if not masks:
            return results
        
        # Read image dimensions
        image = cv2.imread(image_path)
        if image is None:
            h, w = 800, 800
        else:
            h, w = image.shape[:2]
        
        optimized_masks = []
        optimized_categories = []
        optimized_scores = []
        
        # Filter and optimize
        for i, (mask, category, score) in enumerate(zip(masks, categories, scores)):
            if mask is None:
                continue
            
            # Resize
            mask_resized = cv2.resize(mask.astype(float), (w, h))
            
            # Apply morphological operations (optional)
            kernel = np.ones((3, 3), np.uint8)
            mask_cleaned = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel)
            mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
            
            # Binarize
            mask_binary = (mask_cleaned > 0.5).astype(float)
            
            # Filter too small regions
            if mask_binary.sum() < 50:
                continue
            
            optimized_masks.append(mask_binary)
            optimized_categories.append(category)
            optimized_scores.append(score)
        
        # Sort by category and score
        sorted_indices = sorted(range(len(optimized_scores)), 
                               key=lambda i: (optimized_categories[i], optimized_scores[i]), 
                               reverse=True)
        
        return {
            'masks': [optimized_masks[i] for i in sorted_indices],
            'categories': [optimized_categories[i] for i in sorted_indices],
            'scores': [optimized_scores[i] for i in sorted_indices],
            'boxes': [results.get('boxes', [])[i] for i in sorted_indices] if results.get('boxes') else [],
            'prompts_used': results.get('prompts_used', [])
        }
    
    def _save_all_results(self, results, save_dir, base_name):
        """Save all results"""
        # Save numpy file
        npz_path = os.path.join(save_dir, f"{base_name}_results.npz")
        try:
            np.savez_compressed(
                npz_path,
                masks=np.array(results.get('masks', [])),
                categories=results.get('categories', []),
                scores=results.get('scores', []),
                boxes=results.get('boxes', []),
                prompts_used=results.get('prompts_used', [])
            )
            print(f"  ✓ Segmentation data: {npz_path}")
        except Exception as e:
            print(f"  ✗ Save data failed: {e}")
        
        # Save text summary
        txt_path = os.path.join(save_dir, f"{base_name}_summary.txt")
        try:
            with open(txt_path, 'w') as f:
                f.write(f"Segmentation Results Summary\n")
                f.write(f"="*50 + "\n")
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
                
                if results.get('prompts_used'):
                    from collections import Counter
                    prompt_counts = Counter(results['prompts_used'])
                    f.write(f"\nPrompt Statistics:\n")
                    for prompt, count in prompt_counts.most_common(10):
                        f.write(f"  '{prompt}': {count} successful times\n")
            
            print(f"  ✓ Result summary: {txt_path}")
        except Exception as e:
            print(f"  ✗ Save summary failed: {e}")
    
    def _create_all_visualizations(self, image_path, results, save_dir, base_name):
        """Create all visualizations"""
        # 1. Comprehensive visualization
        vis_path = os.path.join(save_dir, f"{base_name}_comprehensive.png")
        try:
            self.visualizer.create_comprehensive_visualization(
                image_path, 
                results, 
                save_path=vis_path
            )
            print(f"  ✓ Comprehensive visualization: {vis_path}")
        except Exception as e:
            print(f"  ✗ Create comprehensive visualization failed: {e}")
        
        # 2. Simple overlay
        overlay_path = os.path.join(save_dir, f"{base_name}_overlay.png")
        try:
            self._create_simple_overlay(image_path, results, overlay_path)
            print(f"  ✓ Overlay: {overlay_path}")
        except Exception as e:
            print(f"  ✗ Create overlay failed: {e}")
        
        # 3. Category mask visualization
        mask_path = os.path.join(save_dir, f"{base_name}_masks.png")
        try:
            self._create_mask_visualization(image_path, results, mask_path)
            print(f"  ✓ Mask visualization: {mask_path}")
        except Exception as e:
            print(f"  ✗ Create mask visualization failed: {e}")
    
    def _create_simple_overlay(self, image_path, results, save_path):
        """Create simple overlay"""
        import matplotlib.pyplot as plt
        
        # Read image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        # Create overlay
        overlay = image_rgb.copy().astype(float) / 255.0
        masks = results.get('masks', [])
        categories = results.get('categories', [])
        
        if not masks:
            return
        
        # Assign colors for each category
        unique_categories = list(set(categories))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_categories)))
        color_map = {cat: colors[i][:3] for i, cat in enumerate(unique_categories)}
        
        # Sort by region size (largest to smallest)
        mask_sizes = [(i, mask.sum()) for i, mask in enumerate(masks)]
        mask_sizes.sort(key=lambda x: x[1], reverse=True)
        
        for i, _ in mask_sizes:
            mask = masks[i]
            category = categories[i]
            
            if mask is None:
                continue
            
            # Resize
            mask_resized = cv2.resize(mask.astype(float), (w, h))
            mask_binary = mask_resized > 0.5
            
            if np.any(mask_binary):
                color = color_map.get(category, np.array([1, 0, 0]))
                
                # Create colored region
                colored_region = np.zeros_like(overlay)
                colored_region[mask_binary] = color
                
                # Overlay
                alpha = 0.3
                mask_alpha = mask_binary.astype(float) * alpha
                mask_alpha_3d = np.stack([mask_alpha] * 3, axis=2)
                
                overlay = overlay * (1 - mask_alpha_3d) + colored_region * mask_alpha_3d
        
        # Create figure
        plt.figure(figsize=(12, 8))
        plt.imshow(np.clip(overlay, 0, 1))
        plt.title(f'Segmentation Overlay - {len(masks)} regions', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_mask_visualization(self, image_path, results, save_path):
        """Create mask visualization"""
        import matplotlib.pyplot as plt
        
        # Read image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        masks = results.get('masks', [])
        categories = results.get('categories', [])
        
        if not masks:
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Original image
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original Image', fontsize=11, fontweight='bold')
        axes[0].axis('off')
        
        # Masks for different categories
        unique_categories = list(set(categories))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_categories)))
        
        for i, category in enumerate(unique_categories[:5]):  # Show at most 5 categories
            if i+1 >= len(axes):
                break
            
            # Merge all masks for this category
            category_mask = np.zeros((h, w), dtype=float)
            for mask, cat in zip(masks, categories):
                if cat == category:
                    mask_resized = cv2.resize(mask.astype(float), (w, h))
                    category_mask = np.maximum(category_mask, mask_resized)
            
            if category_mask.sum() > 0:
                # Create colored mask
                color_mask = np.zeros((h, w, 3))
                for c in range(3):
                    color_mask[:, :, c] = colors[i][c] * category_mask
                
                # Blend
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
        
        # Hide extra subplots
        for j in range(i+2, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle('Category Mask Visualization', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _fallback_segmentation(self, image_path):
        """Fallback segmentation method"""
        print("Executing fallback segmentation...")
        
        # Set image
        if self.sam_adapter.set_image(image_path) is None:
            return {}
        
        # Basic prompts
        basic_prompts = [
            "object", "thing", "region", "area",
            "person", "car", "building", "tree",
            "road", "sky", "water", "grass"
        ]
        
        all_masks = []
        all_categories = []
        all_scores = []
        
        for prompt in basic_prompts:
            print(f"  Trying: '{prompt}'")
            
            self.sam_adapter.current_state = self.sam_adapter.initial_state
            
            try:
                results = self.sam_adapter.execute_text_prompt(prompt, score_threshold=0.1)
                
                if results:
                    for result in results:
                        mask = result.get('mask')
                        score = result.get('score', 0)
                        
                        if mask is not None and mask.sum() > 10:
                            all_masks.append(mask)
                            all_categories.append(prompt)
                            all_scores.append(score)
            except:
                continue
        
        return {
            'masks': all_masks,
            'categories': all_categories,
            'scores': all_scores
        }

def main():
    parser = argparse.ArgumentParser(description='Optimized Intelligent Full Segmentation System')
    parser.add_argument('image_path', help='Input image path')
    parser.add_argument('--output_dir', default='./results', help='Output directory')
    parser.add_argument('--iterations', type=int, default=2, help='Number of iterations')
    parser.add_argument('--batch', action='store_true', help='Batch processing mode')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image does not exist {args.image_path}")
        sys.exit(1)
    
    try:
        # Create system
        system = OptimizedSegmentationSystem()
        
        if args.batch and os.path.isdir(args.image_path):
            # Batch processing
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend([f for f in os.listdir(args.image_path) 
                                  if f.lower().endswith(ext)])
            
            print(f"Found {len(image_files)} images")
            
            for i, img_file in enumerate(sorted(image_files)):
                img_path = os.path.join(args.image_path, img_file)
                print(f"\n{'='*50}")
                print(f"[{i+1}/{len(image_files)}] Processing: {img_file}")
                print(f"{'='*50}")
                
                try:
                    system.process_image(img_path, args.output_dir, args.iterations)
                except Exception as e:
                    print(f"Processing failed: {e}")
                    continue
        else:
            # Single image processing
            system.process_image(args.image_path, args.output_dir, args.iterations)
            
    except KeyboardInterrupt:
        print("\nUser interrupted")
    except Exception as e:
        print(f"\nSystem error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()




