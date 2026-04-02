# test_quick.py
#!/usr/bin/env python3
"""
快速测试SAM3是否工作
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

sys.path.append('.')
from core.sam3_adapter_final import SAM3AdapterFinal

def test_simple():
    """简单测试"""
    print("Simple SAM3 test...")
    
    # 初始化适配器
    adapter = SAM3AdapterFinal()
    
    # 测试图像
    image_path = "/data/users/syh/InstructSAM/datasets/dior/JPEGImages-trainval/09938.jpg"
    
    # 设置图像
    print(f"Setting image: {image_path}")
    adapter.set_image(image_path)
    
    # 测试"car"提示
    print("\nTesting 'car' prompt...")
    results = adapter.execute_text_prompt("car", score_threshold=0.1)
    print(f"Found {len(results)} results")
    
    if results:
        # 显示第一个结果
        first_result = results[0]
        print(f"First result score: {first_result['score']:.3f}")
        print(f"First result bbox: {first_result['bbox']}")
        
        # 可视化
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        mask = first_result['mask']
        mask_resized = cv2.resize(mask.astype(float), (w, h))
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(mask_resized > 0.5, cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')
        
        # 叠加
        overlay = image_rgb.copy()
        colored = np.zeros_like(image_rgb)
        colored[mask_resized > 0.5] = [255, 0, 0]
        overlay = cv2.addWeighted(overlay, 0.6, colored, 0.4, 0)
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return True
    else:
        print("No results found")
        return False

def test_multiple_prompts():
    """测试多个提示词"""
    print("\n" + "="*50)
    print("Testing multiple prompts...")
    
    adapter = SAM3AdapterFinal()
    image_path = "/data/users/syh/InstructSAM/datasets/dior/JPEGImages-trainval/09938.jpg"
    
    adapter.set_image(image_path)
    
    prompts = ["car", "vehicle", "building", "tree", "person"]
    
    all_results = []
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        adapter.current_state = adapter.initial_state
        results = adapter.execute_text_prompt(prompt, score_threshold=0.2)
        print(f"  Results: {len(results)}")
        
        for result in results:
            all_results.append({
                'prompt': prompt,
                'score': result['score'],
                'mask': result['mask']
            })
    
    print(f"\nTotal results across all prompts: {len(all_results)}")
    
    if all_results:
        # 可视化所有结果
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        overlay = image_rgb.copy()
        colors = plt.cm.tab10(np.linspace(0, 1, len(prompts)))
        
        for i, prompt in enumerate(prompts):
            prompt_results = [r for r in all_results if r['prompt'] == prompt]
            
            for result in prompt_results:
                mask = result['mask']
                if mask is None:
                    continue
                
                mask_resized = cv2.resize(mask.astype(float), (w, h))
                mask_binary = mask_resized > 0.5
                
                color = colors[i][:3] * 255
                colored = np.zeros_like(image_rgb)
                colored[mask_binary] = color
                
                overlay = cv2.addWeighted(overlay, 0.7, colored.astype(np.uint8), 0.3, 0)
        
        axes[1].imshow(overlay)
        axes[1].set_title(f'All Results ({len(all_results)} masks)')
        axes[1].axis('off')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_patches = [Patch(color=colors[i][:3], label=prompts[i]) for i in range(len(prompts))]
        axes[1].legend(handles=legend_patches, loc='upper right', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        return True
    
    return False

if __name__ == "__main__":
    success1 = test_simple()
    success2 = test_multiple_prompts()
    
    if success1 or success2:
        print("\n✓ SAM3 is working correctly!")
    else:
        print("\n✗ SAM3 test failed. Please check the configuration.")