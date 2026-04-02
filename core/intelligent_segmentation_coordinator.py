# core/intelligent_segmentation_coordinator.py
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from collections import Counter

class IntelligentSegmentationCoordinator:
    """智能分割协调器，管理Qwen3和SAM3的协作"""
    
    def __init__(self, qwen_agent, sam_adapter):
        self.qwen_agent = qwen_agent
        self.sam_adapter = sam_adapter
        self.segmentation_history = []
        
    def intelligent_full_segmentation(self, image_path, max_iterations=3) -> Dict:
        """
        智能全分割流程
        通过多轮迭代优化分割结果
        """
        print(f"\n{'='*60}")
        print("开始智能全分割流程")
        print(f"{'='*60}")
        
        # 步骤1: 设置图像
        print("\n步骤1: 初始化图像")
        if self.sam_adapter.set_image(image_path) is None:
            print("✗ 图像设置失败")
            return {}
        
        # 步骤2: 智能场景分析
        print("\n步骤2: Qwen3智能场景分析")
        scene_analysis = self.qwen_agent.intelligent_scene_analysis(image_path)
        
        print(f"  分割策略: {scene_analysis.get('segmentation_strategy', 'N/A')}")
        print(f"  预期区域数: {scene_analysis.get('total_expected_regions', 'N/A')}")
        
        # 步骤3: 生成优化提示词
        print("\n步骤3: 生成SAM3优化提示词")
        prompts = self.qwen_agent.generate_sam3_prompts(scene_analysis)
        
        print(f"  生成的提示词 ({len(prompts)}个):")
        for i, p in enumerate(prompts[:10]):  # 显示前10个
            print(f"    {i+1}. '{p['text']}' ({p['type']}, 优先级: {p['priority']})")
        
        if len(prompts) > 10:
            print(f"    ... 还有 {len(prompts)-10} 个提示词")
        
        # 步骤4: 多轮分割
        all_results = self._iterative_segmentation(image_path, prompts, max_iterations)
        
        # 步骤5: 结果后处理
        final_results = self._post_process_results(all_results)
        
        # 步骤6: 结果分析
        self._analyze_final_results(final_results, scene_analysis)
        
        return final_results
    
    def _iterative_segmentation(self, image_path, prompts, max_iterations) -> List[Dict]:
        """迭代分割过程"""
        all_masks = []
        all_boxes = []
        all_scores = []
        all_categories = []
        all_prompts_used = []
        
        iteration = 1
        remaining_prompts = prompts.copy()
        
        while iteration <= max_iterations and remaining_prompts:
            print(f"\n--- 第 {iteration} 轮分割 ---")
            print(f"  待处理提示词: {len(remaining_prompts)} 个")
            
            iteration_masks = []
            iteration_boxes = []
            iteration_scores = []
            iteration_categories = []
            
            # 处理当前轮次的提示词
            for i, prompt_info in enumerate(remaining_prompts[:20]):  # 每轮最多处理20个
                prompt_text = prompt_info["text"]
                category = prompt_info["category"]
                
                print(f"  [{i+1}/{min(20, len(remaining_prompts))}] 提示词: '{prompt_text}'")
                
                # 重置到初始状态
                self.sam_adapter.current_state = self.sam_adapter.initial_state
                
                # 执行分割
                try:
                    results = self.sam_adapter.execute_text_prompt(
                        prompt_text, 
                        score_threshold=0.15  # 降低阈值以捕获更多结果
                    )
                    
                    if results:
                        print(f"    发现 {len(results)} 个区域")
                        
                        for result in results:
                            mask = result.get('mask')
                            bbox = result.get('bbox', [])
                            score = result.get('score', 0)
                            
                            if mask is not None and mask.size > 0 and mask.sum() > 5:
                                iteration_masks.append(mask)
                                iteration_boxes.append(bbox)
                                iteration_scores.append(score)
                                iteration_categories.append(category)
                                all_prompts_used.append(prompt_text)
                    else:
                        print(f"    未发现区域")
                        
                except Exception as e:
                    print(f"    分割失败: {e}")
                    continue
            
            # 收集本轮结果
            if iteration_masks:
                # 应用本轮NMS - 修复这里的问题
                keep_indices = self._apply_adaptive_nms_fixed(iteration_masks, iteration_scores)
                
                for idx in keep_indices:
                    all_masks.append(iteration_masks[idx])
                    all_boxes.append(iteration_boxes[idx])
                    all_scores.append(iteration_scores[idx])
                    all_categories.append(iteration_categories[idx])
                
                print(f"  本轮保留 {len(keep_indices)} 个区域")
            
            # 移除已处理的提示词
            if len(remaining_prompts) > 20:
                remaining_prompts = remaining_prompts[20:]
            else:
                remaining_prompts = []
            
            iteration += 1
            
            # 如果还有提示词，进行中间分析
            if remaining_prompts and iteration <= max_iterations:
                intermediate_results = {
                    'masks': all_masks,
                    'scores': all_scores,
                    'categories': all_categories
                }
                
                # 分析当前结果，优化剩余提示词
                analysis = self.qwen_agent.analyze_segmentation_results(
                    image_path, 
                    intermediate_results
                )
                
                if analysis.get("needs_refinement", False):
                    print(f"\n  Qwen3分析建议补充 {len(analysis['missing_prompts'])} 个提示词")
                    for mp in analysis['missing_prompts'][:5]:
                        remaining_prompts.append({
                            "text": mp,
                            "category": "refinement",
                            "priority": 1,
                            "type": "refinement"
                        })
        
        print(f"\n迭代分割完成，总共尝试了 {len(all_prompts_used)} 个提示词")
        
        return {
            'masks': all_masks,
            'boxes': all_boxes,
            'scores': all_scores,
            'categories': all_categories,
            'prompts_used': all_prompts_used
        }
    
    def _apply_adaptive_nms_fixed(self, masks, scores, base_iou_threshold=0.5) -> List[int]:
        """修复版自适应非极大值抑制"""
        if not masks:
            return []
        
        # 按分数排序
        scores_array = np.array(scores)
        sorted_indices = np.argsort(scores_array)[::-1].tolist()  # 转换为列表
        keep = []
        
        while sorted_indices:  # 现在sorted_indices是列表，可以直接判断是否为空
            current_idx = sorted_indices[0]
            current_mask = masks[current_idx]
            keep.append(current_idx)
            
            if len(sorted_indices) == 1:
                break
            
            # 计算IoU并过滤
            remaining = []
            for idx in sorted_indices[1:]:
                other_mask = masks[idx]
                
                # 自适应阈值：小物体使用更高阈值
                current_size = current_mask.sum()
                other_size = other_mask.sum()
                
                if current_size > 1000 and other_size > 1000:  # 两个都是大物体
                    iou_threshold = base_iou_threshold * 0.8  # 更严格
                elif current_size < 100 or other_size < 100:  # 有小物体
                    iou_threshold = base_iou_threshold * 1.5  # 更宽松
                else:
                    iou_threshold = base_iou_threshold
                
                iou = self._calculate_iou(current_mask, other_mask)
                
                if iou < iou_threshold:
                    remaining.append(idx)
            
            sorted_indices = remaining
        
        return keep
    
    def _calculate_iou(self, mask1, mask2) -> float:
        """计算IoU"""
        try:
            # 确保是二维数组
            if mask1.ndim > 2:
                mask1 = mask1.squeeze()
            if mask2.ndim > 2:
                mask2 = mask2.squeeze()
            
            # 二值化
            mask1_bin = (mask1 > 0.5).astype(np.uint8)
            mask2_bin = (mask2 > 0.5).astype(np.uint8)
            
            # 调整尺寸
            h1, w1 = mask1_bin.shape
            h2, w2 = mask2_bin.shape
            
            if (h1, w1) != (h2, w2):
                mask2_resized = cv2.resize(mask2_bin.astype(float), (w1, h1))
                mask2_bin = (mask2_resized > 0.5).astype(np.uint8)
            
            # 计算交集和并集
            intersection = np.logical_and(mask1_bin, mask2_bin).sum()
            union = np.logical_or(mask1_bin, mask2_bin).sum()
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            print(f"IoU计算失败: {e}")
            return 0.0
    
    def _post_process_results(self, results: Dict) -> Dict:
        """结果后处理"""
        masks = results.get('masks', [])
        scores = results.get('scores', [])
        
        if not masks:
            return results
        
        # 最终去重 - 使用修复版清理
        final_indices = self._final_cleanup_fixed(masks, scores)
        
        final_results = {
            'masks': [masks[i] for i in final_indices],
            'boxes': [results['boxes'][i] for i in final_indices],
            'scores': [scores[i] for i in final_indices],
            'categories': [results['categories'][i] for i in final_indices],
            'prompts_used': results.get('prompts_used', [])
        }
        
        print(f"\n后处理完成: {len(masks)} -> {len(final_results['masks'])} 个区域")
        
        return final_results
    
    def _final_cleanup_fixed(self, masks, scores) -> List[int]:
        """修复版最终清理：去除重复和小区域"""
        if not masks:
            return []
        
        # 按分数排序
        scores_array = np.array(scores)
        sorted_indices = np.argsort(scores_array)[::-1].tolist()  # 转换为列表
        keep = []
        
        for idx in sorted_indices:
            if idx >= len(masks):
                continue
                
            current_mask = masks[idx]
            current_size = current_mask.sum()
            
            # 过滤太小区域
            if current_size < 10:
                continue
            
            # 检查重复
            is_duplicate = False
            for kept_idx in keep:
                if kept_idx >= len(masks):
                    continue
                    
                kept_mask = masks[kept_idx]
                iou = self._calculate_iou(current_mask, kept_mask)
                
                if iou > 0.7:  # 高IoU阈值
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                keep.append(idx)
        
        return keep
    
    def _analyze_final_results(self, results: Dict, scene_analysis: Dict):
        """分析最终结果"""
        print(f"\n{'='*60}")
        print("分割结果分析")
        print(f"{'='*60}")
        
        num_regions = len(results.get('masks', []))
        expected_regions = scene_analysis.get('total_expected_regions', 0)
        
        print(f"发现区域数: {num_regions}")
        print(f"预期区域数: {expected_regions}")
        
        if expected_regions > 0:
            coverage_ratio = num_regions / expected_regions * 100
            print(f"覆盖率: {coverage_ratio:.1f}%")
        
        # 类别统计
        if results.get('categories'):
            category_counts = Counter(results['categories'])
            print(f"\n类别分布:")
            for category, count in category_counts.most_common(10):
                print(f"  {category}: {count}个区域")
        
        # 分数统计
        if results.get('scores'):
            scores = results['scores']
            print(f"\n置信度统计:")
            print(f"  平均值: {np.mean(scores):.3f}")
            print(f"  中位数: {np.median(scores):.3f}")
            print(f"  最小值: {np.min(scores):.3f}")
            print(f"  最大值: {np.max(scores):.3f}")
        
        # 提示词使用统计
        if results.get('prompts_used'):
            prompt_counts = Counter(results['prompts_used'])
            print(f"\n最有效提示词:")
            for prompt, count in prompt_counts.most_common(5):
                print(f"  '{prompt}': {count}次成功")