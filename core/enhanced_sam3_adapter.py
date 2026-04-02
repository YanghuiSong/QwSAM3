# core/enhanced_sam3_adapter.py
import torch
import numpy as np
from PIL import Image
import cv2
import sys
sys.path.append('.')
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from config import Config

class EnhancedSAM3Adapter:
    """增强的SAM3适配器，专门用于全分割"""
    
    def __init__(self, checkpoint=Config.SAM3_CHECKPOINT, device=None):
        print("Loading Enhanced SAM3 adapter...")
        
        # 获取设备字符串
        if device is None:
            device = Config.get_sam3_device()
        
        print(f"  Checkpoint: {checkpoint}")
        print(f"  Device: {device}")
        
        try:
            # 将字符串转换为torch.device对象
            self.device = torch.device(device)
            
            # 加载模型
            self.model = build_sam3_image_model(
                checkpoint_path=checkpoint,
                device='cpu',  # 先加载到CPU
                eval_mode=True,
                compile=False
            )
            
            # 将模型移动到指定设备
            if device != 'cpu':
                self.model = self.model.to(self.device)
            
            # 创建处理器
            self.processor = Sam3Processor(self.model)
            self.model.eval()
            
            # 状态管理
            self.initial_state = None  # 保存初始状态（包含图像特征）
            self.current_state = None
            self.image_size = None
            self.current_image = None
            self._image_loaded = False
            
            print("✓ Enhanced SAM3 adapter loaded successfully")
            
        except Exception as e:
            print(f"✗ Enhanced SAM3 adapter loading failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def set_image(self, image):
        """设置图像并保存初始状态"""
        try:
            # 重置状态
            self._image_loaded = False
            self.initial_state = None
            self.current_state = None
            
            if isinstance(image, str):
                import os
                if not os.path.exists(image):
                    print(f"✗ Image file not found: {image}")
                    return None
                
                try:
                    image = Image.open(image).convert("RGB")
                except Exception as e:
                    print(f"✗ Failed to open image: {e}")
                    return None
            elif isinstance(image, Image.Image):
                if image.mode != 'RGB':
                    image = image.convert("RGB")
            else:
                print(f"✗ Unsupported image type: {type(image)}")
                return None
            
            self.current_image = image
            self.image_size = image.size
            
            print(f"  Setting image, size: {self.image_size}")
            
            # 初始化SAM3状态
            try:
                self.initial_state = self.processor.set_image(image)
                self.current_state = self.initial_state
                self._image_loaded = True
                print("✓ Image state set successfully")
                return self.current_state
            except Exception as e:
                print(f"✗ Failed to set image state: {e}")
                import traceback
                traceback.print_exc()
                return None
            
        except Exception as e:
            print(f"✗ Failed to set image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def reset_to_initial_state(self):
        """重置到初始状态（包含图像特征）"""
        if self.initial_state is not None:
            self.current_state = self.initial_state
            print("✓ Reset to initial state")
            return self.current_state
        else:
            print("⚠ No initial state available")
            return None
    
    def execute_text_prompt(self, text_prompt):
        """执行文本提示（增强版）"""
        if not self._image_loaded or self.current_state is None:
            print("Warning: State is empty, please set image first")
            return []
        
        try:
            print(f"  Executing text prompt: '{text_prompt}'")
            
            # 使用当前状态执行文本提示
            new_state = self.processor.set_text_prompt(
                state=self.current_state,
                prompt=text_prompt
            )
            
            if new_state is None:
                print(f"  Text prompt execution failed")
                return []
            
            # 更新当前状态
            self.current_state = new_state
            
            # 提取结果
            results = self._extract_results(new_state)
            print(f"  Found {len(results)} results")
            
            return results
            
        except Exception as e:
            print(f"✗ Text prompt execution failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def exhaustive_segmentation(self, object_categories, 
                               score_threshold=0.2, iou_threshold=0.5,
                               max_attempts_per_category=3):
        """
        穷举分割：尝试所有物体类别
        
        Args:
            object_categories: 物体类别列表
            score_threshold: 置信度阈值
            iou_threshold: IoU阈值（非极大值抑制）
            max_attempts_per_category: 每个类别最大尝试次数
            
        Returns:
            分割结果字典
        """
        if not self._image_loaded or self.initial_state is None:
            print("Error: Image not set or initial state not available")
            return {'masks': [], 'boxes': [], 'scores': [], 'categories': []}
        
        all_masks = []
        all_boxes = []
        all_scores = []
        all_categories = []
        
        print(f"Starting exhaustive segmentation with {len(object_categories)} categories...")
        
        for i, category in enumerate(object_categories):
            print(f"  [{i+1}/{len(object_categories)}] Trying category: '{category}'")
            
            # 重置到初始状态
            self.reset_to_initial_state()
            
            # 尝试分割当前类别
            attempt_results = []
            for attempt in range(max_attempts_per_category):
                try:
                    # 如果是第二次及以后的尝试，尝试添加变体
                    if attempt > 0:
                        # 尝试复数形式或其他变体
                        if category.endswith('s'):
                            variant = category[:-1]  # 去掉s
                        else:
                            variant = category + 's'  # 加s
                        print(f"    Attempt {attempt+1}: trying variant '{variant}'")
                        results = self.execute_text_prompt(variant)
                    else:
                        results = self.execute_text_prompt(category)
                    
                    if results:
                        # 过滤并收集结果
                        for result in results:
                            score = result.get('score', 0)
                            if score >= score_threshold:
                                mask = result.get('mask')
                                bbox = result.get('bbox', [])
                                
                                if mask is not None and mask.size > 0:
                                    attempt_results.append({
                                        'mask': mask,
                                        'bbox': bbox,
                                        'score': score,
                                        'category': category
                                    })
                        
                        print(f"    Found {len(results)} results, {len(attempt_results)} passed threshold")
                        break  # 成功找到结果，停止尝试
                    
                except Exception as e:
                    print(f"    Attempt {attempt+1} failed: {e}")
                    continue
            
            # 将本次尝试的结果添加到总结果中
            for result in attempt_results:
                all_masks.append(result['mask'])
                all_boxes.append(result['bbox'])
                all_scores.append(result['score'])
                all_categories.append(result['category'])
        
        print(f"\nTotal candidates found: {len(all_masks)}")
        
        # 应用非极大值抑制
        if all_masks:
            indices_to_keep = self._nms_indices(all_masks, all_scores, iou_threshold)
            
            final_masks = [all_masks[i] for i in indices_to_keep]
            final_boxes = [all_boxes[i] for i in indices_to_keep]
            final_scores = [all_scores[i] for i in indices_to_keep]
            final_categories = [all_categories[i] for i in indices_to_keep]
            
            print(f"After NMS: {len(final_masks)} unique results")
            
            return {
                'masks': final_masks,
                'boxes': final_boxes,
                'scores': final_scores,
                'categories': final_categories
            }
        
        return {'masks': [], 'boxes': [], 'scores': [], 'categories': []}
    
    def _nms_indices(self, masks, scores, iou_threshold):
        """非极大值抑制，返回要保留的索引"""
        if not masks:
            return []
        
        # 按分数排序（降序）
        sorted_indices = np.argsort(scores)[::-1]
        keep = []
        
        while len(sorted_indices) > 0:
            current_idx = sorted_indices[0]
            keep.append(current_idx)
            
            if len(sorted_indices) == 1:
                break
            
            # 计算当前mask与所有剩余mask的IoU
            current_mask = masks[current_idx]
            ious = []
            
            for idx in sorted_indices[1:]:
                other_mask = masks[idx]
                iou = self._calculate_iou(current_mask, other_mask)
                ious.append(iou)
            
            # 保留IoU低于阈值的索引
            remaining = []
            for i, idx in enumerate(sorted_indices[1:]):
                if ious[i] < iou_threshold:
                    remaining.append(idx)
            
            sorted_indices = remaining
        
        return keep
    
    def _calculate_iou(self, mask1, mask2):
        """计算两个mask的IoU"""
        try:
            # 二值化
            mask1_binary = (mask1 > 0.5).astype(np.uint8)
            mask2_binary = (mask2 > 0.5).astype(np.uint8)
            
            # 调整到相同尺寸
            h1, w1 = mask1_binary.shape
            h2, w2 = mask2_binary.shape
            
            if (h1, w1) != (h2, w2):
                mask2_binary = cv2.resize(mask2_binary.astype(float), (w1, h1))
                mask2_binary = (mask2_binary > 0.5).astype(np.uint8)
            
            # 计算交集和并集
            intersection = np.logical_and(mask1_binary, mask2_binary).sum()
            union = np.logical_or(mask1_binary, mask2_binary).sum()
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            print(f"IoU calculation failed: {e}")
            return 0.0
    
    def _extract_results(self, state):
        """从状态中提取结果"""
        if state is None:
            return []
        
        try:
            masks = state.get("masks", [])
            boxes = state.get("boxes", [])
            scores = state.get("scores", [])
            
            if not masks:
                return []
            
            results = []
            for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
                if len(mask) > 0:
                    # 处理mask
                    mask_data = None
                    if torch.is_tensor(mask[0]):
                        mask_data = mask[0].cpu().numpy()
                    else:
                        mask_data = mask[0]
                    
                    # 处理box
                    box_data = []
                    if box is not None:
                        if torch.is_tensor(box):
                            box_data = box.cpu().numpy().tolist()
                        elif hasattr(box, 'tolist'):
                            box_data = box.tolist()
                        else:
                            box_data = list(box)
                    
                    # 处理score
                    score_val = 0.0
                    if torch.is_tensor(score):
                        score_val = float(score.cpu().numpy())
                    else:
                        try:
                            score_val = float(score)
                        except:
                            score_val = 0.0
                    
                    # 确保mask有效
                    if mask_data is not None and mask_data.size > 0:
                        results.append({
                            "mask": mask_data,
                            "bbox": box_data,
                            "score": score_val
                        })
            
            return results
            
        except Exception as e:
            print(f"Failed to extract results: {e}")
            return []
    
    def get_status(self):
        """获取状态信息"""
        return {
            "image_loaded": self._image_loaded,
            "initial_state_available": self.initial_state is not None,
            "current_state_available": self.current_state is not None,
            "image_size": self.image_size
        }