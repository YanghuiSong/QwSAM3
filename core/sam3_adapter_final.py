# core/sam3_adapter_final.py
import sys
import torch
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from config import Config
import cv2

class SAM3AdapterFinal:
    """最终修复版SAM3适配器"""
    
    def __init__(self, checkpoint=Config.SAM3_CHECKPOINT, device=None):
        print("Loading SAM3 model (final version)...")
        
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
            self.initial_state = None  # 保存初始状态
            self.current_state = None
            self.image_size = None
            self.current_image = None
            self._image_loaded = False
            
            print("✓ SAM3 model loaded successfully")
            
        except Exception as e:
            print(f"✗ SAM3 model loading failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def set_image(self, image):
        """设置图像"""
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
    
    def execute_text_prompt(self, text_prompt, score_threshold=0.3):
        """执行文本提示 - 完全修复版"""
        if not self._image_loaded or self.current_state is None:
            print("Warning: State is empty, please set image first")
            return []
        
        try:
            print(f"  Executing text prompt: '{text_prompt}'")
            
            # 执行文本提示
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
            results = self._extract_results_final(new_state, score_threshold)
            print(f"  Found {len(results)} results")
            
            return results
            
        except Exception as e:
            print(f"✗ Text prompt execution failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _extract_results_final(self, state, score_threshold=0.3):
        """最终版结果提取 - 正确处理布尔张量"""
        if state is None:
            return []
        
        try:
            # 获取masks、boxes、scores - 它们都是张量
            masks_tensor = state.get("masks")
            boxes_tensor = state.get("boxes")
            scores_tensor = state.get("scores")
            
            # 检查张量是否有效 - 使用numel()而不是布尔判断
            if masks_tensor is None or not torch.is_tensor(masks_tensor):
                print("    No masks tensor found")
                return []
            
            # 检查张量是否有元素
            if masks_tensor.numel() == 0:
                print("    Masks tensor is empty")
                return []
            
            # 获取mask数量
            num_masks = masks_tensor.shape[0]
            print(f"    Number of masks detected: {num_masks}")
            
            results = []
            
            for i in range(num_masks):
                try:
                    # 获取分数
                    score = 0.0
                    if scores_tensor is not None and i < scores_tensor.shape[0]:
                        if torch.is_tensor(scores_tensor):
                            score = float(scores_tensor[i].cpu().numpy())
                        else:
                            score = float(scores_tensor[i])
                    
                    # 跳过低分结果
                    if score < score_threshold:
                        continue
                    
                    # 获取mask - 布尔张量，形状为[1, H, W]
                    mask_item = masks_tensor[i]
                    
                    # mask_item的形状是 [1, H, W]，需要转换为浮点数
                    if mask_item.dim() == 3:  # [1, H, W]
                        # 布尔张量转换为浮点数
                        mask_np = mask_item[0].cpu().numpy().astype(float)
                    else:  # 其他形状，直接转换
                        mask_np = mask_item.cpu().numpy().astype(float)
                    
                    # 获取边界框 - 绝对坐标[x1, y1, x2, y2]
                    bbox = []
                    if boxes_tensor is not None and i < boxes_tensor.shape[0]:
                        box_item = boxes_tensor[i]
                        if torch.is_tensor(box_item):
                            # 转换为归一化中心坐标 [x_center, y_center, width, height]
                            x1, y1, x2, y2 = box_item.cpu().numpy()
                            width = x2 - x1
                            height = y2 - y1
                            x_center = x1 + width / 2
                            y_center = y1 + height / 2
                            
                            # 归一化（使用原始图像尺寸）
                            if hasattr(self, 'image_size') and self.image_size:
                                img_w, img_h = self.image_size
                            else:
                                # 如果没有image_size，使用默认值
                                img_w, img_h = 800, 800
                            
                            bbox = [
                                float(x_center / img_w),
                                float(y_center / img_h),
                                float(width / img_w),
                                float(height / img_h)
                            ]
                    
                    # 确保mask有效
                    if mask_np is not None and mask_np.size > 0:
                        # 检查mask是否包含任何有效像素
                        mask_sum = mask_np.sum()
                        if mask_sum > 10:  # 至少10个像素
                            results.append({
                                "mask": mask_np,
                                "bbox": bbox,
                                "score": score,
                                "index": i
                            })
                            print(f"      Mask {i}: score={score:.3f}, shape={mask_np.shape}, pixels={mask_sum:.0f}")
                        else:
                            print(f"      Mask {i} skipped: too small ({mask_sum:.0f} pixels)")
                    
                except Exception as e:
                    print(f"      Failed to process mask {i}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            print(f"Failed to extract results: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def exhaustive_segmentation(self, object_categories, 
                               score_threshold=0.2, iou_threshold=0.5):
        """穷举分割"""
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
            self.current_state = self.initial_state
            
            # 尝试分割
            try:
                results = self.execute_text_prompt(category, score_threshold)
                
                if results:
                    print(f"    Found {len(results)} results after filtering")
                    
                    # 收集结果
                    for result in results:
                        mask = result.get('mask')
                        bbox = result.get('bbox', [])
                        score = result.get('score', 0)
                        
                        if mask is not None and mask.size > 0:
                            all_masks.append(mask)
                            all_boxes.append(bbox)
                            all_scores.append(score)
                            all_categories.append(category)
                else:
                    print(f"    No results found for '{category}'")
                
            except Exception as e:
                print(f"    Segmentation failed: {e}")
                continue
        
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
        """非极大值抑制"""
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
        """计算IoU"""
        try:
            # 确保是浮点数
            mask1_float = mask1.astype(float)
            mask2_float = mask2.astype(float)
            
            # 二值化
            mask1_binary = (mask1_float > 0.5).astype(np.uint8)
            mask2_binary = (mask2_float > 0.5).astype(np.uint8)
            
            # 调整到相同尺寸
            h1, w1 = mask1_binary.shape
            h2, w2 = mask2_binary.shape
            
            if (h1, w1) != (h2, w2):
                mask2_resized = cv2.resize(mask2_binary.astype(float), (w1, h1))
                mask2_binary = (mask2_resized > 0.5).astype(np.uint8)
            
            # 计算交集和并集
            intersection = np.logical_and(mask1_binary, mask2_binary).sum()
            union = np.logical_or(mask1_binary, mask2_binary).sum()
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            print(f"IoU calculation failed: {e}")
            return 0.0
    
    def get_status(self):
        """获取状态"""
        return {
            "image_loaded": self._image_loaded,
            "initial_state_available": self.initial_state is not None,
            "image_size": self.image_size
        }