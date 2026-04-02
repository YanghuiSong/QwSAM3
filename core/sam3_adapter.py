# core/sam3_adapter.py
import sys
import torch
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from config import Config
import cv2

class SAM3Adapter:
    """SAM3 Adapter - Enhanced version for better full segmentation"""
    
    def __init__(self, checkpoint=Config.SAM3_CHECKPOINT, device=None):
        print("Loading SAM3 model...")
        
        # Get device string
        if device is None:
            self.device_str = Config.get_sam3_device()
        else:
            self.device_str = device
        
        print(f"  Checkpoint: {checkpoint}")
        print(f"  Device: {self.device_str}")
        
        try:
            # Convert string to torch.device object
            self.device = torch.device(self.device_str)
            
            # Load model
            self.model = build_sam3_image_model(
                checkpoint_path=checkpoint,
                device='cpu',  # Load to CPU first
                eval_mode=True,
                compile=False
            )
            
            # Move model to specified device
            if self.device_str != 'cpu':
                self.model = self.model.to(self.device)
            
            # Create processor
            self.processor = Sam3Processor(self.model)
            self.model.eval()
            
            # State management
            self.current_state = None
            self.image_size = None
            self.current_image = None
            self._image_loaded = False  # Mark if image is loaded
            
            print("✓ SAM3 model loaded successfully")
            
        except Exception as e:
            print(f"✗ SAM3 model loading failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def set_image(self, image):
        """Set current processing image - fixed version"""
        try:
            # Reset state
            self._image_loaded = False
            self.current_state = None
            
            if isinstance(image, str):
                # Check if file exists
                import os
                if not os.path.exists(image):
                    print(f"✗ Image file not found: {image}")
                    return None
                
                # Open image
                try:
                    image = Image.open(image).convert("RGB")
                except Exception as e:
                    print(f"✗ Failed to open image: {e}")
                    return None
            elif isinstance(image, Image.Image):
                # Ensure RGB format
                if image.mode != 'RGB':
                    image = image.convert("RGB")
            else:
                print(f"✗ Unsupported image type: {type(image)}")
                return None
            
            self.current_image = image
            self.image_size = image.size
            
            print(f"  Setting image, size: {self.image_size}")
            
            # Initialize SAM3 state
            try:
                self.current_state = self.processor.set_image(image)
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
    
    def execute_text_prompt(self, text_prompt, state=None):
        """Execute text prompt in English"""
        # Check if image is loaded
        if not self._image_loaded or self.current_state is None:
            print("Warning: State is empty, please set image first")
            return []
        
        try:
            if state is None:
                state = self.current_state
            
            if state is None:
                print("Error: SAM3 state is empty")
                return []
            
            print(f"  Executing text prompt: '{text_prompt}'")
            new_state = self.processor.set_text_prompt(
                state=state,
                prompt=text_prompt
            )
            
            if new_state is None:
                print(f"  Text prompt execution failed, returning empty state")
                return []
            
            self.current_state = new_state
            results = self._extract_results(new_state)
            print(f"  Found {len(results)} results")
            return results
            
        except Exception as e:
            print(f"✗ Text prompt execution failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def execute_box_prompt(self, box_coords, label=1, state=None):
        """Execute box prompt"""
        if not self._image_loaded or self.current_state is None:
            print("Warning: State is empty, please set image first")
            return []
        
        try:
            if state is None:
                state = self.current_state
            
            if state is None:
                print("Error: SAM3 state is empty")
                return []
            
            new_state = self.processor.add_geometric_prompt(
                state=state,
                geometric_prompt=box_coords,
                label=label
            )
            
            self.current_state = new_state
            return self._extract_results(new_state)
            
        except Exception as e:
            print(f"Box prompt execution failed: {e}")
            return []
    
    def execute_point_prompt(self, point_coords, label=1, state=None):
        """Execute point prompt"""
        if not self._image_loaded or self.current_state is None:
            print("Warning: State is empty, please set image first")
            return []
        
        try:
            if state is None:
                state = self.current_state
            
            if state is None:
                print("Error: SAM3 state is empty")
                return []
            
            new_state = self.processor.add_geometric_prompt(
                state=state,
                geometric_prompt=point_coords,
                label=label
            )
            
            self.current_state = new_state
            return self._extract_results(new_state)
            
        except Exception as e:
            print(f"Point prompt execution failed: {e}")
            return []
    
    def reset_prompts(self):
        """重置所有提示，但保持图像状态"""
        try:
            if self.current_state is not None:
                # 注意：reset_all_prompts 可能返回 None 或新状态
                new_state = self.processor.reset_all_prompts(self.current_state)
                if new_state is not None:
                    self.current_state = new_state
                    print("✓ Prompts reset")
                else:
                    # 如果返回 None，我们使用原始状态（保持图像特征）
                    print("✓ Prompts reset (using original state)")
            return self.current_state
        except Exception as e:
            print(f"Failed to reset prompts: {e}")
            # 即使失败，也返回当前状态
            return self.current_state
    
    def execute_instruction_sequence(self, instructions, score_threshold=Config.DEFAULT_SCORE_THRESHOLD):
        """Execute instruction sequence"""
        if not self._image_loaded:
            print("Error: Image not set, please call set_image() first")
            return []
        
        all_results = []
        
        for i, instr in enumerate(instructions):
            if not instr:
                continue
                
            instr_type = instr.get('type', '')
            print(f"Executing instruction {i+1}/{len(instructions)}: {instr_type}")
            
            try:
                if instr_type == 'text':
                    content = instr.get('content', '')
                    if not content:
                        print(f"  Warning: Text prompt content is empty")
                        continue
                        
                    results = self.execute_text_prompt(content)
                elif instr_type == 'box':
                    content = instr.get('content', [])
                    if not content:
                        print(f"  Warning: Box prompt content is empty")
                        continue
                        
                    results = self.execute_box_prompt(
                        content, 
                        instr.get('label', 1)
                    )
                elif instr_type == 'point':
                    content = instr.get('content', [])
                    if not content:
                        print(f"  Warning: Point prompt content is empty")
                        continue
                        
                    results = self.execute_point_prompt(
                        content,
                        instr.get('label', 1)
                    )
                else:
                    print(f"  Warning: Unknown instruction type '{instr_type}'")
                    continue
                
                # Filter low-score results
                if results:
                    filtered = self._filter_results(results, score_threshold)
                    if filtered:
                        all_results.extend(filtered)
                        
            except Exception as e:
                print(f"Failed to execute instruction {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Reset prompts for next instruction
            self.reset_prompts()
        
        return all_results
    
    def exhaustive_segmentation(self, object_categories, score_threshold=0.2, iou_threshold=0.5):
        """
        Perform exhaustive segmentation by trying all object categories
        
        Args:
            object_categories: List of object categories to try
            score_threshold: Confidence threshold for filtering
            iou_threshold: IoU threshold for non-maximum suppression
            
        Returns:
            Dictionary with masks, boxes, and scores
        """
        if not self._image_loaded:
            print("Error: Image not set")
            return {'masks': [], 'boxes': [], 'scores': []}
        
        all_masks = []
        all_boxes = []
        all_scores = []
        
        print(f"Starting exhaustive segmentation with {len(object_categories)} categories...")
        
        for i, category in enumerate(object_categories):
            print(f"  [{i+1}/{len(object_categories)}] Trying: '{category}'")
            
            # Reset prompts
            self.reset_prompts()
            
            # Try segmentation
            try:
                results = self.execute_text_prompt(category)
                
                if results:
                    # Filter by score
                    for result in results:
                        score = result.get('score', 0)
                        if score >= score_threshold:
                            mask = result.get('mask')
                            bbox = result.get('bbox', [])
                            
                            if mask is not None:
                                all_masks.append(mask)
                                all_boxes.append(bbox)
                                all_scores.append(score)
                                
                    print(f"    Found {len(results)} potential results")
                else:
                    print(f"    No results found")
                    
            except Exception as e:
                print(f"    Segmentation failed: {e}")
                continue
        
        print(f"\nTotal candidates found: {len(all_masks)}")
        
        # Apply non-maximum suppression
        if all_masks:
            final_masks, final_boxes, final_scores = self._nms_fusion(
                all_masks, all_boxes, all_scores, iou_threshold
            )
            
            print(f"After NMS: {len(final_masks)} results")
            
            return {
                'masks': final_masks,
                'boxes': final_boxes,
                'scores': final_scores
            }
        
        return {'masks': [], 'boxes': [], 'scores': []}
    
    def _nms_fusion(self, masks, boxes, scores, iou_threshold):
        """Non-maximum suppression fusion"""
        if not masks:
            return [], [], []
        
        # Sort by score
        indices = np.argsort(scores)[::-1]
        keep = []
        
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU between current and all others
            current_mask = masks[current]
            ious = []
            
            for idx in indices[1:]:
                other_mask = masks[idx]
                iou = self._calculate_iou(current_mask, other_mask)
                ious.append(iou)
            
            # Keep those with IoU below threshold
            remaining = []
            for i, idx in enumerate(indices[1:]):
                if ious[i] < iou_threshold:
                    remaining.append(idx)
            
            indices = remaining
        
        # Return kept results
        final_masks = [masks[i] for i in keep]
        final_boxes = [boxes[i] for i in keep]
        final_scores = [scores[i] for i in keep]
        
        return final_masks, final_boxes, final_scores
    
    def _calculate_iou(self, mask1, mask2):
        """Calculate IoU between two masks"""
        try:
            # Ensure masks are binary
            mask1_binary = (mask1 > 0.5).astype(np.uint8)
            mask2_binary = (mask2 > 0.5).astype(np.uint8)
            
            # Resize to same dimensions if needed
            if mask1_binary.shape != mask2_binary.shape:
                h, w = mask1_binary.shape
                mask2_binary = cv2.resize(mask2_binary, (w, h))
            
            intersection = np.logical_and(mask1_binary, mask2_binary).sum()
            union = np.logical_or(mask1_binary, mask2_binary).sum()
            
            return intersection / union if union > 0 else 0
        except Exception as e:
            print(f"IoU calculation failed: {e}")
            return 0
    
    def _extract_results(self, state):
        """Extract results from state"""
        if state is None:
            return []
        
        try:
            masks = state.get("masks", [])
            boxes = state.get("boxes", [])
            scores = state.get("scores", [])
            
            if not masks or not boxes or not scores:
                return []
            
            results = []
            for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
                if len(mask) > 0 and len(box) > 0:
                    # Process mask
                    mask_data = None
                    if torch.is_tensor(mask[0]):
                        mask_data = mask[0].cpu().numpy()
                    else:
                        mask_data = mask[0]
                    
                    # Process box
                    box_data = None
                    if torch.is_tensor(box):
                        box_data = box.cpu().numpy()
                    else:
                        box_data = box
                    
                    # Process score
                    score_val = 0.0
                    if torch.is_tensor(score):
                        score_val = float(score.cpu().numpy())
                    else:
                        try:
                            score_val = float(score)
                        except:
                            score_val = 0.0
                    
                    # Ensure mask and box are valid
                    if mask_data is not None and mask_data.size > 0:
                        results.append({
                            "mask": mask_data,
                            "bbox": box_data.tolist() if hasattr(box_data, 'tolist') else box_data,
                            "score": score_val
                        })
            
            return results
            
        except Exception as e:
            print(f"Failed to extract results: {e}")
            return []
    
    def _filter_results(self, results, threshold):
        """Filter results below threshold"""
        filtered = []
        for r in results:
            score = r.get('score', 0)
            if score >= threshold:
                filtered.append(r)
        return filtered
    
    def get_status(self):
        """Get status information"""
        return {
            "image_loaded": self._image_loaded,
            "image_size": self.image_size,
            "state_available": self.current_state is not None
        }