# core/full_segmenter.py
import torch
import numpy as np
import cv2
import json
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import List, Dict, Any, Tuple

class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，用于处理numpy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class FullSegmenter:
    """开放词汇全分割器 - 增强版"""
    
    def __init__(self, qwen_agent, sam3_adapter):
        self.qwen_agent = qwen_agent
        self.sam3_adapter = sam3_adapter
        self.temp_dir = "./temp_results"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 分割配置
        self.score_threshold = 0.2  # 降低阈值以捕获更多结果
        self.iou_threshold = 0.5
    
    def full_segmentation(self, image_path: str) -> Dict[str, Any]:
        """执行开放词汇全分割 - 增强版"""
        print(f"\n=== 开始开放词汇全分割 ===")
        print(f"图像: {image_path}")
        
        # 检查文件
        if not os.path.exists(image_path):
            return {
                "success": False,
                "error": f"图像文件不存在: {image_path}",
                "image_path": image_path
            }
        
        try:
            # 加载图像以获取尺寸
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            h, w = image_np.shape[:2]
        except Exception as e:
            return {
                "success": False,
                "error": f"加载图像失败: {str(e)}",
                "image_path": image_path
            }
        
        # 1. 深度场景分析
        print("\n1. 深度场景分析...")
        try:
            scene_analysis = self.qwen_agent.analyze_scene(image_path, detail_level="high")
            print(f"✓ 场景分析完成")
        except Exception as e:
            print(f"✗ 场景分析失败: {e}")
            scene_analysis = "场景分析失败"
        
        # 2. 基于场景分析生成定制化的分割指令
        print("\n2. 生成定制化分割指令...")
        custom_instructions = self._generate_custom_instructions(scene_analysis)
        
        # 3. 设置图像到SAM3
        print("\n3. 设置图像到SAM3...")
        try:
            # 确保图像设置成功
            state = self.sam3_adapter.set_image(image_path)
            if state is None:
                # 尝试直接使用PIL图像
                state = self.sam3_adapter.set_image(image)
            
            if state is None:
                return {
                    "success": False,
                    "error": "无法设置图像到SAM3",
                    "image_path": image_path
                }
            print("✓ 图像设置成功")
        except Exception as e:
            return {
                "success": False,
                "error": f"设置图像失败: {str(e)}",
                "image_path": image_path
            }
        
        # 4. 执行多层次分割
        print("\n4. 执行多层次分割...")
        all_results = []
        
        # 第一层: 主要物体
        print("  → 第一层: 主要物体分割...")
        main_objects = self._extract_main_objects(scene_analysis)
        main_results = self._segment_with_objects(main_objects, image_path)
        if main_results:
            all_results.extend(main_results)
            print(f"    找到 {len(main_results)} 个主要物体")
        
        # 第二层: 场景元素
        print("  → 第二层: 场景元素分割...")
        scene_elements = ["ground", "sky", "road", "building", "tree", "car", "person"]
        scene_results = self._segment_with_objects(scene_elements, image_path)
        if scene_results:
            # 过滤掉与主要物体重叠的结果
            filtered_scene_results = self._filter_overlapping_results(scene_results, all_results)
            if filtered_scene_results:
                all_results.extend(filtered_scene_results)
                print(f"    找到 {len(filtered_scene_results)} 个场景元素")
        
        # 第三层: 通用物体
        print("  → 第三层: 通用物体分割...")
        generic_objects = ["object", "thing", "item", "structure", "area", "region"]
        generic_results = self._segment_with_objects(generic_objects, image_path)
        if generic_results:
            # 过滤掉与已有结果重叠的结果
            filtered_generic_results = self._filter_overlapping_results(generic_results, all_results)
            if filtered_generic_results:
                all_results.extend(filtered_generic_results)
                print(f"    找到 {len(filtered_generic_results)} 个通用物体")
        
        print(f"\n✓ 多层次分割完成，总共找到 {len(all_results)} 个结果")
        
        # 5. 去重处理
        print("\n5. 去重处理...")
        final_results = self._post_process_results(all_results)
        print(f"去重后保留 {len(final_results)} 个结果")
        
        # 6. 生成全景分割图
        print("\n6. 生成全景分割图...")
        panoptic_path = self._create_panoptic_segmentation(
            image_path, final_results, "开放词汇全分割"
        )
        
        # 7. 保存结果
        print("\n7. 保存分割结果...")
        result_data = self._save_full_segmentation_results(
            image_path, final_results, scene_analysis, panoptic_path
        )
        
        print(f"\n=== 全分割完成 ===")
        print(f"总共分割出 {len(final_results)} 个区域")
        
        if result_data.get("success", False):
            return result_data
        else:
            return {
                "success": False,
                "error": result_data.get("error", "保存结果失败"),
                "image_path": image_path
            }
    
    def _extract_main_objects(self, scene_analysis: str) -> List[str]:
        """从场景分析中提取主要物体（增强版）"""
        import re
        
        objects = []
        
        # 1. 直接从场景分析中提取英文名词
        # 模式匹配：字母单词，长度3-15，可能是物体名词
        english_pattern = r'\b([a-z]{3,15})\b'
        scene_lower = scene_analysis.lower()
        
        # 常见物体词汇（SAM3训练中常见的类别）
        common_objects = [
            # 通用类别
            "person", "people", "human", "man", "woman", "child", "baby",
            "car", "vehicle", "truck", "bus", "motorcycle", "bicycle", "van", "train",
            "building", "house", "structure", "architecture", "shop", "store", "home",
            "tree", "plant", "vegetation", "flower", "grass", "bush", "leaf",
            "road", "street", "pavement", "sidewalk", "path", "lane", "highway",
            "sky", "cloud", "sun", "moon", "star",
            "water", "river", "lake", "sea", "ocean", "pool", "pond",
            "ground", "floor", "earth", "land", "terrain", "field",
            "animal", "dog", "cat", "bird", "horse", "cow", "sheep",
            "furniture", "chair", "table", "sofa", "bed", "desk", "cabinet",
            "electronic", "computer", "phone", "television", "screen", "monitor",
            "food", "fruit", "vegetable", "meal", "drink", "water", "coffee",
            "clothing", "shirt", "pants", "dress", "shoe", "hat", "jacket",
            "equipment", "machine", "tool", "device", "instrument",
            "book", "paper", "document", "magazine", "newspaper",
            "container", "box", "bag", "bottle", "cup", "glass", "plate",
            "light", "lamp", "bulb", "candle", "flashlight",
            "window", "door", "gate", "fence", "wall", "roof", "floor",
            "sign", "symbol", "logo", "label", "text", "letter", "number"
        ]
        
        # 2. 提取英文名词
        english_matches = re.findall(english_pattern, scene_lower)
        for word in english_matches:
            # 检查是否是常见物体词汇
            if word in common_objects and word not in objects:
                objects.append(word)
        
        # 3. 如果没有找到足够物体，添加通用类别
        if len(objects) < 5:
            generic_categories = [
                "object", "thing", "item", "element", "component",
                "region", "area", "zone", "section", "part",
                "material", "substance", "texture", "pattern",
                "shape", "form", "structure", "construction"
            ]
            objects.extend(generic_categories[:10-len(objects)])
        
        # 4. 去重并返回（最多15个）
        unique_objects = []
        for obj in objects:
            if obj not in unique_objects and len(obj) > 2:
                unique_objects.append(obj)
        
        print(f"提取到 {len(unique_objects)} 个物体类别: {unique_objects[:10]}...")
        return unique_objects[:15]
    
    def _segment_with_objects(self, objects: List[str], image_path: str) -> List[Dict]:
        """使用给定的物体列表进行分割"""
        results = []
        
        for obj in objects:
            print(f"    尝试分割: {obj}")
            try:
                # 重置SAM3状态
                self.sam3_adapter.reset_prompts()
                
                # 执行分割
                obj_results = self.sam3_adapter.execute_text_prompt(obj)
                
                if obj_results:
                    print(f"      找到 {len(obj_results)} 个结果")
                    
                    # 过滤低质量结果
                    filtered = []
                    for r in obj_results:
                        score = r.get('score', 0)
                        if score >= self.score_threshold:
                            filtered.append(r)
                    
                    if filtered:
                        print(f"      保留 {len(filtered)} 个高质量结果")
                        results.extend(filtered)
                else:
                    print(f"      未找到结果")
                    
            except Exception as e:
                print(f"      分割失败: {e}")
                continue
        
        return results
    
    def _filter_overlapping_results(self, new_results: List[Dict], existing_results: List[Dict]) -> List[Dict]:
        """过滤掉与已有结果重叠的新结果"""
        if not existing_results:
            return new_results
        
        filtered = []
        
        for new_result in new_results:
            new_mask = new_result.get('mask')
            if new_mask is None:
                continue
            
            # 检查是否与任何已有结果显著重叠
            is_overlapping = False
            for existing_result in existing_results:
                existing_mask = existing_result.get('mask')
                if existing_mask is None:
                    continue
                
                try:
                    iou = self._calculate_iou(new_mask, existing_mask)
                    if iou > 0.3:  # 重叠超过30%
                        is_overlapping = True
                        break
                except:
                    continue
            
            if not is_overlapping:
                filtered.append(new_result)
        
        return filtered
    
    def _post_process_results(self, all_results: List[Dict]) -> List[Dict]:
        """后处理结果"""
        if not all_results:
            return []
        
        # 按分数排序
        sorted_results = sorted(all_results, key=lambda x: x.get('score', 0), reverse=True)
        
        # 基于IoU的去重
        final_results = []
        for i, result in enumerate(sorted_results):
            if i == 0:
                final_results.append(result)
                continue
            
            mask1 = result.get('mask')
            if mask1 is None:
                continue
            
            # 检查与已有结果的IoU
            is_duplicate = False
            for existing_result in final_results:
                mask2 = existing_result.get('mask')
                if mask2 is None:
                    continue
                
                try:
                    iou = self._calculate_iou(mask1, mask2)
                    if iou > self.iou_threshold:
                        is_duplicate = True
                        break
                except:
                    continue
            
            if not is_duplicate:
                final_results.append(result)
        
        return final_results
    
    def _calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """计算IoU"""
        try:
            # 确保掩码是二值化的
            mask1_binary = (mask1 > 0.5).astype(np.uint8)
            mask2_binary = (mask2 > 0.5).astype(np.uint8)
            
            # 调整到相同尺寸
            if mask1_binary.shape != mask2_binary.shape:
                h, w = mask1_binary.shape
                mask2_binary = cv2.resize(mask2_binary, (w, h))
            
            intersection = np.logical_and(mask1_binary, mask2_binary).sum()
            union = np.logical_or(mask1_binary, mask2_binary).sum()
            
            return intersection / union if union > 0 else 0
        except Exception as e:
            print(f"计算IoU失败: {e}")
            return 0
    
    def _create_panoptic_segmentation(self, image_path: str, results: List[Dict], 
                                     title: str) -> str:
        """创建全景分割图"""
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print("无法读取图像")
                return ""
            
            h, w = image.shape[:2]
            
            # 创建全景掩码
            panoptic_mask = np.zeros((h, w, 3), dtype=np.uint8)
            
            if results:
                # 使用颜色映射
                colors = plt.cm.tab20(np.linspace(0, 1, min(len(results), 20)))
                
                for i, result in enumerate(results):
                    mask = result.get('mask')
                    if mask is None or mask.size == 0:
                        continue
                    
                    try:
                        # 缩放掩码到图像尺寸
                        mask_resized = cv2.resize(mask.astype(np.float32), (w, h))
                        binary_mask = (mask_resized > 0.5).astype(np.uint8)
                        
                        if binary_mask.sum() > 0:
                            color = colors[i % len(colors)][:3] * 255
                            colored_region = np.zeros((h, w, 3), dtype=np.uint8)
                            colored_region[binary_mask > 0] = color
                            panoptic_mask = cv2.addWeighted(panoptic_mask, 1, colored_region, 0.8, 0)
                            
                            # 添加边界框（可选）
                            bbox = result.get('bbox', [])
                            if len(bbox) == 4:
                                x_center, y_center, width, height = bbox
                                x1 = int((x_center - width/2) * w)
                                y1 = int((y_center - height/2) * h)
                                x2 = int((x_center + width/2) * w)
                                y2 = int((y_center + height/2) * h)
                                
                                cv2.rectangle(panoptic_mask, (x1, y1), (x2, y2), 
                                            (255, 255, 255), 2)
                    except Exception as e:
                        print(f"处理掩码 {i} 失败: {e}")
                        continue
            else:
                # 如果没有结果，显示原始图像
                panoptic_mask = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 创建可视化
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # 原始图像
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axes[0].imshow(image_rgb)
            axes[0].set_title('原始图像')
            axes[0].axis('off')
            
            # 全景分割掩码
            axes[1].imshow(panoptic_mask)
            axes[1].set_title(f'{title} - {len(results)}个区域')
            axes[1].axis('off')
            
            plt.suptitle(f'开放词汇全分割结果', fontsize=14)
            plt.tight_layout()
            
            # 保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.temp_dir, f"panoptic_{timestamp}.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ 全景图已保存: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"创建全景图失败: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _save_full_segmentation_results(self, image_path: str, results: List[Dict], 
                                       scene_analysis: str, panoptic_path: str) -> Dict[str, Any]:
        """保存全分割结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join(self.temp_dir, f"full_seg_{timestamp}.json")
            
            # 准备结果数据
            result_data = {
                "image_path": str(image_path),
                "timestamp": str(timestamp),
                "total_regions": int(len(results)),
                "scene_analysis": str(scene_analysis[:500]) if scene_analysis else "",
                "panoptic_image": str(panoptic_path) if panoptic_path else "",
                "regions": [],
                "statistics": {
                    "average_score": 0.0,
                    "min_score": 0.0,
                    "max_score": 0.0,
                    "total_area": 0
                }
            }
            
            # 处理区域数据
            scores = []
            total_area = 0
            
            for i, result in enumerate(results):
                # 转换分数
                score = float(result.get("score", 0.0))
                scores.append(score)
                
                # 计算面积
                area = 0
                mask = result.get('mask')
                if mask is not None and hasattr(mask, 'sum'):
                    try:
                        area = int(mask.sum())
                        total_area += area
                    except:
                        pass
                
                # 处理边界框
                bbox = result.get("bbox", [])
                bbox_list = []
                if bbox and hasattr(bbox, '__iter__'):
                    try:
                        bbox_list = [float(x) for x in bbox]
                    except:
                        bbox_list = []
                
                # 尝试推断类别
                category = self._infer_category(result, scene_analysis)
                
                result_data["regions"].append({
                    "id": int(i),
                    "score": score,
                    "bbox": bbox_list,
                    "area_pixels": int(area),
                    "category": category
                })
            
            # 更新统计信息
            if scores:
                result_data["statistics"]["average_score"] = float(np.mean(scores))
                result_data["statistics"]["min_score"] = float(np.min(scores))
                result_data["statistics"]["max_score"] = float(np.max(scores))
                result_data["statistics"]["total_area"] = int(total_area)
            
            # 保存JSON
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            
            print(f"✓ 结果已保存到: {result_path}")
            
            return {
                "success": True,
                "result_path": str(result_path),
                "panoptic_path": str(panoptic_path),
                "num_regions": int(len(results)),
                "data": result_data
            }
            
        except Exception as e:
            print(f"保存结果失败: {e}")
            
            return {
                "success": False,
                "error": f"保存结果失败: {str(e)}"
            }
    
    def _infer_category(self, result: Dict, scene_analysis: str) -> str:
        """尝试推断分割结果的类别"""
        # 基于边界框位置和大小推断
        bbox = result.get("bbox", [])
        if len(bbox) == 4:
            x_center, y_center, width, height = bbox
            
            # 根据位置和大小推断
            if width > 0.3 or height > 0.3:  # 大物体
                if y_center < 0.3:  # 顶部
                    return "sky"
                elif y_center > 0.7:  # 底部
                    return "ground"
                else:
                    return "building"
            elif width > 0.1 or height > 0.1:  # 中等物体
                return "vehicle"
            else:  # 小物体
                return "object"
        
        return "unknown"