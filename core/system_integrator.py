# core/system_integrator.py
import os
import json
import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

class SystemIntegrator:
    """系统集成器：协调千问3-VL和SAM3，实现智能交互分割"""
    
    def __init__(self, qwen_agent, sam3_adapter):
        """
        初始化系统集成器
        
        Args:
            qwen_agent: 千问3-VL智能体
            sam3_adapter: SAM3适配器
        """
        self.qwen_agent = qwen_agent
        self.sam3_adapter = sam3_adapter
        
        # 创建临时目录
        self.temp_dir = "./temp_results"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 初始化全分割器
        from core.full_segmenter import FullSegmenter
        self.full_segmenter = FullSegmenter(qwen_agent, sam3_adapter)
        
        # 状态管理
        self.current_image_path = None
        self.current_image = None
        self.image_description = None
        self.segmentation_history = []
        
        print("系统集成器初始化完成（包含全分割器）")
    
    def process_image(self, image_path: str, user_request: str) -> Dict[str, Any]:
        """
        处理单张图像的完整流程
        
        Args:
            image_path: 图像路径
            user_request: 用户请求
            
        Returns:
            分割结果字典
        """
        print(f"\n处理图像: {image_path}")
        print(f"用户请求: {user_request}")
        
        # 设置当前图像
        self.current_image_path = image_path
        self.current_image = Image.open(image_path).convert("RGB")
        
        # 步骤1: 使用千问3-VL理解图像内容
        print("\n1. 使用千问3-VL理解图像内容...")
        self.image_description = self.qwen_agent.describe_image(image_path)
        print(f"✓ 图像理解完成")
        print(f"图像描述摘要: {self.image_description[:200]}...")
        
        # 步骤2: 生成分割指令
        print("\n2. 生成分割指令...")
        instructions = self.qwen_agent.generate_instructions(
            image_path=image_path,
            user_request=user_request,
            image_description=self.image_description
        )
        
        if not instructions:
            print("✗ 指令生成失败")
            return {
                "success": False,
                "error": "指令生成失败",
                "image_path": image_path
            }
        
        print(f"✓ 指令生成完成")
        print(f"分割策略: {instructions.get('strategy', 'N/A')}")
        print(f"指令数量: {len(instructions.get('instructions', []))}")
        
        # 步骤3: 执行分割
        print("\n3. 执行SAM3分割...")
        
        # 设置图像到SAM3
        self.sam3_adapter.set_image(image_path)
        
        # 执行指令序列
        all_results = []
        try:
            all_results = self.sam3_adapter.execute_instruction_sequence(
                instructions.get('instructions', [])
            )
        except Exception as e:
            print(f"分割执行失败: {e}")
            # 尝试使用默认方式
            all_results = self._fallback_segmentation(user_request)
        
        print(f"✓ 分割完成，找到 {len(all_results)} 个物体")
        
        # 步骤4: 生成可视化结果
        print("\n4. 生成可视化结果...")
        visualization_path = self._visualize_results(
            image_path, 
            all_results, 
            instructions.get('strategy', '')
        )
        
        # 步骤5: 保存会话
        session_data = self._save_session(
            image_path=image_path,
            user_request=user_request,
            image_description=self.image_description,
            instructions=instructions,
            results=all_results,
            visualization_path=visualization_path
        )
        
        return {
            "success": True,
            "image_path": image_path,
            "user_request": user_request,
            "strategy": instructions.get('strategy', ''),
            "num_objects": len(all_results),
            "results": all_results,
            "visualization_path": visualization_path,
            "session_path": session_data['session_path']
        }
    
    def process_refinement(self, refinement_request: str) -> Dict[str, Any]:
        """
        处理细化请求
        
        Args:
            refinement_request: 细化请求
            
        Returns:
            细化结果
        """
        if not self.current_image_path:
            return {
                "success": False,
                "error": "请先处理一张图像"
            }
        
        print(f"\n[细化请求: {refinement_request}]")
        
        # 生成细化指令
        instructions = self.qwen_agent.generate_instructions(
            image_path=self.current_image_path,
            user_request=refinement_request,
            image_description=self.image_description
        )
        
        if not instructions or 'instructions' not in instructions:
            print("✗ 细化指令生成失败")
            return {
                "success": False,
                "error": "细化指令生成失败"
            }
        
        print(f"执行{len(instructions['instructions'])}条细化指令...")
        
        # 执行细化分割
        refinement_results = self.sam3_adapter.execute_instruction_sequence(
            instructions['instructions']
        )
        
        print(f"细化后找到 {len(refinement_results)} 个物体")
        
        # 更新可视化
        if refinement_results:
            # 合并历史结果和新结果
            all_results = self.segmentation_history[-1]['results'] + refinement_results
            visualization_path = self._visualize_results(
                self.current_image_path,
                all_results,
                f"细化: {refinement_request}"
            )
            
            # 更新会话
            session_data = self.segmentation_history[-1]
            session_data['refinements'].append({
                'request': refinement_request,
                'results': refinement_results,
                'timestamp': datetime.now().isoformat()
            })
            
            self._save_json(session_data, session_data['session_path'])
            
            return {
                "success": True,
                "num_new_objects": len(refinement_results),
                "visualization_path": visualization_path,
                "session_path": session_data['session_path']
            }
        else:
            return {
                "success": False,
                "error": "未找到新物体",
                "num_new_objects": 0
            }
    
    def full_segmentation(self, image_path: str, mode: str = "standard") -> Dict[str, Any]:
        """
        执行开放词汇全分割
        
        Args:
            image_path: 图像路径
            mode: 分割模式 ("standard", "interactive", "custom")
            
        Returns:
            全分割结果
        """
        self.current_image_path = image_path
        
        if mode == "interactive":
            return self.full_segmenter.interactive_full_segmentation(image_path)
        else:
            return self.full_segmenter.full_segmentation(image_path)
    
    def custom_category_segmentation(self, image_path: str, categories: str) -> Dict[str, Any]:
        """
        自定义类别分割
        
        Args:
            image_path: 图像路径
            categories: 类别字符串，用逗号分隔
            
        Returns:
            分割结果
        """
        return self.full_segmenter.custom_segmentation(image_path, categories)
    
    def _fallback_segmentation(self, user_request: str) -> List[Dict]:
        """后备分割策略"""
        print("使用后备分割策略...")
        
        # 提取关键词
        import re
        keywords = re.findall(r'\b\w{3,}\b', user_request.lower())
        
        if not keywords:
            keywords = ["object"]
        
        results = []
        for keyword in keywords[:3]:  # 最多尝试3个关键词
            try:
                self.sam3_adapter.reset_prompts()
                keyword_results = self.sam3_adapter.execute_text_prompt(keyword)
                if keyword_results:
                    results.extend(keyword_results)
                    print(f"  找到 '{keyword}': {len(keyword_results)} 个")
            except:
                continue
        
        return results
    
    def _visualize_results(self, image_path: str, results: List[Dict], title: str = "") -> str:
        """生成可视化结果"""
        try:
            # 读取图像
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 创建子图
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 子图1: 原始图像
            axes[0].imshow(image_rgb)
            axes[0].set_title('原始图像')
            axes[0].axis('off')
            
            # 子图2: 分割掩码覆盖
            overlay = image_rgb.copy()
            mask_overlay = np.zeros_like(image_rgb)
            
            colors = plt.cm.tab20(np.linspace(0, 1, min(len(results), 20)))
            
            for i, result in enumerate(results):
                mask = result['mask']
                if len(mask.shape) == 2 and mask.shape[0] > 0 and mask.shape[1] > 0:
                    # 缩放掩码到图像尺寸
                    mask_resized = cv2.resize(mask.astype(np.uint8), 
                                            (image_rgb.shape[1], image_rgb.shape[0]))
                    
                    # 创建彩色掩码
                    color = np.array(colors[i % len(colors)][:3]) * 255
                    colored_mask = np.zeros_like(image_rgb)
                    colored_mask[mask_resized > 0] = color
                    
                    # 叠加掩码
                    mask_overlay = cv2.addWeighted(mask_overlay, 1, colored_mask, 0.7, 0)
            
            # 叠加掩码到原始图像
            overlay = cv2.addWeighted(image_rgb, 0.6, mask_overlay.astype(np.uint8), 0.4, 0)
            axes[1].imshow(overlay)
            axes[1].set_title(f'分割结果 ({len(results)} 个物体)')
            axes[1].axis('off')
            
            # 子图3: 边界框
            bbox_image = image_rgb.copy()
            for i, result in enumerate(results):
                bbox = result['bbox']
                if len(bbox) == 4:
                    # 转换归一化坐标到像素坐标
                    h, w = image_rgb.shape[:2]
                    x_center, y_center, width, height = bbox
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)
                    
                    # 绘制边界框
                    color = colors[i % len(colors)][:3]
                    cv2.rectangle(bbox_image, (x1, y1), (x2, y2), 
                                (color[0]*255, color[1]*255, color[2]*255), 2)
                    
                    # 添加标签
                    label = f"Obj {i+1}: {result['score']:.2f}"
                    cv2.putText(bbox_image, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                               (color[0]*255, color[1]*255, color[2]*255), 1)
            
            axes[2].imshow(bbox_image)
            axes[2].set_title('边界框标注')
            axes[2].axis('off')
            
            # 设置总标题
            if title:
                fig.suptitle(title, fontsize=14)
            
            plt.tight_layout()
            
            # 保存图像
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.temp_dir, f"visualization_{timestamp}.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ 可视化结果已保存: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"可视化失败: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _save_session(self, **kwargs) -> Dict:
        """保存会话数据"""
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "image_path": kwargs.get("image_path", ""),
            "user_request": kwargs.get("user_request", ""),
            "image_description": kwargs.get("image_description", ""),
            "instructions": kwargs.get("instructions", {}),
            "results": kwargs.get("results", []),
            "visualization_path": kwargs.get("visualization_path", ""),
            "refinements": []
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_path = os.path.join(self.temp_dir, f"session_{timestamp}.json")
        
        self._save_json(session_data, session_path)
        
        # 添加到历史
        self.segmentation_history.append(session_data)
        
        print(f"✓ 会话已保存: {session_path}")
        
        return {
            "session_path": session_path,
            "session_data": session_data
        }
    
    def _save_json(self, data: Dict, path: str):
        """保存JSON文件"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存JSON失败: {e}")
            return False
    
    def load_session(self, session_path: str) -> Dict:
        """加载会话"""
        try:
            with open(session_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            self.segmentation_history.append(session_data)
            
            # 恢复状态
            self.current_image_path = session_data.get("image_path")
            self.image_description = session_data.get("image_description")
            
            print(f"✓ 会话加载成功: {session_path}")
            return session_data
        except Exception as e:
            print(f"加载会话失败: {e}")
            return {}
    
    def batch_process(self, input_dir: str, output_dir: str, request_file: str):
        """批量处理图像"""
        if not os.path.exists(input_dir):
            print(f"输入目录不存在: {input_dir}")
            return
        
        if not os.path.exists(request_file):
            print(f"请求文件不存在: {request_file}")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取请求文件
        requests = []
        with open(request_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and ',' in line:
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        requests.append({
                            "image": parts[0].strip(),
                            "request": parts[1].strip()
                        })
        
        print(f"开始批量处理 {len(requests)} 个请求...")
        
        results = []
        for i, req in enumerate(requests):
            image_path = os.path.join(input_dir, req["image"])
            
            if not os.path.exists(image_path):
                print(f"图像不存在: {image_path}")
                continue
            
            print(f"\n[{i+1}/{len(requests)}] 处理: {req['image']}")
            print(f"请求: {req['request']}")
            
            try:
                result = self.process_image(image_path, req["request"])
                
                if result["success"]:
                    # 复制结果到输出目录
                    import shutil
                    if result.get("visualization_path"):
                        vis_name = os.path.basename(result["visualization_path"])
                        shutil.copy(result["visualization_path"], 
                                  os.path.join(output_dir, vis_name))
                    
                    results.append({
                        "image": req["image"],
                        "success": True,
                        "num_objects": result["num_objects"]
                    })
                else:
                    results.append({
                        "image": req["image"],
                        "success": False,
                        "error": result.get("error", "未知错误")
                    })
                    
            except Exception as e:
                print(f"处理失败: {e}")
                results.append({
                    "image": req["image"],
                    "success": False,
                    "error": str(e)
                })
        
        # 保存批量处理结果
        batch_result_path = os.path.join(output_dir, "batch_results.json")
        self._save_json({
            "timestamp": datetime.now().isoformat(),
            "total": len(requests),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "results": results
        }, batch_result_path)
        
        print(f"\n批量处理完成!")
        print(f"成功: {sum(1 for r in results if r['success'])}")
        print(f"失败: {sum(1 for r in results if not r['success'])}")
        print(f"详细结果: {batch_result_path}")
    
    def visualize_result(self, result: Dict):
        """可视化分割结果"""
        if not result.get("success") or not result.get("visualization_path"):
            print("无法可视化结果")
            return
        
        try:
            import matplotlib.pyplot as plt
            from PIL import Image
            
            img = Image.open(result["visualization_path"])
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"分割结果 - {result.get('strategy', '')}")
            plt.show()
        except Exception as e:
            print(f"可视化失败: {e}")