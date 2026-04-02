# core/session_manager.py
import os
import json
import time
from datetime import datetime
from config import Config

class SessionManager:
    """管理整个交互会话，协调各个组件"""
    
    def __init__(self, qwen_agent, sam3_adapter, instruct_parser):
        self.qwen_agent = qwen_agent
        self.sam3_adapter = sam3_adapter
        self.instruct_parser = instruct_parser
        
        # 会话状态
        self.current_session = None
        self.session_history = []
        
        # 创建临时目录
        os.makedirs(Config.TEMP_DIR, exist_ok=True)
    
    def start_session(self, image_path, user_request):
        """开始新的交互会话"""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        session = {
            "session_id": session_id,
            "image_path": image_path,
            "user_request": user_request,
            "start_time": time.time(),
            "steps": [],
            "results": []
        }
        
        self.current_session = session
        print(f"开始新会话: {session_id}")
        print(f"图像: {image_path}")
        print(f"用户请求: {user_request}")
        
        # 1. 设置图像
        self.sam3_adapter.set_image(image_path)
        img_width, img_height = self.sam3_adapter.image_size
        
        # 2. 让千问3分析图像并生成指令
        print("\n[千问3正在分析图像...]")
        instructions_data = self.qwen_agent.generate_instructions(image_path, user_request)
        
        print(f"千问3策略: {instructions_data.get('strategy', 'N/A')}")
        raw_instructions = instructions_data.get("instructions", [])
        
        # 3. 解析指令
        print("\n[正在解析指令...]")
        parsed_instructions = []
        for instr in raw_instructions:
            parsed = self.instruct_parser.parse_instruction(
                instr, img_width, img_height
            )
            parsed_instructions.append(parsed)
        
        # 记录到会话
        session["qwen_instructions"] = instructions_data
        session["parsed_instructions"] = parsed_instructions
        
        # 4. 执行指令序列
        print(f"\n[正在执行{len(parsed_instructions)}条指令...]")
        results = self.sam3_adapter.execute_instruction_sequence(
            parsed_instructions,
            score_threshold=Config.DEFAULT_SCORE_THRESHOLD
        )
        
        session["results"] = results
        session["end_time"] = time.time()
        session["duration"] = session["end_time"] - session["start_time"]
        
        # 保存会话
        self._save_session(session)
        self.session_history.append(session)
        
        return session
    
    def interactive_refinement(self, refinement_request):
        """交互式细化分割结果"""
        if self.current_session is None:
            raise ValueError("没有活跃的会话")
        
        print(f"\n[细化请求: {refinement_request}]")
        
        # 1. 获取当前状态描述
        current_results = self.current_session["results"]
        result_summary = f"当前已分割{len(current_results)}个物体"
        
        # 2. 让千问3生成细化指令
        refinement_data = self.qwen_agent.generate_instructions(
            self.current_session["image_path"],
            f"{refinement_request}。当前状态: {result_summary}",
            self.current_session.get("image_description")
        )
        
        # 3. 解析并执行细化指令
        img_width, img_height = self.sam3_adapter.image_size
        raw_instr = refinement_data.get("instructions", [])
        
        parsed_instr = []
        for instr in raw_instr:
            parsed = self.instruct_parser.parse_instruction(
                instr, img_width, img_height
            )
            parsed_instr.append(parsed)
        
        print(f"执行{len(parsed_instr)}条细化指令...")
        new_results = self.sam3_adapter.execute_instruction_sequence(parsed_instr)
        
        # 合并结果（简单去重）
        all_results = self._merge_results(
            self.current_session["results"], 
            new_results
        )
        
        # 更新会话
        self.current_session["results"] = all_results
        self.current_session["refinements"] = self.current_session.get("refinements", []) + [{
            "request": refinement_request,
            "instructions": refinement_data,
            "new_results": len(new_results)
        }]
        
        # 保存更新
        self._save_session(self.current_session)
        
        return all_results
    
    def _merge_results(self, old_results, new_results, iou_threshold=0.5):
        """合并新旧结果，基于IoU去重"""
        if not old_results:
            return new_results
        if not new_results:
            return old_results
        
        merged = old_results.copy()
        
        for new_res in new_results:
            # 计算与所有旧结果的IoU
            max_iou = 0
            for old_res in old_results:
                iou = self._calculate_iou(new_res["bbox"], old_res["bbox"])
                max_iou = max(max_iou, iou)
            
            # 如果IoU低于阈值，添加新结果
            if max_iou < iou_threshold:
                merged.append(new_res)
        
        return merged
    
    def _calculate_iou(self, bbox1, bbox2):
        """计算两个边界框的IoU"""
        # bbox格式: [x1, y1, x2, y2]
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # 计算交集
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        
        # 计算并集
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _save_session(self, session):
        """保存会话到文件"""
        session_file = os.path.join(
            Config.TEMP_DIR, 
            f"session_{session['session_id']}.json"
        )
        
        # 转换不可JSON序列化的对象
        saveable_session = session.copy()
        if "results" in saveable_session:
            for res in saveable_session["results"]:
                if "mask" in res:
                    res["mask_shape"] = res["mask"].shape
                    del res["mask"]
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(saveable_session, f, indent=2, ensure_ascii=False)
        
        print(f"会话已保存: {session_file}")