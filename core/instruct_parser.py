# core/instruct_parser.py
import re
import json

class InstructParser:
    """解析千问3生成的指令，转换为SAM3可执行的步骤"""
    
    @staticmethod
    def parse_instruction(instruction, image_width, image_height):
        """解析单条指令"""
        instr_type = instruction.get("type", "text")
        content = instruction.get("content", "")
        label = instruction.get("label", 1)  # 默认为正向提示
        
        if instr_type == "text":
            # 提取简单单词或短语（SAM3能理解的形式）
            simple_phrase = InstructParser._extract_simple_phrase(content)
            return {
                "type": "text",
                "content": simple_phrase,
                "action": "set_text_prompt"
            }
        
        elif instr_type == "box":
            # 解析框选坐标，支持多种格式
            box_coords = InstructParser._parse_box_coordinates(
                content, image_width, image_height
            )
            return {
                "type": "box",
                "content": box_coords,  # [x_center, y_center, width, height]
                "label": label,
                "action": "add_geometric_prompt"
            }
        
        elif instr_type == "point":
            # 解析点坐标
            point_coords = InstructParser._parse_point_coordinates(
                content, image_width, image_height
            )
            return {
                "type": "point",
                "content": point_coords,  # [x, y]
                "label": label,
                "action": "add_geometric_prompt"
            }
        
        else:
            raise ValueError(f"未知的指令类型: {instr_type}")
    
    @staticmethod
    def _extract_simple_phrase(text):
        """从复杂描述中提取简单短语"""
        # 移除冠词和复杂修饰
        text = re.sub(r'\b(a|an|the|this|that|these|those)\b', '', text, flags=re.IGNORECASE)
        
        # 提取名词短语（简单实现）
        words = text.strip().split()
        if len(words) <= 3:  # SAM3适合1-3个单词的短语
            return ' '.join(words[:3]).lower()
        else:
            # 取最后一个名词
            for word in reversed(words):
                if len(word) > 3:  # 避免短词
                    return word.lower()
            return words[-1].lower()
    
    @staticmethod
    def _parse_box_coordinates(box_str, img_w, img_h):
        """解析框选坐标，支持多种格式"""
        # 尝试解析JSON格式
        if box_str.startswith("[") and box_str.endswith("]"):
            try:
                coords = json.loads(box_str)
                if len(coords) == 4:
                    # 假设是归一化坐标 [x_center, y_center, width, height]
                    if all(0 <= c <= 1 for c in coords):
                        return coords
                    # 如果是像素坐标，转换为归一化
                    elif img_w > 0 and img_h > 0:
                        return [
                            coords[0] / img_w,  # x_center
                            coords[1] / img_h,  # y_center
                            coords[2] / img_w,  # width
                            coords[3] / img_h   # height
                        ]
            except:
                pass
        
        # 尝试解析文本描述
        patterns = [
            r"box:\s*\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]",
            r"bbox:\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)",
            r"坐标:\s*([\d\.]+)[,\s]+([\d\.]+)[,\s]+([\d\.]+)[,\s]+([\d\.]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, box_str)
            if match:
                coords = [float(match.group(i)) for i in range(1, 5)]
                # 转换为归一化坐标
                if img_w > 0 and img_h > 0 and any(c > 1 for c in coords):
                    # 假设是 [x_min, y_min, x_max, y_max] 像素格式
                    x_min, y_min, x_max, y_max = coords
                    x_center = (x_min + x_max) / 2 / img_w
                    y_center = (y_min + y_max) / 2 / img_h
                    width = (x_max - x_min) / img_w
                    height = (y_max - y_min) / img_h
                    return [x_center, y_center, width, height]
                else:
                    # 假设已经是归一化坐标
                    return coords
        
        # 默认返回图像中心的框
        return [0.5, 0.5, 0.3, 0.3]
    
    @staticmethod
    def _parse_point_coordinates(point_str, img_w, img_h):
        """解析点坐标"""
        # 类似框选坐标的解析逻辑
        if point_str.startswith("[") and point_str.endswith("]"):
            try:
                coords = json.loads(point_str)
                if len(coords) == 2:
                    if all(0 <= c <= 1 for c in coords):
                        return coords
                    elif img_w > 0 and img_h > 0:
                        return [coords[0] / img_w, coords[1] / img_h]
            except:
                pass
        
        patterns = [
            r"point:\s*\[([\d\.]+),\s*([\d\.]+)\]",
            r"坐标:\s*([\d\.]+)[,\s]+([\d\.]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, point_str)
            if match:
                coords = [float(match.group(i)) for i in range(1, 3)]
                if img_w > 0 and img_h > 0 and any(c > 1 for c in coords):
                    return [coords[0] / img_w, coords[1] / img_h]
                else:
                    return coords
        
        # 默认返回图像中心点
        return [0.5, 0.5]