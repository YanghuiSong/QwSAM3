# core/object_detector.py
import torch
from transformers import AutoModelForObjectDetection, AutoImageProcessor

class ObjectDetector:
    """物体检测辅助器，为SAM3提供初始候选"""
    
    def __init__(self, model_name="facebook/detr-resnet-50"):
        print("Loading object detector...")
        
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.model.eval()
        print("✓ Object detector loaded")
    
    def detect_objects(self, image_path, confidence_threshold=0.3):
        """检测图像中的物体，返回候选框"""
        from PIL import Image
        
        image = Image.open(image_path).convert("RGB")
        
        # 预处理
        inputs = self.processor(images=image, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 后处理
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=confidence_threshold
        )[0]
        
        # 提取框和标签
        boxes = []
        labels = []
        scores = []
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            # 转换框格式为归一化中心坐标
            x_min, y_min, x_max, y_max = box.tolist()
            width = x_max - x_min
            height = y_max - y_min
            x_center = x_min + width / 2
            y_center = y_min + height / 2
            
            # 归一化
            img_w, img_h = image.size
            norm_box = [
                x_center / img_w,
                y_center / img_h,
                width / img_w,
                height / img_h
            ]
            
            boxes.append(norm_box)
            labels.append(self.model.config.id2label[label.item()])
            scores.append(score.item())
        
        return boxes, labels, scores