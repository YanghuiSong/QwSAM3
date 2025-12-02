# QwSAM3: 基于SAM3的开放词汇遥感图像分割

![Open Vocabulary Segmentation Results](https://github.com/YanghuiSong/QwSAM3/blob/main/figures/open_vocab_result.jpg?raw=true)

## 项目简介

QwSAM3是一个基于SAM3（Segment Anything Model 3）的训练-free框架，专注于开放词汇遥感图像分割任务。本项目利用强大的SAM3模型和Qwen3-VL多模态大模型，实现了无需训练即可进行开放词汇目标识别与分割的功能，特别适用于遥感图像分析场景。

## 主要特点

- 🌐 **开放词汇识别**：无需预定义类别，可识别用户指定的任意类别（如"basketball court", "grass", "sports field"等）
- 📸 **高精度分割**：基于SAM3的先进分割能力，提供高质量的掩码输出
- 🔍 **多场景适应**：适用于道路、体育场地、草地、屋顶等多种遥感场景
- 💡 **端到端流程**：从图像输入到结果可视化，全流程自动化

## 结果展示

以下展示了QwSAM3在不同场景下的分割效果，所有结果均基于同一输入图像生成：

### 1. 体育场地检测
![Sports Field Detection](https://github.com/YanghuiSong/QwSAM3/blob/main/figures/Sports%20field_detection.jpg?raw=true)
*体育场地分割结果，清晰识别出多个运动场地*

### 2. 篮球场检测
![Basketball Court Detection](https://github.com/YanghuiSong/QwSAM3/blob/main/figures/basketball%20court_detection.jpg?raw=true)
*精准识别篮球场区域，分割边界清晰*

### 3. 草地检测
![Grass Detection](https://github.com/YanghuiSong/QwSAM3/blob/main/figures/grass_detection.jpg?raw=true)
*对草地区域进行精确分割，区分不同草地区域*

### 4. 路径检测
![Path Detection](https://github.com/YanghuiSong/QwSAM3/blob/main/figures/path_detection.jpg?raw=true)
*识别并分割图像中的小径和步道*

### 5. 道路检测
![Road Detection](https://github.com/YanghuiSong/QwSAM3/blob/main/figures/road_detection.jpg?raw=true)
*道路区域分割，准确识别主要道路和次要道路*

### 6. 屋顶检测
![Roof Detection](https://github.com/YanghuiSong/QwSAM3/blob/main/figures/roof_detection.jpg?raw=true)
*对建筑物屋顶进行精确分割*

### 7. 树木检测
![Tree Detection](https://github.com/YanghuiSong/QwSAM3/blob/main/figures/tree_detection.jpg?raw=true)
*树木区域分割，区分不同树种和大小*

### 8. 开放词汇综合结果
![Open Vocabulary Results](https://github.com/YanghuiSong/QwSAM3/blob/main/figures/open_vocab_result.jpg?raw=true)
*同时识别多种类别（体育场地、篮球场、草地、道路等）的综合结果*

### 9. SAM3测试输出
![SAM3 Test Output](https://github.com/YanghuiSong/QwSAM3/blob/main/figures/sam3_test_output.jpg?raw=true)
*SAM3模型的原始分割输出，作为后续处理的基础*

## 技术对比

| 模型 | 开放词汇 | 无需训练 | 高精度 | 多场景适应 | 速度 |
|------|----------|----------|--------|------------|------|
| QwSAM3 | ✅ | ✅ | ✅ | ✅ | ⚡ |
| 传统方法 | ❌ | ❌ | ⚠️ | ⚠️ | ⚡ |
| 专用模型 | ❌ | ❌ | ✅ | ❌ | ⚡ |

*QwSAM3在保持高精度的同时，实现了开放词汇和无需训练的核心优势*

## 使用方法

1. **安装依赖**:
```bash
# 安装SAM3
git clone https://github.com/facebookresearch/segment-anything-3.git
cd segment-anything-3
pip install -e .

# 安装Qwen2.5-VL
pip install transformers accelerate
```

2. **运行示例**:
```python
from qwsam3 import QwSAM3

# 初始化模型
qwsam = QwSAM3(model_path="Qwen/Qwen2.5-VL-7B-Instruct")

# 执行开放词汇分割
results = qwsam.segment(
    image_path="path/to/your/image.jpg",
    prompts=["basketball court", "grass", "sports field"]
)

# 保存结果
results.save("output_result.jpg")
```

## 项目结构

```
QwSAM3/
├── qwsam3.py             # 核心模型实现
├── examples/             # 示例脚本
├── figures/              # 结果图片
├── requirements.txt      # 依赖列表
└── README.md             # 本文件
```

## 依赖

- Python 3.8+
- PyTorch 2.0+
- SAM3 (Segment Anything Model 3)
- Qwen2.5-VL (Qwen2.5-VL-7B-Instruct)
- OpenCV
- Pillow

## 贡献

欢迎提交Issue和Pull Request！请遵循以下步骤：
1. Fork项目
2. 创建新分支 (`git checkout -b feature/your-feature`)
3. 提交更改
4. 提交Pull Request

## 许可

本项目采用MIT许可证。详情请参阅[LICENSE](LICENSE)文件。

---

**QwSAM3: 让遥感图像分析变得简单、灵活、高效！**
