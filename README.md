# QwSAM3: Interactive Segmentation System Based on SAM3 and Qwen

## Overview

QwSAM3 is an interactive intelligent segmentation system that combines the power of SAM3 (Segment Anything Model 3) and Qwen (Alibaba's large language model) to provide precise and interactive image/video segmentation capabilities. The system integrates vision foundation models with language understanding capabilities to achieve more accurate and interactive segmentation results compared to traditional automatic segmentation systems.

### Key Features

- **Intelligent Instruction Parsing**: Utilizes Qwen to parse user's natural language inputs for intuitive control over segmentation tasks.
- **Coordinated Segmentation**: Coordinates object detectors and SAM3 adapters for enhanced segmentation accuracy.
- **Session Management**: Maintains interactive context states for multi-turn interactions.
- **Result Visualization**: Provides enhanced visualization of segmentation results.
- **Evaluation Tools**: Built-in evaluation scripts for various datasets (COCO, YTVIS, etc.).

## Architecture

The system follows a layered agent architecture:

- **Interaction Layer**: [demo_intelligent_segmentation.py](./demo_intelligent_segmentation.py) serves as the entry point for handling user input.
- **Coordination Layer (Core)**: [intelligent_segmentation_coordinator.py](./core/intelligent_segmentation_coordinator.py) acts as the core brain, invoking [qwen_agent.py](./core/qwen_agent.py) for intent parsing and scheduling [object_detector.py](./core/object_detector.py) and [sam3_adapter.py](./core/sam3_adapter.py).
- **Model Layer (SAM3)**: Encapsulates SAM3's inference, tracking, and post-processing logic ([sam3/agent](./sam3/agent), [sam3/model](./sam3/model)).
- **Tool Layer**: Includes evaluation ([sam3/eval](./sam3/eval)), training ([sam3/train](./sam3/train)), and performance optimization libraries ([sam3/perflib](./sam3/perflib)).

### Technical Components

- **Multi-modal Fusion**: Uses Qwen (LLM) to process text instructions and SAM3 (VLM/CV) for visual feature processing.
- **Modular Adapters**: Through [sam3_adapter.py](./core/sam3_adapter.py) series files to abstract underlying model details for easy replacement or upgrades.
- **Acceleration Optimization**: Incorporates Triton kernels ([sam3/perflib/triton](./sam3/perflib/triton)) for NMS and connected component analysis acceleration.

## Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- Torchvision >= 0.15.0
- Transformers >= 4.36.0
- OpenCV >= 4.8.0
- CUDA-compatible GPU (recommended for optimal performance)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YanghuiSong/QwSAM3.git
   cd QwSAM3
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the intelligent segmentation demo:
   ```bash
   python demo_intelligent_segmentation.py --image_path <path_to_image>
   ```

2. For quick testing:
   ```bash
   python test_quick.py
   ```

## Core Modules

- **[core/](./core/)**: Business logic core
  - [intelligent_segmentation_coordinator.py](./core/intelligent_segmentation_coordinator.py): Core coordinator that connects LLM and CV models
  - [qwen_agent.py](./core/qwen_agent.py) / [qwen_enhanced_agent.py](./core/qwen_enhanced_agent.py): Large language model agents for instruction understanding
  - [sam3_adapter*.py](./core/sam3_adapter.py): SAM3 model encapsulation and adaptation
  - [object_detector.py](./core/object_detector.py): Object detection module for providing initial prompts
  - [session_manager.py](./core/session_manager.py): Manages multi-turn interaction session states

- **[sam3/](./sam3/)**: Lower-level algorithm library
  - [agent/](./sam3/agent/): Native SAM3 agent logic, including clients and inference flows
  - [model/](./sam3/model/): Model definitions including Encoder, Decoder, ViTDet, Tracker structures
  - [eval/](./sam3/eval/): Complete evaluation toolkit supporting HOTA, TETA, COCO, YTVIS metrics
  - [train/](./sam3/train/): Training pipeline with DataLoader, Loss functions, and Optimizer configs
  - [perflib/](./sam3/perflib/): Performance optimization library with Triton-accelerated operators

- **[visualization/](./visualization/)**: Result visualization modules with enhanced visualizers

## Acknowledgments

We would like to thank the following projects for their contributions to our work:

- [SAM3](https://github.com/facebookresearch/sam3/tree/main) for the foundational Segment Anything Model 3 implementation
- [SegEarth-OV-3](https://github.com/earth-insights/SegEarth-OV-3) for valuable insights in earth observation segmentation
- [InstructSAM](https://github.com/yunusserhat/instructsam) for inspiration on instruction-driven segmentation approaches

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.