# 深度思考AI大模型训练项目

这个项目提供了一个完整的训练框架，用于训练具有深度思考能力的500M参数规模大语言模型。

## 项目特点

- **完整的训练流程**：包括数据加载、模型定义、训练循环和评估
- **优化的训练策略**：支持分布式训练、梯度累积、混合精度训练等
- **深度思考能力增强**：特别设计的数据集和训练方法，增强模型的推理和思考能力
- **灵活的配置**：丰富的命令行参数，可以根据需要调整训练参数
- **可靠的检查点保存**：支持断点续训和最佳模型保存

## 项目结构

```
.
├── README.md                 # 项目说明文档
├── train_500m_model.py         # 主训练脚本
└── sample_data/              # 示例数据集
    └── deep_thinking_examples.jsonl  # 深度思考示例数据
```

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.20+
- 至少一张支持CUDA的GPU（推荐多卡环境）
- 至少8GB GPU内存（用于500M参数模型）

## 安装依赖

```bash
pip install torch transformers datasets wandb tqdm numpy
```

## 使用方法

### 基本训练

```bash
python train_500m_model.py --output_dir ./output --dataset_path ./sample_data/deep_thinking_examples.jsonl
```

### 使用自定义参数训练

```bash
python train_500m_model.py \
  --output_dir ./output \
  --model_name_or_path EleutherAI/gpt-neo-125M \
  --dataset_path ./your_dataset.jsonl \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --fp16
```

### 分布式训练

```bash
torchrun --nproc_per_node=4 train_500m_model.py \
  --output_dir ./output \
  --dataset_path ./your_dataset.jsonl
```

## 数据集格式

训练脚本支持以下格式的数据：

1. JSONL格式，每行包含一个JSON对象，格式如下：

```json
{"input": "问题或指令", "output": "期望的回答或响应"}
```

2. 也支持以下格式：

```json
{"instruction": "指令", "response": "响应"}
{"prompt": "提示", "completion": "完成"}
{"text": "完整文本"}
```

## 主要特性

### 深度思考能力

本训练框架特别关注模型的深度思考能力，通过以下方式增强：

- 使用包含推理模式的训练数据
- 支持多步推理训练
- 优化模型结构以适应复杂推理任务

### 优化训练

- **混合精度训练**：使用FP16或BF16加速训练
- **梯度累积**：支持大批量训练
- **分布式训练**：支持多GPU和多节点训练
- **检查点保存**：定期保存模型，支持断点续训
- **早停机制**：根据验证集性能自动停止训练

## 许可证

MIT