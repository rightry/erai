#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
深度思考AI大模型训练脚本 (3B参数规模)

此脚本实现了一个完整的大规模语言模型训练流程，包括：
- 数据加载与预处理
- 模型定义与初始化
- 分布式训练设置
- 训练循环与优化
- 评估与检查点保存
- 日志记录与可视化

适用于训练具有深度思考能力的3B参数规模模型。
"""

import os
import time
import math
import json
import logging
import argparse
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoConfig,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset

import wandb
import numpy as np
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingArguments:
    """训练参数配置"""
    # 基本参数
    output_dir: str = "./output"
    model_name_or_path: str = "EleutherAI/gpt-neo-1.3B"  # 基础模型，将扩展到3B
    tokenizer_name: Optional[str] = None
    dataset_path: str = "path/to/dataset"  # 数据集路径
    dataset_format: str = "jsonl"  # 数据集格式
    
    # 训练参数
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16  # 梯度累积步数
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_train_epochs: int = 3
    max_steps: int = -1  # 覆盖epochs
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    
    # 模型参数
    hidden_size: int = 2560  # 扩展到3B参数所需的隐藏层大小
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 10240
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-5
    
    # 序列长度
    max_seq_length: int = 1024
    
    # 优化参数
    fp16: bool = True  # 混合精度训练
    bf16: bool = False  # BF16精度
    
    # 分布式训练
    local_rank: int = -1
    deepspeed_config: Optional[str] = None
    
    # 保存与评估
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 100
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # 其他
    seed: int = 42
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None
    
    # 深度思考能力增强
    use_reasoning_patterns: bool = True  # 启用推理模式训练
    reasoning_patterns_ratio: float = 0.3  # 推理样本在训练中的比例
    multi_step_reasoning: bool = True  # 多步推理
    
    # 日志与监控
    use_wandb: bool = True
    wandb_project: str = "deep-thinking-ai-3b"


class DeepThinkingDataset(Dataset):
    """用于深度思考AI训练的数据集"""
    
    def __init__(self, dataset_path, tokenizer, max_seq_length, dataset_format="jsonl"):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # 加载数据集
        logger.info(f"Loading dataset from {dataset_path}")
        if dataset_format == "jsonl":
            self.data = self._load_jsonl(dataset_path)
        elif dataset_format == "huggingface":
            self.data = load_dataset(dataset_path)
            if isinstance(self.data, dict):
                self.data = self.data["train"]
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")
        
        logger.info(f"Loaded {len(self.data)} examples")
    
    def _load_jsonl(self, path):
        """加载JSONL格式的数据集"""
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 处理不同格式的数据
        if isinstance(item, dict):
            if "input" in item and "output" in item:
                text = f"问题: {item['input']}\n回答: {item['output']}"
            elif "instruction" in item and "response" in item:
                text = f"指令: {item['instruction']}\n回应: {item['response']}"
            elif "prompt" in item and "completion" in item:
                text = f"{item['prompt']}{item['completion']}"
            elif "text" in item:
                text = item["text"]
            else:
                # 尝试使用数据集中的第一个字符串字段
                for key, value in item.items():
                    if isinstance(value, str):
                        text = value
                        break
                else:
                    raise ValueError(f"Unsupported data format: {item}")
        elif isinstance(item, str):
            text = item
        else:
            raise ValueError(f"Unsupported data format: {item}")
        
        # 标记化
        encodings = self.tokenizer(text, truncation=True, max_length=self.max_seq_length)
        
        # 创建输入ID和注意力掩码
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        
        # 确保长度一致
        if len(input_ids) < self.max_seq_length:
            padding_length = self.max_seq_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(input_ids)  # 对于自回归任务，标签与输入相同
        }


class DeepThinkingModelTrainer:
    """深度思考AI模型训练器"""
    
    def __init__(self, args):
        self.args = args
        self.setup_environment()
        
        # 初始化tokenizer
        self.tokenizer = self.init_tokenizer()
        
        # 初始化模型
        self.model = self.init_model()
        
        # 设置分布式训练
        self.setup_distributed()
        
        # 准备数据集
        self.train_dataloader, self.eval_dataloader = self.prepare_data()
        
        # 设置优化器和学习率调度器
        self.optimizer, self.lr_scheduler = self.setup_optimization()
        
        # 设置混合精度训练
        self.scaler = GradScaler() if self.args.fp16 else None
        
        # 训练状态
        self.global_step = 0
        self.best_eval_loss = float("inf")
        
        # 设置日志记录
        self.setup_logging()
    
    def setup_environment(self):
        """设置训练环境"""
        # 设置随机种子
        transformers.set_seed(self.args.seed)
        
        # 创建输出目录
        if self.is_main_process():
            os.makedirs(self.args.output_dir, exist_ok=True)
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置分布式训练环境
        if self.args.local_rank != -1:
            torch.cuda.set_device(self.args.local_rank)
    
    def is_main_process(self):
        """检查是否为主进程"""
        return self.args.local_rank == -1 or dist.get_rank() == 0
    
    def init_tokenizer(self):
        """初始化tokenizer"""
        tokenizer_name = self.args.tokenizer_name or self.args.model_name_or_path
        logger.info(f"Loading tokenizer from {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # 确保tokenizer有pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def init_model(self):
        """初始化模型"""
        logger.info(f"Loading model from {self.args.model_name_or_path}")
        
        # 加载配置
        config = AutoConfig.from_pretrained(self.args.model_name_or_path)
        
        # 修改配置以达到3B参数规模
        config.hidden_size = self.args.hidden_size
        config.num_hidden_layers = self.args.num_hidden_layers
        config.num_attention_heads = self.args.num_attention_heads
        config.intermediate_size = self.args.intermediate_size
        config.hidden_dropout_prob = self.args.hidden_dropout_prob
        config.attention_probs_dropout_prob = self.args.attention_dropout_prob
        config.max_position_embeddings = self.args.max_position_embeddings
        
        # 加载模型
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name_or_path,
                config=config,
                torch_dtype=torch.bfloat16 if self.args.bf16 else torch.float16 if self.args.fp16 else torch.float32
            )
        except:
            logger.warning("Failed to load pretrained model, initializing from scratch")
            model = AutoModelForCausalLM.from_config(config)
        
        # 移动模型到设备
        model = model.to(self.device)
        
        return model
    
    def setup_distributed(self):
        """设置分布式训练"""
        if self.args.local_rank != -1:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            self.model = DDP(
                self.model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True
            )
    
    def prepare_data(self):
        """准备数据集和数据加载器"""
        # 创建训练数据集
        train_dataset = DeepThinkingDataset(
            dataset_path=self.args.dataset_path,
            tokenizer=self.tokenizer,
            max_seq_length=self.args.max_seq_length,
            dataset_format=self.args.dataset_format
        )
        
        # 创建评估数据集 (使用训练数据的一小部分)
        eval_size = min(1000, int(len(train_dataset) * 0.1))
        train_size = len(train_dataset) - eval_size
        
        # 分割数据集
        train_dataset, eval_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, eval_size],
            generator=torch.Generator().manual_seed(self.args.seed)
        )
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Eval dataset size: {len(eval_dataset)}")
        
        # 创建数据收集器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # 不使用掩码语言建模，而是使用因果语言建模
        )
        
        # 创建数据加载器
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if self.args.local_rank != -1 else None
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset) if self.args.local_rank != -1 else None
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            pin_memory=True,
            num_workers=4
        )
        
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            sampler=eval_sampler,
            collate_fn=data_collator,
            pin_memory=True,
            num_workers=4
        )
        
        return train_dataloader, eval_dataloader
    
    def setup_optimization(self):
        """设置优化器和学习率调度器"""
        # 准备优化器
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        
        # 计算总训练步数
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
        else:
            t_total = len(self.train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        
        # 准备学习率调度器
        warmup_steps = int(t_total * self.args.warmup_ratio)
        
        if self.args.lr_scheduler_type == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        else:
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        
        return optimizer, lr_scheduler
    
    def setup_logging(self):
        """设置日志记录"""
        if self.is_main_process() and self.args.use_wandb:
            wandb.init(project=self.args.wandb_project, config=vars(self.args))
    
    def save_checkpoint(self, save_path, is_best=False):
        """保存检查点"""
        if not self.is_main_process():
            return
        
        # 准备保存的状态
        state_dict = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
        
        # 保存模型
        save_dict = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "args": vars(self.args),
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
        }
        
        if self.scaler is not None:
            save_dict["scaler"] = self.scaler.state_dict()
        
        torch.save(save_dict, save_path)
        logger.info(f"Saved checkpoint to {save_path}")
        
        # 如果是最佳模型，复制一份
        if is_best:
            best_path = os.path.join(os.path.dirname(save_path), "best_model.pt")
            torch.save(save_dict, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, load_path):
        """加载检查点"""
        if not os.path.exists(load_path):
            logger.warning(f"Checkpoint {load_path} does not exist, skipping loading")
            return
        
        logger.info(f"Loading checkpoint from {load_path}")
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # 加载模型
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])
        
        # 加载优化器和学习率调度器
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        
        # 加载其他状态
        self.global_step = checkpoint["global_step"]
        self.best_eval_loss = checkpoint["best_eval_loss"]
        
        # 加载混合精度训练状态
        if self.scaler is not None and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        
        logger.info(f"Loaded checkpoint from {load_path} (global_step: {self.global_step})")
    
    def train(self):
        """训练模型"""
        logger.info("***** Starting training *****")
        logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.args.max_steps}")
        
        # 训练循环
        self.model.train()
        completed_steps = 0
        train_loss = 0.0
        
        progress_bar = tqdm(range(self.args.max_steps), disable=not self.is_main_process())
        for epoch in range(int(self.args.num_train_epochs)):
            if self.args.local_rank != -1:
                self.train_dataloader.sampler.set_epoch(epoch)
            
            epoch_iterator = self.train_dataloader
            
            for step, batch in enumerate(epoch_iterator):
                # 将批次移动到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                with autocast(enabled=self.args.fp16 or self.args.bf16):
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.args.gradient_accumulation_steps
                
                # 反向传播
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                train_loss += loss.item()
                
                # 梯度累积
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or step == len(epoch_iterator) - 1:
                    # 梯度裁剪
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    
                    # 优化器步骤
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.lr_scheduler.step()
                    self.model.zero_grad()
                    
                    completed_steps += 1
                    self.global_step += 1
                    progress_bar.update(1)
                    
                    # 日志记录
                    if self.global_step % self.args.logging_steps == 0:
                        avg_loss = train_loss / self.args.logging_steps / self.args.gradient_accumulation_steps
                        lr = self.lr_scheduler.get_last_lr()[0]
                        
                        logger.info(f"Step {self.global_step}: loss = {avg_loss:.4f}, lr = {lr:.8f}")
                        
                        if self.is_main_process() and self.args.use_wandb:
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/learning_rate": lr,
                                "train/epoch": epoch + step / len(epoch_iterator),
                                "train/step": self.global_step,
                            })
                        
                        train_loss = 0.0
                    
                    # 评估
                    if self.global_step % self.args.eval_steps == 0:
                        eval_results = self.evaluate()
                        self.model.train()
                        
                        # 保存最佳模型
                        if eval_results["eval_loss"] < self.best_eval_loss:
                            self.best_eval_loss = eval_results["eval_loss"]
                            self.save_checkpoint(
                                os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}.pt"),
                                is_best=True
                            )
                    
                    # 保存检查点
                    if self.global_step % self.args.save_steps == 0:
                        self.save_checkpoint(os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}.pt"))
                        
                        # 删除旧检查点
                        if self.args.save_total_limit > 0:
                            self.cleanup_checkpoints()
                
                if self.args.max_steps > 0 and completed_steps >= self.args.max_steps:
                    break
            
            if self.args.max_steps > 0 and completed_steps >= self.args.max_steps:
                break
        
        # 保存最终模型
        if self.is_main_process():
            self.save_checkpoint(os.path.join(self.args.output_dir, "final_model.pt"))
            
            # 保存为HuggingFace格式
            if hasattr(self.model, "module"):
                self.model.module.save_pretrained(os.path.join(self.args.output_dir, "hf_model"))
            else:
                self.model.save_pretrained(os.path.join(self.args.output_dir, "hf_model"))
            
            self.tokenizer.save_pretrained(os.path.join(self.args.output_dir, "hf_model"))
        
        return self.global_step, train_loss
    
    def evaluate(self):
        """评估模型"""
        logger.info("***** Running evaluation *****")
        
        self.model.eval()
        eval_loss = 0.0
        eval_steps = 0
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating", disable=not self.is_main_process()):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.no_grad():
                with autocast(enabled=self.args.fp16 or self.args.bf16):
                    outputs = self.model(**batch)
            
            loss = outputs.loss
            eval_loss += loss.item()
            eval_steps += 1
        
        # 计算平均损失
        eval_loss = eval_loss / eval_steps
        perplexity = math.exp(eval_loss)
        
        results = {
            "eval_loss": eval_loss,
            "eval_perplexity": perplexity,
        }
        
        logger.info(f"Evaluation results: {results}")
        
        if self.is_main_process() and self.args.use_wandb:
            wandb.log({
                "eval/loss": eval_loss,
                "eval/perplexity": perplexity,
                "eval/step": self.global_step,
            })
        
        return results
    
    def cleanup_checkpoints(self):
        """清理旧检查点，只保留最新的几个"""
        if not self.is_main_process():
            return
        
        checkpoints = [f for f in os.listdir(self.args.output_dir) if f.startswith("checkpoint-") and f.endswith(".pt")]
        if len(checkpoints) <= self.args.save_total_limit:
            return
        
        # 按步骤排序
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))
        
        # 删除旧检查点
        checkpoints_to_delete = checkpoints[:-self.args.save_total_limit]
        for checkpoint in checkpoints_to_delete:
            os.remove(os.path.join(self.args.output_dir, checkpoint))
            logger.info(f"Deleted old checkpoint: {checkpoint}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练深度思考AI大模型")
    
    # 添加TrainingArguments中的所有参数
    for field_name, field in TrainingArguments.__dataclass_fields__.items():
        field_type = field.type
        if field_type == bool:
            parser.add_argument(f"--{field_name}", action="store_true", help=f"Enable {field_name}")
        elif field_type == Optional[str]:
            parser.add_argument(f"--{field_name}", type=str, default=None, help=f"Optional string for {field_name}")
        elif field_type == str:
            default_value = getattr(TrainingArguments, field_name)
            parser.add_argument(f"--{field_name}", type=str, default=default_value, help=f"String for {field_name}")
        elif field_type == int:
            default_value = getattr(TrainingArguments, field_name)
            parser.add_argument(f"--{field_name}", type=int, default=default_value, help=f"Integer for {field_name}")
        elif field_type == float:
            default_value = getattr(TrainingArguments, field_name)
            parser.add_argument(f"--{field_name}", type=float, default=default_value, help=f"Float for {field_name}")
    
    args = parser.parse_args()
    return args


def create_sample_dataset(output_path, num_samples=100):
    """创建示例数据集用于测试"""
    if os.path.exists(output_path):
        logger.info(f"Sample dataset already exists at {output_path}")
        return
    
    # 检查示例数据文件是否存在
    sample_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "deep_thinking_examples.jsonl")
    if not os.path.exists(sample_data_path):
        logger.error(f"示例数据文件不存在: {sample_data_path}")
        raise FileNotFoundError(f"示例数据文件不存在: {sample_data_path}")
    
    logger.info(f"从 {sample_data_path} 复制示例数据到 {output_path}")
    
    # 从示例文件加载数据
    samples = []
    with open(sample_data_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    
    # 如果需要，通过随机复制来增加样本数量
    if len(samples) < num_samples:
        logger.info(f"通过随机复制增加样本数量从 {len(samples)} 到 {num_samples}")
        while len(samples) < num_samples:
            idx = np.random.randint(0, len(samples))
            samples.append(samples[idx].copy())
    
    # 截取所需数量的样本
    samples = samples[:num_samples]
    
    # 保存为JSONL格式
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    logger.info(f"创建了包含 {len(samples)} 个示例的数据集，保存到 {output_path}")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 将命令行参数转换为TrainingArguments对象
    training_args = TrainingArguments()
    for key, value in vars(args).items():
        if hasattr(training_args, key):
            setattr(training_args, key, value)
    
    # 如果数据集路径是默认值，创建示例数据集
    if training_args.dataset_path == "path/to/dataset":
        sample_dataset_path = os.path.join(os.path.dirname(training_args.output_dir), "sample_dataset.jsonl")
        create_sample_dataset(sample_dataset_path)
        training_args.dataset_path = sample_dataset_path
    
    # 创建并启动训练器
    trainer = DeepThinkingModelTrainer(training_args)
    
    # 加载检查点（如果存在）
    checkpoint_path = os.path.join(training_args.output_dir, "latest_checkpoint.pt")
    if os.path.exists(checkpoint_path):
        trainer.load_checkpoint(checkpoint_path)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()