#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
深度思考AI大模型训练脚本 (500M参数规模)

此脚本实现了一个完整的大规模语言模型训练流程，包括：
- 数据加载与预处理
- 模型定义与初始化
- 分布式训练设置
- 训练循环与优化
- 评估与检查点保存
- 日志记录与可视化

适用于训练具有深度思考能力的500M参数规模模型。
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
    model_name_or_path: str = "EleutherAI/gpt-neo-125M"  # 基础模型，将扩展到500M
    tokenizer_name: Optional[str] = None
    dataset_path: str = "path/to/dataset"  # 数据集路径
    dataset_format: str = "jsonl"  # 数据集格式
    
    # 训练参数
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # 梯度累积步数
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_train_epochs: int = 3
    max_steps: int = -1  # 覆盖epochs
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    
    # 模型参数 - 调整为500M参数规模
    hidden_size: int = 1024  # 调整为500M参数所需的隐藏层大小
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    intermediate_size: int = 4096
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
    wandb_project: str = "deep-thinking-ai-500m"


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
        
        # 修改配置以达到1B参数规模
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
    
    def train(self):
        """训练模型"""
        logger.info("***** 开始训练 *****")
        logger.info(f"  训练设备 = {self.device}")
        logger.info(f"  每设备训练批次大小 = {self.args.per_device_train_batch_size}")
        logger.info(f"  梯度累积步数 = {self.args.gradient_accumulation_steps}")
        logger.info(f"  总批次大小 = {self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps}")
        
        # 设置模型为训练模式
        self.model.train()
        
        # 训练循环
        total_steps = 0
        train_loss = 0.0
        train_iterator = tqdm(range(int(self.args.num_train_epochs)), desc="Epoch", disable=not self.is_main_process())
        
        for _ in train_iterator:
            epoch_iterator = tqdm(self.train_dataloader, desc="Iteration", disable=not self.is_main_process())
            
            for step, batch in enumerate(epoch_iterator):
                # 将批次移动到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                with autocast() if self.args.fp16 else nullcontext():
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    loss = loss / self.args.gradient_accumulation_steps
                
                # 反向传播
                if self.args.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                train_loss += loss.item()
                
                # 梯度累积
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or step == len(epoch_iterator) - 1:
                    # 梯度裁剪
                    if self.args.fp16:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    
                    # 优化器步骤
                    if self.args.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.lr_scheduler.step()
                    self.model.zero_grad()
                    
                    self.global_step += 1
                    total_steps += 1
                    
                    # 日志记录
                    if self.global_step % self.args.logging_steps == 0 and self.is_main_process():
                        avg_loss = train_loss / total_steps
                        logger.info(f"Step {self.global_step}: loss = {avg_loss}")
                        
                        if self.args.use_wandb:
                            wandb.log({"train/loss": avg_loss, "train/lr": self.lr_scheduler.get_last_lr()[0]})
                    
                    # 评估
                    if self.global_step % self.args.eval_steps == 0:
                        eval_results = self.evaluate()
                        
                        # 保存最佳模型
                        if eval_results["eval_loss"] < self.best_eval_loss:
                            self.best_eval_loss = eval_results["eval_loss"]
                            self.save_model("best_model")
                        
                        # 恢复训练模式
                        self.model.train()
                    
                    # 保存检查点
                    if self.global_step % self.args.save_steps == 0:
                        self.save_model(f"checkpoint-{self.global_step}")
                
                # 检查是否达到最大步数
                if 0 < self.args.max_steps <= self.global_step:
                    epoch_iterator.close()
                    break
            
            # 检查是否达到最大步数
            if 0 < self.args.max_steps <= self.global_step:
                train_iterator.close()
                break
        
        # 保存最终模型
        self.save_model("final_model")
        
        return {"train_loss": train_loss / total_steps}
    
    def evaluate(self):
        """评估模型"""
        logger.info("***** 开始评估 *****")
        
        # 设置模型为评估模式
        self.model.eval()
        
        eval_loss = 0.0
        eval_steps = 0
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating", disable=not self.is_main_process()):
            # 将批次移动到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs.loss
            
            eval_loss += loss.item()
            eval_steps += 1
        
        # 计算平均损失
        eval_loss = eval_loss / eval_steps
        
        # 记录评估结果
        results = {"eval_loss": eval_loss}
        
        logger.info(f"***** 评估结果 *****")
        logger.info(f"  评估损失 = {eval_loss}")
        
        if self.is_main_process() and self.args.use_wandb:
            wandb.log({"eval/loss": eval_loss})
        
        return results
    
    def save_model(self, output_dir_name):
        """保存模型"""
        if not self.is_main_process():
            return
        
        output_dir = os.path.join(self.args.output_dir, output_dir_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取要保存的模型
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        
        # 保存模型和配置
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存训练参数
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        
        logger.info(f"模型保存到 {output_dir}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="深度思考AI大模型训练脚本 (1B参数规模)")
    
    # 将TrainingArguments中的所有参数添加到解析器
    for field_name, field in TrainingArguments.__dataclass_fields__.items():
        field_type = field.type
        default = field.default
        
        # 处理不同类型的参数
        if field_type == bool or field_type == "bool":
            parser.add_argument(f"--{field_name}", action="store_true", default=default)
        elif field_type == Optional[str] or field_type == "Optional[str]":
            parser.add_argument(f"--{field_name}", type=str, default=default)
        elif field_type == str or field_type == "str":
            parser.add_argument(f"--{field_name}", type=str, default=default)
        elif field_type == int or field_type == "int":
            parser.add_argument(f"--{field_name}", type=int, default=default)
        elif field_type == float or field_type == "float":
            parser.add_argument(f"--{field_name}", type=float, default=default)
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建训练器
    trainer = DeepThinkingModelTrainer(args)
    
    # 训练模型
    trainer.train()


if __name__ == "__main__":
    main()