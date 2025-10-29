import logging
import os
import random
import json
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    IntervalStrategy,
)
from datasets import Dataset
from transformers.tokenization_utils_base import BatchEncoding
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NERTrainingConfig:
    """NER训练配置"""
    model_path: str = field(default="./chinese-roberta-wwm-ext")
    output_dir: str = field(default="./ner_multilabel_model")
    max_length: int = field(default=128)
    num_train_epochs: int = field(default=5)
    batch_size: int = field(default=32)
    learning_rate: float = field(default=3e-5)
    eval_split: float = field(default=0.1)
    seed: int = field(default=42)
    warmup_ratio: float = field(default=0.1)
    weight_decay: float = field(default=0.01)
    gradient_accumulation_steps: int = field(default=1)
    fp16: bool = field(default=True)
    early_stopping_patience: int = field(default=3)
    label_smoothing: float = field(default=0.1)

class MultiLabelNERForHF(PreTrainedModel):
    """改进的多标签NER模型"""
    
    def __init__(
        self,
        config: AutoConfig,
        base_model: nn.Module,
        num_categories: int,
        dropout_prob: float = 0.1
    ):
        super().__init__(config)
        self._base_model = base_model
        
        # 获取隐藏层维度
        self.hidden_size = getattr(config, "hidden_size", None)
        if self.hidden_size is None:
            self.hidden_size = getattr(base_model.config, "hidden_size")
            if self.hidden_size is None:
                raise ValueError("无法获取hidden_size")
                
        # 添加dropout和层标准化
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # 分类器
        self.classifiers = nn.ModuleDict({
            "label": nn.Linear(self.hidden_size, config.num_labels),
            "category": nn.Linear(self.hidden_size, num_categories)
        })
        
        # 损失函数权重
        self.loss_weights = getattr(config, "loss_weights", {
            "label": 1.0,
            "category": 1.0
        })
        
        # 标签平滑
        self.label_smoothing = getattr(config, "label_smoothing", 0.0)
        self.loss_fct = nn.BCEWithLogitsLoss(label_smoothing=self.label_smoothing)
        
        # 初始化
        self.init_weights()
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        categories: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        # 使用自动混合精度
        with torch.cuda.amp.autocast(enabled=True):
            # 前向传播基础模型
            outputs = self._base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                **kwargs
            )
            
            # 获取隐藏状态并应用dropout和归一化
            hidden = outputs.last_hidden_state
            hidden = self.dropout(hidden)
            hidden = self.layer_norm(hidden)
            
            # 计算logits
            logits = {
                name: classifier(hidden)
                for name, classifier in self.classifiers.items()
            }
            
            loss = None
            if labels is not None and categories is not None:
                # 计算损失
                device = hidden.device
                dtype = hidden.dtype
                
                labels = labels.to(device=device, dtype=dtype)
                categories = categories.to(device=device, dtype=dtype)
                
                losses = {}
                for name, target in [("label", labels), ("category", categories)]:
                    losses[name] = self.loss_fct(
                        logits[name].view(-1, logits[name].size(-1)),
                        target.view(-1, target.size(-1))
                    )
                
                # 应用权重
                loss = sum(
                    self.loss_weights[name] * loss_val
                    for name, loss_val in losses.items()
                )
                
            return {
                "loss": loss,
                "label_logits": logits["label"],
                "category_logits": logits["category"],
                "hidden_states": hidden
            }

@dataclass
class OptimizedDataCollator:
    """优化的数据整理器"""
    
    tokenizer: AutoTokenizer
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __post_init__(self):
        self.pad_token_id = (
            self.tokenizer.pad_token_id or
            self.tokenizer.eos_token_id
        )
    
    def __call__(
        self,
        features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        try:
            batch_size = len(features)
            if not batch_size:
                raise ValueError("Empty batch")
                
            # 提取标签
            labels = [f.pop("labels") for f in features]
            categories = [f.pop("categories") for f in features]
            
            # 验证
            if not all(isinstance(l, list) for l in labels):
                raise ValueError("Invalid labels format")
            if not all(isinstance(c, list) for c in categories):
                raise ValueError("Invalid categories format")
                
            # Padding
            batch = self.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt"
            )
            
            # 准备标签tensor
            seq_len = batch["input_ids"].shape[1]
            num_labels = len(labels[0][0])
            num_categories = len(categories[0][0])
            
            padded_labels = torch.zeros(
                (batch_size, seq_len, num_labels),
                dtype=torch.float32
            )
            padded_cats = torch.zeros(
                (batch_size, seq_len, num_categories),
                dtype=torch.float32
            )
            
            # 填充标签
            for i, (lbl, cat) in enumerate(zip(labels, categories)):
                cur_len = min(len(lbl), seq_len)
                if cur_len > 0:
                    padded_labels[i, :cur_len] = torch.tensor(
                        lbl[:cur_len],
                        dtype=torch.float32
                    )
                    padded_cats[i, :cur_len] = torch.tensor(
                        cat[:cur_len],
                        dtype=torch.float32
                    )
            
            batch["labels"] = padded_labels
            batch["categories"] = padded_cats
            
            return batch
            
        except Exception as e:
            logger.error(f"数据整理错误: {str(e)}")
            raise

class NERTrainer:
    """优化的NER训练器"""
    
    def __init__(self, config: NERTrainingConfig):
        self.config = config
        self.writer = SummaryWriter(
            os.path.join(config.output_dir, "tensorboard")
        )
        
        # 初始化
        self._initialize_resources()
        
    def _initialize_resources(self):
        """初始化资源"""
        try:
            # 验证路径
            if not os.path.exists(self.config.model_path):
                raise ValueError(f"模型路径不存在: {self.config.model_path}")
                
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                use_fast=True,
                add_prefix_space=True
            )
            
            # 初始化映射
            self.label2id: Dict[str, int] = {}
            self.id2label: Dict[int, str] = {}
            self.category2id: Dict[str, int] = {}
            self.id2category: Dict[int, str] = {}
            
        except Exception as e:
            raise RuntimeError(f"资源初始化失败: {str(e)}")
    
    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        # 分别处理label和category的预测
        label_preds = predictions[0]
        category_preds = predictions[1]
        
        # 将logits转换为预测标签
        label_preds = (torch.sigmoid(torch.tensor(label_preds)) > 0.5).numpy()
        category_preds = (torch.sigmoid(torch.tensor(category_preds)) > 0.5).numpy()
        
        # 计算指标
        metrics = {}
        for name, preds, target in [
            ("label", label_preds, labels[0]),
            ("category", category_preds, labels[1])
        ]:
            precision, recall, f1, _ = precision_recall_fscore_support(
                target.flatten(),
                preds.flatten(),
                average='binary'
            )
            accuracy = accuracy_score(target.flatten(), preds.flatten())
            
            metrics.update({
                f"{name}_precision": precision,
                f"{name}_recall": recall,
                f"{name}_f1": f1,
                f"{name}_accuracy": accuracy
            })
            
        return metrics
    
    def train(
        self,
        train_data: List[Dict[str, Any]]
    ):
        """训练入口"""
        try:
            # 设置随机种子
            self._set_seed()
            
            # 数据预处理
            train_dataset, eval_dataset = self._prepare_datasets(train_data)
            
            # 初始化模型
            model = self._initialize_model()
            
            # 训练参数
            training_args = self._get_training_args()
            
            # 配置Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=OptimizedDataCollator(
                    tokenizer=self.tokenizer,
                    max_length=self.config.max_length
                ),
                compute_metrics=self.compute_metrics,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=self.config.early_stopping_patience
                    )
                ]
            )
            
            # 训练
            trainer.train()
            
            # 保存
            self._save_model(model)
            
            # 清理
            self.writer.close()
            
        except Exception as e:
            logger.error(f"训练失败: {str(e)}")
            raise
            
    def _set_seed(self):
        """设置随机种子"""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
            
    def _prepare_datasets(
        self,
        data: List[Dict[str, Any]]
    ) -> Tuple[Dataset, Optional[Dataset]]:
        """准备数据集"""
        processed = self.convert_to_multilabel_bio_with_category(data)
        
        # 划分数据集
        random.shuffle(processed)
        split_idx = math.ceil(len(processed) * (1 - self.config.eval_split))
        
        train_data = processed[:split_idx]
        eval_data = processed[split_idx:] if split_idx < len(processed) else None
        
        # 转换为Dataset对象
        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data) if eval_data else None
        
        # Tokenization
        train_dataset = self._tokenize_dataset(train_dataset)
        if eval_dataset:
            eval_dataset = self._tokenize_dataset(eval_dataset)
            
        return train_dataset, eval_dataset
        
    def _get_training_args(self) -> TrainingArguments:
        """获取训练参数"""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_train_epochs,
            save_strategy=IntervalStrategy.EPOCH,
            evaluation_strategy=IntervalStrategy.EPOCH,
            logging_steps=50,
            fp16=self.config.fp16 and torch.cuda.is_available(),
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            seed=self.config.seed
        )
