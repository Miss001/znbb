# -*- coding: utf-8 -*-
# ---------------------------
# 导入依赖
# ---------------------------
from transformers import (
    AutoTokenizer,       # 用于加载预训练分词器
    AutoModel,           # 用于加载预训练模型
    AutoConfig,          # 用于加载/修改模型配置
    PreTrainedModel,     # 所有HF模型的基类
    Trainer,             # 高级训练接口
    TrainingArguments,   # 训练参数配置
)
from datasets import Dataset  # 用于创建和处理数据集
from typing import List, Dict, Any, Optional
import json
import math
import random
import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers.tokenization_utils_base import BatchEncoding  # 返回类型


# ---------------------------
# Multi-label NER 模型定义
# ---------------------------
class MultiLabelNERForHF(PreTrainedModel):
    """
    支持多标签NER的token分类模型封装
    输入:
      - input_ids, attention_mask
      - labels: shape (batch, seq_len, num_labels) float/binary
      - categories: shape (batch, seq_len, num_categories) float/binary
    输出:
      - dict 包含 loss, label_logits, category_logits
    """
    def __init__(self, config: AutoConfig, base_model: nn.Module, num_categories: int):
        super().__init__(config)
        self._base_model = base_model  # 预训练基础模型
        # 获取隐藏层维度
        hidden_size = getattr(config, "hidden_size", None) or getattr(base_model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("无法从config或base_model.config中获取hidden_size")
        # 分类器: 标签预测
        self.label_classifier = nn.Linear(hidden_size, config.num_labels)
        # 分类器: 类别预测
        self.category_classifier = nn.Linear(hidden_size, num_categories)
        # 多标签二分类损失
        self.loss_fct = nn.BCEWithLogitsLoss()
        # HF模型初始化
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        categories: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        # 移除Trainer可能传入的不支持参数
        kwargs.pop("num_items_in_batch", None)
        # 前向传播基础模型
        outputs = self._base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # 获取最后隐藏层
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]  # fallback
        # 线性层生成logits
        label_logits = self.label_classifier(hidden)       # (batch, seq_len, num_labels)
        category_logits = self.category_classifier(hidden) # (batch, seq_len, num_categories)

        loss = None
        if labels is not None and categories is not None:
            # 将标签转到同一device和dtype
            device = label_logits.device
            labels = labels.to(device=device, dtype=torch.float)
            categories = categories.to(device=device, dtype=torch.float)
            # reshape为 (batch*seq_len, num_labels) 计算BCE
            loss_label = self.loss_fct(label_logits.view(-1, label_logits.size(-1)),
                                       labels.view(-1, labels.size(-1)))
            loss_category = self.loss_fct(category_logits.view(-1, category_logits.size(-1)),
                                         categories.view(-1, categories.size(-1)))
            loss = loss_label + loss_category

        return {"loss": loss, "label_logits": label_logits, "category_logits": category_logits}


# ---------------------------
# 数据批处理器: pad & convert labels
# ---------------------------
@dataclass
class DataCollatorForMultiLabelTokenClassification:
    """
    用于对多标签NER数据进行padding，并将labels/categories转换为torch.FloatTensor
    """
    tokenizer: AutoTokenizer
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 提取labels和categories
        labels = [f.pop("labels") for f in features]         # shape: [seq_len, num_labels]
        categories = [f.pop("categories") for f in features] # shape: [seq_len, num_categories]

        # 使用tokenizer.pad对input_ids, attention_mask等进行padding
        batch = self.tokenizer.pad(
            features,
            padding="max_length",
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )

        # 获取batch大小和序列长度
        batch_size = batch["input_ids"].shape[0]
        seq_len = batch["input_ids"].shape[1]

        # 获取label/cat数量
        num_labels = len(labels[0][0]) if len(labels) > 0 and len(labels[0]) > 0 else 0
        num_categories = len(categories[0][0]) if len(categories) > 0 and len(categories[0]) > 0 else 0

        # 初始化全零tensor
        padded_labels = torch.zeros((batch_size, seq_len, num_labels), dtype=torch.float)
        padded_cats = torch.zeros((batch_size, seq_len, num_categories), dtype=torch.float)

        # 将原始标签复制到padded tensor中
        for i in range(batch_size):
            lbl = labels[i]
            cat = categories[i]
            cur_len = min(len(lbl), seq_len)
            if cur_len > 0:
                padded_labels[i, :cur_len, :] = torch.tensor(lbl[:cur_len], dtype=torch.float)
            if len(cat) > 0:
                padded_cats[i, :cur_len, :] = torch.tensor(cat[:cur_len], dtype=torch.float)

        batch["labels"] = padded_labels
        batch["categories"] = padded_cats
        return batch


# ---------------------------
# NER训练器封装
# ---------------------------
class NERTrainer:
    def __init__(self, model_path: str = "./chinese-roberta-wwm-ext", output_dir: str = "./ner_multilabel_model", max_length: int = 128):
        self.model_path = model_path
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        # 标签/类别映射
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.category2id: Dict[str, int] = {}
        self.id2category: Dict[int, str] = {}
        self.max_length = max_length

    def convert_to_multilabel_bio_with_category(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将输入的文本和标签转换为多标签BIO矩阵，同时附加类别信息
        输入: [{"text": str, "labels": [[start, end, label, category], ...]}]
        输出: [{"tokens": list(str), "ner_label_tags": [[0/1]], "ner_category_tags": [[0/1]]}]
        """
        # 收集所有标签/类别
        all_labels = set()
        all_categories = set()
        for ex in data:
            for start, end, label, category in ex.get("labels", []):
                all_labels.add(label)
                all_categories.add(category if category is not None else "UNK")

        # 构建映射
        self.label2id = {label: i for i, label in enumerate(sorted(all_labels))}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.category2id = {cat: i for i, cat in enumerate(sorted(all_categories))}
        self.id2category = {i: cat for cat, i in self.category2id.items()}

        result = []
        for ex in data:
            text = ex["text"]
            tokens = list(text)  # 中文按字符级分词
            L = len(tokens)
            num_labels = len(self.label2id)
            num_cats = len(self.category2id)
            # 初始化标签矩阵
            label_matrix = [[0] * num_labels for _ in range(L)]
            category_matrix = [[0] * num_cats for _ in range(L)]
            # 填充矩阵
            for start, end, label, category in ex.get("labels", []):
                if start >= L or end > L or start < 0 or end <= start:
                    continue
                label_id = self.label2id.get(label)
                cat = category if category is not None else "UNK"
                cat_id = self.category2id.get(cat)
                if label_id is None or cat_id is None:
                    continue
                for i in range(start, end):
                    label_matrix[i][label_id] = 1
                    category_matrix[i][cat_id] = 1
            result.append({
                "tokens": tokens,
                "ner_label_tags": label_matrix,
                "ner_category_tags": category_matrix
            })
        return result

    def tokenize_and_align_labels(self, examples: Dict[str, List[Any]]) -> BatchEncoding:
        """
        将字符级tokens转换为模型输入ID，并对齐labels和categories
        """
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_attention_mask=True
        )
        batch_label_ids = []
        batch_category_ids = []

        # 对齐子词级别标签
        for i in range(len(examples["tokens"])):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_matrix = examples["ner_label_tags"][i]
            category_matrix = examples["ner_category_tags"][i]

            seq_label_ids = []
            seq_cat_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    seq_label_ids.append([0] * len(self.label2id))
                    seq_cat_ids.append([0] * len(self.category2id))
                else:
                    seq_label_ids.append(label_matrix[word_idx])
                    seq_cat_ids.append(category_matrix[word_idx])
            batch_label_ids.append(seq_label_ids)
            batch_category_ids.append(seq_cat_ids)

        tokenized_inputs["labels"] = batch_label_ids
        tokenized_inputs["categories"] = batch_category_ids
        return tokenized_inputs

    def train(
        self,
        train_data: List[Dict[str, Any]],
        num_train_epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        eval_split: float = 0.1,
        seed: int = 42,
    ):
        """
        多标签NER训练入口
        """
        # 设置随机种子
        random.seed(seed)
        torch.manual_seed(seed)

        # 数据预处理
        processed = self.convert_to_multilabel_bio_with_category(train_data)
        random.shuffle(processed)
        split_idx = math.ceil(len(processed) * (1 - eval_split))
        train_subset = processed[:split_idx]
        eval_subset = processed[split_idx:] if split_idx < len(processed) else []

        train_dataset = Dataset.from_list(train_subset)
        eval_dataset = Dataset.from_list(eval_subset) if eval_subset else None

        # tokenization
        train_tokenized = train_dataset.map(
            lambda ex: self.tokenize_and_align_labels(ex),
            batched=True,
            remove_columns=["tokens", "ner_label_tags", "ner_category_tags"]
        )
        eval_tokenized = None
        if eval_dataset:
            eval_tokenized = eval_dataset.map(
                lambda ex: self.tokenize_and_align_labels(ex),
                batched=True,
                remove_columns=["tokens", "ner_label_tags", "ner_category_tags"]
            )

        # 加载预训练基础模型
        _base_model = AutoModel.from_pretrained(self.model_path).to("cpu")
        config = AutoConfig.from_pretrained(self.model_path)
        config.num_labels = len(self.label2id)
        config.id2label = self.id2label
        config.label2id = self.label2id

        # 初始化多标签NER模型
        model = MultiLabelNERForHF(config=config, base_model=_base_model, num_categories=len(self.category2id)).to("cpu")

        # 数据collator
        data_collator = DataCollatorForMultiLabelTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=self.max_length
        )

        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            save_strategy="epoch",
            logging_steps=50,
            fp16=False, #torch.cuda.is_available(),
            save_total_limit=3,
            logging_dir=os.path.join(self.output_dir, "logs"),
            seed=seed,
            use_cpu=True  # 强制使用CPU,
        )

        # Trainer封装训练
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        trainer.train()

        # 保存模型和映射
        os.makedirs(self.output_dir, exist_ok=True)
        model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        mapping_path = os.path.join(self.output_dir, "label_map.json")
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump({
                "id2label": self.id2label,
                "label2id": self.label2id,
                "id2category": self.id2category,
                "category2id": self.category2id
            }, f, ensure_ascii=False, indent=2)

        print(f"训练完成，模型已保存到: {self.output_dir}")
        print(f"标签映射已保存到: {mapping_path}")
