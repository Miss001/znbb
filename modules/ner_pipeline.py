# modules/ner_pipeline.py
# -*- coding: utf-8 -*-
import torch
import numpy as np
from typing import Dict, List, Any
from functools import lru_cache
from modules.global_resources import GlobalResources
from datasource.factory import get_static_es_client


class NERPipeline:
    """NER 实体识别 + 字典映射解析（兼容多标签模型）"""

    def __init__(self, debug: bool = False):
        # ---- 加载全局配置 ----
        config = GlobalResources.get_config()
        es_conf = config.get_section("elasticsearch")

        # ---- 加载静态 ES 客户端 ----
        self.es = get_static_es_client("elasticsearch")

        # ---- 加载模型与映射 ----
        self.model, self.tokenizer, self.id2label, self.id2category = GlobalResources.get_ner_resources()

        # ---- 模型设备 ----
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

        # ---- 目标索引 ----
        self.index = es_conf["dict_index"]
        self.debug = debug

    # ===========================================================
    # NER 实体识别（兼容多标签输出）
    # ===========================================================
    def predict_entities(self, text: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """识别文本中的实体，返回带类别信息的结构"""
        tokens = list(text)
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            label_logits = outputs["label_logits"]
            category_logits = outputs["category_logits"]

        label_probs = torch.sigmoid(label_logits).cpu().numpy()[0]
        category_probs = torch.sigmoid(category_logits).cpu().numpy()[0]

        entities = []
        start = None
        current_label, current_cat = None, None

        for i in range(len(tokens)):
            if i >= len(label_probs):
                break
            label_indices = np.where(label_probs[i] > threshold)[0]
            cat_indices = np.where(category_probs[i] > threshold)[0]

            labels = [self.id2label[idx] for idx in label_indices] if len(label_indices) > 0 else []
            cats = [self.id2category[idx] for idx in cat_indices] if len(cat_indices) > 0 else []

            if labels and cats:
                if start is None:
                    start = i
                    current_label = labels[0]
                    current_cat = cats[0]
            else:
                if start is not None:
                    entities.append({
                        "text": text[start:i],
                        "label": current_label,
                        "category": current_cat
                    })
                    start = None

        if start is not None:
            entities.append({
                "text": text[start:],
                "label": current_label,
                "category": current_cat
            })

        if self.debug:
            print(f"[DEBUG] Predicted entities: {entities}")

        return entities

    # ===========================================================
    # 字典映射（ES 查询）
    # ===========================================================
    @lru_cache(maxsize=1024)
    def _cached_search(self, category: str, text: str):
        """根据类别 + 文本在字典索引中查找对应代码"""
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"ct": text}},
                        {"match": {"zdbh": category}}
                    ]
                }
            },
            "_source": ["dm"],
            "size": 1,
        }
        results = self.es.search(index_name=self.index, query_body=query)
        if results and len(results) > 0:
            return results[0]["_source"].get("dm")
        return None

    # ===========================================================
    # 实体映射解析
    # ===========================================================
    def resolve_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for ent in entities:
            dm = self._cached_search(ent["category"], ent["text"])
            results.append({
                "name": ent["text"],
                "type": ent["label"],
                "category": ent["category"],
                "resolved_value": dm
            })
        return results

    # ===========================================================
    # 主入口
    # ===========================================================
    def process_query(self, text: str):
        entities = self.predict_entities(text)
        resolved = self.resolve_entities(entities)
        return {
            "original_text": text,
            "entities": resolved
        }
