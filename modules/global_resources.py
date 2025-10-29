# -*- coding: utf-8 -*-
"""
GlobalResources
-----------------
全局资源单例管理模块。

功能概述：
统一管理系统中所有**共享型资源**，确保全局只初始化一次、避免重复加载。
包括：
- ConfigManager（配置管理器）
- Embedding 模型（向量模型）
- NER 模型 / Tokenizer / LabelMap
- HTTP Session（请求会话）
- 动态 / 静态数据源（如数据库、Elasticsearch 等）
"""

import os
import json
import requests
from typing import Tuple, Dict, Any

from config.ConfigManager import ConfigManager
from sentence_transformers import SentenceTransformer
import torch
from datasource.factory import create_datasource, get_static_es_client


class GlobalResources:
    """全局单例资源管理器（Singleton Pattern）"""

    # ========= 类级别静态变量（全局唯一） =========
    _config: ConfigManager = None                  # 配置管理器单例
    _embed_model: SentenceTransformer = None       # 向量化模型（SentenceTransformer）
    _ner_model: Any = None                         # NER 模型实例
    _ner_tokenizer: Any = None                     # NER 模型对应的 Tokenizer
    _id2label: Dict[str, str] = None               # NER 标签映射表
    _id2category: Dict[str, str] = None            # NER 类别映射表
    _http_session: requests.Session = None         # HTTP 请求会话（Session）
    _datasource_cache: Dict[str, Any] = {}         # 动态数据源缓存（按名称）
    _static_es: Any = None                         # 静态 Elasticsearch 客户端实例

    # 模型缓存（支持多个模型同时缓存，例如不同的 NER 或 Embedding 模型）
    _model_cache: Dict[str, Dict[str, Any]] = {}

    # ========= ConfigManager 获取 =========
    @classmethod
    def get_config(cls) -> ConfigManager:
        """获取全局配置管理器（ConfigManager 单例）"""
        if cls._config is None:
            cls._config = ConfigManager()
        return cls._config

    # ========= Embedding 模型加载与缓存 =========
    @classmethod
    def get_embedding_model(cls) -> SentenceTransformer:
        """
        加载或返回全局唯一的 SentenceTransformer 模型。
        模型路径由配置项 embedding.local_model_path 指定。
        """
        if cls._embed_model is None:
            conf = cls.get_config().get_section("embedding")
            model_path = conf.get("local_model_path")

            # 校验路径合法性
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"embedding.local_model_path 无效: {model_path}")

            # 如果模型路径已存在于缓存中，则直接复用
            if model_path in cls._model_cache:
                cls._embed_model = cls._model_cache[model_path]["model"]
            else:
                # 加载模型并加入缓存
                cls._embed_model = SentenceTransformer(model_path)
                cls._model_cache[model_path] = {"model": cls._embed_model}
        return cls._embed_model

    # ========= NER 模型 + Tokenizer + LabelMap 加载 =========
    @classmethod
    def get_ner_resources(
        cls, model_path: str = "./model/ner_small_model"
    ) -> Tuple[Any, Any, Dict[str, str], Dict[str, str]]:
        """
        获取自定义多标签 NER 模型及其依赖资源。
        返回：
            (model, tokenizer, id2label, id2category)
        模型结构兼容 MultiLabelNERForHF 类。
        """
        if cls._ner_model is None:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"NER 模型路径不存在: {model_path}")

            # 优先从缓存中取
            if model_path in cls._model_cache:
                cached = cls._model_cache[model_path]
                cls._ner_model = cached["model"]
                cls._ner_tokenizer = cached["tokenizer"]
                cls._id2label = cached["id2label"]
                cls._id2category = cached["id2category"]
            else:
                # ===== 加载 NER 模型依赖 =====
                from transformers import AutoTokenizer, AutoConfig, AutoModel
                from model.multi_ner_trainer import MultiLabelNERForHF  # 自定义多标签 NER 模型类

                # 1️⃣ 加载分词器
                tokenizer = AutoTokenizer.from_pretrained(model_path)

                # 2️⃣ 加载模型配置和底层 encoder（如 BERT、RoBERTa）
                config = AutoConfig.from_pretrained(model_path)
                base_model = AutoModel.from_pretrained(model_path)

                # 3️⃣ 加载标签映射文件（id2label / id2category）
                label_map_file = os.path.join(model_path, "label_map.json")
                if not os.path.exists(label_map_file):
                    raise FileNotFoundError(f"缺少 label_map.json 文件: {label_map_file}")

                with open(label_map_file, "r", encoding="utf-8") as f:
                    map_data = json.load(f)
                    id2label = {int(k): v for k, v in map_data.get("id2label", {}).items()}
                    id2category = {int(k): v for k, v in map_data.get("id2category", {}).items()}

                # 4️⃣ 初始化自定义 MultiLabelNER 模型
                model = MultiLabelNERForHF(config, base_model, num_categories=len(id2category))

                # 5️⃣ 加载模型权重
                model.load_state_dict(
                    torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu"),
                    strict=False
                )

                # 6️⃣ 更新全局变量与缓存
                cls._ner_model = model
                cls._ner_tokenizer = tokenizer
                cls._id2label = id2label
                cls._id2category = id2category
                cls._model_cache[model_path] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "id2label": id2label,
                    "id2category": id2category
                }

        return cls._ner_model, cls._ner_tokenizer, cls._id2label, cls._id2category

    # ========= HTTP 会话 =========
    @classmethod
    def get_http_session(cls) -> requests.Session:
        """
        获取全局 HTTP Session，复用连接以提升请求性能。
        """
        if cls._http_session is None:
            cls._http_session = requests.Session()
        return cls._http_session

    # ========= 动态数据源 =========
    @classmethod
    def get_datasource(cls, name: str):
        """
        按名称动态创建数据源（例如 MySQL / Hive / Kafka 等）。
        若已创建则直接从缓存返回。
        """
        if name not in cls._datasource_cache:
            cls._datasource_cache[name] = create_datasource(name)
        return cls._datasource_cache[name]

    # ========= 静态 Elasticsearch 客户端 =========
    @classmethod
    def get_static_es_client(cls):
        """
        获取静态 Elasticsearch 客户端实例（用于长期连接）。
        """
        if cls._static_es is None:
            cls._static_es = get_static_es_client()
        return cls._static_es

    # ========= 清理缓存（重置所有全局资源）=========
    @classmethod
    def clear_cache(cls):
        """
        清除所有全局缓存对象。
        适用于模型热更新或环境重载场景。
        """
        cls._embed_model = None
        cls._ner_model = None
        cls._ner_tokenizer = None
        cls._id2category = None
        cls._id2label = None
        cls._http_session = None
        cls._datasource_cache.clear()
        cls._static_es = None
        cls._model_cache.clear()
