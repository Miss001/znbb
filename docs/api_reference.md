# znbb 公共 API 文档

本文档覆盖仓库中所有对外公开的类、函数与主要组件，概述其职责、关键参数、返回值以及典型用法示例，方便二次开发与集成。

## 快速索引
- [配置管理 `config`](#配置管理config)
  - [`ConfigManager`](#configmanager)
- [数据源抽象与工厂 `datasource`](#数据源抽象与工厂datasource)
  - [`Datasource` 抽象基类](#datasource-抽象基类)
  - [`MySQLDatasource` / `OracleDatasource`](#mysqldatasource--oracledatasource)
  - [`ESDatasource`](#esdatasource)
  - [`APIDatasource`](#apidatasource)
  - [`create_datasource`](#create_datasource)
  - [`BaseESClient` 与 `get_static_es_client`](#baseesclient-与-get_static_es_client)
- [全局资源管理 `modules.global_resources`](#全局资源管理-modulesglobal_resources)
  - [`GlobalResources`](#globalresources)
- [业务模块](#业务模块)
  - [`MetricRetriever`](#metricretriever)
  - [`NERPipeline`](#nerpipeline)
  - [`NL2SQLGenerator`](#nl2sqlgenerator)
  - [`Prompt`](#prompt)
- [多标签 NER 训练工具 `model.multi_ner_trainer`](#多标签-ner-训练工具-modelmulti_ner_trainer)
  - [`NERTrainingConfig`](#nertrainingconfig)
  - [`MultiLabelNERForHF`](#multilabelnerforhf)
  - [`OptimizedDataCollator`](#optimizeddatacollator)
  - [`NERTrainer`](#nertrainer)

> **提示**：文中示例均为精简版本，未展示的异常处理、日志与安全校验请根据实际场景补充。

---

## 配置管理`config`

### `ConfigManager`
**位置**：`config/ConfigManager.py`

线程安全的懒加载单例，用于读取 `config.ini` 并支持后台热重载。默认每 30 秒检测一次配置文件更改，自动刷新内部缓存。

**构造参数**
- `config_path: str | None`：配置文件路径，默认指向仓库根目录下的 `config/config.ini`。
- `reload_interval: int`：后台检测间隔（秒）。

**主要方法**
- `get_section(section_name: str) -> Dict[str, str]`：返回指定节的键值映射，结果存在内存缓存中。

**使用示例**
```python
from config.ConfigManager import ConfigManager

cfg = ConfigManager()  # 单例获取
llm_conf = cfg.get_section("llm_api")
print(llm_conf["url"], llm_conf.get("timeout", 60))
```

**注意事项**
- 若配置文件不存在，`get_section` 会抛出 `ValueError`，请提前校验。
- 类初始化时会启动守护线程；在短生命周期脚本中使用时可适当调大 `reload_interval`。

---

## 数据源抽象与工厂`datasource`

### `Datasource` 抽象基类
**位置**：`datasource/dynamic_sources.py`

所有动态数据源需继承该类并实现 `execute(query) -> List[Dict]`，返回统一的字典列表结果。

### `MySQLDatasource` / `OracleDatasource`
封装 SQLAlchemy 引擎，提供基础的 SQL 查询执行能力。

**构造参数**
- `user`, `password`, `host`, `port`, `database/service_name`

**核心方法**
- `execute(query: str) -> List[Dict[str, Any]]`
  - 执行 SQL，若无返回行则返回空列表。
  - 捕获异常后返回包含 `error` 与原始 `sql` 的单行字典。

**示例**
```python
from datasource.dynamic_sources import MySQLDatasource

ds = MySQLDatasource("user", "pwd", "127.0.0.1", 3306, "analytics")
rows = ds.execute("SELECT id, name FROM users LIMIT 5")
```

### `ESDatasource`
封装 Elasticsearch SQL 能力，同时兼容原生客户端调用与 `_xpack/sql` REST 接口。

**构造参数**
- `hosts: List[str]`：例如 `['https://es-1:9200']`
- `version: int`：ES 主版本号，决定导入 `elasticsearch` 还是 `elasticsearch7`。
- `username`, `password`: 可选认证信息。

**行为**
- 若缺少官方 SDK，则回退到 HTTP POST 调用。
- 查询失败时返回包含 `error` 与原始 `query` 的字典。

**示例**
```python
es = ESDatasource(["https://es.local:9200"], version=8, username="elastic", password="secret")
results = es.execute("SELECT * FROM metrics LIMIT 10")
```

### `APIDatasource`
通用 HTTP POST 数据源，适合对接 RESTful/GraphQL 等服务。

**构造参数**
- `base_url: str`
- `headers: dict | None`

**示例**
```python
api_ds = APIDatasource("https://api.example.com/query", headers={"Authorization": "Bearer xxx"})
payload = {"metric_id": "pv", "range": "2024-01"}
resp = api_ds.execute(payload)
```

### `create_datasource`
**位置**：`datasource/factory.py`

根据配置节自动实例化对应的 `Datasource` 子类。

**参数**
- `sourcename: str`：`config.ini` 中的节名。

**返回值**
- 对应的 `Datasource` 实例。

**示例配置片段**
```ini
[mysql_analytics]
type = mysql
host = 127.0.0.1
port = 3306
user = root
password = secret
database = analytics
```

**示例调用**
```python
from datasource.factory import create_datasource

ds = create_datasource("mysql_analytics")
data = ds.execute("SELECT COUNT(*) AS total FROM visits")
```

### `BaseESClient` 与 `get_static_es_client`
**位置**：`datasource/static_sources.py`

长连接 Elasticsearch 客户端，内部通过 `ConfigManager` 读取连接信息，并提供重试逻辑。

**构造参数**
- `es_section: str`：配置节名称，默认 `elasticsearch`。
- `debug: bool`
- `max_retries: int`

**主要方法**
- `search(index_name: str, query_body: Dict) -> List[Dict]`
- `exists(index_name: str) -> bool`

**示例**
```python
from datasource.factory import get_static_es_client

client = get_static_es_client("elasticsearch")
query = {"query": {"match_all": {}}, "size": 5}
hits = client.search("metrics_index", query)
```

`get_static_es_client` 通过单例缓存 `BaseESClient`，避免重复初始化。

---

## 全局资源管理 `modules.global_resources`

### `GlobalResources`
集中管理跨模块共享的重型对象（配置、模型、HTTP 会话等）。所有方法均为类方法，可直接调用。

**方法概览**
- `get_config() -> ConfigManager`
- `get_embedding_model() -> SentenceTransformer`
- `get_ner_resources(model_path: str | None) -> (model, tokenizer, id2label, id2category)`
- `get_http_session() -> requests.Session`
- `get_datasource(name: str)`：内部调用 `create_datasource` 并缓存。
- `get_static_es_client()`：懒加载静态 ES 客户端。
- `clear_cache()`：清理所有缓存对象，方便进行模型热更新或测试隔离。

**示例：加载嵌入模型并复用数据源**
```python
from modules.global_resources import GlobalResources

embedder = GlobalResources.get_embedding_model()
vector = embedder.encode(["示例问题"])[0]

olap_ds = GlobalResources.get_datasource("mysql_analytics")
rows = olap_ds.execute("SELECT 1")
```

**注意事项**
- `get_embedding_model` 与 `get_ner_resources` 依赖文件系统中的模型目录，请确保配置路径存在。
- `get_ner_resources` 会导入 `model.multi_ner_trainer.MultiLabelNERForHF`，需提前安装 HuggingFace 及 PyTorch。

---

## 业务模块

### `MetricRetriever`
**位置**：`modules/metric_retriever.py`

负责基于向量相似度在 Elasticsearch 中检索指标与表结构信息。

**构造参数**
- `debug: bool = False`

**公开方法**
- `retrieve(query: str, top_k: int = 5) -> Dict`
  - 返回格式：`{"metrics": [...], "schemas": [...]}`。
  - `metrics` 列表元素包含 `_source` 字段与相似度 `score`。
  - `schemas` 聚合结果去重后按分数排序。

**示例**
```python
retriever = MetricRetriever()
result = retriever.retrieve("上季度销售额")
for metric in result["metrics"]:
    print(metric["name"], metric["score"])
```

**依赖**
- `elasticsearch` 配置节中必须包含 `metrics_index` 与 `schema_index`。

### `NERPipeline`
**位置**：`modules/ner_pipeline.py`

多标签 NER 推理与字典映射流程。

**构造参数**
- `debug: bool = False`

**主要方法**
- `predict_entities(text: str, threshold: float = 0.5) -> List[Dict]`
  - 返回实体片段及标签、类别。
- `resolve_entities(entities: List[Dict]) -> List[Dict]`
  - 调用 ES 字典索引，将实体映射到业务代码。
- `process_query(text: str) -> Dict`
  - 综合识别与映射，输出结构：
    ```python
    {
        "original_text": text,
        "entities": [
            {
                "name": "发案数",
                "type": "指标",
                "category": "crime",
                "resolved_value": "A001"
            }
        ]
    }
    ```

**示例**
```python
pipeline = NERPipeline()
result = pipeline.process_query("查询上周刑事发案数")
```

**注意事项**
- 需要 GPU 支持以获得最佳性能，自动回退 CPU。
- `resolve_entities` 使用 LRU 缓存，重复查询同一实体时命中缓存。

### `NL2SQLGenerator`
**位置**：`modules/nl2sql_generator.py`

封装 LLM 调用，将自然语言问题转为 SQL。

**构造参数**
- 从 `llm_api` 配置节读取 `model, url, api_key, timeout, max_tokens, temperature`。

**公开方法**
- `generate_sql(user_query: str, retrieved_results: Optional[Dict], org_context: Optional[dict], ner_entities: Optional[List[Dict]]) -> dict`
  - 返回模型生成的 JSON 结构，若解析失败则返回 `{"sql": "", "error": "生成失败"}`。

**用法示例**
```python
generator = NL2SQLGenerator()

retrieved = MetricRetriever().retrieve("查询上月销售额", top_k=3)
ner_result = NERPipeline().process_query("查询上月销售额")

sql_result = generator.generate_sql(
    user_query="查询上月销售额",
    retrieved_results=retrieved,
    org_context={"org_code": "110000", "org_level": "province"},
    ner_entities=ner_result["entities"]
)
print(sql_result.get("sql"))
```

**调试建议**
- `_call_llm` 中已包含异常打印，可结合 `debug` 日志排查请求失败原因。

### `Prompt`
**位置**：`modules/Prompt.py`

构建 NL2SQL 提示词的工具类。

**公开方法**
- `build_sql_prompt(user_query, retrieved_results=None, org_context=None, ner_entities=None) -> str`
  - 汇总指标、NER、组织、时间策略信息并生成结构化提示。

**依赖**
- `config.PolicyManager.TimePolicyManager`
- `config.PolicyManager.OrgPolicyManager`

> 当前仓库未包含 `PolicyManager`，请确保在部署时补齐或自行实现同名接口，需提供 `list_templates()` 与 `get_filter_by_level()` 方法。

**示例**
```python
prompt_text = Prompt.build_sql_prompt(
    user_query="统计本季度新增用户数",
    retrieved_results={"metrics": [], "schemas": []},
    org_context={"org_code": "320100", "org_level": "city"},
    ner_entities=[{"name": "新增用户", "type": "metric", "category": "user", "resolved_value": "U001"}]
)
print(prompt_text)
```

---

## 多标签 NER 训练工具 `model.multi_ner_trainer`

### `NERTrainingConfig`
数据类，定义训练超参。常用字段：
- `model_path`: 预训练模型目录。
- `output_dir`: 训练产出目录。
- `max_length`, `batch_size`, `num_train_epochs` 等。

**示例**
```python
from model.multi_ner_trainer import NERTrainingConfig

config = NERTrainingConfig(
    model_path="./chinese-roberta-wwm-ext",
    output_dir="./ner_output",
    num_train_epochs=10,
    batch_size=16
)
```

### `MultiLabelNERForHF`
基于 HuggingFace `PreTrainedModel` 的多标签 NER 模型封装，提供双头输出（实体标签、类别标签）。

**构造参数**
- `config`: `AutoConfig`
- `base_model`: 预训练模型主体
- `num_categories`: 类别数
- `dropout_prob`

**前向输出**
- `{"loss": loss, "label_logits": Tensor, "category_logits": Tensor, "hidden_states": Tensor}`

**注意**
- 使用 `torch.cuda.amp.autocast` 做混合精度推理。
- `config.num_labels` 必须提前设置。

### `OptimizedDataCollator`
Tokenizer 驱动的数据整理器，实现多标签对齐填充，返回 PyTorch 张量。

**参数**
- `tokenizer: AutoTokenizer`
- `padding: bool`
- `max_length`

**调用**
```python
collator = OptimizedDataCollator(tokenizer=tokenizer, max_length=128)
batch = collator(features)
```

### `NERTrainer`
封装训练流程（数据划分、Trainer 配置、模型保存）。

**构造参数**
- `config: NERTrainingConfig`

**关键方法**
- `train(train_data: List[Dict[str, Any]])`
- 内部辅助方法 `_prepare_datasets`, `_initialize_model`, `_get_training_args` 等。

**训练数据格式期望**
```python
{
    "tokens": ["公", "安", "部"],
    "labels": [[0,1,0,...], ...],
    "categories": [[1,0,...], ...]
}
```

**训练示例（伪代码）**
```python
from model.multi_ner_trainer import NERTrainer, NERTrainingConfig

config = NERTrainingConfig(output_dir="./ner_model")
trainer = NERTrainer(config)

# 准备数据
training_samples = json.load(open("train.json"))

trainer.train(training_samples)
```

**额外提示**
- 训练日志输出到 TensorBoard：`<output_dir>/tensorboard`。
- 默认使用早停策略（`EarlyStoppingCallback`），可通过配置调整。

---

## 常见问题排查
- **缺少 `config.ini`**：所有工厂方法都会依赖 `ConfigManager`，请确保提供必要配置节。
- **未安装依赖库**：`sentence_transformers`、`transformers`、`datasets`、`sqlalchemy`、`elasticsearch[8|7]` 均为必备依赖。
- **PolicyManager 未找到**：需自行实现 `config/PolicyManager.py` 并提供 `TimePolicyManager`、`OrgPolicyManager` 类。
- **Elasticsearch 证书问题**：若启用 HTTPS，请在配置中提供 `use_ssl=true` 与 `ca_certs` 路径。

---

## 版本与维护建议
- 调整配置或模型后，调用 `GlobalResources.clear_cache()` 以便重新加载。
- 在服务启动时预热关键资源（如嵌入模型、NER 模型）以减少首请求延迟。
- 如需扩展数据源类型，可继承 `Datasource` 并在 `create_datasource` 中新增分支。

