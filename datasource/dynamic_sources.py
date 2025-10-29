from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from urllib.parse import urljoin
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Any


# ====== 抽象基类 ======
class Datasource(ABC):
    """所有数据源的抽象基类"""

    @abstractmethod
    def execute(self, query: Any) -> List[Dict[str, Any]]:
        """执行查询"""
        pass

class _SQLDatasourceBase(Datasource):
    """通用 SQL 数据源基类"""

    def __init__(self, url: str, pool_size: int = 10, max_overflow: int = 5):
        self.engine: Engine = create_engine(
            url, pool_pre_ping=True, pool_size=pool_size, max_overflow=max_overflow
        )

    def execute(self, query: str) -> List[Dict[str, Any]]:
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                if result.returns_rows:
                    cols = result.keys()
                    return [dict(zip(cols, row)) for row in result.fetchall()]
                return []
        except Exception as e:
            return [{"error": str(e), "sql": query}]


class MySQLDatasource(_SQLDatasourceBase):
    def __init__(self, user, password, host, port, database):
        url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        super().__init__(url)


class OracleDatasource(_SQLDatasourceBase):
    def __init__(self, user, password, host, port, service_name):
        url = f"oracle+cx_oracle://{user}:{password}@{host}:{port}/?service_name={service_name}"
        super().__init__(url)


class ESDatasource(Datasource):
    """ES-SQL 动态查询"""

    def __init__(self, hosts: List[str], version: int, username: str = None, password: str = None):
        self.hosts, self.version, self.username, self.password = hosts, version, username, password
        self._init_client()

    def _init_client(self):
        try:
            if self.version >= 8:
                from elasticsearch import Elasticsearch
            else:
                from elasticsearch7 import Elasticsearch
            self.es_client = Elasticsearch(
                self.hosts,
                http_auth=(self.username, self.password) if self.username else None,
                verify_certs=False,
            )
            self.use_client = True
        except ImportError:
            self.es_client, self.use_client = None, False

    def execute(self, query: str) -> List[Dict[str, Any]]:
        try:
            if not self.use_client:
                url = urljoin(self.hosts[0].rstrip("/") + "/", "_xpack/sql")
                r = requests.post(url, json={"query": query}, auth=(self.username, self.password), verify=False)
                r.raise_for_status()
                resp = r.json()
            else:
                resp = self.es_client.sql.query(body={"query": query})

            cols = [c["name"] for c in resp.get("columns", [])] or []
            rows = resp.get("rows", [])
            if not cols and rows:
                cols = [f"col_{i}" for i in range(len(rows[0]))]
            return [dict(zip(cols, r)) for r in rows]
        except Exception as e:
            return [{"error": str(e), "query": query}]


class APIDatasource(Datasource):
    """通过 API 获取数据"""

    def __init__(self, base_url: str, headers: Optional[dict] = None):
        self.base_url = base_url
        self.headers = headers or {}

    def execute(self, query: Any) -> List[Dict[str, Any]]:
        try:
            resp = requests.post(self.base_url, json=query, headers=self.headers, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return [{"error": str(e), "query": query}]
