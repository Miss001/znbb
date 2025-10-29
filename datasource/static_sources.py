from elasticsearch8 import Elasticsearch
from config.ConfigManager import ConfigManager
from typing import Dict, Any, List
import time
import traceback

class BaseESClient:
    """静态 Elasticsearch 客户端（长期持有连接）"""

    def __init__(self, es_section="elasticsearch", debug=False, max_retries=3):
        self.debug = debug
        self.max_retries = max_retries
        conf = ConfigManager().get_section(es_section)

        use_ssl = conf.get("use_ssl", "false").lower() == "true"
        host_conf = {
            "host": conf.get("host", "localhost"),
            "port": int(conf.get("port", 9200)),
            "scheme": "https" if use_ssl else "http",
        }

        es_kwargs = {
            "hosts": [host_conf],
            "basic_auth": (conf.get("username"), conf.get("password")) if conf.get("username") else None,
        }

        if use_ssl:
            es_kwargs["verify_certs"] = True
            if conf.get("ca_certs"):
                es_kwargs["ca_certs"] = conf.get("ca_certs")

        self.es = Elasticsearch(**es_kwargs)

    def search(self, index_name: str, query_body: Dict[str, Any]) -> List[Dict[str, Any]]:
        """带重试机制"""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.es.search(index=index_name, body=query_body)
                return resp.get("hits", {}).get("hits", [])
            except Exception as e:
                if self.debug:
                    print(f"[Retry {attempt}] ES search failed: {e}")
                    traceback.print_exc()
                time.sleep(1)
        return []

    def exists(self, index_name: str) -> bool:
        try:
            return self.es.indices.exists(index=index_name)
        except Exception:
            return False
