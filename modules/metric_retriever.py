from datasource.factory import get_static_es_client
from modules.global_resources import GlobalResources
import numpy as np

class MetricRetriever:
    """指标检索器"""

    def __init__(self, debug: bool = False):
        self.debug = debug
        conf = GlobalResources.get_config().get_section("elasticsearch")

        self.es_client = get_static_es_client("elasticsearch")
        self.metrics_index = conf["metrics_index"]
        self.schema_index = conf["schema_index"]
        self.embed_model = GlobalResources.get_embedding_model()

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _search_index(self, index_name: str, vector: np.ndarray, top_k: int):
        body = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": vector.tolist()},
                    },
                }
            },
        }
        return self.es_client.search(index_name, body)

    def retrieve(self, query: str, top_k=5):
        q_emb = self._normalize(self.embed_model.encode([query])[0])
        metric_hits = self._search_index(self.metrics_index, q_emb, top_k)
        metrics = [{**hit["_source"], "score": hit["_score"] - 1.0} for hit in metric_hits]

        schemas = []
        for m in metrics:
            emb = self._normalize(np.array(m["embedding"]))
            hits = self._search_index(self.schema_index, emb, top_k)
            for h in hits:
                schemas.append({**h["_source"], "metric_id": m.get("metric_id"), "score": h["_score"] - 1.0})

        unique = {s["tablename"]: s for s in schemas}.values()
        return {"metrics": metrics, "schemas": sorted(unique, key=lambda x: x["score"], reverse=True)}
