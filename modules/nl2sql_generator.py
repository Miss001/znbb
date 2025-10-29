# generator/NL2SQLGenerator.py
import json
import re
from typing import List, Dict, Optional
from modules.Prompt import Prompt
from modules.global_resources import GlobalResources


class NL2SQLGenerator:
    """自然语言转 SQL 模块，统一封装 LLM 调用"""

    def __init__(self):
        cfg = GlobalResources.get_config().get_section("llm_api")
        self.model = cfg.get("model")
        self.url = cfg.get("url")
        self.api_key = cfg.get("api_key", "")
        self.timeout = int(cfg.get("timeout", 60))
        self.max_tokens = int(cfg.get("max_tokens", 2048))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.session = GlobalResources.get_http_session()

        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    # -----------------------------
    # LLM 调用核心方法
    # -----------------------------
    def _call_llm(self, messages: List[Dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        try:
            resp = self.session.post(
                self.url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[LLM调用失败] {e}")
            if 'resp' in locals():
                print("[调试信息] 响应:", resp.text[:500])
            return ""

    # -----------------------------
    # JSON 提取辅助
    # -----------------------------
    @staticmethod
    def extract_json_from_model_output(text: str) -> dict:
        """从模型输出中提取 JSON 结构"""
        # 1. 尝试直接解析
        try:
            return json.loads(text.strip())
        except:
            pass

        # 2. 尝试匹配代码块或 {}
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S)
        jtext = match.group(1).strip() if match else re.search(r"\{.*\}", text, flags=re.S)
        if isinstance(jtext, re.Match):
            jtext = jtext.group(0).strip()

        if not jtext:
            return {}

        # 3. 清理字符串再尝试解析
        try:
            return json.loads(jtext)
        except:
            try:
                jtext2 = jtext.replace("'", '"').replace(",}", "}")
                return json.loads(jtext2)
            except:
                return {}

    # -----------------------------
    # 主逻辑：NL → SQL
    # -----------------------------
    def generate_sql(
            self,
            user_query: str,
            retrieved_results: Optional[Dict[str, List[Dict]]] = None,
            org_context: Optional[dict] = None,
            ner_entities: Optional[List[Dict]] = None
    ) -> dict:
        """输入自然语言，结合上下文信息生成 SQL"""
        prompt = Prompt.build_sql_prompt(
            user_query, retrieved_results, org_context, ner_entities
        )

        messages = [
            {"role": "system", "content": "你是一个 SQL 生成专家，只返回 JSON 格式的 SQL 结果。"},
            {"role": "user", "content": prompt}
        ]

        model_output = self._call_llm(messages)
        sql_result = self.extract_json_from_model_output(model_output)
        return sql_result or {"sql": "", "error": "生成失败"}
