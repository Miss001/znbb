from typing import Dict, List, Optional
from datetime import datetime
from config.PolicyManager import TimePolicyManager, OrgPolicyManager

class Prompt:

    SQL_SYNTAX_MAP = {
        "mysql": "MYSQL",
        "oracle": "Oracle",
        "es": "Elasticsearch SQL"
    }

    @staticmethod
    def build_sql_prompt(
        user_query: str,
        retrieved_results: Optional[Dict[str, List[Dict]]] = None,
        org_context: Optional[dict] = None,
        ner_entities: Optional[List[Dict]] = None
    ) -> str:
        """构造 LLM SQL 生成提示文本"""
        # ---------------------- 初始化 ----------------------
        time_manager = TimePolicyManager()
        org_manager = OrgPolicyManager()
        time_templates = time_manager.list_templates()
        today_str = datetime.utcnow().strftime("%Y-%m-%d")
        org_code = org_context.get("org_code", "未指定") if org_context else "未指定"
        org_level = org_context.get("org_level", "未指定") if org_context else "未指定"

        lines = [
            f"【用户查询描述】\n{user_query}",
            f"【系统当前日期】{today_str}（UTC）"
        ]

        # ---------------------- 候选指标信息 ----------------------
        metrics = retrieved_results.get("metrics", []) if retrieved_results else []
        for m in metrics:
            metric_id = m.get("metric_id", "unknown")
            name = m.get("name", "unknown")
            description = m.get("description", "无描述")
            table_name = m.get("tables", "unknown")
            filters_str = m.get("filters", "无必要条件")
            source = m.get("sourcename", "").lower()

            syntax = Prompt.SQL_SYNTAX_MAP.get(source, "通用SQL语法")
            metric_lines = [
                f"\n▶ 指标ID: {metric_id}",
                f"  指标名称: {name}",
                f"  指标描述: {description}",
                f"  来源表: {table_name}",
                f"  必要条件: {filters_str}",
                f"  SQL 生成语法要求为：{syntax}"
            ]

            # 组织机构策略
            org_filters_conf = m.get("org_filters")
            if org_filters_conf:
                org_filters_str = org_manager.get_filter_by_level(org_filters_conf, org_level)
                metric_lines.append(f"  [组织机构查询策略说明]: {org_filters_str}")

            # 时间策略
            time_filters = m.get("time_filters", [])
            if time_filters and time_templates:
                metric_lines.append("  [时间策略说明]:")
                for tf in time_filters:
                    tpl_name = tf.get("policy")
                    field = tf.get("field")
                    tpl = time_templates.get(tpl_name)
                    if tpl:
                        expr = tpl["expression"].replace("{field}", field)
                        metric_lines.append(
                            f"    - 字段: {field}\n"
                            f"      格式: {tpl.get('format','unknown')}\n"
                            f"      描述: {tpl.get('description','无描述')}\n"
                            f"      示例表达式: {expr}"
                        )
            lines.extend(metric_lines)

        # ---------------------- NER 实体信息 ----------------------
        if ner_entities:
            lines.append("\n【NER 实体信息】")
            for ent in ner_entities:
                resolved = ent.get("resolved_value")
                lines.append(
                    f"- 实体: {ent.get('name','unknown')}, 类型: {ent.get('type','unknown')}" +
                    (f", 对应值: {resolved}" if resolved else ", 未匹配到字典值")
                )

        # ---------------------- 表结构信息 ----------------------
        schemas = retrieved_results.get("schemas", []) if retrieved_results else []
        if schemas:
            lines.append("\n【表结构信息】")
            for s in schemas:
                table_name = s.get("tablename", "unknown")
                fields = s.get("columns", [])
                if fields:
                    field_strs = [f"{f.get('name')} - {f.get('description','无描述')} ({f.get('type','unknown')})" for f in fields]
                    lines.append(f"- {table_name}: {', '.join(field_strs)}")
                else:
                    lines.append(f"- {table_name}: 无字段信息")

        # ========================== 当前组织机构上下文 ==========================
        if org_context:
            lines.append("\n【当前组织机构上下文】")
            lines.append(f"- 组织代码: {org_code}")
            lines.append(f"- 当前层级: {org_level}")
            lines.append(
                "\n组织机构规则说明：\n"
                "- 若查询目标为当前层级自身数据，使用“默认”组织条件。\n"
                "- 若查询目标为下级汇总数据（省→市、市→分局、分局→派出所），使用“分组查询下级”条件。\n"
                "- 请将占位符 {org_code} 替换为实际组织代码。"
            )

        # ========================== 输出要求 ==========================
        lines.append(
            "\n【输出要求】\n"
            "请根据上述信息生成完整 SQL 查询：\n"
            "- 自动选择最匹配的指标（来源表、业务规则与用户查询最匹配）\n"
            "- SQL 必须可直接执行\n"
            "- 包含正确时间及组织机构过滤条件\n"
            "- 若实体信息存在，请根据实体值生成对应 SQL 条件\n"
            "- 输出为 JSON 对象，字段如下：\n"
            "  - chosen_metric_id: 选择的指标ID\n"
            "  - sql: 完整可执行 SQL\n"
            "  - confidence: 模型选择置信度（0~1）"
        )

        prompt = "\n".join(lines)
        print(prompt)
        return prompt
