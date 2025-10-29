# -*- coding: utf-8 -*-
from config.ConfigManager import ConfigManager

# ====== 动态数据源 ======
# 假设这里是你的已有实现
from .dynamic_sources import MySQLDatasource, OracleDatasource, ESDatasource, APIDatasource, Datasource
# 静态 ES 客户端
from .static_sources import BaseESClient

# ====== 数据源工厂方法 ======
def create_datasource(sourcename: str) -> Datasource:
    """
    根据配置文件创建数据源实例
    :param sourcename: 配置节名称
    :return: Datasource 实例
    """
    conf = ConfigManager().get_section(sourcename)
    ds_type = conf.get("type", "").lower()

    if ds_type == "mysql":
        return MySQLDatasource(
            conf["user"],
            conf["password"],
            conf["host"],
            conf.get("port", 3306),
            conf["database"]
        )
    elif ds_type == "oracle":
        return OracleDatasource(
            conf["user"],
            conf["password"],
            conf["host"],
            conf.get("port", 1521),
            conf["service_name"]
        )
    elif ds_type == "es":
        hosts = [h.strip() for h in conf["hosts"].split(",")]
        version = int(conf.get("version", 8))
        return ESDatasource(
            hosts,
            version,
            conf.get("username"),
            conf.get("password")
        )
    elif ds_type == "api":
        return APIDatasource(
            conf["base_url"],
            conf.get("headers")
        )
    else:
        raise ValueError(f"Unsupported datasource type: {ds_type}")


# ====== 静态 ES 客户端（单例） ======
_es_client_instance = None

def get_static_es_client(section: str = "elasticsearch") -> BaseESClient:
    """
    返回单例静态 ES 客户端
    :param section: 配置节名称
    :return: BaseESClient 实例
    """
    global _es_client_instance
    if _es_client_instance is None:
        _es_client_instance = BaseESClient(es_section=section)
    return _es_client_instance
