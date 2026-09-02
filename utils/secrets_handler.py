"""密钥统一解析：兼容本地 .env / 系统环境变量 / Streamlit Cloud secrets。

- 本地开发：.env 由 model/factory.py 的 load_dotenv() 注入 os.environ。
- Streamlit Cloud：secrets 不自动注入 os.environ，必须显式同步，
  因此应用入口(agent_app.py)会先调用 load_cloud_secrets()。
"""
import os
from typing import Optional


def load_cloud_secrets() -> None:
    """将 Streamlit secrets 同步到 os.environ（仅在 streamlit 运行时生效）。"""
    try:
        import streamlit as st  # noqa: PLC0415

        secrets = getattr(st, "secrets", None)
        if secrets is None:
            return
        for key, value in secrets.items():
            if isinstance(value, str) and value.strip():
                os.environ.setdefault(key, value)
    except Exception:
        # 非 streamlit 环境（CLI 脚本）静默跳过
        pass


def get_dashscope_api_key() -> Optional[str]:
    """按优先级返回 DashScope API Key：环境变量 > 云 secrets > 空。"""
    load_cloud_secrets()
    value = os.getenv("DASHSCOPE_API_KEY", "")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
