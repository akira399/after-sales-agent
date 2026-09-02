"""模型运行时配置 —— 支持「访问者自带 Key / 部署者显式提供的平台 Key」两种模式。

安全设计（重点）：
1. API Key 只存在于浏览器会话(st.session_state)或部署者环境变量中，
   绝不写入日志、代码或任何落盘文件。
2. 平台预置 Key（部署者 .env / 云端 Secrets）默认【不】自动生效。
   访问者必须显式选择：
   a. 填写自己的 Key（推荐，各用各的互不串用）；
   b. 或点击「改用平台预置 Key」显式启用部署者额度。
3. 非 Streamlit 环境（CLI / scripts/*.py）直接使用环境变量 Key，方便本地开发。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Optional

from dotenv import load_dotenv

# 自动加载项目根目录 .env（本地开发密钥）
load_dotenv()

# DashScope(阿里云百炼) OpenAI 兼容端点
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass
class ModelConfig:
    """一套可用的模型接入配置。"""

    provider: str = "dashscope"  # dashscope | openai | custom
    base_url: str = DASHSCOPE_BASE_URL
    api_key: str = ""
    chat_model: str = "qwen3.7-plus"
    embed_model: str = "text-embedding-v4"
    rerank_model: str = "qwen3-rerank"

    # ------------------------------------------------------------------
    def is_ready(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def fingerprint(self) -> str:
        """配置指纹：用于判断模型配置是否变化（变化则需重建 Agent/RAG）。"""
        raw = "|".join([self.provider, self.base_url, self.api_key[-6:],
                        self.chat_model, self.embed_model, self.rerank_model])
        return sha1(raw.encode("utf-8")).hexdigest()[:12]

    def display_label(self) -> str:
        """用于 UI 展示，绝不含完整 key。"""
        tail = f"...{self.api_key[-6:]}" if len(self.api_key) > 10 else "(未设置)"
        return f"{self.chat_model} @ {self.provider} [key{tail}]"


def _load_env_config() -> ModelConfig:
    """从环境变量 / .env 读取部署者预置配置（本地与私有部署模式）。"""
    from utils.config_handler import rag_conf

    cfg = ModelConfig()
    key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if key:
        cfg.api_key = key
    # 以 config/rag.yml 为准的模型名（与预建索引一致）
    cfg.chat_model = str(rag_conf.get("chat_model_name") or cfg.chat_model)
    cfg.embed_model = str(rag_conf.get("embedding_model_name") or cfg.embed_model)
    cfg.rerank_model = str(rag_conf.get("rerank_model_name") or cfg.rerank_model)
    return cfg


def _session_state() -> Optional[dict]:
    """安全返回 streamlit session_state；仅在 streamlit run 运行期返回。

    CLI（bare mode）下 streamlit.session_state 不生效，必须走环境变量配置。
    """
    try:
        import streamlit.runtime  # noqa: PLC0415
        import streamlit  # noqa: PLC0415

        if not streamlit.runtime.exists():
            return None
        return streamlit.session_state
    except Exception:
        return None


def _st_key() -> str:
    return "model_config"


def get_effective_config() -> ModelConfig:
    """获取"真正用于构建模型"的配置。

    - Streamlit 环境：只认会话配置（访问者自填 或 显式启用平台 Key）。
      会话无配置时返回空配置（is_ready=False），由 UI 引导填写。
    - 非 Streamlit 环境（CLI/scripts）：直接使用环境变量/.env 配置。
    """
    ss = _session_state()
    if ss is not None:
        cfg = ss.get(_st_key())
        if isinstance(cfg, ModelConfig):
            return cfg
        return ModelConfig()  # 空配置：UI 层会引导填写
    return _load_env_config()


def get_session_config() -> ModelConfig:
    """（兼容命名）返回当前会话生效配置，语义同 get_effective_config。"""
    return get_effective_config()


def save_session_config(cfg: ModelConfig) -> None:
    """把访问者填写的模型配置存入当前会话（不进日志、不落盘）。"""
    ss = _session_state()
    if ss is None:
        raise RuntimeError("save_session_config 只能在 Streamlit 运行时调用")
    ss[_st_key()] = cfg
    _invalidate_session_cache()


def use_platform_config() -> None:
    """访问者显式选择：启用部署者预置的平台 Key。"""
    ss = _session_state()
    if ss is None:
        return
    cfg = _load_env_config()
    if cfg.is_ready():
        ss[_st_key()] = cfg
        _invalidate_session_cache()


def clear_session_config() -> None:
    """清除会话中的模型配置，回退到"未配置"引导。"""
    ss = _session_state()
    if ss is None:
        return
    ss.pop(_st_key(), None)
    _invalidate_session_cache()


def has_platform_key() -> bool:
    """部署者是否在 .env / Secrets 预置了可用 Key，且当前会话未自填。"""
    ss = _session_state()
    if ss is None:
        return _load_env_config().is_ready()
    if isinstance(ss.get(_st_key()), ModelConfig):
        return False  # 访问者已自填，不再提示平台 Key
    return _load_env_config().is_ready()


def _invalidate_session_cache() -> None:
    """配置变化后，清除按旧配置构建的进程/会话缓存组件。"""
    ss = _session_state()
    if ss is None:
        return
    for key in ("agent_obj", "rag_obj", "agent_fp"):
        ss.pop(key, None)
