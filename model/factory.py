"""模型工厂：根据当前会话的 ModelConfig 动态创建模型实例。

安全设计：不缓存全局单例 —— 每个会话根据访问者填写的 Key 构建自己的模型对象，
避免多用户共享同一 Key 被刷额度。非 Streamlit 环境（CLI/脚本）回退到环境变量。
"""
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings

from model.runtime_config import ModelConfig, get_session_config
from utils.logger_handler import logger

# DashScope OpenAI 兼容端点
DASHSCOPE_URL_FALLBACK = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# langchain 惯例占位 key（非空即可，真正鉴权失败时抛错便于提示）
_EMPTY_KEY = "EMPTY"


def build_chat_model(cfg: Optional[ModelConfig] = None) -> ChatOpenAI:
    """构建 Chat 模型。DashScope / OpenAI / 自定义端点均走 OpenAI 兼容协议。"""
    cfg = cfg or get_session_config()
    return ChatOpenAI(
        model=cfg.chat_model,
        api_key=cfg.api_key if cfg.api_key.strip() else _EMPTY_KEY,
        base_url=cfg.base_url or DASHSCOPE_URL_FALLBACK,
    )


def build_embed_model(cfg: Optional[ModelConfig] = None) -> Embeddings:
    """构建 Embedding 模型。

    注意：预建 FAISS 索引维度 = 1024（text-embedding-v4）。
    更换 embed_model 后若维度不同，检索会失败，需重新执行 python -m rag.vector_store。
    """
    cfg = cfg or get_session_config()
    if cfg.provider == "dashscope":
        return DashScopeEmbeddings(
            model=cfg.embed_model,
            dashscope_api_key=cfg.api_key if cfg.api_key.strip() else _EMPTY_KEY,
        )
    # OpenAI / 自定义兼容端点
    return OpenAIEmbeddings(
        model=cfg.embed_model,
        api_key=cfg.api_key if cfg.api_key.strip() else _EMPTY_KEY,
        base_url=cfg.base_url or None,
    )


# ------------------------------------------------------------------
# 兼容旧调用点：动态 getter，不再是模块级单例
# ------------------------------------------------------------------
def get_chat_model() -> ChatOpenAI:
    return build_chat_model()


def get_embed_model() -> Embeddings:
    return build_embed_model()


def log_model_info() -> None:
    cfg = get_session_config()
    logger.info(f"[model] 当前生效配置：{cfg.display_label()} 指纹={cfg.fingerprint()}")
