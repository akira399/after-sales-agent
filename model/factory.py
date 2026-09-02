import os
from abc import ABC, abstractmethod
from typing import Optional
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from utils.config_handler import rag_conf
from utils.secrets_handler import get_dashscope_api_key

# 自动加载项目根目录 .env（密钥不写死在代码/config 中）
load_dotenv()


def _resolve_dashscope_api_key() -> Optional[str]:
    key = get_dashscope_api_key()
    if key:
        return key

    config_value = rag_conf.get("dashscope_api_key")
    if isinstance(config_value, str) and config_value.strip():
        return config_value.strip()

    return None


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | ChatOpenAI]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | ChatOpenAI]:
        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            api_key=_resolve_dashscope_api_key(),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | ChatOpenAI]:
        return DashScopeEmbeddings(
            model=rag_conf["embedding_model_name"],
            dashscope_api_key=_resolve_dashscope_api_key(),
        )


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
