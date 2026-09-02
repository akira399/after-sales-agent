"""
DashScope Rerank 重排序模块。

检索后候选文档进行二次精排，使用语义模型重新评估 query-document 相关度，
筛选出最相关的 Top-N 片段送入 LLM，提升回答质量。
"""

from http import HTTPStatus
from typing import Optional

import dashscope
from langchain_core.documents import Document

from model.runtime_config import ModelConfig
from utils.config_handler import rag_conf
from utils.logger_handler import logger


class RerankerService:
    """DashScope TextReRank 封装，基于 qwen3-rerank 做候选片段精排。

    模型与 Key 绑定到传入的 ModelConfig，不设进程级全局 Key，
    保证多会话（每个访问者自带 Key）互不串用。
    """

    def __init__(self, model_config: Optional[ModelConfig] = None):
        self._model = (model_config.rerank_model if model_config else None) or rag_conf.get(
            "rerank_model_name", "qwen3-rerank"
        )
        self._api_key = (model_config.api_key.strip() if model_config and model_config.api_key else None)
        if self._api_key:
            dashscope.api_key = self._api_key

    def rerank(
        self,
        query: str,
        docs: list[Document],
        top_n: Optional[int] = None,
    ) -> list[Document]:
        """对候选文档列表重排序并返回 top_n 条。"""
        if not docs:
            return []

        if top_n is None or top_n <= 0 or top_n > len(docs):
            top_n = len(docs)

        texts = [d.page_content for d in docs]

        call_kwargs: dict = {
            "model": self._model,
            "query": query,
            "documents": texts,
            "top_n": top_n,
            "return_documents": True,
        }
        if self._api_key:
            call_kwargs["api_key"] = self._api_key

        resp = dashscope.TextReRank.call(**call_kwargs)

        if resp.status_code != HTTPStatus.OK:
            logger.warning(f"[Rerank] 调用失败 code={resp.status_code} msg={resp.message}，回退原始排序")
            return docs[:top_n]

        results = resp.output.get("results") or []
        if not results:
            logger.warning("[Rerank] 返回结果为空，回退原始排序")
            return docs[:top_n]

        reranked: list[Document] = []
        for item in results:
            idx = item.get("index", -1)
            if isinstance(idx, int) and 0 <= idx < len(docs):
                reranked.append(docs[idx])

        if len(reranked) < len(results):
            logger.warning(
                f"[Rerank] 部分索引越界，有效结果 {len(reranked)}/{len(results)}"
            )

        logger.info(f"[Rerank] 重排序完成 query='{query[:80]}' {len(docs)}→{len(reranked)}")
        return reranked
