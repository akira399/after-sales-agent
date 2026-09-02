"""
DashScope Rerank 重排序模块。

检索后候选文档进行二次精排，使用语义模型重新评估 query-document 相关度，
筛选出最相关的 Top-N 片段送入 LLM，提升回答质量。
"""

from http import HTTPStatus
from typing import Optional

import dashscope
from langchain_core.documents import Document

from model.factory import _resolve_dashscope_api_key
from utils.config_handler import rag_conf
from utils.logger_handler import logger


class RerankerService:
    """DashScope TextReRank 封装，基于 qwen3-rerank 做候选片段精排。"""

    def __init__(self):
        api_key = _resolve_dashscope_api_key()
        if api_key:
            dashscope.api_key = api_key
        self._model = rag_conf.get("rerank_model_name", "qwen3-rerank")

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

        resp = dashscope.TextReRank.call(
            model=self._model,
            query=query,
            documents=texts,
            top_n=top_n,
            return_documents=True,
        )

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
