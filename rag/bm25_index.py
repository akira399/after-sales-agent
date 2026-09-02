"""
BM25 关键词检索索引。

基于 ChromaDB 中已入库的全部文档分片构建 BM25Okapi 索引，
使用 jieba 分词以支持中文关键词匹配。
"""

import pickle
import os
from typing import Optional

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from utils.config_handler import faiss_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class Bm25IndexService:
    """BM25 关键词索引，与 ChromaDB 向量库并行维护。"""

    def __init__(self):
        self._index: Optional[BM25Okapi] = None
        self._docs: list[Document] = []
        self._tokenized_corpus: list[list[str]] = []
        self._persist_dir = get_abs_path(faiss_conf["persist_directory"])
        self._state_path = os.path.join(self._persist_dir, "bm25_index.pkl")

    # ------------------------------------------------------------------
    # 分词
    # ------------------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return list(jieba.cut(text))

    # ------------------------------------------------------------------
    # 构建索引
    # ------------------------------------------------------------------
    def build(self, docs: list[Document]) -> None:
        if not docs:
            logger.warning("[BM25] 文档列表为空，跳过索引构建")
            return
        self._docs = list(docs)
        self._tokenized_corpus = [self._tokenize(d.page_content) for d in self._docs]
        self._index = BM25Okapi(self._tokenized_corpus)
        logger.info(f"[BM25] 索引构建完成，共 {len(self._docs)} 条文档")

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 10) -> list[tuple[Document, float]]:
        if self._index is None or not self._docs:
            logger.warning("[BM25] 索引尚未构建，返回空结果")
            return []
        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)
        # 按分数降序取 top-k
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top = indexed_scores[:k]
        return [(self._docs[idx], float(score)) for idx, score in top if score > 0]

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self) -> None:
        if self._index is None:
            return
        os.makedirs(self._persist_dir, exist_ok=True)
        with open(self._state_path, "wb") as f:
            pickle.dump(
                {
                    "docs": self._docs,
                    "tokenized_corpus": self._tokenized_corpus,
                },
                f,
            )
        logger.info(f"[BM25] 索引已持久化到 {self._state_path}")

    def load(self) -> bool:
        if not os.path.exists(self._state_path):
            return False
        try:
            with open(self._state_path, "rb") as f:
                data = pickle.load(f)
            self._docs = data["docs"]
            self._tokenized_corpus = data["tokenized_corpus"]
            self._index = BM25Okapi(self._tokenized_corpus)
            logger.info(f"[BM25] 索引已从 {self._state_path} 加载，共 {len(self._docs)} 条")
            return True
        except Exception as e:
            logger.warning(f"[BM25] 加载持久化索引失败: {e}")
            return False

    def invalidate(self) -> None:
        self._index = None
        self._docs = []
        self._tokenized_corpus = []
        if os.path.exists(self._state_path):
            os.remove(self._state_path)
            logger.info(f"[BM25] 已删除持久化索引文件 {self._state_path}")

    @property
    def is_ready(self) -> bool:
        return self._index is not None and len(self._docs) > 0
