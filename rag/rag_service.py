
"""
知识库检索问答：先对用户问题做识别并生成检索用语，再向量检索；检索结果全部写入上下文后作答。
检索阶段合并「检索用语」与「用户原句」两路混合检索结果，交替合并去重，
其中每路混合检索使用 BM25（关键词）+ 向量（语义）的 RRF 融合，结合稀疏与稠密两种信号。

流程：Query → 改写 → 混合检索(RRF) → Rerank → 上下文压缩 → 生成答案（含引用）
"""
from itertools import zip_longest
import os
from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from model.factory import build_chat_model
from model.runtime_config import ModelConfig
from rag.bm25_index import Bm25IndexService
from rag.reranker import RerankerService
from rag.vector_store import VectorStoreService
from utils.config_handler import faiss_conf, rag_conf
from utils.prompt_loader import load_kb_retrieve_qa_prompts, load_query_understand_prompts


class RagSummarizeService(object):
    def __init__(self, model_config: Optional[ModelConfig] = None):
        self._k = int(faiss_conf.get("k", 5))
        self._bm25_weight = float(faiss_conf.get("bm25_weight", 0.5))
        self._rrf_k = int(faiss_conf.get("rrf_k", 60))
        self._rerank_enabled = bool(rag_conf.get("rerank_model_name"))
        self.vector_store = VectorStoreService()
        self._vs = self.vector_store.vector_store
        self._bm25 = Bm25IndexService()
        self._bm25.load()
        self._reranker = RerankerService(model_config=model_config)
        self.system_prompt_text = load_kb_retrieve_qa_prompts()

        # 模型对象绑定到传入的会话配置（未传则按当前会话/环境动态获取）
        self.model = build_chat_model(model_config)
        self.model_config = model_config

        self._query_understand_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", load_query_understand_prompts()),
                ("human", "{user_question}"),
            ]
        )
        self._query_understand_chain = (
            self._query_understand_prompt | self.model | StrOutputParser()
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt_text),
                (
                    "human",
                    "### 历史对话（最近五轮）\n"
                    "{chat_history}\n\n"
                    "### 检索用语（经问题识别后用于向量检索）\n"
                    "{retrieve_query}\n\n"
                    "以下为本次向量检索返回的全部片段（共 {fragment_count} 条，按相似度由高到低排列，"
                    "须全部纳入理解后再作答）。\n\n"
                    "{context}\n\n"
                    "### 用户原始问题\n{input}",
                ),
            ]
        )
        self.chain = self.prompt_template | self.model | StrOutputParser()

    def reload_bm25(self) -> None:
        """增量索引后重新加载 BM25 索引，保证检索结果与向量库同步。"""
        self._bm25.load()

    @staticmethod
    def _doc_dedupe_key(doc: Document) -> int:
        src = ""
        if doc.metadata:
            src = str(doc.metadata.get("source", ""))
        return hash((doc.page_content[:800], src))

    # ------------------------------------------------------------------
    # 单路混合检索（BM25 + 向量）→ RRF 融合
    # ------------------------------------------------------------------
    def _hybrid_search(self, query: str, top: int) -> list[Document]:
        """对单条 query 执行 BM25 + 向量两路检索，使用 RRF 融合排序。"""
        q = (query or "").strip()
        if not q:
            return []

        # 向量检索
        vector_docs = list(self._vs.similarity_search(q, k=top))

        # BM25 关键词检索
        bm25_results: list[tuple[Document, float]] = []
        if self._bm25.is_ready:
            bm25_results = self._bm25.search(q, k=top)

        # 构建 doc → 统一标识的映射，用于 RRF 累加
        doc_by_key: dict[int, Document] = {}
        rrf_score: dict[int, float] = {}

        def _add(rank: int, doc: Document, weight: float):
            key = self._doc_dedupe_key(doc)
            if key not in doc_by_key:
                doc_by_key[key] = doc
            rrf_score[key] = rrf_score.get(key, 0.0) + weight / (self._rrf_k + rank)

        vec_weight = 1.0 - self._bm25_weight
        bm25_weight = self._bm25_weight

        for rank, d in enumerate(vector_docs, start=1):
            _add(rank, d, vec_weight)
        for rank, (d, _score) in enumerate(bm25_results, start=1):
            _add(rank, d, bm25_weight)

        # 按 RRF 得分降序排列
        sorted_keys = sorted(rrf_score.keys(), key=lambda k: rrf_score[k], reverse=True)
        return [doc_by_key[k] for k in sorted_keys[:top]]

    # ------------------------------------------------------------------
    # 双路混合检索 → 合并 → Rerank 精排
    # ------------------------------------------------------------------
    def retriever_docs(self, retrieval_query: str, user_question: Optional[str] = None) -> list[Document]:
        """
        双路混合检索：先用识别后的检索用语检索，再并入用户原句检索结果，交替合并去重。
        合并后的候选集送入 Rerank 模型精排，最终保留至多 self._k 条送入 LLM。

        Rerank 候选池需远大于最终 k，因此 inner_k 取 k*3 和 30 的较大值，
        确保两路合并后有足够多的候选供精排模型筛选。
        """
        uq = (user_question or "").strip()
        rq = (retrieval_query or "").strip()
        inner_k = max(self._k * 3, 30)

        rq_docs = self._hybrid_search(rq, inner_k) if rq else []
        uq_docs = self._hybrid_search(uq, inner_k) if uq else []

        # 单路场景：直接对一组候选 rerank
        if not rq or rq == uq:
            source_docs = uq_docs
        else:
            # 双路交替合并去重
            merged: list[Document] = []
            seen: set[int] = set()
            for a, b in zip_longest(rq_docs, uq_docs):
                for d in (a, b):
                    if d is None:
                        continue
                    key = self._doc_dedupe_key(d)
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(d)
            source_docs = merged

        # Rerank 精排：用用户原句作为 rerank 查询文本，更贴近用户真实意图
        if self._rerank_enabled and len(source_docs) > self._k:
            return self._reranker.rerank(uq or rq, source_docs, top_n=self._k)

        return source_docs[: self._k]

    @staticmethod
    def _build_context(context_docs: list[Document], user_question: str = "") -> str:
        # Lost-in-middle 缓解：头尾交替排列，高相关文档放首位和末位
        reordered: list[Document] = []
        mid = len(context_docs)
        left, right = 0, mid - 1
        toggle = True
        while left <= right:
            if toggle:
                reordered.append(context_docs[left])
                left += 1
            else:
                reordered.append(context_docs[right])
                right -= 1
            toggle = not toggle

        context = ""
        for i, doc in enumerate(reordered, start=1):
            text = doc.page_content.strip()
            source = doc.metadata.get("source", "") if doc.metadata else ""
            source_name = os.path.basename(source) if source else "未知来源"
            context += f"【片段{i} | 来源：{source_name}】\n{text}\n\n"
        if not context.strip():
            return (
                "（未检索到任何知识库片段：向量集合可能为空，或尚未将 data 目录下 pdf/txt 全量入库。"
                "请将你自己的业务文档放入 data（含子目录），运行全量重建向量库后再提问。）"
            )
        return context

    def refine_retrieval_query(self, user_question: str) -> str:
        """识别用户意图并生成用于向量检索的用语。"""
        text = (self._query_understand_chain.invoke({"user_question": user_question}) or "").strip()
        if not text:
            return user_question.strip()
        return text[:500]

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        if not history:
            return "（无）"
        lines: list[str] = []
        for h in history[-10:]:  # 最多取最近 5 轮（10 条）
            role = "用户" if h.get("role") == "user" else "助手"
            text = (h.get("content") or "").strip()
            lines.append(f"{role}：{text}")
        return "\n".join(lines)

    def generate_answer(
        self,
        user_question: str,
        retrieve_query: str,
        context_docs: list[Document],
        chat_history: Optional[list[dict]] = None,
    ) -> str:
        context = self._build_context(context_docs, user_question)
        return self.chain.invoke(
            {
                "input": user_question,
                "retrieve_query": retrieve_query,
                "context": context,
                "chat_history": self._format_history(chat_history or []),
                "fragment_count": len(context_docs),
            }
        )

    def rag_summarize(self, query: str, chat_history: Optional[list[dict]] = None) -> str:
        rq = self.refine_retrieval_query(query)
        docs = self.retriever_docs(rq, query)
        return self.generate_answer(query, rq, docs, chat_history=chat_history)


if __name__ == "__main__":
    rag = RagSummarizeService()

    print(rag.rag_summarize("这个知识库项目是做什么的"))
