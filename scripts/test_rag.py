"""RAG 政策问答冒烟测试：验证 改写→混合检索→Rerank→作答 全管线。

用法：python scripts/test_rag.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.rag_service import RagSummarizeService

QUESTIONS = [
    "我买的耳机进水了，能保修吗？",
    "七天无理由退货的七天是从哪天开始算的？",
    "退货的运费应该谁承担？",
    "我签收了5天，现在想无理由退货还来得及吗？假设今天是签收后第6天。",
]


def main() -> None:
    rag = RagSummarizeService()
    print(f"[INFO] RAG 服务已加载 | k={rag._k} | rerank={rag._rerank_enabled}")

    for q in QUESTIONS:
        print("\n" + "=" * 60)
        print(f"问题：{q}")

        rq = rag.refine_retrieval_query(q)
        print(f"改写检索词：{rq}")

        docs = rag.retriever_docs(rq, q)
        print(f"命中片段数：{len(docs)}")
        for i, d in enumerate(docs[:3], 1):
            src = os.path.basename(d.metadata.get("source", ""))
            print(f"  [{i}] {src} | {d.page_content[:70].strip()}...")

        ans = rag.generate_answer(q, rq, docs, chat_history=None)
        print(f"\n回答：{ans}")


if __name__ == "__main__":
    main()
