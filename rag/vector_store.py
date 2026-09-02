import os
import pickle

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from utils.config_handler import faiss_conf
from model.factory import build_embed_model
from model.runtime_config import get_session_config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type
from utils.logger_handler import logger


class VectorStoreService:
    def __init__(self):
        self.persist_directory = get_abs_path(faiss_conf["persist_directory"])
        os.makedirs(self.persist_directory, exist_ok=True)

        try:
            self.vector_store = self._load_index()
            logger.info(f"[VectorStoreService] 从 FAISS 加载向量库：{self.persist_directory}")
        except FileNotFoundError:
            self.vector_store = self._create_empty_faiss()
            logger.info(f"[VectorStoreService] 创建新的 FAISS 向量库：{self.persist_directory}")
        except Exception as e:
            # 跨 Python 版本/平台 pickle 不兼容等异常：优雅降级为空库，避免应用崩溃
            logger.warning(f"[VectorStoreService] 加载索引失败({e})，回退空向量库，可运行 load_document() 重建")
            self.vector_store = self._create_empty_faiss()

        logger.info(f"[VectorStoreService] 向量库持久化目录（绝对路径）：{self.persist_directory}")

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=faiss_conf["chunk_size"],
            chunk_overlap=faiss_conf["chunk_overlap"],
            separators=faiss_conf["separators"],
            length_function=len,
        )

        self._all_chunks: list[Document] = []
        self._chunks_loaded = False

    # ------------------------------------------------------------------
    # 序列化（使用 Python I/O 绕过 FAISS C++ 中文路径问题）
    # ------------------------------------------------------------------
    def _index_path(self) -> str:
        return os.path.join(self.persist_directory, "faiss_index.pkl")

    def _save_index(self) -> None:
        with open(self._index_path(), "wb") as f:
            pickle.dump({
                "index": faiss.serialize_index(self.vector_store.index),
                "docstore": self.vector_store.docstore,
                "index_to_docstore_id": self.vector_store.index_to_docstore_id,
            }, f)

    def _load_index(self) -> FAISS:
        if not os.path.exists(self._index_path()):
            raise FileNotFoundError("FAISS index file not found")
        with open(self._index_path(), "rb") as f:
            data = pickle.load(f)
        return FAISS(
            embedding_function=build_embed_model(),
            index=faiss.deserialize_index(data["index"]),
            docstore=data["docstore"],
            index_to_docstore_id=data["index_to_docstore_id"],
        )

    @staticmethod
    def _create_empty_faiss() -> FAISS:
        dim = 1024  # DashScope text-embedding-v4 dimension
        index = faiss.IndexFlatL2(dim)
        docstore = InMemoryDocstore()
        index_to_docstore_id: dict[int, str] = {}
        return FAISS(
            embedding_function=build_embed_model(),
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

    def _ensure_chunks_loaded(self) -> None:
        if self._chunks_loaded:
            return
        self._all_chunks = self._load_all_chunks_from_db()
        self._chunks_loaded = True

    def _load_all_chunks_from_db(self) -> list[Document]:
        try:
            docs = list(self.vector_store.docstore._dict.values())
            if docs:
                logger.info(f"[VectorStoreService] 从 FAISS 恢复 {len(docs)} 条分片")
            return docs
        except Exception:
            logger.warning("[VectorStoreService] 无法从 FAISS 读取分片，从空集合开始")
            return []

    def get_all_chunks(self) -> list[Document]:
        self._ensure_chunks_loaded()
        return list(self._all_chunks)

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": faiss_conf["k"]})

    # ------------------------------------------------------------------
    # 文件级操作
    # ------------------------------------------------------------------
    def _delete_file_vectors(self, file_path: str) -> int:
        ids_to_delete = []
        for idx_key, doc_id in list(self.vector_store.index_to_docstore_id.items()):
            doc = self.vector_store.docstore._dict.get(doc_id)
            if doc and doc.metadata.get("source") == file_path:
                ids_to_delete.append(idx_key)
        if ids_to_delete:
            self.vector_store.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def _get_file_documents(self, read_path: str) -> list[Document]:
        if read_path.endswith("txt"):
            return txt_loader(read_path)
        if read_path.endswith("pdf"):
            return pdf_loader(read_path)
        return []

    def _ingest_file(self, path: str, cancel_check=None) -> list[Document]:
        documents = self._get_file_documents(path)
        if not documents:
            return []
        split_docs = self.spliter.split_documents(documents)
        if not split_docs:
            return []

        batch_size = 20
        for i in range(0, len(split_docs), batch_size):
            if cancel_check and cancel_check():
                logger.warning(f"[加载知识库] 用户取消嵌入 {path}（已入库 {i} 条）")
                self._delete_file_vectors(path)
                return []
            batch = split_docs[i : i + batch_size]
            self.vector_store.add_documents(batch)

        self._save_index()

        return split_docs

    # ------------------------------------------------------------------
    # 全量重建
    # ------------------------------------------------------------------
    def _clear_all_vectors(self) -> None:
        total = len(self.vector_store.docstore._dict)
        if total:
            self.vector_store = self._create_empty_faiss()
            self._save_index()
            logger.info(f"[加载知识库] 已清空集合内原有向量，共 {total} 条")

    # ------------------------------------------------------------------
    # 增量索引（基于文件哈希）
    # ------------------------------------------------------------------
    def load_document(self, cancel_check=None) -> None | str:
        from rag.file_hash_tracker import FileHashTracker

        self._ensure_chunks_loaded()

        if cancel_check is None:
            cancel_check = lambda: False

        data_dir = get_abs_path(faiss_conf["data_path"])
        allowed_types = tuple(faiss_conf["allow_knowledge_file_type"])
        allowed_files_path: list[str] = list(listdir_with_allowed_type(data_dir, allowed_types))

        if not allowed_files_path:
            logger.warning(f"[加载知识库] 目录 {faiss_conf['data_path']} 下未发现允许的文档文件")
            return

        tracker = FileHashTracker()
        new_or_changed, unchanged, deleted = tracker.scan(allowed_files_path)

        # 无历史记录 → 全量重建
        if not tracker._previous:
            logger.info("[增量索引] 未发现哈希记录，执行全量重建")
            self._clear_all_vectors()
            self._all_chunks = []
            for path in allowed_files_path:
                if cancel_check():
                    logger.warning("[增量索引] 用户取消索引")
                    return "cancelled"
                try:
                    split_docs = self._ingest_file(path, cancel_check)
                    self._all_chunks.extend(split_docs)
                    logger.info(f"[加载知识库] {path} 入库成功，分片数 {len(split_docs)}")
                except Exception as e:
                    logger.error(f"[加载知识库] {path} 加载失败：{str(e)}", exc_info=True)
            tracker.save()
            self._rebuild_bm25()
            return

        total_added = 0
        total_removed = 0

        # 删除已移除的文件
        for fp in deleted:
            n = self._delete_file_vectors(fp)
            self._all_chunks = [d for d in self._all_chunks if d.metadata.get("source") != fp]
            total_removed += n
            logger.info(f"[增量索引] 已删除 {fp}（{n} 条分片）")

        # 更新已修改的文件
        for fp in new_or_changed:
            if cancel_check():
                logger.warning("[增量索引] 用户取消索引")
                return "cancelled"
            try:
                n = self._delete_file_vectors(fp)
                total_removed += n
                self._all_chunks = [d for d in self._all_chunks if d.metadata.get("source") != fp]
                split_docs = self._ingest_file(fp, cancel_check)
                self._all_chunks.extend(split_docs)
                total_added += len(split_docs)
                logger.info(f"[增量索引] {fp} 已更新（移除 {n} 条 → 新增 {len(split_docs)} 条）")
            except Exception as e:
                logger.error(f"[增量索引] {fp} 更新失败：{str(e)}", exc_info=True)

        if unchanged:
            logger.info(f"[增量索引] {len(unchanged)} 个文件未变化，跳过")

        logger.info(
            f"[增量索引] 完成 — 新增/更新 {total_added} 条，移除 {total_removed} 条，"
            f"当前总计 {len(self._all_chunks)} 条"
        )

        tracker.save()
        self._save_index()

        if new_or_changed or deleted:
            self._rebuild_bm25()

    def delete_document(self, file_path: str) -> int:
        self._ensure_chunks_loaded()
        abs_path = os.path.abspath(file_path)
        n = self._delete_file_vectors(abs_path)
        self._all_chunks = [d for d in self._all_chunks if d.metadata.get("source") != abs_path]

        from rag.file_hash_tracker import FileHashTracker
        tracker = FileHashTracker()
        tracker.remove_file(abs_path)

        if os.path.exists(abs_path):
            os.remove(abs_path)

        if n > 0:
            self._save_index()
            self._rebuild_bm25()
        logger.info(f"[删除文档] {file_path} — 移除 {n} 条分片")
        return n

    def _rebuild_bm25(self) -> None:
        if self._all_chunks:
            from rag.bm25_index import Bm25IndexService
            bm25 = Bm25IndexService()
            bm25.build(self._all_chunks)
            bm25.save()
            logger.info(f"[加载知识库] BM25 索引已同步重建，共 {len(self._all_chunks)} 条")


if __name__ == "__main__":
    import sys
    vs = VectorStoreService()
    vs.load_document()

    retriever = vs.get_retriever()
    res = retriever.invoke("知识库")
    for r in res:
        text = r.page_content.encode("gbk", errors="replace").decode("gbk", errors="replace")
        print(text)
        print("-" * 20)
    print(f"\n检索完成，共返回 {len(res)} 条结果", file=sys.stderr)
