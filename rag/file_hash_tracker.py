"""
文件哈希跟踪与增量索引。

每次入库前计算各文件的 MD5，与上次记录对比：
- 新文件 / 已修改 → 清掉旧分片后重新入库
- 未变化 → 跳过
- 已删除 → 清掉旧分片

哈希记录持久化到 faiss_db/file_hashes.yml，与向量库在同一目录下。
"""

import hashlib
import os
from typing import Optional

import yaml

from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class FileHashTracker:
    """文件哈希跟踪器，用于增量索引的去重判断。"""

    def __init__(self):
        persist_dir = get_abs_path("faiss_db")
        os.makedirs(persist_dir, exist_ok=True)
        self._tracker_path = os.path.join(persist_dir, "file_hashes.yml")
        self._current: dict[str, str] = {}
        self._previous: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 哈希计算
    # ------------------------------------------------------------------
    @staticmethod
    def compute_md5(filepath: str) -> Optional[str]:
        try:
            h = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            logger.warning(f"[增量索引] 无法计算 {filepath} 的 MD5，将视为新文件")
            return None

    # ------------------------------------------------------------------
    # 加载 / 保存
    # ------------------------------------------------------------------
    def load_previous(self) -> None:
        if os.path.exists(self._tracker_path):
            try:
                with open(self._tracker_path, "r", encoding="utf-8") as f:
                    self._previous = yaml.safe_load(f) or {}
            except Exception:
                logger.warning("[增量索引] 无法加载哈希记录，将全量重建")
                self._previous = {}
        else:
            self._previous = {}

    def save(self) -> None:
        with open(self._tracker_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._current, f, allow_unicode=True)
        logger.info(f"[增量索引] 哈希记录已保存到 {self._tracker_path}")

    # ------------------------------------------------------------------
    # 差异对比
    # ------------------------------------------------------------------
    def scan(self, file_paths: list[str]) -> tuple[list[str], list[str], list[str]]:
        """
        扫描文件列表，返回 (new_or_changed, unchanged, deleted)。
        - new_or_changed: 新增或内容变化的文件
        - unchanged: 内容未变的文件
        - deleted: 上次存在但本次不在列表中的文件
        """
        self.load_previous()
        self._current = {}
        new_or_changed: list[str] = []
        unchanged: list[str] = []

        for fp in sorted(file_paths):
            abs_path = os.path.abspath(fp)
            md5 = self.compute_md5(abs_path)
            if md5 is None:
                new_or_changed.append(fp)
                self._current[fp] = "__unknown__"
                continue
            self._current[fp] = md5

            prev_md5 = self._previous.get(fp)
            if prev_md5 == md5:
                unchanged.append(fp)
            else:
                new_or_changed.append(fp)

        deleted = [fp for fp in self._previous if fp not in self._current]

        return new_or_changed, unchanged, deleted

    def remove_file(self, file_path: str) -> None:
        """从哈希记录中移除单个文件，用于手动删除文档后保持一致性。"""
        self.load_previous()
        self._current = dict(self._previous)
        self._current.pop(file_path, None)
        self.save()
