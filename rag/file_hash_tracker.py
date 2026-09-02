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
    # 相对路径 key（保证跨机器可移植，避免他人/云端 clone 后误判全量重建）
    # ------------------------------------------------------------------
    def _make_key(self, abs_path: str) -> str:
        try:
            rel = os.path.relpath(abs_path, get_abs_path("."))
        except ValueError:  # 跨盘符（Windows）无法 relpath，退回绝对路径
            return abs_path
        return rel.replace("\\", "/")

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
        - new_or_changed: 新增或内容变化的文件（原路径）
        - unchanged: 内容未变的文件（原路径）
        - deleted: 上次存在但本次不在列表中的文件（绝对路径）

        哈希记录的 key 统一使用相对项目根的路径（见 _make_key），
        保证在不同机器 / 云端部署时不会因绝对路径不同而误判全量重建。
        """
        self.load_previous()
        self._current = {}
        new_or_changed: list[str] = []
        unchanged: list[str] = []

        for fp in sorted(file_paths):
            abs_path = os.path.abspath(fp)
            key = self._make_key(abs_path)
            md5 = self.compute_md5(abs_path)
            if md5 is None:
                new_or_changed.append(fp)
                self._current[key] = "__unknown__"
                continue
            self._current[key] = md5

            prev_md5 = self._previous.get(key)
            if prev_md5 == md5:
                unchanged.append(fp)
            else:
                new_or_changed.append(fp)

        # deleted：把相对 key 转回绝对路径，便于调用方直接用于文件删除
        deleted = []
        for key in self._previous:
            if key not in self._current:
                if os.path.isabs(key):
                    deleted.append(key)  # 兼容历史绝对路径记录
                else:
                    deleted.append(get_abs_path(key))

        return new_or_changed, unchanged, deleted

    def remove_file(self, file_path: str) -> None:
        """从哈希记录中移除单个文件，用于手动删除文档后保持一致性。"""
        self.load_previous()
        self._current = dict(self._previous)
        key = self._make_key(os.path.abspath(file_path))
        # 同时兼容历史绝对路径记录
        self._current.pop(key, None)
        self._current.pop(os.path.abspath(file_path), None)
        self.save()
