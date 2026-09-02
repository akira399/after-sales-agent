import os
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):
    """
    递归列出目录下（含子目录）所有符合后缀的文件路径。
    解决知识库文件放在 data 子文件夹时无法被扫描入库的问题。
    """
    files = []

    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return ()

    for root, _dirs, filenames in os.walk(path):
        for f in filenames:
            if f.endswith(allowed_types):
                files.append(os.path.join(root, f))

    return tuple(sorted(files))


def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    return PyPDFLoader(filepath, passwd).load()


def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath, encoding="utf-8").load()
