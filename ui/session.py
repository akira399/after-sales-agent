"""会话状态管理 — 统一管理 st.session_state 的读写，页面层不需要知道键名。"""
import os

import streamlit as st

# ---- session_state 键名（集中定义，避免散落） ----
_KEY_MESSAGES = "agent_messages"
_KEY_ATTACHED_IMG = "_attached_img"
_KEY_LAST_UPLOAD = "_last_main_upload"


# ---- 初始化 ----
def init():
    if _KEY_MESSAGES not in st.session_state:
        st.session_state[_KEY_MESSAGES] = []


# ---- 消息读写 ----
def get_messages() -> list[dict]:
    return st.session_state[_KEY_MESSAGES]


def add_user_message(text: str, llm_text: str,
                     img_b64: str | None = None, img_mime: str = "image/jpeg"):
    msg: dict = {"role": "user", "content": llm_text, "text": text}
    if img_b64:
        msg["content"] = [
            {"type": "text", "text": llm_text},
            {"type": "image_url", "image_url": {"url": f"data:{img_mime};base64,{img_b64}"}},
        ]
        msg["image_base64"] = img_b64
        msg["image_mime"] = img_mime
    st.session_state[_KEY_MESSAGES].append(msg)


def add_assistant_message(content: str, annotated_images: list[str] | None = None):
    st.session_state[_KEY_MESSAGES].append({
        "role": "assistant",
        "content": content,
        "annotated_images": annotated_images or [],
    })


# ---- 图片附件 ----
def get_attached_image() -> str | None:
    path = st.session_state.get(_KEY_ATTACHED_IMG)
    if path and os.path.exists(path):
        return path
    return None


def set_attached_image(path: str | None):
    st.session_state[_KEY_ATTACHED_IMG] = path


def clear_attached_image():
    """清除附件状态（路径 + 上传名）+ 删除临时文件。"""
    path = st.session_state.get(_KEY_ATTACHED_IMG)
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
    st.session_state[_KEY_ATTACHED_IMG] = None
    st.session_state[_KEY_LAST_UPLOAD] = None


def get_last_upload_name() -> str | None:
    return st.session_state.get(_KEY_LAST_UPLOAD)


def set_last_upload_name(name: str):
    st.session_state[_KEY_LAST_UPLOAD] = name
