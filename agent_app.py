r"""
ReAct Agent Streamlit app entrypoint.

Run:
    streamlit run agent_app.py
"""

import base64
import os
import tempfile
import warnings

import streamlit as st

from utils.config_handler import faiss_conf
from utils.path_tool import get_abs_path
from utils.secrets_handler import load_cloud_secrets

warnings.filterwarnings("ignore", message=".*coroutine.*expire_cache.*")

# 同步 Streamlit Cloud secrets 到环境变量（必须在导入 chat_service 之前执行，
# 因为 model.factory 会在 import 期创建 chat_model 并读取 API Key）
load_cloud_secrets()

from agent import chat_service  # noqa: E402
from ui import render, session  # noqa: E402


def _encode_image(img_path: str) -> tuple[str | None, str]:
    ext = os.path.splitext(img_path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
    }
    mime = mime_map.get(ext, "image/jpeg")
    try:
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8"), mime
    except OSError:
        return None, "image/jpeg"


def _build_llm_text(user_input: str, img_path: str | None) -> tuple[str, str | None, str]:
    img_b64, img_mime = None, "image/jpeg"
    llm_text = user_input

    if img_path and os.path.exists(img_path):
        img_b64, img_mime = _encode_image(img_path)
        llm_text = (
            f"{user_input}\n\n"
            f"(The current conversation includes an attached image: {img_path}. "
            "Decide whether image classification or detection tools are needed.)"
        )

    return llm_text, img_b64, img_mime


def _save_uploaded_image(uploaded_img) -> None:
    last_name = session.get_last_upload_name()
    if uploaded_img.name == last_name:
        return

    tmp_dir = tempfile.gettempdir()
    new_path = os.path.join(tmp_dir, f"chat_{uploaded_img.name}")
    with open(new_path, "wb") as f:
        f.write(uploaded_img.getbuffer())

    session.set_attached_image(new_path)
    session.set_last_upload_name(uploaded_img.name)
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="ReAct Agent 问答", layout="wide")
    st.title("ReAct Agent 智能问答")

    faiss_abs = get_abs_path(faiss_conf.get("persist_directory", "faiss_db"))
    data_abs = get_abs_path(faiss_conf.get("data_path", "data"))
    st.caption(
        f"向量库：`{faiss_abs}` | 数据：`{data_abs}` | "
        f"k={faiss_conf.get('k', '')} | 多轮对话 | 报告模式 | CV 推理"
    )

    render.render_sidebar()

    session.init()
    render.render_chat_history(session.get_messages())

    attached = session.get_attached_image()
    if attached and os.path.exists(attached):
        render.render_image_attachment(
            attached,
            on_remove=lambda: (session.clear_attached_image(), st.rerun()),
        )

    if not attached:
        uploaded_img = st.file_uploader(
            "附加图片（可选）",
            type=["jpg", "jpeg", "png", "bmp"],
            key="main_img_upload",
            label_visibility="visible",
        )
        if uploaded_img is not None:
            _save_uploaded_image(uploaded_img)

    user_input = st.chat_input("输入问题，Agent 会自主思考并调用工具")
    if not user_input:
        return

    img_path = session.get_attached_image()
    llm_text, img_b64, img_mime = _build_llm_text(user_input, img_path)

    session.add_user_message(user_input, llm_text, img_b64, img_mime)
    render.render_user_message(user_input, img_path)

    history = session.get_messages()[:-1]
    final_answer = ""

    with st.chat_message("assistant"):
        event_renderer = render.AgentStreamRenderer()
        for event in chat_service.run_agent(llm_text, history, img_b64, img_mime):
            if event.get("type") == "stream_end":
                break
            result = event_renderer.handle(event)
            if result:
                final_answer = result

    if final_answer:
        session.add_assistant_message(final_answer, event_renderer.annotated_images)
    else:
        st.warning("Agent 未产出最终回答，请检查配置后重试。")

    session.clear_attached_image()
    st.rerun()


if __name__ == "__main__":
    main()
