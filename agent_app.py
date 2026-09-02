r"""
「售后小蜜」电商售后智能客服 Agent —— Streamlit 入口。

Run:
    streamlit run agent_app.py

模型接入：
    访问者可在侧栏填写自己的 DashScope / OpenAI 兼容 API Key（仅存于本浏览器会话）；
    或（在部署者预置 Key 时）显式启用平台 Key。未配置前不开放问答，避免误耗额度。
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

# 同步 Streamlit Cloud secrets 到环境变量（必须在导入 chat_service 之前执行）
load_cloud_secrets()

from ui import model_panel, render, session  # noqa: E402
from agent import chat_service  # noqa: E402


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
            "You may mention you've received the image, but image analysis tools "
            "are not available in this demo.)"
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


def _render_welcome() -> None:
    st.title("🛒 售后小蜜 · 电商售后智能客服 Agent")
    st.caption(
        "基于 LangGraph ReAct 编排多工具，混合检索 RAG 政策知识库，"
        "规则引擎兜底合规判定。"
    )


def main() -> None:
    st.set_page_config(page_title="售后小蜜 · Agent 智能客服", layout="wide")

    # 侧栏：模型接入面板（key 仅存会话）
    model_panel.render_model_panel()

    # 侧栏：可用工具清单
    render.render_sidebar()

    session.init()
    history = session.get_messages()

    # 顶部状态栏
    _render_welcome()
    faiss_abs = get_abs_path(faiss_conf.get("persist_directory", "faiss_db"))
    data_abs = get_abs_path(faiss_conf.get("data_path", "data"))
    st.caption(
        f"知识库：`{data_abs}` | 索引：`{faiss_abs}` | "
        f"k={faiss_conf.get('k', '')} | 模型来源：{model_panel.config_source_label()}"
    )

    if not model_panel.is_model_ready():
        # 引导页：未配置模型时展示项目说明，不开放问答
        st.info(
            "### 👋 欢迎体验「售后小蜜」\n\n"
            "这是一个**多工具售后客服 Agent**，可查询订单/物流/退款、检索售后政策知识库、"
            "并用规则引擎做合规判定（如七天无理由窗口）。\n\n"
            "在左侧 **⚙️ 模型接入** 面板中，填入你自己的 **DashScope（阿里云百炼）** 或 "
            "**OpenAI 兼容** API Key 即可开始。\n\n"
            "> 安全说明：Key 仅保存在当前浏览器会话中，服务器不存储、不记日志；"
            "多人访问可各用各的 Key，互不串用。"
        )
        try:
            has_plat = model_panel.has_platform_available()
        except Exception:
            has_plat = False
        if has_plat:
            st.caption("检测到部署者已预置平台 Key，可在侧栏选择'改用平台预置 Key'。")
        st.stop()

    render.render_chat_history(history)

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

    final_answer = ""
    with st.chat_message("assistant"):
        event_renderer = render.AgentStreamRenderer()
        try:
            for event in chat_service.run_agent(llm_text, history[:-1], img_b64, img_mime):
                if event.get("type") == "stream_end":
                    if event.get("status") != "ok":
                        st.error(event.get("error", "Agent 流异常"))
                    break
                result = event_renderer.handle(event)
                if result:
                    final_answer = result
        except Exception as e:
            st.error(f"Agent 执行出错：{e}")

    if final_answer:
        session.add_assistant_message(final_answer)
    else:
        st.warning("Agent 未产出最终回答，请检查模型配置后重试。")

    session.clear_attached_image()
    st.rerun()


if __name__ == "__main__":
    main()
