r"""
「售后小蜜」电商售后智能客服 Agent —— Streamlit 入口。

Run:
    streamlit run agent_app.py

模型接入：
    打开页面即可看到对话框。未配置模型前发送消息会提示先配置。
    - 部署者已在 .env / Secrets 预置 Key：打开即可使用（平台 Key）；
    - 未预置 Key（公开 Demo）：在侧栏填写自己的 DashScope / OpenAI 兼容 Key。
    Key 仅存于当前浏览器会话，不落盘、不写日志。
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


def _render_setup_hint() -> None:
    """模型未就绪时的引导横幅（不阻断对话框）。"""
    source = model_panel.config_source_label()
    if model_panel.is_model_ready():
        return
    st.warning(
        "#### 👋 开始前，先配置模型\n\n"
        "当前尚未检测到可用的 API Key。请在左侧 **⚙️ 模型接入** 面板中填入你自己的 "
        "**DashScope（阿里云百炼）** 或 **OpenAI 兼容** API Key 后即可对话。\n\n"
        "> 安全说明：Key 仅保存在当前浏览器会话中，服务器不存储、不记日志；"
        "多人访问可各用各的 Key，互不串用。"
    )


def main() -> None:
    st.set_page_config(page_title="售后小蜜 · Agent 智能客服", layout="wide")

    # 侧栏：模型接入面板 + 可用工具
    model_panel.render_model_panel()
    render.render_sidebar()

    session.init()
    history = session.get_messages()

    _render_welcome()
    faiss_abs = get_abs_path(faiss_conf.get("persist_directory", "faiss_db"))
    data_abs = get_abs_path(faiss_conf.get("data_path", "data"))
    st.caption(
        f"知识库：`{data_abs}` | 索引：`{faiss_abs}` | "
        f"k={faiss_conf.get('k', '')} | 模型：{model_panel.config_source_label()}"
    )

    # 未配置模型时展示引导（但不隐藏对话框）
    _render_setup_hint()

    # 聊天历史
    render.render_chat_history(history)

    # 图片附件
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

    # 发送前校验模型配置：未配置则明确提示并保留输入（不调用 Agent）
    if not model_panel.is_model_ready():
        st.error("模型尚未配置，请先在左侧 ⚙️ 模型接入 面板填写 API Key。")
        session.add_user_message(user_input, user_input)  # 保留用户输入可见
        st.stop()

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
