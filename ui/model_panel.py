"""模型接入面板 —— 让访问者填入自己的 API Key 来使用本应用。

安全说明：
- API Key 仅写入 st.session_state（浏览器会话内存），不落盘、不打日志、不进 git。
- 会话刷新/关闭后 Key 即消失；多人访问各用自己的 Key，互不串用。
- 若部署者在 .env / 云端 Secrets 中预置了 Key，仅作为"平台演示额度"供访问者
  显式选择启用，默认引导访问者填写自己的 Key，避免消耗部署者额度。
"""
import requests
import streamlit as st

from model.factory import DASHSCOPE_URL_FALLBACK
from model.runtime_config import (
    ModelConfig,
    get_session_config,
    has_platform_key,
    save_session_config,
    use_platform_config,
)

# 常见预设
_PRESETS = {
    "DashScope(阿里云百炼)": {
        "base_url": DASHSCOPE_URL_FALLBACK,
        "chat_model": "qwen3.7-plus",
        "embed_model": "text-embedding-v4",
        "rerank_model": "qwen3-rerank",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "chat_model": "gpt-4o-mini",
        "embed_model": "text-embedding-3-small",
        "rerank_model": "",
    },
}


def _test_chat_connection(cfg: ModelConfig) -> tuple[bool, str]:
    """向 OpenAI 兼容端点发一次最小对话，验证 Key 可用性。"""
    if not cfg.api_key.strip():
        return False, "尚未填写 API Key。"
    url = (cfg.base_url or DASHSCOPE_URL_FALLBACK).rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg.chat_model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 5,
    }
    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {cfg.api_key.strip()}"},
            timeout=15,
        )
    except requests.exceptions.RequestException as e:
        return False, f"网络错误：{e}"
    if resp.status_code == 200:
        return True, "连接成功，API Key 有效。"
    try:
        msg = resp.json().get("error", {}).get("message", resp.text[:120])
    except Exception:
        msg = resp.text[:120]
    return False, f"HTTP {resp.status_code}：{msg}"


def _session_config() -> ModelConfig | None:
    """仅返回访问者主动填写的配置（不含平台 fallback）。"""
    if "model_config" in st.session_state:
        return st.session_state["model_config"]
    return None


def config_source_label() -> str:
    """当前生效配置来源标识（供页面顶部展示）。"""
    cfg = get_session_config()
    if not cfg.is_ready():
        return "未配置"
    own = _session_config()
    if own is not None and own.api_key:
        return f"访问者自带 Key：{cfg.display_label()}"
    return f"平台预置 Key：{cfg.display_label()}"


def is_model_ready() -> bool:
    """是否有可用配置（访问者自填或显式启用平台 Key 均可）。"""
    return get_session_config().is_ready()


def has_platform_available() -> bool:
    """部署者是否预置 Key 且当前会话未自填。"""
    return has_platform_key() and _session_config() is None


def render_model_panel() -> None:
    """渲染模型接入面板（放在侧栏）。"""
    st.sidebar.subheader("⚙️ 模型接入")

    # ---- 当前来源状态提示 ----
    own = _session_config()
    if own is not None and own.api_key:
        st.sidebar.success(f"当前使用 **你填写的 Key**（{own.chat_model}）")
    elif has_platform_available():
        st.sidebar.warning(
            "当前将使用**平台预置 Key**。若这是公开演示，访问者会消耗部署者额度。"
        )
    else:
        st.sidebar.warning("尚未配置模型。请填写你自己的 API Key 后使用。")

    # ---- 访问者自填表单 ----
    with st.sidebar.expander("🔑 填写自己的 API Key", expanded=own is None):
        preset = st.selectbox(
            "服务商预设", list(_PRESETS.keys()), key="mp_preset",
            help="选择预设自动填充默认地址与模型名，仍可手动修改。",
        )
        p = _PRESETS[preset]

        with st.form("model_config_form", clear_on_submit=False):
            base_url = st.text_input("Base URL", value=p["base_url"], key="mp_base_url")
            api_key = st.text_input(
                "API Key", type="password",
                placeholder="sk-...",
                key="mp_api_key",
                help="仅存于本浏览器会话，不落盘、不进日志。",
            )
            chat_model = st.text_input("对话模型", value=p["chat_model"], key="mp_chat")
            embed_model = st.text_input("Embedding 模型", value=p["embed_model"], key="mp_embed")
            rerank_model = st.text_input(
                "Rerank 模型（可留空=不精排）", value=p["rerank_model"], key="mp_rerank"
            )

            c1, c2 = st.columns(2)
            test_clicked = c1.form_submit_button("测试连接", type="secondary")
            save_clicked = c2.form_submit_button("保存并使用", type="primary")

        if test_clicked:
            tmp = ModelConfig(
                base_url=base_url.strip() or p["base_url"],
                api_key=api_key.strip(),
                chat_model=chat_model.strip() or p["chat_model"],
                embed_model=embed_model.strip() or p["embed_model"],
                rerank_model=rerank_model.strip(),
            )
            ok, msg = _test_chat_connection(tmp)
            if ok:
                st.sidebar.success(msg)
            else:
                st.sidebar.error(msg)

        if save_clicked:
            if not api_key.strip():
                st.sidebar.error("请填写 API Key。")
            else:
                cfg = ModelConfig(
                    base_url=base_url.strip() or p["base_url"],
                    api_key=api_key.strip(),
                    chat_model=chat_model.strip() or p["chat_model"],
                    embed_model=embed_model.strip() or p["embed_model"],
                    rerank_model=rerank_model.strip(),
                )
                save_session_config(cfg)
                st.sidebar.success("已保存，Agent 将按你的配置重建。")
                st.rerun()

    # ---- 明确启用平台预置 Key ----
    if has_platform_available():
        if st.sidebar.button("改用平台预置 Key", use_container_width=True):
            use_platform_config()
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "安全提示：Key 只保存在你的浏览器会话，服务器不存储。刷新页面后需重新填写。"
    )
