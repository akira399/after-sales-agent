"""UI 渲染层 — 只负责显示，不操作状态、不调用 Agent。"""
import base64
import os

import streamlit as st

# 工具名称中文映射
TOOL_LABELS: dict[str, str] = {
    "search_policy": "售后政策知识库检索",
    "query_order": "查询订单详情",
    "query_user_orders": "查询我的订单",
    "query_logistics": "查询物流轨迹",
    "query_refund": "查询退款进度",
    "check_return_window": "判定退货窗口(规则引擎)",
    "judge_freight": "判定运费承担方(规则引擎)",
    "check_warranty": "判定保修范围(规则引擎)",
    "get_user_id": "获取用户 ID",
    "get_current_date": "获取当前日期",
    "fill_context_for_ticket": "触发工单生成模式",
}


# ================================================================
# 侧栏
# ================================================================
def render_sidebar():
    st.sidebar.subheader(f"可用工具（{len(TOOL_LABELS)} 个）")
    for _, label in TOOL_LABELS.items():
        st.sidebar.caption(f"- {label}")


# ================================================================
# 历史消息
# ================================================================
def render_chat_history(messages: list[dict]):
    for msg in messages:
        with st.chat_message(msg["role"]):
            display_text = msg.get("text") or (
                msg["content"] if isinstance(msg["content"], str) else ""
            )
            st.markdown(display_text)
            if msg.get("image_base64"):
                st.image(base64.b64decode(msg["image_base64"]), width=200)
            for i, img_b64 in enumerate(msg.get("annotated_images", [])):
                img_bytes = base64.b64decode(img_b64)
                st.image(img_bytes)
                st.download_button(
                    label="下载标注图",
                    data=img_bytes,
                    file_name=f"detect_result_{i}.jpg",
                    mime="image/jpeg",
                    key=f"dl_hist_{i}_{hash(img_b64[:50])}",
                )


# ================================================================
# 用户消息
# ================================================================
def render_user_message(text: str, img_path: str | None = None):
    with st.chat_message("user"):
        st.markdown(text)
        if img_path and os.path.exists(img_path):
            st.image(img_path, width=200)


# ================================================================
# Agent 流式事件
# ================================================================
class AgentStreamRenderer:
    """消费 chat_service 产出的事件流，逐步渲染工具调用和最终回答。"""

    def __init__(self):
        self._shown_tool_calls: set[str] = set()
        self._shown_tool_results: set[str] = set()
        self.annotated_images: list[str] = []

    def handle(self, event: dict) -> str | None:
        """处理一个事件。返回最终回答文本（仅 final_answer 事件有值）。"""
        etype = event.get("type")

        if etype == "tool_call":
            self._render_tool_call(event["call_id"], event["name"], event["args"])
        elif etype == "tool_result":
            self._render_tool_result(
                event["call_id"], event["name"], event["result"]
            )
        elif etype == "final_answer":
            return self._render_final(event["content"])

        return None

    def _render_tool_call(self, call_id: str, name: str, args: dict):
        if call_id in self._shown_tool_calls:
            return
        self._shown_tool_calls.add(call_id)
        label = TOOL_LABELS.get(name, name)
        with st.status(f"调用工具：{label}", expanded=False):
            st.caption(f"**入参**：{args}")
            st.caption("状态：完成 ✓")

    def _render_tool_result(self, call_id: str, name: str, result_text: str):
        if call_id in self._shown_tool_results:
            return
        self._shown_tool_results.add(call_id)
        label = TOOL_LABELS.get(name, name)
        with st.expander(f"工具返回：{label}", expanded=False):
            st.text(result_text[:2000])

    @staticmethod
    def _render_final(content: str) -> str:
        st.markdown(content)
        return content


# ================================================================
# 图片附件提示
# ================================================================
def render_image_attachment(img_path: str, on_remove):
    c1, c2 = st.columns([0.9, 0.1])
    with c1:
        st.caption(f"📎 {os.path.basename(img_path)}")
    with c2:
        if st.button("✕", key="rm_img", help="移除图片"):
            on_remove()
