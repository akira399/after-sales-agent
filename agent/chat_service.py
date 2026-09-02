"""Agent 服务层 — 封装一次完整的 Agent 问答流程，以事件流形式产出。

事件契约（按顺序）：
  stream_start  →  tool_call*  →  tool_result*  →  final_answer?  →  stream_end

会话隔离设计：
  不使用 @st.cache_resource 做进程级缓存（否则多访问者共享同一 Agent/API Key）。
  Agent 与 RAG 实例缓存在 st.session_state，按模型配置指纹失效重建，
  保证每位访问者只用自己的 Key。
"""
from typing import Generator

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage

from agent.react_agent import ReactAgent
from model.runtime_config import ModelConfig, get_session_config
from rag.rag_service import RagSummarizeService
from utils.logger_handler import logger


def _ensure_session_components() -> tuple[ReactAgent, ModelConfig]:
    """确保当前会话已构建 Agent 与 RAG 实例（按配置指纹重建）。

    返回 (agent, config)。进程内可缓存的最重资源是向量索引，
    这里按会话缓存 agent/rag，指纹变化才重建。
    """
    cfg = get_session_config()
    fp = cfg.fingerprint()

    agent = st.session_state.get("agent_obj")
    if agent is not None and st.session_state.get("agent_fp") == fp:
        return agent, cfg

    logger.info(f"[chat_service] 按会话配置重建 Agent/RAG（{cfg.display_label()}）")
    # 先构建 RAG（search_policy 工具通过 session_state["rag_obj"] 获取当前会话实例）
    rag = RagSummarizeService(model_config=cfg)
    st.session_state["rag_obj"] = rag
    # 再构建 Agent
    agent = ReactAgent(model_config=cfg)
    st.session_state["agent_obj"] = agent
    st.session_state["agent_fp"] = fp
    return agent, cfg


def _get_session_agent() -> ReactAgent:
    """兼容性入口：返回当前会话 Agent（会话不存在时按默认配置创建）。"""
    agent, _ = _ensure_session_components()
    return agent


def run_agent(
    query: str,
    history: list[dict],
    image_base64: str | None = None,
    image_mime: str = "image/jpeg",
) -> Generator[dict, None, None]:
    """执行一次 Agent 问答，逐事件 yield。调用方通过事件驱动 UI 渲染。

    始终以 stream_end 事件收尾，调用方据此判断流是否完整结束。
    """
    agent, cfg = _ensure_session_components()

    try:
        final_content = ""

        for chunk in agent.agent.stream(
            {"messages": agent._build_messages(query, history, image_base64, image_mime)},
            stream_mode="values",
            context={"report": False},
        ):
            messages = chunk.get("messages", [])
            if not messages:
                continue

            for msg in messages:
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        yield {
                            "type": "tool_call",
                            "call_id": tc.get("id", ""),
                            "name": tc.get("name", "?"),
                            "args": tc.get("args", {}),
                        }

                if isinstance(msg, ToolMessage):
                    yield {
                        "type": "tool_result",
                        "call_id": getattr(msg, "tool_call_id", ""),
                        "name": getattr(msg, "name", "?"),
                        "result": str(msg.content) if msg.content else "(空)",
                    }

            # 持续收集最终回答，但不 yield — stream_mode="values" 会重复推送同一条
            # 消息，在循环内 yield 会导致页面重复渲染
            last = messages[-1]
            if isinstance(last, AIMessage) and last.content:
                fc = getattr(last, "tool_calls", None)
                if not fc:
                    final_content = last.content

        # 流结束后，统一发一次 final_answer
        if final_content:
            yield {"type": "final_answer", "content": final_content}

        yield {"type": "stream_end", "status": "ok"}

    except Exception as e:
        logger.error(f"[chat_service] Agent 流异常: {e}", exc_info=True)
        yield {"type": "stream_end", "status": "error", "error": str(e)}
