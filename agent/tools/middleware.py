from typing import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger
from utils.prompt_loader import load_ticket_prompt

# 敏感字段：日志打印前对含这些 key 的值做脱敏，防止 API Key 等写入日志
_SENSITIVE_KEYS = ("api_key", "apikey", "key", "token", "secret", "authorization", "password")


def _sanitize(value, depth: int = 0) -> object:
    """递归脱敏：任何 dict/list 中出现敏感键名对应值一律打码。"""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(k, str) and k.lower() in _SENSITIVE_KEYS and depth < 3:
                out[k] = "***REDACTED***"
            else:
                out[k] = _sanitize(v, depth + 1)
        return out
    if isinstance(value, list):
        return [_sanitize(v, depth + 1) for v in value]
    return value


def _safe_repr(value, max_len: int = 500) -> str:
    """转字符串并截断，避免超长内容刷屏日志。"""
    try:
        s = str(_sanitize(value))
    except Exception:
        s = "<unprintable>"
    return s[:max_len] if len(s) > max_len else s


# 缓存工单提示词文本，避免重复读取文件
_TICKET_PROMPT = load_ticket_prompt()


@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{_safe_repr(request.tool_call.get('args', {}))}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        # 工单模式：调用该工具后翻转上下文标记，下轮模型调用切换为工单生成角色
        if request.tool_call['name'] == "fill_context_for_ticket":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e


@before_model
def log_before_model(
        state: AgentState,
        runtime: Runtime,
) -> dict | None:
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")

    last_msg = state['messages'][-1]
    last_content = ""
    if hasattr(last_msg, 'content') and last_msg.content:
        content = last_msg.content
        if isinstance(content, list):
            # 多模态消息（图片+文字），提取文本部分用于日志
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict)]
            last_content = " ".join(text_parts).strip()
        else:
            last_content = str(content).strip()
    logger.debug(f"[log_before_model]{type(last_msg).__name__} | {_safe_repr(last_content, max_len=300)}")

    # 工单模式：替换 system prompt 为工单生成角色
    if runtime.context.get("report"):
        existing = state.get("messages", [])
        filtered = [m for m in existing if not isinstance(m, SystemMessage)]
        new_system = SystemMessage(content=_TICKET_PROMPT)
        logger.info("[log_before_model] 已切换到工单生成模式提示词")
        return {"messages": [new_system] + filtered}

    return None
