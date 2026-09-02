import base64
import os
from typing import Optional

from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_main_prompt
from agent.tools.agent_tools import (search_policy, query_order, query_user_orders, query_logistics,
                                     query_refund, check_return_window, judge_freight, check_warranty,
                                     get_user_id, get_current_date, fill_context_for_ticket)
from agent.tools.middleware import monitor_tool, log_before_model


def _encode_image(image_path: str) -> tuple[str, str] | tuple[None, None]:
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".bmp": "image/bmp",
    }
    mime = mime_map.get(ext, "image/jpeg")
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8"), mime
    except Exception:
        return None, None


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_main_prompt(),
            tools=[search_policy, query_order, query_user_orders, query_logistics,
                   query_refund, check_return_window, judge_freight, check_warranty,
                   get_user_id, get_current_date, fill_context_for_ticket],
            middleware=[monitor_tool, log_before_model],
        )

    @staticmethod
    def _build_messages(
        query: str,
        chat_history: Optional[list[dict]] = None,
        image_base64: Optional[str] = None,
        image_mime: str = "image/jpeg",
    ) -> list[dict]:
        msgs: list[dict] = []
        if chat_history:
            for h in chat_history[-10:]:
                content = h.get("content", "")
                # 历史消息只保留纯文本，避免 base64 图片撑爆上下文
                if isinstance(content, list):
                    text_parts = [p["text"] for p in content if p.get("type") == "text"]
                    content = " ".join(text_parts) if text_parts else ""
                msgs.append({"role": h["role"], "content": content})

        # 当前消息：多模态（图+文）或纯文本
        if image_base64:
            msgs.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{image_mime};base64,{image_base64}"
                    }},
                ],
            })
        else:
            msgs.append({"role": "user", "content": query})

        return msgs

    def execute_stream(
        self,
        query: str,
        chat_history: Optional[list[dict]] = None,
        image_path: Optional[str] = None,
    ):
        img_b64, img_mime = None, "image/jpeg"
        if image_path:
            img_b64, img_mime = _encode_image(image_path)

        input_dict = {
            "messages": self._build_messages(query, chat_history, img_b64, img_mime),
        }

        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
