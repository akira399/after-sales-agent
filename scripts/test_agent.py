"""Agent 全链路冒烟测试：验证 ReAct 编排 + 工具调用 + 规则引擎。

用法：python scripts/test_agent.py
输出：每个问题的工具调用轨迹 + 最终回答
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import AIMessage, ToolMessage

from agent.react_agent import ReactAgent

QUESTIONS = [
    "我 8 月 28 号签收的耳机，现在还能七天无理由退货吗？",
    "我的退款为什么被驳回了？订单号 DX20260720001",
    "订单 DX20260830004 的货到哪了？怎么还没到",
]


def run_one(agent: ReactAgent, question: str) -> None:
    print("\n" + "=" * 70)
    print(f"用户：{question}")

    seen_tools: set[str] = set()
    final = ""

    for chunk in agent.agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
        context={"report": False},
    ):
        msgs = chunk.get("messages", [])
        if not msgs:
            continue

        for m in msgs:
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    key = f"{tc['name']}|{tc.get('id','')}"
                    if key in seen_tools:
                        continue
                    seen_tools.add(key)
                    print(f"  → 调用工具: {tc['name']}({tc.get('args', {})})")
            if isinstance(m, ToolMessage):
                content = str(m.content or "")
                print(f"  ← 工具返回: {m.name} => {content[:150].replace(chr(10), ' / ')}")

        last = msgs[-1]
        if isinstance(last, AIMessage) and last.content and not getattr(last, "tool_calls", None):
            final = last.content

    print(f"\n助手回答：\n{final}")


def main() -> None:
    agent = ReactAgent()
    tool_names = [t.name for t in agent.agent.tools] if hasattr(agent.agent, "tools") else []
    print(f"[INFO] Agent 已加载，工具数：{len(tool_names)}")
    print(f"[INFO] 工具清单：{tool_names}")

    for q in QUESTIONS:
        run_one(agent, q)


if __name__ == "__main__":
    main()
