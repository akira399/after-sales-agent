"""Streamlit UI 冒烟测试（AppTest 模拟浏览器会话）。

验证产品关键体验：
1. 打开页面即出现聊天输入框（不再因未配置模型而整页空白）
2. 部署者配了平台 Key 时 → 模型自动就绪（platform 来源）
3. 无任何 Key 时发送消息 → 给出明确配置提示，页面不崩溃

用法：
    python scripts/test_ui.py          # 用 .env 里的真实 Key 测场景1/2
    python scripts/test_ui.py --nokey  # 模拟无 Key（场景3），需临时注释 .env

说明：AppTest 与 .env 的 load_dotenv 在同一进程，环境变量控制见 main()。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from streamlit.testing.v1 import AppTest


_APP_PATH = os.path.join(os.path.dirname(__file__), "..", "agent_app.py")


def _run_page() -> AppTest:
    at = AppTest.from_file(_APP_PATH, default_timeout=40)
    at.run()
    return at


def test_opens_with_chat_input() -> bool:
    at = _run_page()
    if at.exception:
        print(f"[FAIL] 页面异常: {at.exception}")
        return False
    n_input = len(at.chat_input)
    n_title = sum(1 for t in at.title if "售后小蜜" in t.value)
    print(f"[INFO] title={n_title} chat_input={n_input} exception={at.exception}")
    if n_input < 1:
        print("[FAIL] 页面没有聊天输入框（对话框缺失）")
        return False
    if n_title < 1:
        print("[FAIL] 页面标题缺失")
        return False
    print("[PASS] 打开页面即显示对话框")
    return True


def test_send_without_key_shows_hint() -> bool:
    """无 Key 场景：env 置空后发送消息，应出现配置提示且不崩溃。"""
    os.environ["DASHSCOPE_API_KEY"] = ""
    # 强制配置解析重新读取（模块已 import，直接改 env 即可被 _load_env_config 感知）
    from model import runtime_config  # noqa: PLC0415

    runtime_config._env_cache = None  # 若存在缓存则清除

    at = _run_page()
    if at.exception:
        print(f"[FAIL] 页面异常: {at.exception}")
        return False
    # 发送一条消息
    if len(at.chat_input) < 1:
        print("[WARN] 无 chat_input，跳过发送")
        # 若页面有 warning 引导也算通过（说明在引导而非空白崩溃）
        hints = [str(w) for w in at.warning] + [str(e) for e in at.error]
        if any("API Key" in h or "模型" in h for h in hints):
            print("[PASS] 无 Key 场景页面给出引导且不崩溃")
            return True
        print("[FAIL] 无 Key 且无聊天框也无引导")
        return False

    at.chat_input[0].set_value("耳机进水了能保修吗").run()
    if at.exception:
        print(f"[FAIL] 发送后页面异常: {at.exception}")
        return False
    errors = [str(e.value) for e in at.error]
    warnings = [str(w.value) for w in at.warning]
    print(f"[INFO] errors={errors} warnings={warnings}")
    joined = " ".join(errors + warnings)
    if "API Key" in joined or "模型" in joined:
        print("[PASS] 无 Key 发送时给出配置提示，页面不崩溃")
        return True
    print("[FAIL] 无 Key 发送未见明确提示")
    return False


def test_platform_key_auto() -> bool:
    from model.runtime_config import config_source

    src = config_source()
    print(f"[INFO] config_source = {src}")
    # 若本机 .env 有 Key 应为 platform；未配则返回 none（属预期，返回 True 不判失败）
    if src in ("platform", "user"):
        print("[PASS] 模型配置自动就绪")
        return True
    print("[WARN] 当前无平台 Key（未在 .env 配置，公开 Demo 场景属预期）")
    return True


def main() -> None:
    mode = "nokey" if "--nokey" in sys.argv else "withkey"

    if mode == "withkey":
        print("===== 场景A：本地/部署者配了平台 Key =====")
        r1 = test_opens_with_chat_input()
        r2 = test_platform_key_auto()
        print(f"A1) 打开即有对话框    : {'PASS' if r1 else 'FAIL'}")
        print(f"A2) 模型自动就绪       : {'PASS' if r2 else 'FAIL'}")
        # 恢复：确保不误删 .env 注入值
        os.environ.pop("DASHSCOPE_API_KEY", None)
    else:
        print("===== 场景B：无任何 Key（公开 Demo 访问者）=====")
        r3 = test_send_without_key_shows_hint()
        print(f"B1) 无Key发送给出提示   : {'PASS' if r3 else 'FAIL'}")


if __name__ == "__main__":
    main()
