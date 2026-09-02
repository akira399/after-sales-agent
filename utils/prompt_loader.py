from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


def _load_prompt(key: str) -> str:
    """通用加载：按 yaml 中的 key 读取对应路径的文本内容。"""
    try:
        path = get_abs_path(prompts_conf[key])
    except KeyError as e:
        logger.error(f"[prompt_loader]yaml 中缺少 {key} 配置项")
        raise e
    try:
        return open(path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[prompt_loader]读取 {key} 失败：{str(e)}")
        raise e


def load_kb_retrieve_qa_prompts():
    """知识库检索问答系统提示词（kb_retrieve_qa.txt）"""
    return _load_prompt("kb_retrieve_qa_prompt_path")


def load_query_understand_prompts():
    """用户问题识别 → 检索用语（query_understand.txt）"""
    return _load_prompt("query_understand_prompt_path")


def load_main_prompt():
    """ReAct Agent 主提示词 —— 含思考准则、工具调用规则（main_prompt.txt）"""
    return _load_prompt("main_prompt_path")


def load_ticket_prompt():
    """工单生成专用提示词（ticket_prompt.txt）"""
    return _load_prompt("ticket_prompt_path")


if __name__ == "__main__":
    print(load_kb_retrieve_qa_prompts())
