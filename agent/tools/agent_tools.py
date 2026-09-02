"""售后客服 Agent 工具集（替代原项目的棉花质检/天气工具）。

设计要点：
1. 业务数据读取与规则判定分离：CSV 只提供事实，合规结论一律由 return_rules 纯函数产出。
2. 工具返回结构化文本，便于 LLM 理解，也便于 UI 展示。
3. 查不到就明确说查不到，绝不臆造订单/退款信息。
"""
import csv
import os
from datetime import date, datetime

from langchain_core.tools import tool

from agent.tools.return_rules import (
    check_no_reason_window,
    judge_freight_payer,
    judge_warranty,
    parse_date,
)
from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

# 演示用当前登录用户（真实系统应由会话态/鉴权提供）
CURRENT_USER_ID = "1001"

# 商品保修月数表（真实系统应查商品库）
WARRANTY_MONTHS = {
    "无线蓝牙耳机Pro": 12,
    "65W氮化镓充电器": 12,
    "智能手表WatchX": 24,
}

_rag = RagSummarizeService()
_external_dir = get_abs_path(agent_conf.get("external_data_dir", "data/external"))


def _read_csv(filename: str) -> list[dict]:
    """读取 CSV 全部行，文件缺失或异常时返回空列表。"""
    path = os.path.join(_external_dir, filename)
    if not os.path.exists(path):
        logger.warning(f"[tools] 数据文件不存在: {path}")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        logger.error(f"[tools] 读取 {filename} 失败: {e}", exc_info=True)
        return []


def _bool(v: str | None) -> bool:
    return str(v).strip() in ("是", "true", "True", "1", "yes")


# ------------------------------------------------------------------
# 知识库检索
# ------------------------------------------------------------------
@tool(description="检索售后政策、退换货规则、保修条款、商品FAQ知识库。凡涉及规则/政策/保修范围的问题都应先调用本工具。")
def search_policy(query: str) -> str:
    return _rag.rag_summarize(query, chat_history=None)


# ------------------------------------------------------------------
# 业务数据查询
# ------------------------------------------------------------------
@tool(description="按订单号查询订单详情，返回商品、金额、下单/签收日期、状态、是否含运费险、是否已激活。用户未给订单号时可先用 query_user_orders 查。")
def query_order(order_id: str) -> str:
    for row in _read_csv("orders.csv"):
        if row.get("order_id", "").strip() == order_id.strip():
            return (
                f"订单 {row['order_id']}｜用户 {row['user_id']}\n"
                f"商品：{row['product_name']}｜金额：{row['price']} 元\n"
                f"下单：{row['order_date']}｜签收：{row['sign_date'] or '未签收'}\n"
                f"状态：{row['status']}｜运费险：{row['has_insurance']}｜已激活：{row['activated']}"
            )
    return f"未查询到订单 {order_id}，请与用户核对订单号是否正确。"


@tool(description="查询当前登录用户近期的全部订单列表（订单号、商品、金额、签收日）。当用户说'我的订单''我买的东西'而未给订单号时使用。")
def query_user_orders() -> str:
    rows = [r for r in _read_csv("orders.csv") if r.get("user_id") == CURRENT_USER_ID]
    if not rows:
        return f"用户 {CURRENT_USER_ID} 暂无订单记录。"
    lines = [f"用户 {CURRENT_USER_ID} 共 {len(rows)} 笔订单："]
    for r in rows:
        lines.append(
            f"- {r['order_id']}｜{r['product_name']}｜{r['price']}元｜"
            f"签收：{r['sign_date'] or '未签收'}｜状态：{r['status']}"
        )
    return "\n".join(lines)


@tool(description="按订单号查询物流轨迹，返回各节点时间、描述、地点与承运商。用于回答'到哪了''为什么还没到'。")
def query_logistics(order_id: str) -> str:
    rows = [r for r in _read_csv("logistics.csv") if r.get("order_id", "").strip() == order_id.strip()]
    if not rows:
        return f"未查询到订单 {order_id} 的物流记录，可能尚未发货或单号有误。"
    lines = [f"订单 {order_id} 物流轨迹（最新在后）："]
    for r in rows:
        lines.append(f"- {r['node_time']}｜{r['node_desc']}｜{r['location']}｜{r['operator']}")
    return "\n".join(lines)


@tool(description="按退款单号查询退款进度；也可按订单号查该订单关联的退款。返回状态、金额、原因、驳回理由、预计到账日。")
def query_refund(refund_id: str = "", order_id: str = "") -> str:
    rows = _read_csv("refunds.csv")
    hit = None
    for r in rows:
        if refund_id and r.get("refund_id", "").strip() == refund_id.strip():
            hit = r
            break
        if order_id and r.get("order_id", "").strip() == order_id.strip():
            hit = r
            break
    if not hit:
        return "未查询到对应退款记录，请核对退款单号或订单号。"

    txt = (
        f"退款单 {hit['refund_id']}｜订单 {hit['order_id']}\n"
        f"金额：{hit['amount']} 元｜状态：{hit['status']}\n"
        f"申请时间：{hit['apply_time']}｜原因：{hit['reason']}"
    )
    if hit.get("reject_reason"):
        txt += f"\n驳回理由：{hit['reject_reason']}"
    if hit.get("expect_arrive"):
        txt += f"\n预计到账：{hit['expect_arrive']}"
    return txt


# ------------------------------------------------------------------
# 合规判定（规则引擎，防幻觉核心）
# ------------------------------------------------------------------
@tool(description="判定某订单是否仍可七天无理由退货。会自动读取订单签收日与激活状态，按'签收次日起算'的法定口径计算，绝不用自己的算术代替该工具的结论。")
def check_return_window(order_id: str) -> str:
    order = None
    for row in _read_csv("orders.csv"):
        if row.get("order_id", "").strip() == order_id.strip():
            order = row
            break
    if not order:
        return f"未查询到订单 {order_id}，无法判定退货窗口，请核对订单号。"

    result = check_no_reason_window(
        sign_date=order.get("sign_date"),
        today=date.today(),
        activated=_bool(order.get("activated")),
    )
    return f"【规则引擎判定】{result.to_text()}"


@tool(description="判定退货运费由谁承担。is_quality_issue 为 true 表示质量问题/商家责任，false 表示无理由退货。")
def judge_freight(order_id: str, is_quality_issue: bool) -> str:
    has_insurance = False
    for row in _read_csv("orders.csv"):
        if row.get("order_id", "").strip() == order_id.strip():
            has_insurance = _bool(row.get("has_insurance"))
            break
    payer, detail = judge_freight_payer(is_quality_issue, has_insurance)
    return f"【规则引擎判定】运费承担方：{payer}。{detail}"


@tool(description="判定商品是否在保修范围。damage_type 取值：quality(质量故障)、water(进水)、drop(跌落)、self_repair(私拆)、wear(外观磨损)。")
def check_warranty(order_id: str, damage_type: str) -> str:
    order = None
    for row in _read_csv("orders.csv"):
        if row.get("order_id", "").strip() == order_id.strip():
            order = row
            break
    if not order:
        return f"未查询到订单 {order_id}，无法核算保修。"

    months = WARRANTY_MONTHS.get(order["product_name"].strip(), 12)
    ok, detail = judge_warranty(
        sign_date=order.get("sign_date"),
        warranty_months=months,
        damage_type=damage_type,
        today=date.today(),
    )
    return f"【规则引擎判定】{'在保，可免费维修/换新' if ok else '不予保修'}。{detail}（{order['product_name']}保修期 {months} 个月）"


# ------------------------------------------------------------------
# 基础信息
# ------------------------------------------------------------------
@tool(description="获取当前登录用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    return CURRENT_USER_ID


@tool(description="获取当前真实日期，返回格式 YYYY-MM-DD。判定退货窗口、保修期前必须先调用本工具获取今天日期。")
def get_current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


# ------------------------------------------------------------------
# 工单生成（触发报告模式）
# ------------------------------------------------------------------
@tool(description="无入参，无返回值。当用户表达投诉、要求升级处理、要求转人工时调用本工具，触发系统切换为工单生成模式，为后续生成结构化工单注入上下文。")
def fill_context_for_ticket() -> str:
    return "已触发工单生成模式，请按要求输出结构化工单。"
