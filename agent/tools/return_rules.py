"""售后合规判定规则引擎（纯函数，可单测）。

设计原则：售后窗口、运费承担这类"确定性合规判定"绝不让 LLM 自由发挥，
由纯函数算出确定结论，LLM 只负责理解用户语义并解释结论 —— 从机制上杜绝政策幻觉。

面试官可重点考察：为什么把规则下沉？答：避免 LLM 算错日期/误判责任方导致合规风险。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

# 法定无理由退货窗口（自然日）
NO_REASON_WINDOW_DAYS = 7
# 质量问题换新窗口（演示店铺承诺，优于法定）
QUALITY_EXCHANGE_DAYS = 15
# 退款到账承诺时效（收到退回商品后的工作日）
REFUND_PROCESS_DAYS = 3


@dataclass
class WindowResult:
    """无理由退货窗口判定结果。"""

    eligible: bool
    days_used: int
    days_left: int
    deadline: str
    reason: str

    def to_text(self) -> str:
        return (
            f"{'符合' if self.eligible else '不符合'}七天无理由退货条件。"
            f"（签收日 {self._sign} → 已过 {self.days_used} 天，剩余 {self.days_left} 天，"
            f"截止 {self.deadline}）。{self.reason}"
        )

    # 内部字段，便于格式化输出
    _sign: str = ""


def parse_date(value: str) -> date | None:
    """宽松解析日期字符串，支持 YYYY-MM-DD / YYYY-MM-DD HH:MM。"""
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def check_no_reason_window(
    sign_date: str | date | None,
    today: date | None = None,
    activated: bool = False,
) -> WindowResult:
    """判定订单是否仍在七天无理由退货窗口内。

    规则依据：《网络购买商品七日无理由退货暂行办法》第十条 —— 七日期间自签收次日开始起算。

    Args:
        sign_date: 签收日期（空字符串/None 表示尚未签收）
        today: 当前日期（可注入，便于单测）
        activated: 商品是否已激活（激活产生使用痕迹的可激活设备，价值明显贬损）

    Returns:
        WindowResult 判定结果
    """
    today = today or date.today()
    sign = sign_date if isinstance(sign_date, date) else parse_date(sign_date or "")

    if sign is None:
        return WindowResult(
            eligible=False,
            days_used=-1,
            days_left=-1,
            deadline="—",
            reason="订单尚未签收，七日窗口未起算；未发货可直接申请全额退款，已发货需拦截或拒收。",
            _sign="未签收",
        )

    # 关键：从签收次日开始起算（第 1 天 = 签收日 + 1）
    start = sign + timedelta(days=1)
    days_used = (today - start).days + 1  # 含当天的自然日计数
    deadline = sign + timedelta(days=NO_REASON_WINDOW_DAYS)
    days_left = (deadline - today).days

    result = WindowResult(
        eligible=False,
        days_used=max(days_used, 0),
        days_left=days_left,
        deadline=deadline.isoformat(),
        reason="",
        _sign=sign.isoformat(),
    )

    if days_used < 1:
        result.reason = "签收日尚未开始起算（次日起算）。"
        return result

    if days_used > NO_REASON_WINDOW_DAYS:
        result.reason = f"已超出七日无理由退货窗口（法定自签收次日起算满 {NO_REASON_WINDOW_DAYS} 天），不可再主张无理由退货；若属质量问题仍可依消法第二十四条主张。 "
        return result

    if activated:
        result.eligible = False
        result.reason = (
            "虽在窗口内，但该商品已激活（产生激活/授权等数据使用痕迹），"
            "依据商品不完好判定标准，不适用无理由退货。"
        )
        return result

    result.eligible = True
    result.reason = (
        f"在窗口内且商品未激活，可申请七天无理由退货；寄回运费由消费者承担"
        f"（若含运费险由保险理赔），退回商品需保持完好并连同赠品一并寄回。"
    )
    return result


def judge_freight_payer(
    is_quality_issue: bool,
    has_insurance: bool,
) -> tuple[str, str]:
    """判定退货运费承担方。

    Args:
        is_quality_issue: 是否属质量问题（商家责任）
        has_insurance: 是否购买运费险

    Returns:
        (承担方, 说明)
    """
    if is_quality_issue:
        return "商家", "属质量问题，依消法第二十四条，运输等必要费用由经营者承担。"
    if has_insurance:
        return "运费险理赔", "无理由退货本由消费者承担运费，但订单含运费险，理赔额度内由保险公司承担。"
    return "消费者", "无理由退货，商品退回运费依法由消费者承担（有约定从约定）。"


def judge_warranty(
    sign_date: str | date | None,
    warranty_months: int,
    damage_type: str,
    today: date | None = None,
) -> tuple[bool, str]:
    """判定是否在保修范围。

    Args:
        sign_date: 签收日期
        warranty_months: 保修月数
        damage_type: 损坏类型，取值 quality(质量故障)/water(进水)/drop(跌落)/self_repair(私拆)/wear(外观磨损)
        today: 当前日期

    Returns:
        (是否保修, 说明)
    """
    today = today or date.today()
    sign = sign_date if isinstance(sign_date, date) else parse_date(sign_date or "")

    human_damage = {
        "water": "进水",
        "drop": "跌落摔坏",
        "self_repair": "私自拆修",
        "wear": "外观磨损/划痕",
    }

    if damage_type in human_damage:
        return False, (
            f"{human_damage[damage_type]}属人为损坏，不在保修范围（保修仅覆盖非人为质量故障）。"
            f"可提供付费维修，需寄回检测。"
        )

    if sign is None:
        return False, "订单尚未签收，无法核算保修期。"

    # 按月近似计算保修到期
    y, m = sign.year, sign.month + warranty_months
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    d = min(sign.day, [31, 29 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 28,
                       31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    expire = date(y, m, d)

    if today > expire:
        return False, f"已过保修期（保修截止 {expire.isoformat()}，共 {warranty_months} 个月），仅支持付费维修。"

    return True, f"在保修期内（保修截止 {expire.isoformat()}），非人为质量故障可免费维修/换新，运费商家承担。"


if __name__ == "__main__":
    # 自测：今天是 2026-09-02
    today = date(2026, 9, 2)
    print(check_no_reason_window("2026-09-01", today).to_text())   # 签收1天 → 符合
    print(check_no_reason_window("2026-08-28", today).to_text())   # 签收6天 → 符合
    print(check_no_reason_window("2026-08-20", today).to_text())   # 超期 → 不符合
    print(check_no_reason_window("", today).to_text())             # 未签收
    print(check_no_reason_window("2026-09-01", today, activated=True).to_text())  # 已激活
    print(judge_freight_payer(False, True))
    print(judge_warranty("2026-08-12", 12, "water", today))
