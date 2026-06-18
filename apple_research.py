"""
练习 2：Apple (AAPL.US) 完整投研数据拉取
=========================================
用途：一次性拉取行情、财报、估值、评级、资讯五维数据，
      为人工投研解读提供原材料。

用法：在 Jupyter 中复制以下各段代码逐段执行，
      或 python3 apple_research.py 一次性运行。
"""

import os
import time
from dotenv import load_dotenv
load_dotenv()

from longbridge.openapi import (
    Config, QuoteContext, FundamentalContext, ContentContext,
    Period, AdjustType, FinancialReportKind,
)

# ============================================================
# 0. 初始化连接
# ============================================================
config = Config.from_apikey(
    app_key=os.getenv("LONGBRIDGE_APP_KEY"),
    app_secret=os.getenv("LONGBRIDGE_APP_SECRET"),
    access_token=os.getenv("LONGBRIDGE_ACCESS_TOKEN"),
)

quote_ctx = QuoteContext(config)
fund_ctx = FundamentalContext(config)
content_ctx = ContentContext(config)

SYMBOL = "AAPL.US"

print("=" * 60)
print(f"  {SYMBOL} 投研数据拉取")
print("=" * 60)


# ============================================================
# 1. 实时行情 + K线量价
# ============================================================
print("\n" + "=" * 60)
print("一、行情与量价")
print("=" * 60)

# 1a. 实时行情
q = quote_ctx.quote([SYMBOL])[0]
today_vol = q.volume
change_pct = (q.last_done - q.prev_close) / q.prev_close * 100 if q.prev_close else 0

print(f"""
  当前价格:   {q.last_done} USD
  昨日收盘:   {q.prev_close} USD
  涨跌幅:     {change_pct:+.2f}%
  今日最高:   {q.high} USD
  今日最低:   {q.low} USD
  振幅:       {(q.high - q.low) / q.prev_close * 100:.2f}%
  成交量:     {today_vol:,} 股
  成交额:     {q.turnover:,.2f} USD
  交易状态:   {q.trade_status}
  数据时间:   {q.timestamp}
""")

time.sleep(1)

# 1b. 近 5 日 K 线 → 算均量
candles = quote_ctx.candlesticks(
    symbol=SYMBOL, period=Period.Day, count=5,
    adjust_type=AdjustType.ForwardAdjust,
)

print("  近 5 日量价：")
total_vol = 0
for c in candles:
    total_vol += c.volume
    print(f"    {c.timestamp} | C:{c.close:>8} | 量:{c.volume:>12,}")

avg_vol = total_vol / len(candles) if len(candles) > 0 else 0
vol_ratio = today_vol / avg_vol if avg_vol > 0 else 1

print(f"\n  近 5 日均量: {avg_vol:,.0f} 股")
print(f"  量比(今日/均量): {vol_ratio:.2f}")

if vol_ratio > 1.5:
    vol_judge = "放量"
elif vol_ratio > 1.2:
    vol_judge = "温和放量"
elif vol_ratio < 0.8:
    vol_judge = "缩量"
else:
    vol_judge = "量能正常"

direction = "上涨" if change_pct > 0 else "下跌"
print(f"  → 今日为{vol_judge}{direction}")


# ============================================================
# 2. 财报数据（利润表）
# ============================================================
time.sleep(2)
print("\n" + "=" * 60)
print("二、利润表（Income Statement）— 最新季度 vs 同比")
print("=" * 60)

income = fund_ctx.financial_report(SYMBOL, kind=FinancialReportKind.IncomeStatement)

# 提取核心指标的最新值
core_fields = ['EPS', 'OperatingRevenue', 'NetProfit',
               'OperatingIncome', 'GrossMgn', 'NetProfitMargin',
               'ROE', 'ProfitQuality']

for indicator_block in income.list.get('IS', {}).get('indicators', []):
    for account in indicator_block.get('accounts', []):
        field = account.get('field')
        if field not in core_fields:
            continue
        name = account.get('name', field)
        values = account.get('values', [])
        if not values:
            continue
        latest = values[0]  # 最新季度
        period = latest.get('period', '')
        val = latest.get('value', 'N/A')
        yoy = latest.get('yoy', '')
        rank = account.get('industry_ranking', '')

        rank_str = f"（行业排名 {rank}）" if rank else ""
        yoy_str = f"（同比 {float(yoy):+.2f}%）" if yoy and yoy != '' else ""

        print(f"  {name}: {val}  {rank_str}{yoy_str}")


# ============================================================
# 3. 估值数据
# ============================================================
time.sleep(1)
print("\n" + "=" * 60)
print("三、估值数据")
print("=" * 60)

valuation = fund_ctx.valuation(SYMBOL)
for attr in dir(valuation):
    if not attr.startswith('_'):
        val = getattr(valuation, attr)
        if not callable(val) and val is not None:
            print(f"  {attr}: {val}")


# ============================================================
# 4. 公司概况
# ============================================================
time.sleep(1)
print("\n" + "=" * 60)
print("四、公司概况")
print("=" * 60)

company = fund_ctx.company(SYMBOL)
key_attrs = [a for a in dir(company) if not a.startswith('_')]
for attr in key_attrs:
    val = getattr(company, attr)
    if not callable(val) and val is not None and val != "":
        print(f"  {attr}: {val}")


# ============================================================
# 5. 机构评级
# ============================================================
time.sleep(1)
print("\n" + "=" * 60)
print("五、机构评级")
print("=" * 60)

ratings = fund_ctx.institution_rating(SYMBOL)
print(f"  共 {len(ratings)} 条评级记录（展示前 5 条）")
for i, r in enumerate(ratings[:5]):
    print(f"\n  第 {i+1} 条:")
    for attr in dir(r):
        if not attr.startswith('_'):
            val = getattr(r, attr)
            if not callable(val) and val is not None and val != "":
                print(f"    {attr}: {val}")


# ============================================================
# 6. 新闻资讯
# ============================================================
time.sleep(1)
print("\n" + "=" * 60)
print("六、近期新闻（前 3 条）")
print("=" * 60)

news_list = content_ctx.news(SYMBOL)
print(f"  共 {len(news_list)} 条新闻（展示前 3 条）")
for i, news in enumerate(news_list[:3]):
    print(f"\n  --- 新闻 {i+1} ---")
    for attr in dir(news):
        if not attr.startswith('_'):
            val = getattr(news, attr)
            if not callable(val) and val is not None and val != "":
                val_str = str(val)
                if len(val_str) > 200:
                    val_str = val_str[:200] + "..."
                print(f"    {attr}: {val_str}")


# ============================================================
# 7. 汇总输出：供人工撰写解读
# ============================================================
print("\n" + "=" * 60)
print("七、数据拉取完成")
print("=" * 60)
print("""
  接下来请根据以上数据，撰写 AAPL 投研解读报告，覆盖：

  1. 量价判断 —— 当前价格、成交量 vs 均量，方向判断
  2. 收入与利润 —— 营收/净利润的绝对值与增速对比
  3. 盈利能力 —— 毛利率、净利率、ROE 各说明什么
  4. 利润质量 —— 经营现金流 vs 会计利润
  5. 估值水位 —— PE/PB 当前处于什么位置（需要对比历史）
  6. 风险提示 —— 基于数据的 3 个具体风险点

  模板见 apple_research.py 文件末尾的注释。
""")
