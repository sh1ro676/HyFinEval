# -*- coding: utf-8 -*-
"""生成演示用 PDF：招商银行 2023 年报摘要（中文、可提取、分 3 页）。"""
from fpdf import FPDF

FONT = r"C:/Windows/Fonts/simhei.ttf"

class PDF(FPDF):
    def header(self):
        self.set_font("CJK", "", 10)
        self.set_text_color(120)
        self.cell(0, 8, "招商银行（600036）2023 年年度报告摘要（演示样例）", align="R")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("CJK", "", 9)
        self.set_text_color(150)
        self.cell(0, 10, f"第 {self.page_no()} 页 / 共 3 页", align="C")


pdf = PDF(orientation="P", unit="mm", format="A4")
pdf.add_font("CJK", "", FONT)
pdf.add_font("CJK", "B", FONT)
pdf.set_auto_page_break(auto=True, margin=18)

# 第 1 页：经营概览
pdf.add_page()
pdf.set_font("CJK", "B", 15)
pdf.set_text_color(20)
pdf.cell(0, 10, "一、经营概览", ln=True)
pdf.set_font("CJK", "", 11)
pdf.set_text_color(40)
lines = [
    "招商银行 2023 年实现营业收入 3,391.23 亿元，同比增长 3.78%。",
    "归属于本行股东净利润 1,466.02 亿元，同比增长 6.22%。",
    "截至 2023 年末，集团资产总额 11.03 万亿元，较上年末增长 8.77%。",
    "其中，贷款和垫款总额 6.52 万亿元，客户存款总额 8.16 万亿元。",
    "净利息收入 2,146.31 亿元，非利息净收入 1,244.92 亿元。",
    "成本收入比 32.52%，保持同业较优水平。",
]
for t in lines:
    pdf.multi_cell(0, 7, t)
    pdf.ln(1)

# 第 2 页：资产质量与资本
pdf.add_page()
pdf.set_font("CJK", "B", 15)
pdf.set_text_color(20)
pdf.cell(0, 10, "二、资产质量与资本充足", ln=True)
pdf.set_font("CJK", "", 11)
pdf.set_text_color(40)
lines2 = [
    "2023 年末不良贷款率 0.95%，较上年末下降 0.01 个百分点。",
    "拨备覆盖率 437.70%，风险抵补能力充足。",
    "核心一级资本充足率 13.10%，一级资本充足率 15.05%，",
    "资本充足率 17.88%，均满足监管要求并留有缓冲。",
    "关注类贷款余额占比 1.10%，资产质量整体稳健。",
]
for t in lines2:
    pdf.multi_cell(0, 7, t)
    pdf.ln(1)

# 第 3 页：分红与回报
pdf.add_page()
pdf.set_font("CJK", "B", 15)
pdf.set_text_color(20)
pdf.cell(0, 10, "三、分红与股东回报", ln=True)
pdf.set_font("CJK", "", 11)
pdf.set_text_color(40)
lines3 = [
    "2023 年度现金分红方案：每股派发现金红利 1.972 元（含税）。",
    "全年合计拟派发现金红利约 497.34 亿元，分红比例 33.92%。",
    "以 2023 年末股价估算，股息率约 5.6%。",
    "董事会建议实施中期分红，提升股东回报连续性。",
    "（注：本文件为演示样例，数据仅供产品功能展示，不构成投资建议。）",
]
for t in lines3:
    pdf.multi_cell(0, 7, t)
    pdf.ln(1)

out = "D:/finance_samples/demo_upload_sample.pdf"
pdf.output(out)
print("written:", out)
