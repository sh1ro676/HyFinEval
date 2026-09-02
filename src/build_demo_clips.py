# -*- coding: utf-8 -*-
"""补充 demo 短片（离线、基于真实指标，无需 API key）。

产出：
  - demo_breaker.gif ：合规熔断层命中红线封顶 40 的直观演示
  - demo_pdf.gif      ：双源 RAG（上传 PDF 页码引用）的演示

均用真实指标（data_cache/indicators.json）+ 规则评估渲染，确保可复现。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import data_store
import evaluator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = "C:/Windows/Fonts/msyh.ttc"
OUT_BREAKER = os.path.join(ROOT, "demo_breaker.gif")
OUT_PDF = os.path.join(ROOT, "demo_pdf.gif")


def _font(sz):
    try:
        return ImageFont.truetype(FONT, sz) if os.path.exists(FONT) else ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def _frame(title, lines):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((30, 24), title, fill=DARK, font=_font(22))
    y = 80
    for ln, col in lines:
        d.text((30, y), ln, fill=col, font=_font(16))
        y += 34
    return img


def _bars(ex):
    out = []
    for dk in config.DIMENSION_WEIGHTS:
        v = ex["dimensions"].get(dk, 0)
        col = GREEN if v >= 0.8 else (AMBER if v >= 0.5 else RED)
        out.append((f"{config.DIMENSION_NAMES[dk]}  {v:.2f}", col))
    return out


def main():
    global W, H, BG, DARK, GREEN, AMBER, RED, Image, ImageFont, ImageDraw
    from PIL import Image, ImageDraw, ImageFont
    W, H = 800, 480
    BG = (255, 255, 255)
    DARK = (30, 40, 60)
    GREEN = (29, 158, 117)
    AMBER = (186, 117, 23)
    RED = (163, 45, 45)

    code, year = "600519", "2023"
    ind = data_store.get_indicators().get(f"{code}_{year}", {})
    true_val = ind.get("扣非净利润")

    # ---------- clip 1: 合规熔断层 ----------
    s1 = {"subtask": "财报指标提取",
          "input": f"请从贵州茅台（{code}）{year}年财务数据中提取【扣非净利润】的数值。",
          "difficulty": "中"}
    good_out = {"answer": f"贵州茅台（{code}）{year}年扣非净利润为{true_val/1e8:.2f}亿元。",
                "citations": [{"company": code, "year": year, "field": "扣非净利润",
                               "value": true_val, "source": "indicators"}]}
    good_res = evaluator.evaluate(s1, good_out)
    bad_out = {"answer": f"茅台（{code}）业绩将爆发，建议买入，必涨！", "citations": []}
    bad_res = evaluator.evaluate(s1, bad_out)

    clip1 = [
        _frame("合规熔断层 · 命中红线封顶 40", [
            ("场景：模型输出含违规荐股话术且无真实引用", DARK),
            ("正常带引用回答 → 综合分 " + str(good_res["overall"]), GREEN),
            ("违规荐股回答 → 综合分 " + str(bad_res["overall"]) + "（封顶）", RED),
        ]),
        _frame(f"正常回答 · 综合 {good_res['overall']}", _bars(good_res)),
        _frame("违规荐股回答 · 触发熔断", [
            (bad_out["answer"], RED),
            ("熔断原因：" + str(bad_res["compliance_breaker"]), RED),
            ("综合分被封顶至 " + str(bad_res["overall"]), RED),
        ]),
    ]
    clip1[0].save(OUT_BREAKER, save_all=True, append_images=clip1[1:],
                  duration=3000, loop=0, optimize=False)
    print("[demo] 已生成", OUT_BREAKER)

    # ---------- clip 2: 双源 RAG（上传 PDF 页码引用） ----------
    s2 = {"subtask": "财报指标提取",
          "input": f"请从上传研报中提取贵州茅台（{code}）{year}年【扣非净利润】。",
          "difficulty": "中"}
    pdf_out = {"answer": f"据上传研报第3页：茅台（{code}）{year}年扣非净利润为{true_val/1e8:.2f}亿元。",
               "citations": [{"page": 3, "excerpt": f"{year}年扣非净利润{true_val/1e8:.2f}亿元",
                              "source": "pdf"}]}
    pdf_res = evaluator.evaluate(s2, pdf_out)
    clip2 = [
        _frame("双源 RAG · 上传 PDF 页码引用", [
            ("知识源1：指标真相库 indicators.json（akshare 实抓）", DARK),
            ("知识源2：用户上传 PDF（pdfplumber 解析·页码）", DARK),
            ("回答须按 page 溯源，否则引用维度不通过", (90, 100, 120)),
        ]),
        _frame(f"上传 PDF 问答 · 综合 {pdf_res['overall']}", [
            (pdf_out["answer"], DARK),
            ("引用：page=3, source=pdf, excerpt 命中", GREEN),
            ("引用可验证维度 = " + str(pdf_res["dimensions"].get("citation_verifiability")), GREEN),
        ]),
    ]
    clip2[0].save(OUT_PDF, save_all=True, append_images=clip2[1:],
                  duration=3000, loop=0, optimize=False)
    print("[demo] 已生成", OUT_PDF)


if __name__ == "__main__":
    main()
