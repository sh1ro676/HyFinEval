# -*- coding: utf-8 -*-
"""构建 HyFinEval 的 ≤2 分钟 demo 素材（离线、基于真实指标）。

产出：
  - demo.html  ：自动轮播的「应用 → 引用溯源 → 7 维评分」网页 demo（浏览器打开即播放）
  - demo.gif   ：同上内容的 GIF（若系统有中文字体则生成；无字体则跳过）

示例数据来自本地真实指标（data_cache/indicators.json），经 baseline 生成 + 规则评估，
确保 demo 内容真实可复现、无需 API key。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import data_store
import baseline_app
import evaluator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_HTML = os.path.join(ROOT, "demo.html")
OUT_GIF = os.path.join(ROOT, "demo.gif")


def make_example(code, year, subtask, question):
    sample = {"subtask": subtask, "input": question, "difficulty": "中"}
    out = baseline_app.baseline_generate(sample)
    res = evaluator.evaluate(sample, out)
    return {
        "code": code, "year": year, "subtask": subtask, "question": question,
        "answer": out.get("answer", ""),
        "citations": out.get("citations") or [],
        "overall": res["overall"],
        "dims": res["dimensions"],
        "failure": res["failure_mode"],
    }


def main():
    ex_a = make_example(
        "600519", "2023", "财报指标提取",
        "请从贵州茅台（600519）2023年财务数据中提取【扣非净利润】的数值。")
    ex_b = make_example(
        "600519", "2023", "公告摘要",
        "请摘要贵州茅台（600519）2023年相关公告的关键要素，不编造具体数字。")

    # ---------- 1) demo.html（自动轮播） ----------
    def dim_rows(ex):
        rows = ""
        for d, w in config.DIMENSION_WEIGHTS.items():
            v = ex["dims"].get(d, 0)
            pct = int(v * 100)
            color = "#1D9E75" if v >= 0.8 else ("#BA7517" if v >= 0.5 else "#A32D2D")
            rows += (
                f'<div class="row"><span class="lab">{config.DIMENSION_NAMES[d]}</span>'
                f'<span class="bar"><i style="width:{pct}%;background:{color}"></i></span>'
                f'<span class="val">{v:.2f}</span></div>')
        return rows

    def cite_html(ex):
        if not ex["citations"]:
            return '<p class="muted">（无结构化 citations）</p>'
        items = "".join(
            f"<li>{c.get('field')} = {c.get('value')} "
            f"[{c.get('company')} {c.get('year')}]</li>"
            for c in ex["citations"] if isinstance(c, dict))
        return f'<ul class="cite">{items}</ul>'

    slides = f'''
    <section class="slide">
      <h1>📊 HyFinEval · 财报速读助手</h1>
      <p class="lead">基于腾讯混元 Hy3 的金融分析应用：指标提取 / 财务问答 / 公告摘要。</p>
      <p class="muted">真实用户场景：基金 / 行业分析师把一份财报从 30 分钟读到 1 分钟——
      输出带引用、可溯源，并实时给出 7 维评分。</p>
    </section>
    <section class="slide">
      <h2>① 输入</h2>
      <div class="kv"><b>股票代码</b> {ex_a['code']}（{data_store.company_name(ex_a['code'])}）
        &nbsp; <b>报告期</b> {ex_a['year']} &nbsp; <b>任务</b> {ex_a['subtask']}</div>
      <div class="q">{ex_a['question']}</div>
    </section>
    <section class="slide">
      <h2>② 模型输出 + ③ 引用溯源</h2>
      <div class="a">{ex_a['answer']}</div>
      {cite_html(ex_a)}
    </section>
    <section class="slide">
      <h2>④ 7 维评分（综合 {ex_a['overall']}）</h2>
      {dim_rows(ex_a)}
      <p class="muted">失败模式：{ex_a['failure']}</p>
    </section>
    <section class="slide">
      <h2>另一示例 · 公告摘要</h2>
      <div class="q">{ex_b['question']}</div>
      <div class="a">{ex_b['answer']}</div>
      <h2>7 维评分（综合 {ex_b['overall']}）</h2>
      {dim_rows(ex_b)}
    </section>
    '''

    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HyFinEval Demo</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0;font-family:-apple-system,"Microsoft YaHei",sans-serif;}}
body{{background:#0f1729;color:#e8edf5;}}
.wrap{{max-width:880px;margin:0 auto;padding:24px;}}
.slide{{display:none;animation:fade 1s;background:#16213e;border:1px solid #2a3a5e;
  border-radius:14px;padding:28px;min-height:420px;}}
.slide.active{{display:block;}}
h1{{font-size:26px;margin-bottom:12px;}}
h2{{font-size:20px;margin:8px 0 14px;color:#9fe1cb;}}
.lead{{font-size:16px;margin-bottom:10px;}}
.muted{{color:#8b9bbd;font-size:13px;margin-top:10px;}}
.q{{background:#0f1729;border-left:3px solid #1D9E75;padding:12px 14px;border-radius:8px;margin:10px 0;}}
.a{{background:#0f1729;padding:12px 14px;border-radius:8px;line-height:1.7;}}
.kv{{font-size:14px;margin-bottom:10px;}}
.cite{{margin:10px 0 0 18px;font-size:13px;color:#9fe1cb;}}
.row{{display:flex;align-items:center;gap:10px;margin:7px 0;font-size:13px;}}
.lab{{width:160px;flex:none;}}
.bar{{flex:1;height:10px;background:#0f1729;border-radius:6px;overflow:hidden;}}
.bar i{{display:block;height:100%;}}
.val{{width:42px;text-align:right;color:#9fe1cb;}}
.barwrap{{margin-top:8px;}}
@keyframes fade{{from{{opacity:0}}to{{opacity:1}}}}
</style></head><body><div class="wrap">
{slides}
<div class="barwrap"><div id="prog" style="height:4px;background:#1D9E75;width:0;transition:width .3s"></div></div>
</div>
<script>
var s=document.querySelectorAll('.slide');var i=0;s[0].classList.add('active');
var N=s.length,T=4000;
setInterval(function(){{s[i].classList.remove('active');i=(i+1)%N;s[i].classList.add('active');
  document.getElementById('prog').style.width=((i+1)/N*100)+'%';}},T);
</script></body></html>'''
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print("[demo] 已生成", OUT_HTML)

    # ---------- 2) demo.gif（若中文字体可用） ----------
    try:
        from PIL import Image, ImageDraw, ImageFont
        font_path = "C:/Windows/Fonts/msyh.ttc"
        font = ImageFont.truetype(font_path, 22) if os.path.exists(font_path) else ImageFont.load_default()
        font_s = ImageFont.truetype(font_path, 16) if os.path.exists(font_path) else ImageFont.load_default()
    except Exception as e:
        print("[demo] 跳过 GIF（无 PIL 或中文字体）：", e)
        return

    W, H = 800, 480
    BG = (255, 255, 255)
    DARK = (30, 40, 60)
    GREEN = (29, 158, 117)
    AMBER = (186, 117, 23)
    RED = (163, 45, 45)

    def frame(title, lines):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((30, 24), title, fill=DARK, font=font)
        y = 80
        for ln, col in lines:
            d.text((30, y), ln, fill=col, font=font_s)
            y += 34
        return img

    def bar_lines(ex):
        out = []
        for dk, w in config.DIMENSION_WEIGHTS.items():
            v = ex["dims"].get(dk, 0)
            col = GREEN if v >= 0.8 else (AMBER if v >= 0.5 else RED)
            out.append((f"{config.DIMENSION_NAMES[dk]}  {v:.2f}", col))
        return out

    frames = [
        frame("HyFinEval · 财报速读助手", [
            ("基于腾讯混元 Hy3 的金融分析应用", DARK),
            ("场景：分析师读财报 → 1 分钟带引用解读", (90, 100, 120)),
            ("输出可溯源 + 实时 7 维评分", (90, 100, 120)),
        ]),
        frame("① 输入", [
            (f"代码 600519（贵州茅台）  年份 2023  任务 财报指标提取", DARK),
            (ex_a["question"], (90, 100, 120)),
        ]),
        frame("② 输出 + ③ 引用溯源", [
            (ex_a["answer"], DARK),
        ] + [(f"  引用：{c.get('field')} = {c.get('value')} [{c.get('company')} {c.get('year')}]", GREEN)
             for c in ex_a["citations"] if isinstance(c, dict)]),
        frame(f"④ 7 维评分（综合 {ex_a['overall']}）", bar_lines(ex_a)),
        frame("另一示例 · 公告摘要", [
            (ex_b["question"], DARK),
            (ex_b["answer"], (90, 100, 120)),
            (f"综合评分：{ex_b['overall']}", GREEN),
        ]),
    ]
    frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:],
                  duration=3500, loop=0, optimize=False)
    print("[demo] 已生成", OUT_GIF)


if __name__ == "__main__":
    main()
