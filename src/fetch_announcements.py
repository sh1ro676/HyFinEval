# -*- coding: utf-8 -*-
"""从巨潮资讯网抓取真实公告原文，构建公告摘要任务的"开卷"知识源。

背景
----
原先 16 条「公告摘要」样本只有巨潮披露列表的**标题**，没有正文，模型只能闭卷
输出"该类公告通常应包含…具体数值以原文为准"的通用框架 —— 这导致输出之间缺乏
可判别的信息增量差异，人工三档标注的效度存疑（见 README E.5/E.6）。

本脚本补齐这一环：抓真实公告 PDF → 抽取正文 → 落盘为结构化原文库，供后续
开卷样本生成与 faithfulness（忠实度）核验使用。

用法
----
    python src/fetch_announcements.py                      # 默认配置全量抓取
    python src/fetch_announcements.py --per-type 3         # 每类每公司抓 3 篇
    python src/fetch_announcements.py --companies 600519,000858
    python src/fetch_announcements.py --dry-run            # 只列不下载

依赖：pypdf（仅本脚本需要，已装在隔离 venv；缺失时自动提示）
输出：data_cache/announcement_texts.json；PDF 缓存于 raw/ann_pdf/
"""
import argparse
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "data_cache", "announcement_texts.json")
PDF_DIR = os.path.join(ROOT, "raw", "ann_pdf")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
HEADERS = {
    "User-Agent": UA,
    "Accept": "*/*",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
}
STATIC = "http://static.cninfo.com.cn/"

# 目标公司（代码 -> 简称，仅用于校验抓到的公告归属）
COMPANIES = {
    "600519": "贵州茅台",
    "000858": "五粮液",
    "300750": "宁德时代",
    "002594": "比亚迪",
    "600036": "招商银行",
    "601318": "中国平安",
    "601012": "隆基绿能",
    "600887": "伊利股份",
    "002415": "海康威视",
    "000333": "美的集团",
}

# ---- A 股交易所判定：按股票代码前缀映射，取代简化的 "6 开头=沪" 判断 ----
# 巨潮资讯的 column 参数：sse=上交所 / szse=深交所。
# 前缀规则（A 股实际编码）：
#   沪市（sse）：600/601/603/605 主板，688 科创板，689 科创板存托凭证
#   深市（szse）：000/001 深主板，002/003 中小板(已并入主板)，300/301 创业板
#   例外（代码不以 6 开头但属沪市，或反之）：
#     900xxx B 股 -> 沪市；200xxx B 股 -> 深市（本项目不涉及 B 股，映射已含以防扩展）
# 巨潮查询接口对 6 位 A 股代码用上述两列即可，前缀不匹配时默认按首位兜底。
_SSE_PREFIXES = ("600", "601", "603", "605", "688", "689", "900")
_SZSE_PREFIXES = ("000", "001", "002", "003", "300", "301", "200")


def exchange_of(code):
    """按代码前缀判定交易所巨潮 column（sse/szse）；无法识别时按首位数字兜底。"""
    code = str(code).strip()
    if code.startswith(_SSE_PREFIXES):
        return "sse"
    if code.startswith(_SZSE_PREFIXES):
        return "szse"
    # 兜底：沪市主板均以 6 开头（历史行为），其余归深市
    return "sse" if code.startswith("6") else "szse"


# 公告类型 -> 标题关键词（按顺序首个命中为准）
# 顺序有讲究：「限制性股票/股票期权」必须先于「回购」判断，否则
# "关于部分限制性股票回购注销完成的公告" 会被误分为股份回购（实为股权激励）。
TYPE_RULES = [
    ("股权激励", ["激励计划", "股票期权", "限制性股票"]),
    ("利润分配/分红", ["分红", "派息", "权益分派", "利润分配"]),
    ("股份回购", ["回购"]),
    ("股东/高管增持", ["增持"]),
    ("业绩预告/快报", ["业绩说明会", "业绩预告", "业绩快报", "生产经营情况"]),
]


def _post(url, data):
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=body, headers=dict(
        HEADERS, **{"Content-Type": "application/x-www-form-urlencoded"}))
    return json.loads(urllib.request.urlopen(req, timeout=25).read().decode("utf-8"))


def get_orgid(code):
    """用巨潮搜索接口取 orgId；优先取 code 完全匹配的条目。"""
    try:
        res = _post("http://www.cninfo.com.cn/new/information/topSearch/query",
                    {"keyWord": code, "maxNum": 10}) or []
    except Exception as e:
        print("  [orgId] 查询失败 %s: %s" % (code, e))
        return None
    for x in res:
        if x.get("code") == code:
            return x.get("orgId")
    return res[0].get("orgId") if res else None


def query_list(code, orgid, start, end, column, max_pages=8):
    """拉取某公司一段时间内的全部公告（临时公告，不限类别）。

    巨潮每页上限 30 条，必须翻页才能覆盖完整时间窗；否则只会拿到最近 30 条，
    早于该窗口的公告（例如 2022 年）会被静默漏掉。
    """
    out, total = [], None
    for page in range(1, max_pages + 1):
        data = {
            "pageNum": page, "pageSize": 30, "column": column, "tabName": "fulltext",
            "stock": "%s,%s" % (code, orgid), "searchkey": "", "secid": "",
            "plate": "", "category": "", "trade": "",
            "seDate": "%s~%s" % (start, end),
            "sortName": "", "sortType": "", "isHLtitle": "true",
        }
        try:
            res = _post("http://www.cninfo.com.cn/new/hisAnnouncement/query", data)
        except Exception as e:
            print("   [第%d页] 查询失败：%s" % (page, e))
            break
        anns = (res or {}).get("announcements") or []
        if total is None:
            total = (res or {}).get("totalAnnouncement") or 0
        out.extend(anns)
        if not anns or len(out) >= total:
            break
        time.sleep(0.3)
    return {"announcements": out, "totalAnnouncement": total or len(out)}


def classify(title):
    for t, kws in TYPE_RULES:
        if any(k in title for k in kws):
            return t
    return None


def clean_text(raw):
    """清洗 PDF 抽取文本：去页码行、压缩空行、规整空白。"""
    lines = []
    for ln in raw.splitlines():
        s = ln.strip()
        # 纯页码 / 过短噪声行
        if re.fullmatch(r"\d{1,3}", s):
            continue
        if re.fullmatch(r"[-—–_·\.\s]{0,6}", s):
            continue
        lines.append(s)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def pdf_to_text(pdf_bytes):
    try:
        from pypdf import PdfReader
    except ImportError:
        print("  缺少 pypdf：pip install pypdf")
        return None
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception:
        return None
    buf = []
    for pg in reader.pages:
        try:
            buf.append(pg.extract_text() or "")
        except Exception:
            continue
    return clean_text("\n".join(buf))


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        with open(dest, "rb") as f:
            return f.read()
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            data = urllib.request.urlopen(req, timeout=40).read()
            if len(data) < 1024:
                raise ValueError("too small")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(data)
            time.sleep(0.4)
            return data
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--companies", default=",".join(COMPANIES))
    ap.add_argument("--per-type", type=int, default=2, help="每公司每类最多抓几篇")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2024-06-30")
    ap.add_argument("--min-chars", type=int, default=300, help="正文最短字数（过滤扫描件）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(PDF_DIR, exist_ok=True)
    codes = [c.strip() for c in args.companies.split(",") if c.strip()]
    total_ok = 0
    records = []

    for code in codes:
        name = COMPANIES.get(code, code)
        column = exchange_of(code)
        orgid = get_orgid(code)
        if not orgid:
            print("[%s %s] 无法获取 orgId，跳过" % (code, name))
            continue
        try:
            res = query_list(code, orgid, args.start, args.end, column)
        except Exception as e:
            print("[%s %s] 列表查询失败：%s" % (code, name, e))
            continue
        anns = res.get("announcements") or []
        print("[%s %s] 列表 %d 条" % (code, name, len(anns)))

        # 按类型分桶，每桶取前 N 篇
        buckets = {}
        for a in anns:
            title = (a.get("announcementTitle") or "").strip()
            t = classify(title)
            if not t:
                continue
            # 过滤法律意见书/会议资料等低信息密度附件（除非确属该类型唯一来源）
            if any(x in title for x in ("法律意见书", "会议资料", "章程", "管理办法")):
                continue
            buckets.setdefault(t, []).append(a)

        for t, items in buckets.items():
            picked = items[: args.per_type]
            for a in picked:
                title = (a.get("announcementTitle") or "").strip()
                url = STATIC + (a.get("adjunctUrl") or "")
                ts = a.get("announcementTime")
                date = ""
                if ts:
                    date = time.strftime("%Y-%m-%d", time.localtime(ts / 1000))
                fname = re.sub(r"[^\w.\-]", "_", code + "_" + date + "_" + title)[:90] + ".pdf"
                dest = os.path.join(PDF_DIR, fname)
                print("   [%s] %s | %s" % (t, title[:44], date))
                if args.dry_run:
                    records.append({"code": code, "title": title, "type": t, "time": date})
                    continue
                data = download(url, dest)
                if not data:
                    print("     -> 下载失败")
                    continue
                text = pdf_to_text(data)
                if not text or len(text) < args.min_chars:
                    print("     -> 正文过短(%s 字符)，判为扫描件，丢弃"
                          % (len(text) if text else 0))
                    continue
                records.append({
                    "code": code, "name": name, "title": title, "type": t,
                    "time": date, "url": url, "chars": len(text), "text": text,
                })
                total_ok += 1
                print("     -> OK %d 字" % len(text))

    if args.dry_run:
        print("\n[dry-run] 未下载，共列出 %d 个候选" % len(records))
        return

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    print("\n完成：成功 %d 篇 -> %s" % (total_ok, OUT_PATH))
    dist = {}
    for r in records:
        dist[r["type"]] = dist.get(r["type"], 0) + 1
    for k, v in sorted(dist.items(), key=lambda x: -x[1]):
        print("  %-14s %d 篇" % (k, v))


if __name__ == "__main__":
    main()
