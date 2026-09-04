# -*- coding: utf-8 -*-
"""为重建后的「公告摘要」样本调用 Hy3 生成输出（开卷/闭卷配对）。

与 src/build_label_pool.py 的分工：那个脚本面向指标类样本（走 indicators RAG），
本脚本面向公告类样本——上下文是**公告原文本身**，不走指标表，因此单独实现。

用法
----
    python src/gen_announcement_outputs.py                # 全部生成
    python src/gen_announcement_outputs.py --limit 6      # 先跑 6 条试通
    python src/gen_announcement_outputs.py --only 开卷     # 只生成开卷
    python src/gen_announcement_outputs.py --ids FIN-105,FIN-107

输出
----
    data_cache/ann_outputs.json
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import hy3_app  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "data_cache", "ann_samples.json")
OUT = os.path.join(ROOT, "data_cache", "ann_outputs.json")

SYSTEM_OPEN = (
    "你是一名严谨的金融公告分析助手。已提供【公告原文】。请输出一个 JSON 对象，结构：\n"
    "{\"answer\":\"多小节分析（Markdown，必须含 ## 公告类型 / ## 关键要素 / "
    "## 投资者要点 三个小节，有实质内容，禁止一两句话敷衍）\","
    "\"citations\":[{\"source\":\"announcement\",\"code\":\"股票代码\","
    "\"title\":\"公告标题\",\"excerpt\":\"原文摘录（不超过60字）\"}]}\n"
    "规则：\n"
    "1. 只输出一个 JSON 对象，不要任何额外解释文字。\n"
    "2. answer 必须分小节、有实质内容。\n"
    "3. 所有具体数值、比例、日期必须来自【公告原文】，并在 citations 中给出对应 excerpt。\n"
    "4. 不得引入原文之外的任何具体数字；原文未提及的内容写「原文未披露」。\n"
    "5. 关键要素要写全：涉及主体、金额/比例、关键日期、审议程序等，逐条列出。\n"
)

SYSTEM_CLOSED = (
    "你是一名严谨的金融公告分析助手。当前为闭卷模式，未提供公告原文，只有公告标题。"
    "请输出一个 JSON 对象，结构：\n"
    "{\"answer\":\"多小节分析（Markdown，必须含 ## 公告类型 / ## 关键要素 / "
    "## 投资者要点 三个小节，有实质内容，禁止一两句话敷衍）\","
    "\"citations\":[]}\n"
    "规则：\n"
    "1. 只输出一个 JSON 对象，不要任何额外解释文字。\n"
    "2. 可以基于金融知识说明【该类公告通常应包含】哪些关键要素。\n"
    "3. 不得编造本公告的具体金额、比例、日期；未掌握的具体数值写"
    "「需以公告原文为准」。\n"
    "4. citations 为空数组。\n"
)


def build_messages(sample):
    system = SYSTEM_OPEN if sample["mode"] == "开卷" else SYSTEM_CLOSED
    user = sample["input"] + "\n\n请只输出 JSON。"
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def parse_out(text):
    """解析模型返回的 JSON；失败时降级为纯文本答案。

    关键：形如 '{\n "answer": "## 公告类型...' 这样**没有闭合括号**的截断响应
    必须判为失败（返回 None）以便重试，不能当答案收下——否则会得到 34 字的残片，
    混进标注池后是纯噪声。
    """
    if not text or text.startswith("[HY3_ERROR]"):
        return None
    stripped = text.strip()
    if stripped.startswith("{") and "}" not in stripped:
        return None
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1:
        try:
            obj = json.loads(text[s:e + 1])
            if isinstance(obj, dict) and obj.get("answer"):
                return {
                    "answer": str(obj["answer"]).strip(),
                    "citations": obj.get("citations") or [],
                }
        except Exception:
            pass
    return {"answer": text.strip(), "citations": [], "_parse_fallback": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", choices=["开卷", "闭卷"], default=None)
    ap.add_argument("--ids", default="")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8,
                    help="并发线程数（推理模型单条约 60s，串行跑 124 条太慢）")
    ap.add_argument("--pairs", type=int, default=0,
                    help="只生成 N 组开卷/闭卷配对，按等距抽样保持公司/类型多样")
    args = ap.parse_args()

    if not os.path.exists(SAMPLES):
        print("未找到 %s，请先运行 src/build_announcement_samples.py" % SAMPLES)
        return
    samples = json.load(open(SAMPLES, encoding="utf-8"))
    if args.only:
        samples = [s for s in samples if s["mode"] == args.only]

    # 分层抽样：按配对组等距抽取，避免只取到前几家公司的同类公告
    if args.pairs:
        opens = [s for s in samples if s["mode"] == "开卷"]
        if args.pairs < len(opens):
            step = len(opens) / args.pairs
            chosen = [opens[int(i * step)] for i in range(args.pairs)]
            keep = set()
            for s in chosen:
                keep.add(s["id"])
                keep.add(s["pair_id"])
            samples = [s for s in samples if s["id"] in keep]

    if args.ids:
        want = set(x.strip() for x in args.ids.split(",") if x.strip())
        samples = [s for s in samples if s["id"] in want]
    if args.limit:
        samples = samples[: args.limit]

    done = {}
    if os.path.exists(OUT):
        for r in json.load(open(OUT, encoding="utf-8")):
            done[r["id"]] = r
    todo = [s for s in samples if s["id"] not in done]
    print("待生成 %d 条（已完成 %d 条，将跳过），并发 %d"
          % (len(todo), len(done), args.workers))
    if not todo:
        print("无待生成样本")
        return

    results = list(done.values())
    results_lock = __import__("threading").Lock()

    def gen_one(s):
        """生成单条；失败返回 None（调用方负责记录）。"""
        for attempt in range(3):
            temp = 0.2 if attempt == 0 else 0.6
            out = hy3_app.call_hy3(build_messages(s), max_tokens=8192,
                                   temperature=temp)
            parsed = parse_out(out)
            if parsed and not parsed.get("_parse_fallback"):
                return s, parsed
            time.sleep(1.5 * (attempt + 1))
        # 三次都失败：最后一次若有内容也接受（宽松兜底）
        return s, (parse_out(out) or None)

    finished = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(gen_one, s): s for s in todo}
        for fut in as_completed(futs):
            s, parsed = fut.result()
            finished += 1
            if not parsed:
                print("[%d/%d] %s ✗ 失败" % (finished, len(todo), s["id"]))
                continue
            rec = {
                "id": s["id"], "subtask": s["subtask"],
                "difficulty": s["difficulty"],
                "input": s["input"], "output": parsed["answer"],
                "citations": parsed["citations"],
                "mode": s["mode"], "pair_id": s["pair_id"],
                "type": s["type"], "code": s["code"], "name": s["name"],
                "title": s["title"], "time": s["time"],
                "n_facts": s["n_facts"],
                "ground_truth_facts": s["ground_truth_facts"],
                "_context_for_labeler": s.get("context_text", ""),
            }
            if parsed.get("_parse_fallback"):
                rec["_parse_fallback"] = True
            with results_lock:
                results.append(rec)
                # 每条落盘：进程被杀也不丢已完成的部分。
                # 必须原子写（临时文件 + rename）：直接覆写时若进程被杀在
                # json.dump 中途，会留下半截 JSON，下次启动无法解析（已发生过）。
                tmp = OUT + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=1)
                os.replace(tmp, OUT)
            print("[%d/%d] %s %s ✓ %d 字"
                  % (finished, len(todo), s["id"], s["mode"], len(rec["output"])))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    ok = sum(1 for r in results if r["id"] in {s["id"] for s in samples})
    print("\n完成：本批可用 %d 条，文件累计 %d 条 -> %s" % (ok, len(results), OUT))


if __name__ == "__main__":
    main()
