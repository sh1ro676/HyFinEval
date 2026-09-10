# -*- coding: utf-8 -*-
"""真实成对比较：让严格审计员 Hy3 裁判直接比较 "A vs B 哪个更好"。

改进（2026-09-10）：
- 统一使用严格审计员角色（委员会中对齐度最高的角色）
- 支持从 label_pool.json 直接加载（28 条人工标注样本）
- 邻居采样基于严格审计员绝对分排序
- 新增 Bradley-Terry 聚合与人工标注对齐验证

采样策略（高效）：按严格审计员分数排序后，每个样本只与前后 2 个邻居比较，
聚焦"排名接近、最难区分"的 pair。每对做左右互换消除位置偏置。

用法：
  python src/pairwise_judge.py --workers 10
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import hy3_app
import rubric


def load_json(path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full = os.path.join(root, path) if not os.path.isabs(path) else path
    with open(full, encoding="utf-8") as f:
        return json.load(f)


PAIRWISE_PROMPT_STRICT = (
    "你是一位四大会计事务所的严格审计员。请将以下两份模型输出进行直接比较，"
    "判断哪一份质量更高。你的评估标准极其严格：\n"
    "1) 数字是否精确——任何未注明来源的具体数字都视为可疑；\n"
    "2) 引用是否可追溯——每一条信息都必须有明确出处；\n"
    "3) 对不确定信息是否诚实声明——说\"以原文为准\"比硬编数字得分高；\n"
    "4) 有没有胡编乱造或免责声明充数。\n\n"
    "只回答一个字：A 或 B。不要解释理由。"
)


def compare_pair(sample_a, output_a, sample_b, output_b, swap: bool = False):
    """调用 Hy3 比较两个输出。swap=True 时互换 A/B 位置以检测位置偏置。
    含 429 限流重试（最多 3 次，退避 3/6/12 秒）。"""
    import time
    if not config.USE_HY3:
        return None
    if swap:
        sample_a, sample_b = sample_b, sample_a
        output_a, output_b = output_b, output_a

    text_a = output_a.get("answer", "") if isinstance(output_a, dict) else str(output_a)
    text_b = output_b.get("answer", "") if isinstance(output_b, dict) else str(output_b)

    # 截断到 1000 字符，控制 prompt 长度
    text_a = text_a[:1000]
    text_b = text_b[:1000]

    user = (
        f"【题目】{sample_a.get('input', '')}\n\n"
        f"--- 输出 A ---\n{text_a}\n\n"
        f"--- 输出 B ---\n{text_b}\n\n"
        f"哪一份输出质量更高？只回答 A 或 B。"
    )

    for attempt in range(3):
        res = hy3_app.call_hy3([
            {"role": "system", "content": PAIRWISE_PROMPT_STRICT},
            {"role": "user", "content": user},
        ], temperature=0.0, max_tokens=256)
        if res and not res.startswith("[HY3_ERROR]"):
            break
        # 429 或其他错误，退避重试
        time.sleep(3 * (2 ** attempt))
    else:
        return None

    res = res.strip().upper()
    if "A" in res and "B" not in res:
        winner = "B" if swap else "A"
    elif "B" in res and "A" not in res:
        winner = "A" if swap else "B"
    else:
        return None
    return {"winner": winner, "swap": swap}


def bt_mle_from_pairwise(pairs_wins, ids, max_iter=100, tol=1e-5):
    """从 pairwise 胜负结果做 Bradley-Terry MLE 聚合。"""
    n = len(ids)
    if n < 2:
        return {}
    idx_map = {sid: i for i, sid in enumerate(ids)}
    abilities = [0.0] * n

    for _ in range(max_iter):
        old = abilities[:]
        for i in range(n):
            sid_i = ids[i]
            numer = 0.0
            denom = 0.0
            for j in range(n):
                if i == j:
                    continue
                sid_j = ids[j]
                w_ij = pairs_wins.get((sid_i, sid_j), 0)
                w_ji = pairs_wins.get((sid_j, sid_i), 0)
                total = w_ij + w_ji
                if total == 0:
                    continue
                numer += total * (w_ij / total)
                pi = 1.0 / (1.0 + (abilities[j] - abilities[i]))
                denom += total * pi
            if denom > 0:
                abilities[i] = numer / denom
        if all(abs(abilities[i] - old[i]) < tol for i in range(n)):
            break

    # z-score 标准化
    import statistics
    mu = statistics.mean(abilities)
    sd = statistics.stdev(abilities) if n > 1 else 1.0
    if sd == 0:
        sd = 1.0
    return {sid: (abilities[idx_map[sid]] - mu) / sd for sid in ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data_cache/label_pool.json", help="样本来源")
    ap.add_argument("--auditor-scores", default="src/results/strict_auditor_all.json", help="严格审计员绝对分（用于排序采样）")
    ap.add_argument("--out", default="src/results/pairwise_judge.json")
    ap.add_argument("--workers", type=int, default=3, help="并发数（建议 ≤3 避免 429 限流）")
    ap.add_argument("--neighbor", type=int, default=2, help="每个样本与前后几个邻居比较")
    args = ap.parse_args()

    if not config.USE_HY3:
        print("错误：未检测到 HY3_API_KEY，无法进行成对比较。")
        return

    # ---- 加载样本 ----
    samples_raw = load_json(args.source)
    # label_pool.json 是 list
    if isinstance(samples_raw, list):
        samples_data = samples_raw
    elif isinstance(samples_raw, dict) and "samples" in samples_raw:
        samples_data = samples_raw["samples"]
    else:
        samples_data = list(samples_raw.values())

    # 加载严格审计员分数用于排序
    auditor_scores = {}
    if os.path.exists(args.auditor_scores):
        aud = load_json(args.auditor_scores)
        for r in aud.get("samples", []):
            if r.get("status") == "ok":
                auditor_scores[r["id"]] = r["overall"]

    samples = []
    for s in samples_data:
        sid = s.get("id")
        if not sid:
            continue
        samples.append({
            "id": sid,
            "input": s.get("input", ""),
            "output": {"answer": s.get("output", ""), "citations": s.get("citations", [])},
            "auditor_score": auditor_scores.get(sid, 50.0),
        })

    n = len(samples)
    if n < 3:
        print(f"有效样本仅 {n} 条，不足以做 pairwise。")
        return

    # 按严格审计员分数排序
    samples.sort(key=lambda x: x["auditor_score"])
    id_to_idx = {s["id"]: i for i, s in enumerate(samples)}

    # 生成邻居 pair
    pairs = []
    seen = set()
    for i in range(n):
        for offset in range(1, args.neighbor + 1):
            j = i + offset
            if j >= n:
                continue
            a, b = samples[i], samples[j]
            key = tuple(sorted([a["id"], b["id"]]))
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"a": a["id"], "b": b["id"], "sample_a": a, "sample_b": b})

    print(f"样本数: {n} | 生成 pair 数: {len(pairs)} | 含左右互换共 {len(pairs)*2} 次调用")

    results = []
    import time
    # 串行执行，避免 429 限流；每对之间加 3 秒延迟
    for idx, p in enumerate(pairs):
        # 原始顺序
        r = compare_pair(p["sample_a"], p["sample_a"]["output"],
                         p["sample_b"], p["sample_b"]["output"], False)
        if r:
            results.append({"a": p["a"], "b": p["b"], "swap": False, "winner": r["winner"]})
        else:
            print(f"  {p['a']} vs {p['b']} (正常) 失败")
        time.sleep(3)
        # 互换顺序
        r = compare_pair(p["sample_a"], p["sample_a"]["output"],
                         p["sample_b"], p["sample_b"]["output"], True)
        if r:
            results.append({"a": p["a"], "b": p["b"], "swap": True, "winner": r["winner"]})
        else:
            print(f"  {p['a']} vs {p['b']} (互换) 失败")
        time.sleep(3)
        print(f"  进度: {idx+1}/{len(pairs)} pair 完成")

    # ---- 统计位置偏置 ----
    consistent = 0
    total_swap_pairs = 0
    pair_results = {}
    for r in results:
        key = tuple(sorted([r["a"], r["b"]]))
        if key not in pair_results:
            pair_results[key] = {"normal": None, "swap": None}
        if r["swap"]:
            pair_results[key]["swap"] = r["winner"]
        else:
            pair_results[key]["normal"] = r["winner"]

    for key, vals in pair_results.items():
        if vals["normal"] is not None and vals["swap"] is not None:
            total_swap_pairs += 1
            if (vals["normal"] == key[0] and vals["swap"] == key[0]) or \
               (vals["normal"] == key[1] and vals["swap"] == key[1]):
                consistent += 1

    swap_consistency = consistent / total_swap_pairs if total_swap_pairs > 0 else 0

    # ---- Bradley-Terry 聚合 ----
    pairs_wins = {}
    for r in results:
        if not r.get("winner"):
            continue
        winner = r["winner"]
        loser = r["b"] if winner == r["a"] else r["a"]
        pairs_wins[(winner, loser)] = pairs_wins.get((winner, loser), 0) + 1

    ids = [s["id"] for s in samples]
    bt_abilities = bt_mle_from_pairwise(pairs_wins, ids)

    # ---- 与人工标注对齐（如果 source 是 label_pool）----
    human_align = {}
    human_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_cache/human_labels.json")
    if os.path.exists(human_path):
        human = load_json("data_cache/human_labels.json")
        def band_to_score(b):
            return {"优": 90, "中": 60, "差": 30}.get(b, 60)

        bt_scores_list = []
        human_scores_list = []
        human_bands_list = []
        for sid in ids:
            if sid not in bt_abilities:
                continue
            h = human.get(sid)
            if not h:
                continue
            ann = h.get("A") or h.get("B") or next(iter(h.values()), None) if isinstance(h, dict) else None
            if not ann:
                continue
            bt_scores_list.append(bt_abilities[sid])
            human_scores_list.append(band_to_score(ann.get("band", "中")))
            human_bands_list.append(ann.get("band", "中"))

        if len(bt_scores_list) >= 2:
            import stats_utils
            rho = stats_utils.spearman(bt_scores_list, human_scores_list)
            mae = sum(abs(bt_scores_list[i] - human_scores_list[i]) for i in range(len(bt_scores_list))) / len(bt_scores_list)
            auto_bands = ["优" if s >= 0.5 else "中" if s >= -0.5 else "差" for s in bt_scores_list]
            def bidx(b):
                return {"优": 2, "中": 1, "差": 0}.get(b, 1)
            kappa = stats_utils.quad_weighted_kappa(
                [bidx(b) for b in auto_bands],
                [bidx(b) for b in human_bands_list],
                k=3,
            )
            human_align = {
                "n": len(bt_scores_list),
                "kappa": round(kappa, 3) if kappa is not None else None,
                "spearman": round(rho, 3) if rho is not None else None,
                "mae": round(mae, 2),
            }

    report = {
        "n_samples": n,
        "n_pairs": len(pairs),
        "n_comparisons": len(results),
        "swap_consistency": round(swap_consistency, 3),
        "bt_abilities": bt_abilities,
        "human_alignment": human_align,
        "comparisons": results,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"成对比较完成：{len(results)}/{len(pairs)*2} 次调用成功")
    print(f"左右互换一致性: {swap_consistency:.1%} ({consistent}/{total_swap_pairs})")
    if human_align:
        print(f"BT聚合 vs 人工标注: κ={human_align['kappa']}, ρ={human_align['spearman']}, MAE={human_align['mae']}, n={human_align['n']}")
    print(f"结果已写入 {args.out}")


if __name__ == "__main__":
    main()
