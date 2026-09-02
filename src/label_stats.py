# -*- coding: utf-8 -*-
"""人工对齐统计（B 组，轻量标注版）。

读取 build_label_pool.py 生成的 {id: auto_score} 与 human_label.py 标注的
{id: {annotator: {band, flags}}}，计算：
  - auto vs 人工 的 Spearman 秩相关（合法性证据，基于三档映射分）
  - 平均绝对分差 MAE
  - 两位标注者间的 二次加权 Kappa（在 优/中/差 三档上，序值一致性）
  - 各可观测勾选项的 正向信号命中率，及（若有两人）标注者间一致率
写出 results/human_alignment.json 并打印摘要。

用法：  python src/label_stats.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

BAND_MAP = {"优": 90, "中": 60, "差": 30}
BAND_IDX = {"优": 2, "中": 1, "差": 0}
FLAGS = ["format_ok", "cited", "no_redline", "no_contra"]
FLAG_DESC = {
    "format_ok": "格式完整",
    "cited": "引用可溯源",
    "no_redline": "无合规红线",
    "no_contra": "无自相矛盾/硬伤",
}


def spearman(a, b):
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0] * len(x)
        for i, v in enumerate(order):
            r[v] = i + 1
        return r
    if len(a) < 2:
        return None
    ra, rb = rank(a), rank(b)
    m = sum(ra) / len(ra)
    cov = sum((ra[i] - m) * (rb[i] - m) for i in range(len(ra)))
    va = sum((x - m) ** 2 for x in ra) ** 0.5
    vb = sum((x - m) ** 2 for x in rb) ** 0.5
    return cov / (va * vb) if va and vb else None


def quad_weighted_kappa(ba, bb, k=3):
    """在 k 档（已为 0..k-1 整数）上计算二次加权 Kappa。"""
    n = len(ba)
    hist = {}
    for x, y in zip(ba, bb):
        hist[(x, y)] = hist.get((x, y), 0) + 1
    row = [sum(hist.get((i, j), 0) for j in range(k)) for i in range(k)]
    col = [sum(hist.get((i, j), 0) for i in range(k)) for j in range(k)]
    po = sum(hist.get((i, i), 0) for i in range(k)) / n
    pe = sum(row[i] * col[i] for i in range(k)) / (n * n)
    wsum = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            wsum += w * hist.get((i, j), 0) / n
    return 1 - wsum / (1 - pe) if (1 - pe) else 1.0


def _score(entry):
    """标注条目 -> 映射分（兼容旧占位 int）。"""
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, dict) and entry.get("band") in BAND_MAP:
        return float(BAND_MAP[entry["band"]])
    return None


def _idx(entry):
    if isinstance(entry, dict) and entry.get("band") in BAND_IDX:
        return BAND_IDX[entry["band"]]
    return None


def _valid(entry):
    return isinstance(entry, dict) and "band" in entry


def vget(lab, i, ann, *keys):
    node = lab[i].get(ann, {})
    for k in keys:
        if not isinstance(node, dict):
            return False
        node = node.get(k)
    return bool(node)


def main():
    pool = json.load(open(os.path.join(config.ROOT_DIR, "data_cache", "label_pool.json"), encoding="utf-8"))
    lab = {}
    lp = os.path.join(config.ROOT_DIR, "data_cache", "human_labels.json")
    if os.path.exists(lp):
        lab = json.load(open(lp, encoding="utf-8"))

    auto = {p["id"]: p["auto_score"] for p in pool}

    A = {i: _score(v.get("A")) for i, v in lab.items() if isinstance(v, dict) and _score(v.get("A")) is not None}
    B = {i: _score(v.get("B")) for i, v in lab.items() if isinstance(v, dict) and _score(v.get("B")) is not None}
    out = {"n_labeled_A": len(A), "n_labeled_B": len(B)}

    def _align(tag, H):
        common = [i for i in H if i in auto]
        if not common:
            return
        ra = [auto[i] for i in common]
        rh = [H[i] for i in common]
        rho = spearman(ra, rh)
        mad = sum(abs(ra[k] - rh[k]) for k in range(len(ra))) / len(ra)
        out[f"auto_vs_human{tag}_spearman"] = round(rho, 3) if rho is not None else None
        out[f"auto_vs_human{tag}_mae"] = round(mad, 3)
        out[f"n_align{tag}"] = len(common)

    _align("A", A)
    _align("B", B)

    # 标注者间 Kappa（三档）
    Aidx = {i: _idx(v.get("A")) for i, v in lab.items() if _valid(v.get("A"))}
    Bidx = {i: _idx(v.get("B")) for i, v in lab.items() if _valid(v.get("B"))}
    both = [i for i in Aidx if i in Bidx]
    if len(both) >= 2:
        out["inter_annotator_kappa"] = round(quad_weighted_kappa([Aidx[i] for i in both],
                                                                [Bidx[i] for i in both], k=3), 3)
        out["n_both"] = len(both)

    # 可观测勾选项统计
    flag_stats = {}
    for fk in FLAGS:
        aids = [i for i, v in lab.items() if _valid(v.get("A")) and isinstance(v["A"].get("flags"), dict)]
        if aids:
            rate_a = sum(1 for i in aids if vget(lab, i, "A", "flags", fk)) / len(aids)
            rec = {"positive_rate_A": round(rate_a, 3), "n_A": len(aids)}
            bids = [i for i in aids if _valid(lab[i].get("B")) and isinstance(lab[i]["B"].get("flags"), dict)]
            if bids:
                agree = sum(1 for i in bids if vget(lab, i, "A", "flags", fk) == vget(lab, i, "B", "flags", fk))
                rec["agreement_AB"] = round(agree / len(bids), 3)
                rec["n_both"] = len(bids)
            flag_stats[FLAG_DESC[fk]] = rec
    out["flags"] = flag_stats

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "human_alignment.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("=" * 50)
    print("人工对齐统计（轻量标注）：")
    for k, v in out.items():
        print(f"  {k} = {v}")
    print(f"结果已写入 {config.RESULTS_DIR}/human_alignment.json")


if __name__ == "__main__":
    main()
