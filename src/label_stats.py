"""人工对齐统计（B 组）。

读取 build_label_pool.py 生成的 {id: auto_score} 与 human_label.py 标注的
{id: {annotator: score}}，计算：
  - auto vs 人工 的 Spearman 秩相关（合法性证据）
  - 两位标注者间的 加权 Kappa（quadratic，序值一致性）
  - 平均绝对分差
写出 results/human_alignment.json 并打印摘要。

用法：  python src/label_stats.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import data_store


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


def quad_weighted_kappa(a, b, k=5):
    """把 0–100 分桶为 k 档，计算二次加权 Kappa（序值一致性）。"""
    def band(x):
        return min(k - 1, int(x * k / 100))
    ba, bb = [band(x) for x in a], [band(x) for x in b]
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


def main():
    pool = json.load(open(os.path.join(config.ROOT_DIR, "data_cache", "label_pool.json"), encoding="utf-8"))
    lab = {}
    lp = os.path.join(config.ROOT_DIR, "data_cache", "human_labels.json")
    if os.path.exists(lp):
        lab = json.load(open(lp, encoding="utf-8"))

    auto = {p["id"]: p["auto_score"] for p in pool}
    # 取标注者 A（若有 B 则取两人均标的部分做 Kappa）
    A = {i: v["A"] for i, v in lab.items() if "A" in v}
    B = {i: v["B"] for i, v in lab.items() if "B" in v}
    common = [i for i in A if i in auto]

    out = {"n_labeled_A": len(A), "n_labeled_B": len(B)}
    if common:
        ra = [auto[i] for i in common]
        rh = [A[i] for i in common]
        rho = spearman(ra, rh)
        mad = sum(abs(ra[i] - rh[i]) for i in range(len(ra))) / len(ra)
        out["auto_vs_humanA_spearman"] = round(rho, 3) if rho is not None else None
        out["auto_vs_humanA_mae"] = round(mad, 3)
    both = [i for i in A if i in B]
    if both:
        ka = [A[i] for i in both]
        kb = [B[i] for i in both]
        out["inter_annotator_kappa"] = round(quad_weighted_kappa(ka, kb), 3)
        out["n_both"] = len(both)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "human_alignment.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("=" * 50)
    print("人工对齐统计：")
    for k, v in out.items():
        print(f"  {k} = {v}")
    print(f"结果已写入 {config.RESULTS_DIR}/human_alignment.json")


if __name__ == "__main__":
    main()
