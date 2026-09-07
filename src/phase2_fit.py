# -*- coding: utf-8 -*-
"""Phase 2：用 28 条人工标注池做 per-subtask 维度重加权 / 拟合打分器。

目标：在已提交基线（auto_score vs 人工 A 的 Spearman ≈ 0.635）之上，
通过留一交叉验证（LOO-CV）拟合 7 维（factual/citation/completeness/format/
safety/computation/calibration）的线性打分器，测排序相关性的诚实上界。

方法学（严格避免过拟合）：
  - 特征 X：每条样本经 evaluator 重算的 7 维分数（[0,1]）。
  - 目标 y：人工标注者 A 的数值分（0–100）。
  - 拟合：以 y 的秩为因变量，对 X 做最小二乘回归得到权重向量 w；
          Spearman(w·X, y) ≈ Pearson(rank(w·X), rank(y))（单调变换下等价）。
  - 诚实估计：全局 LOO-CV——每次留 1 条，在余下 27 条上拟合，对该条预测，
          汇总 28 条 LOO 预测后算 Spearman，作为泛化上界估计。
  - 诊断：per-subtask 组内 in-sample Spearman，看哪些子任务重加权收益最大。

输出：results/phase2.json（含基线/拟合/LOO/per-subtask 数字与权重）。

用法：  python src/phase2_fit.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import numpy as np
import evaluator


BAND_ORDER = {"优": 3.0, "中": 2.0, "差": 1.0}


def _human_score(labels, pid):
    v = labels.get(pid, {}).get("A")
    if isinstance(v, dict):
        return BAND_ORDER.get(v.get("band"))
    return v


DIMS = [
    "factual_accuracy",
    "citation_verifiability",
    "completeness",
    "format",
    "safety_no_hallucination",
    "computation",
    "calibration",
]


def rank_vec(x):
    """平均秩（处理并列）。"""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    sorted_x = np.asarray(x)[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 2:
        return None
    ra, rb = rank_vec(a), rank_vec(b)
    da, db = ra - ra.mean(), rb - rb.mean()
    va, vb = (da @ da) ** 0.5, (db @ db) ** 0.5
    return float((da @ db) / (va * vb)) if va and vb else None


def fit_weights(X, y):
    """最小二乘回归 rank(y) ~ [X, 1]，返回权重向量 w（含截距在末位）。"""
    Xa = np.hstack([X, np.ones((len(X), 1))])
    ry = rank_vec(y)
    w, *_ = np.linalg.lstsq(Xa, ry, rcond=None)
    return w


def predict(X, w):
    Xa = np.hstack([X, np.ones((len(X), 1))])
    return Xa @ w


def main():
    pool_path = os.path.join(config.ROOT_DIR, "data_cache", "label_pool.json")
    lab_path = os.path.join(config.ROOT_DIR, "data_cache", "human_labels.json")
    pool = json.load(open(pool_path, encoding="utf-8"))
    labels = json.load(open(lab_path, encoding="utf-8"))

    # ---- 1. 重算 28 条 7 维（带缓存）----
    cache_path = os.path.join(config.ROOT_DIR, "data_cache", "auto_dims_pool.json")
    if os.path.exists(cache_path):
        dims_cache = json.load(open(cache_path, encoding="utf-8"))
    else:
        dims_cache = {}

    rows = []
    for p in pool:
        pid = p["id"]
        if pid in dims_cache:
            dvec = dims_cache[pid]
        else:
            output = {"answer": p["output"], "citations": p.get("citations", [])}
            res = evaluator.evaluate(p, output, use_hy3_judge=False)
            dvec = {k: float(res["dimensions"][k]) for k in DIMS}
            dims_cache[pid] = dvec
        rows.append({
            "id": pid,
            "subtask": p["subtask"],
            "difficulty": p.get("difficulty"),
            "auto_score": p.get("auto_score"),
            "human_A": _human_score(labels, pid),
            "dims": dvec,
        })
    json.dump(dims_cache, open(cache_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # 仅保留有人类标注 A 的
    rows = [r for r in rows if r["human_A"] is not None]
    rows.sort(key=lambda r: r["id"])
    ids = [r["id"] for r in rows]
    subtasks = [r["subtask"] for r in rows]
    X = np.array([[r["dims"][k] for k in DIMS] for r in rows], float)
    y = np.array([r["human_A"] for r in rows], float)
    auto_score = np.array([r["auto_score"] for r in rows], float)
    n = len(rows)

    # ---- 2. 基线：auto_score vs 人工 A ----
    base_rho = spearman(auto_score, y)

    # ---- 3. 全局线性拟合打分器 ----
    w_global = fit_weights(X, y)
    fitted_global = predict(X, w_global)
    insample_global = spearman(fitted_global, y)

    # LOO-CV（全局）
    loo_pred = np.empty(n)
    for i in range(n):
        mask = np.ones(n, bool)
        mask[i] = False
        w = fit_weights(X[mask], y[mask])
        loo_pred[i] = predict(X[i:i + 1], w)[0]
    loo_global = spearman(loo_pred, y)

    # ---- 4. per-subtask 重加权（诊断）----
    uniq_sub = []
    for s in subtasks:
        if s not in uniq_sub:
            uniq_sub.append(s)
    per_sub = {}
    for s in uniq_sub:
        idx = [i for i in range(n) if subtasks[i] == s]
        if len(idx) < 3:
            per_sub[s] = {"n": len(idx), "insample_spearman": None,
                          "note": "样本过少，跳过"}
            continue
        Xs = X[idx]
        ys = y[idx]
        ws = fit_weights(Xs, ys)
        fs = predict(Xs, ws)
        rho_s = spearman(fs, ys)
        per_sub[s] = {
            "n": len(idx),
            "insample_spearman": round(rho_s, 3) if rho_s is not None else None,
            "weights": {DIMS[k]: round(float(ws[k]), 3) for k in range(len(DIMS))},
        }

    # per-subtask 组合得分（in-sample 上界，仅作诊断）：每条用其所属子任务的 w
    combined = np.empty(n)
    for i in range(n):
        s = subtasks[i]
        if s in per_sub and per_sub[s].get("insample_spearman") is not None:
            ws = fit_weights(X[[j for j in range(n) if subtasks[j] == s]],
                             y[[j for j in range(n) if subtasks[j] == s]])
            combined[i] = predict(X[i:i + 1], ws)[0]
        else:
            combined[i] = fitted_global[i]
    combined_insample = spearman(combined, y)

    # ---- 4b. 稳健变体 1：per-subtask 偏置校正（每子任务 1 个参数，抗过拟合）----
    # 修正各子任务的系统性偏差：corrected_i = auto_score_i + b_{subtask(i)}，
    # b_s = mean(human_A_s) - mean(auto_score_s)。LOO：留一样本时用其余样本估 b_s。
    sub_of = subtasks
    corrected = np.empty(n)
    for i in range(n):
        s = sub_of[i]
        others = [j for j in range(n) if sub_of[j] == s and j != i]
        if others:
            b = np.mean(y[others]) - np.mean(auto_score[others])
        else:
            b = 0.0
        corrected[i] = auto_score[i] + b
    pst_offset_loo = spearman(corrected, y)
    # in-sample 偏置（用全部样本估 b）
    b_all = {}
    for s in uniq_sub:
        idx = [j for j in range(n) if sub_of[j] == s]
        b_all[s] = np.mean(y[idx]) - np.mean(auto_score[idx])
    corr_ins = np.array([auto_score[i] + b_all[sub_of[i]] for i in range(n)])
    pst_offset_insample = spearman(corr_ins, y)

    # ---- 4c. 稳健变体 2：岭回归全局拟合（λ 选 LOO 最优）----
    def ridge_fit(Xm, ym, lam):
        Xa = np.hstack([Xm, np.ones((len(Xm), 1))])
        ry = rank_vec(ym)
        A = Xa.T @ Xa + lam * np.eye(Xa.shape[1])
        b = Xa.T @ ry
        return np.linalg.solve(A, b)

    best_lam, best_loo = None, -2.0
    loo_ridge = np.empty(n)
    for lam in [0.1, 1.0, 5.0, 20.0, 100.0]:
        pred = np.empty(n)
        for i in range(n):
            m = np.ones(n, bool)
            m[i] = False
            w = ridge_fit(X[m], y[m], lam)
            pred[i] = predict(X[i:i + 1], w)[0]
        r = spearman(pred, y)
        if r is not None and r > best_loo:
            best_loo, best_lam = r, lam
            loo_ridge[:] = pred
    ridge_loo = best_loo
    ridge_lam = best_lam

    # ---- 5. 当前维度权重（对照）----
    base_weights = {k: config.DIMENSION_WEIGHTS.get(k) for k in DIMS}
    fitted_weights = {DIMS[k]: round(float(w_global[k]), 4) for k in range(len(DIMS))}

    out = {
        "n": n,
        "baseline_auto_score_spearman": round(base_rho, 3),
        "global_fit_insample_spearman": round(insample_global, 3),
        "global_fit_loo_cv_spearman": round(loo_global, 3),
        "per_subtask_offset_insample_spearman": round(pst_offset_insample, 3),
        "per_subtask_offset_loo_cv_spearman": round(pst_offset_loo, 3),
        "ridge_global_loo_cv_spearman": round(ridge_loo, 3),
        "ridge_best_lambda": ridge_lam,
        "per_subtask_combined_insample_spearman": round(combined_insample, 3),
        "base_dimension_weights": base_weights,
        "fitted_global_weights": fitted_weights,
        "per_subtask": per_sub,
        "ids": ids,
        "note": ("诚实上界以 global_fit_loo_cv_spearman 为准；"
                 "insample 数字为乐观上界（含过拟合）。"),
    }
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(config.RESULTS_DIR, "phase2.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Phase 2 — per-subtask 维度重加权 / 拟合打分器 (LOO-CV)")
    print("=" * 60)
    print(f"样本数 n               = {n}")
    print(f"基线 auto_score ρ      = {base_rho:.3f}")
    print(f"全局拟合 in-sample ρ   = {insample_global:.3f}  (乐观上界)")
    print(f"全局拟合 LOO-CV ρ      = {loo_global:.3f}  (诚实泛化；<基线→过拟合)")
    print(f"per-subtask 偏置校正 ρ = in-s {pst_offset_insample:.3f} / LOO {pst_offset_loo:.3f}")
    print(f"岭回归全局(λ={ridge_lam}) LOO = {ridge_loo:.3f}")
    print(f"per-subtask 组合 in-s. = {combined_insample:.3f}  (诊断)")
    print("-" * 60)
    print("拟合全局权重（对照当前等权/预设权重）：")
    for k in DIMS:
        print(f"  {k:24s} base={base_weights.get(k)}  fitted={fitted_weights[k]}")
    print("-" * 60)
    print("per-subtask 组内 in-sample Spearman：")
    for s, v in per_sub.items():
        print(f"  {s:12s} n={v.get('n')}  ρ={v.get('insample_spearman')}")
    print(f"\n结果已写入 {config.RESULTS_DIR}/phase2.json")


if __name__ == "__main__":
    main()
