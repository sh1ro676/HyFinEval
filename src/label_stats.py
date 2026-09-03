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
import stats_utils

BAND_MAP = {"优": 90, "中": 60, "差": 30}
BAND_IDX = {"优": 2, "中": 1, "差": 0}
FLAGS = ["format_ok", "cited", "no_redline", "no_contra"]
FLAG_DESC = {
    "format_ok": "格式完整",
    "cited": "引用可溯源",
    "no_redline": "无合规红线",
    "no_contra": "无自相矛盾/硬伤",
}


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
        rho = stats_utils.spearman(ra, rh)
        tau = stats_utils.kendall_tau_b(ra, rh)
        mad = sum(abs(ra[k] - rh[k]) for k in range(len(ra))) / len(ra)
        out[f"auto_vs_human{tag}_spearman"] = round(rho, 3) if rho is not None else None
        out[f"auto_vs_human{tag}_kendall"] = round(tau, 3) if tau is not None else None
        out[f"auto_vs_human{tag}_mae"] = round(mad, 3)
        out[f"n_align{tag}"] = len(common)
        # 并列强度：并列越重，秩相关越保守，需一并报告以便解读
        out[f"n_unique_auto{tag}"] = len(set(ra))
        out[f"n_unique_human{tag}"] = len(set(rh))

    _align("A", A)
    _align("B", B)

    subtask = {p["id"]: p.get("subtask") or "未分类" for p in pool}

    # ---------- 子任务级拆分：评估器效度强烈依赖「有无可核实真值」 ----------
    by_sub = {}
    for i in A:
        by_sub.setdefault(subtask.get(i, "未分类"), []).append(i)
    per_subtask = {}
    for st in sorted(by_sub):
        ids = [i for i in by_sub[st] if i in auto]
        if len(ids) < 3:
            continue
        ra = [auto[i] for i in ids]
        rh = [A[i] for i in ids]
        rho = stats_utils.spearman(ra, rh)
        tau = stats_utils.kendall_tau_b(ra, rh)
        dist = {"优": 0, "中": 0, "差": 0}
        for i in ids:
            b = lab[i]["A"].get("band")
            if b in dist:
                dist[b] += 1
        per_subtask[st] = {
            "n": len(ids),
            "spearman": round(rho, 3) if rho is not None else None,
            "kendall": round(tau, 3) if tau is not None else None,
            "auto_avg": round(sum(ra) / len(ra), 1),
            "human_band_dist": dist,
        }
    out["per_subtask"] = per_subtask

    # ---------- 三档一致率：把连续 auto 分映射到与人工同尺度 ----------
    # 阈值依据：99 为满分档（各维度全 1.0），85 以下出现明确短板
    def _auto_band(s):
        return 2 if s >= 95 else (1 if s >= 85 else 0)

    band_names = ["差", "中", "优"]
    ids_all = [i for i in A if i in auto]
    if ids_all:
        ab = [_auto_band(auto[i]) for i in ids_all]
        hb = [BAND_IDX[lab[i]["A"]["band"]] for i in ids_all]
        exact = sum(1 for x, y in zip(ab, hb) if x == y) / len(ab)
        adj = sum(1 for x, y in zip(ab, hb) if abs(x - y) <= 1) / len(ab)
        out["band_agreement_exact"] = round(exact, 3)
        out["band_agreement_adjacent"] = round(adj, 3)
        out["band_kappa_auto_vs_human"] = round(stats_utils.quad_weighted_kappa(ab, hb, k=3), 3)
        out["band_threshold_note"] = "auto 分档阈值：>=95 优 / 85~95 中 / <85 差"
        # 混淆矩阵（行=auto档，列=人工档）
        cm = {band_names[r]: {band_names[c]: 0 for c in range(3)} for r in range(3)}
        for x, y in zip(ab, hb):
            cm[band_names[x]][band_names[y]] += 1
        out["confusion_matrix"] = cm

    # ---------- 逐维度诊断：哪个维度与人工判断脱节 ----------
    dim_keys = None
    for p in pool:
        if isinstance(p.get("auto_dims"), dict) and p["auto_dims"]:
            dim_keys = list(p["auto_dims"].keys())
            break
    if dim_keys:
        dims_stat = {}
        for d in dim_keys:
            vals, hum = [], []
            for p in pool:
                if p["id"] in A and isinstance(p.get("auto_dims"), dict) and d in p["auto_dims"]:
                    vals.append(float(p["auto_dims"][d]))
                    hum.append(BAND_IDX[lab[p["id"]]["A"]["band"]])
            if len(vals) < 3:
                continue
            rho = stats_utils.spearman(vals, hum)
            dims_stat[d] = {
                "spearman_vs_human": round(rho, 3) if rho is not None else None,
                "n_unique_values": len(set(vals)),
                "mode_share": round(max(vals.count(v) for v in set(vals)) / len(vals), 3),
            }
        out["per_dimension"] = dims_stat

    # 标注者间 Kappa（三档）
    Aidx = {i: _idx(v.get("A")) for i, v in lab.items() if _valid(v.get("A"))}
    Bidx = {i: _idx(v.get("B")) for i, v in lab.items() if _valid(v.get("B"))}
    both = [i for i in Aidx if i in Bidx]
    if len(both) >= 2:
        out["inter_annotator_kappa"] = round(stats_utils.quad_weighted_kappa([Aidx[i] for i in both],
                                                                            [Bidx[i] for i in both], k=3), 3)
        out["n_both"] = len(both)
    else:
        # 单人标注：无法估计标注者间一致性，如实记录为缺失而非省略
        out["inter_annotator_kappa"] = None
        out["n_both"] = 0
        out["kappa_note"] = "仅单人标注，标注者间一致性未估计"

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
