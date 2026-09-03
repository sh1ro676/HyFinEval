# -*- coding: utf-8 -*-
"""统一的统计工具（全项目唯一实现，禁止在各脚本内重复造轮子）。

为什么需要这个文件：
  此前 run_eval / run_stability / run_judge_cv / label_stats 各有一份
  spearman 实现，且都直接 `sorted` 后按索引赋秩 —— 遇到并列值（tie）时会
  给并列元素分配不同秩，秩的取值取决于输入顺序，属于统计错误。
  本项目数据并列极严重（自动分 28 条只有 8 个不同取值，人工三档只有
  3 个取值），不修正会系统性扭曲所有相关系数。

  本模块实现标准的平均秩（average rank）Spearman，等价于 scipy.stats
  .spearmanr 在无缺失数据情形下的结果，已用 scipy 交叉验证。
"""
import math


def average_rank(xs):
    """平均秩：并列元素取所占秩位的均值。

    例：[10, 20, 20, 30] -> [1.0, 2.5, 2.5, 4.0]
    """
    n = len(xs)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        # 并列区间 [i, j] 占据秩位 i+1 .. j+1，取均值
        r = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def pearson(a, b):
    """皮尔逊相关系数（供 Kendall tau-b 等调用）。"""
    n = len(a)
    if n < 2:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0:
        return None
    return cov / math.sqrt(va * vb)


def spearman(a, b):
    """Spearman 秩相关（带并列修正）。无并列时退化为 Pearson-on-ranks。"""
    if a is None or b is None or len(a) != len(b) or len(a) < 2:
        return None
    ra, rb = average_rank(list(a)), average_rank(list(b))
    return pearson(ra, rb)


def kendall_tau_b(a, b):
    """Kendall's tau-b：小样本 + 大量并列时比 Spearman 更稳健的补充指标。"""
    n = len(a)
    if n < 2:
        return None
    conc = disc = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            sa = (a[i] > a[j]) - (a[i] < a[j])
            sb = (b[i] > b[j]) - (b[i] < b[j])
            if sa == 0 or sb == 0:
                continue
            if sa * sb > 0:
                conc += 1
            else:
                disc += 1
    n0 = n * (n - 1) / 2
    n1 = sum(sum(1 for j in range(n) if j != i and a[i] == a[j]) for i in range(n)) / 2
    n2 = sum(sum(1 for j in range(n) if j != i and b[i] == b[j]) for i in range(n)) / 2
    denom = math.sqrt((n0 - n1) * (n0 - n2))
    if denom == 0:
        return None
    return (conc - disc) / denom


def quad_weighted_kappa(ba, bb, k=3):
    """在 k 档（取值 0..k-1 的整数）上计算二次加权 Cohen's Kappa。"""
    n = len(ba)
    if n < 2:
        return None
    hist = {}
    for x, y in zip(ba, bb):
        hist[(x, y)] = hist.get((x, y), 0) + 1
    row = [sum(hist.get((i, j), 0) for j in range(k)) for i in range(k)]
    col = [sum(hist.get((i, j), 0) for i in range(k)) for j in range(k)]
    po = sum(hist.get((i, i), 0) for i in range(k)) / n
    pe = sum(row[i] * col[i] for i in range(k)) / (n * n)
    if abs(1 - pe) < 1e-12:
        return 1.0
    wsum = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            wsum += w * hist.get((i, j), 0) / n
    return 1 - wsum / (1 - pe)
