# -*- coding: utf-8 -*-
"""Bradley-Terry 模型：从成对比较胜负矩阵估计全局能力值（ability scores）。

用于将大量 "A 比 B 好" 的成对判断聚合成全局排序分。
"""
import math
from typing import Dict, List, Optional, Tuple


def bt_mle(
    wins: Dict[Tuple[str, str], int],
    n_items: int,
    item_ids: List[str],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Optional[Dict[str, float]]:
    """Bradley-Terry 最大似然估计（迭代版）。

    Args:
        wins: {(winner_id, loser_id): count} —— winner 击败 loser 的次数
        n_items: 项目总数
        item_ids: 项目ID列表（顺序与能力值向量对应）
        max_iter: 最大迭代次数
        tol: 收敛阈值

    Returns:
        {item_id: ability_score}，能力值已做 softmax 归一化（和为1，方便解释）
    """
    if n_items < 2:
        return None

    # 初始化：均匀能力值
    abilities = [1.0 / n_items] * n_items
    id_to_idx = {item_ids[i]: i for i in range(n_items)}

    # 预处理：每个样本的总胜场和总败场
    win_counts = [0.0] * n_items
    loss_counts = [0.0] * n_items
    for (w, l), c in wins.items():
        if w in id_to_idx and l in id_to_idx:
            win_counts[id_to_idx[w]] += c
            loss_counts[id_to_idx[l]] += c

    for _ in range(max_iter):
        new_abilities = [0.0] * n_items
        for i in range(n_items):
            if win_counts[i] == 0:
                # 从未赢过：给一个极小值避免除零
                new_abilities[i] = 1e-8
                continue
            denom = 0.0
            for j in range(n_items):
                if i == j:
                    continue
                # i 与 j 的所有比较次数（无论胜负）
                n_ij = wins.get((item_ids[i], item_ids[j]), 0) + wins.get((item_ids[j], item_ids[i]), 0)
                if n_ij > 0:
                    denom += n_ij / (abilities[i] + abilities[j])
            if denom > 0:
                new_abilities[i] = win_counts[i] / denom
            else:
                new_abilities[i] = abilities[i]

        # 归一化
        total = sum(new_abilities)
        if total > 0:
            new_abilities = [a / total for a in new_abilities]

        # 检查收敛
        delta = max(abs(new_abilities[i] - abilities[i]) for i in range(n_items))
        abilities = new_abilities
        if delta < tol:
            break

    return {item_ids[i]: abilities[i] for i in range(n_items)}


def pairwise_from_scores(
    scores: Dict[str, float],
    ids: List[str],
) -> Dict[Tuple[str, str], int]:
    """从绝对分推导出成对比较胜负（模拟 pairwise）。

    如果 scores[A] > scores[B]，则计为 A 击败 B 一次。
    相等时不计入（避免 tie 带来的歧义）。
    """
    wins = {}
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = ids[i], ids[j]
            sa, sb = scores.get(a, 0), scores.get(b, 0)
            if sa > sb:
                wins[(a, b)] = 1
            elif sb > sa:
                wins[(b, a)] = 1
    return wins


def pairwise_consistency(wins: Dict[Tuple[str, str], int]) -> float:
    """计算传递性违例率：A>B, B>C 但 C>A 的三元组比例。

    返回 0.0~1.0，越低越好（0 表示完全传递）。
    """
    items = set()
    for w, l in wins.keys():
        items.add(w)
        items.add(l)
    items = list(items)
    n = len(items)
    if n < 3:
        return 0.0

    # 构建胜负矩阵
    beats = {a: set() for a in items}
    for (w, l), c in wins.items():
        if c > 0:
            beats[w].add(l)

    violations = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = items[i], items[j], items[k]
                total += 1
                # 检查所有 6 种可能的循环
                ab = b in beats.get(a, set())
                ba = a in beats.get(b, set())
                bc = c in beats.get(b, set())
                cb = b in beats.get(c, set())
                ca = a in beats.get(c, set())
                ac = c in beats.get(a, set())
                # 有方向信息才算
                has_dir = sum([ab, ba, bc, cb, ca, ac])
                if has_dir < 2:
                    continue
                # 检测循环：a>b, b>c, c>a
                if (ab and bc and ca) or (ac and cb and ba):
                    violations += 1
    return violations / total if total > 0 else 0.0
