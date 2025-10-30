import numpy as np


def calculate_rank(score, target, filter_list):
    score = score.copy()
    target_score = score[target]
    score[filter_list] = target_score - 1
    score[target] = target_score
    rank = np.sum(score > target_score) + 1
    # rank = np.sum(score > target_score) + np.sum(score == target_score) // 2 + 1
    return rank


def metrics(rank_list):
    mr = np.mean(rank_list)
    mrr = np.mean(1 / rank_list)
    hit10 = np.sum(rank_list < 11) / len(rank_list)
    hit3 = np.sum(rank_list < 4) / len(rank_list)
    hit1 = np.sum(rank_list < 2) / len(rank_list)
    return mr, mrr, hit10, hit3, hit1


def get_rank(score, ent, filter_list):
    score = score.copy()
    return calculate_rank(score, ent, filter_list)


def get_topK(score, ent, filter_list, topK):
    score = score.copy()
    target_score = score[ent]
    score[filter_list] = target_score - 1
    score[ent] = target_score
    indices = np.argsort(-score, kind='stable')  # 负号表示降序
    vals = score[indices]
    return indices[:topK], vals[:topK]
