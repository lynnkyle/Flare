# from dataset import KG
#
# kg = KG(data="MKG-W", max_vis_len=-1)
#
# query_dict = set()
#
# for triple in kg.train:
#     h, r, t = triple
#     query_dict.add((h, r, -1))
#     query_dict.add((-1, r, t))
#
# num = 0
# for triple in kg.test:
#     h, r, t = triple
#     if (h, r, -1) in query_dict:
#         print(h, r, -1)
#         num += 1
#     if (-1, r, t) in query_dict:
#         num += 1
#         print(-1, r, t)
#
# print(num)
import torch

score = torch.tensor([
    [2.1, 0.3, 1.7, 4.2],
    [0.5, 3.8, 1.2, 0.9]
])

label = torch.tensor([3, 1])

masked_score = score.clone()
masked_score.scatter_(1, label.unsqueeze(1), -float('inf'))
print(masked_score)
