# from dataset import KG
#
# dataset = KG('DB15K', -1)
#
#
# def count_exact_overlap(train, test):
#     train_set = set(map(tuple, train))
#     test_set = set(map(tuple, test))
#     overlap = train_set & test_set
#     return len(overlap), overlap
#
#
# def load_triples(file_path):
#     triplets = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f.readlines():
#             h, r, t = line.strip().split('\t')
#             triplets.append((int(h), int(r), int(t)))
#     return triplets
#
#
# t = load_triples('/mnt/data2/zhz/lzy/Flare/data/valid.txt')
#
# print(len(t))
# lenth, res = count_exact_overlap(dataset.valid, t)
# print(lenth)

# def text_files_are_equal(file1, file2):
#     with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
#         for idx, (line1, line2) in enumerate(zip(f1, f2)):
#             if line1.rstrip('\n') != line2.rstrip('\n'):
#                 print(idx)
#                 return False
#     return True
#
#
# print(text_files_are_equal("/mnt/data2/zhz/lzy/Flare/data/train.txt", "/mnt/data2/zhz/lzy/Flare/data/DB15K/train.txt"))
# print(text_files_are_equal("/mnt/data2/zhz/lzy/Flare/data/valid.txt", "/mnt/data2/zhz/lzy/Flare/data/DB15K/valid.txt"))
# print(text_files_are_equal("/mnt/data2/zhz/lzy/Flare/data/test.txt", "/mnt/data2/zhz/lzy/Flare/data/DB15K/test.txt"))

import json

with open("/mnt/data2/zhz/lzy/Flare/data/DB15K/valid.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(len(data))
