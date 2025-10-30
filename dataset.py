import os
import random
import torch
from torch.utils.data import Dataset, DataLoader


class KG(Dataset):
    def __init__(self, data, max_vis_len):
        super().__init__()

        self.dir = f'data/{data}'
        self.ent2id = {}
        self.id2ent = []
        self.rel2id = {}
        self.id2rel = []

        with open(os.path.join(self.dir, 'entities.txt'), 'r') as f:
            lines = f.readlines()
            for _, line in enumerate(lines):
                self.ent2id[line.strip()] = _
                self.id2ent.append(line.strip())
        self.num_ent = len(self.ent2id)

        with open(os.path.join(self.dir, 'relations.txt'), 'r') as f:
            lines = f.readlines()
            for _, line in enumerate(lines):
                self.rel2id[line.strip()] = _
                self.id2rel.append(line.strip())
        self.num_rel = len(self.rel2id)

        self.train = []
        with open(os.path.join(self.dir, 'train2id.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                h, r, t = line.strip().split('\t')
                self.train.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        self.valid = []
        with open(os.path.join(self.dir, 'valid2id.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                h, r, t = line.strip().split('\t')
                self.valid.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        self.test = []
        with open(os.path.join(self.dir, 'test2id.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                h, r, t = line.strip().split('\t')
                self.test.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        self.filter_dict = {}
        for data_filter in [self.train, self.valid, self.test]:
            for triple in data_filter:
                h, r, t = triple
                if (-1, r, t) not in self.filter_dict:
                    self.filter_dict[(-1, r, t)] = []
                self.filter_dict[(-1, r, t)].append(h)
                if (h, r, -1) not in self.filter_dict:
                    self.filter_dict[(h, r, -1)] = []
                self.filter_dict[(h, r, -1)].append(t)

        self.max_vis_len_ent = max_vis_len
        self.max_vis_len_rel = max_vis_len

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        h, r, t = self.train[idx]
        if random.random() < 0.5:
            masked_triple = [self.num_ent + self.num_rel, r + self.num_ent, t + self.num_rel]
            label = h
        else:
            masked_triple = [h + self.num_rel, r + self.num_ent, self.num_ent + self.num_rel]
            label = t
        return torch.tensor(masked_triple), torch.tensor(label)

    # def collate_fn(self, batch):
    #     data = torch.tensor([item[0] for item in batch])
    #     label = torch.tensor([item[1] for item in batch])
    #     return data, label


if __name__ == '__main__':
    dataset = KG('DB15K', -1)
    dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=True)
    for data in dataloader:
        print(data)
