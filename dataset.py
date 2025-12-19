import os
import random
import torch
import copy
import json
from torch.nn.utils.rnn import pad_sequence
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

        train_file = 'train2id.txt'
        valid_file = "valid2id.txt"
        test_file = "test2id.txt"
        if data == "MKG-W" or data == "MKG-Y":
            train_file = "train.txt"
            valid_file = "valid.txt"
            test_file = "test.txt"
        self.train = []
        with open(os.path.join(self.dir, train_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                h, r, t = line.strip().split('\t')
                self.train.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        self.valid = []
        with open(os.path.join(self.dir, valid_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                h, r, t = line.strip().split('\t')
                self.valid.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        self.test = []
        with open(os.path.join(self.dir, test_file), 'r') as f:
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

        self.negative_sampling = {}

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


class KG_Pre(Dataset):
    def __init__(self, data, max_vis_len):
        super().__init__()

        self.dir = f'data/{data}'
        self.ent2id = {}
        self.id2ent = []
        self.rel2id = {}
        self.id2rel = []

        with open(os.path.join(self.dir, 'entity2id.txt'), 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                ent, _ = line.strip().split(' ')
                self.ent2id[ent] = int(_)
                self.id2ent.append(ent)
        self.num_ent = len(self.ent2id)

        with open(os.path.join(self.dir, 'relation2id.txt'), 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                rel, _ = line.strip().split(' ')
                self.rel2id[rel] = int(_)
                self.id2rel.append(rel)
        self.num_rel = len(self.rel2id)

        self.train = []
        with open(os.path.join(self.dir, 'train.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                h, r, t = line.strip().split('\t')
                self.train.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        self.valid = []
        with open(os.path.join(self.dir, 'valid.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                h, r, t = line.strip().split('\t')
                self.valid.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        self.test = []
        with open(os.path.join(self.dir, 'test.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                h, r, t = line.strip().split('\t')
                self.test.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        self.filter_dict = {}
        for data_filter in [self.train, self.valid, self.test]:
            for triple in data_filter:
                h, r, t = map(int, triple)
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


def make_data_module(args, tokenizer, logger=None):
    data_module = KGDataModule(args, tokenizer, logger)
    data_collator = KGDataCollator(args, tokenizer)
    return {
        'train_dataset': data_module.train_dataset,
        'eval_dataset': data_module.eval_dataset,
        'data_collator': data_collator
    }


class KGDataModule(object):
    def __init__(self, args, tokenizer, logger=None):
        self.args = args
        self.tokenizer = tokenizer

        train_example = json.load(open(args.train_path, 'r', encoding='utf-8'))
        eval_example = json.load(open(args.eval_path, 'r', encoding='utf-8'))
        test_example = json.load(open(args.test_path, 'r', encoding='utf-8'))

        self.train_dataset = KGDataset(train_example)
        self.eval_dataset = KGDataset(eval_example)
        self.test_dataset = KGDataset(test_example)


IGNORE_INDEX = -100


class KGDataCollator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.source_max_len = args.source_max_len
        self.target_max_len = args.target_max_len

    def __call__(self, instances):
        sources = [f"{self.tokenizer.bos_token} {example['input']}" for example in instances]
        targets = [f"{example['output']} {self.tokenizer.eos_token}" for example in instances]

        # Tokenize(source：输入,含提示词, target:标签)
        tokenized_sources = self.tokenizer(sources, max_length=self.source_max_len,
                                           truncation=True, add_special_tokens=False)
        tokenized_targets = self.tokenizer(targets, max_length=self.target_max_len, truncation=True,
                                           add_special_tokens=False)
        source_input_ids = tokenized_sources["input_ids"]
        target_input_ids = tokenized_targets["input_ids"]

        # LLAMA Input(data_dict) Construction
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(source_input_ids, target_input_ids):
            input_ids.append(torch.tensor(tokenized_source + tokenized_target))
            labels.append(
                torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
            )
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
            'labels': labels
        }
        if self.args.model_class == 'KGELlama':
            data_dict['query_ids'] = torch.LongTensor([
                example['query_id'] for example in instances
            ])
            data_dict['entity_ids'] = torch.LongTensor([
                example['entity_ids'] for example in instances
            ])
        else:
            raise NotImplementedError
        return data_dict


class KGDataset(Dataset):
    def __init__(self, example):
        self.data = example
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    dataset = KG('DB15K', -1)
    dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=True)
    for data in dataloader:
        print(data)
