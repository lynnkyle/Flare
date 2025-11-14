import argparse
import os.path
import random
import json
import numpy as np

import torch
from torch import nn
import logging

from dataset import KG_Pre
from model import FormerAlign
from merge_tokens import get_entity_visual_tokens, get_entity_textual_tokens
from utils import calculate_rank, metrics, get_rank, get_topK


@torch.no_grad()
def valid_eval_metric(valid_or_test):
    rank_list = []
    ent_embs, rel_embs, _, _ = model()  # [!!!important]不要放在循环内, 导致测试时速度变慢
    for triple in valid_or_test:
        # for triple in tqdm(valid_or_test):
        h, r, t = triple
        head_score = \
            model.score(torch.tensor([[kg.num_ent + kg.num_rel, r + kg.num_ent, t + kg.num_rel]]).cuda(), ent_embs,
                        rel_embs)[0].detach().cpu().numpy()  # [batch_size, num_entity]
        head_rank = calculate_rank(head_score, h, kg.filter_dict[(-1, r, t)])
        rank_list.append(head_rank)
        tail_score = \
            model.score(torch.tensor([[h + kg.num_rel, r + kg.num_ent, kg.num_ent + kg.num_rel]]).cuda(), ent_embs,
                        rel_embs)[0].detach().cpu().numpy()  # [batch_size, num_entity]
        tail_rank = calculate_rank(tail_score, t, kg.filter_dict[(h, r, -1)])
        rank_list.append(tail_rank)
    rank_list = np.array(rank_list)
    mr, mrr, hit10, hit3, hit1 = metrics(rank_list)
    return mr, mrr, hit10, hit3, hit1


@torch.no_grad()
def save_numpy(args, valid_or_test, topK=20):
    query_list = []
    rank_list = []
    topk_list = []
    topk_score_list = []
    query_embeds = []
    ent_embs, rel_embs, _, _ = model()  # [!!!important]不要放在循环内, 导致测试时速度变慢
    for triple in valid_or_test:
        # for triple in tqdm(valid_or_test):
        h, r, t = triple
        query_list.append(f'(?, {r}, {t})')
        head_score = \
            model.score(torch.tensor([[kg.num_ent + kg.num_rel, r + kg.num_ent, t + kg.num_rel]]).cuda(), ent_embs,
                        rel_embs)[0].detach().cpu().numpy()  # [batch_size, num_entity]
        head_rank = get_rank(head_score, h, kg.filter_dict[(-1, r, t)])
        rank_list.append(head_rank)
        topks, topk_scores = get_topK(head_score, h, kg.filter_dict[(-1, r, t)], topK)
        topk_list.append(topks)
        topk_score_list.append(topk_scores)
        query_embeds.append(
            model.query(torch.tensor([[kg.num_ent + kg.num_rel, r + kg.num_ent, t + kg.num_rel]]).cuda(), ent_embs,
                        rel_embs)[0].detach().cpu().numpy()
        )

        query_list.append(f'({h}, {r}, ?)')
        tail_score = \
            model.score(torch.tensor([[h + kg.num_rel, r + kg.num_ent, kg.num_ent + kg.num_rel]]).cuda(), ent_embs,
                        rel_embs)[0].detach().cpu().numpy()  # [batch_size, num_entity]
        tail_rank = get_rank(tail_score, t, kg.filter_dict[(h, r, -1)])
        rank_list.append(tail_rank)
        topks, topk_scores = get_topK(tail_score, t, kg.filter_dict[(h, r, -1)], topK)
        topk_list.append(topks)
        topk_score_list.append(topk_scores)
        query_embeds.append(
            model.query(torch.tensor([[kg.num_ent + kg.num_rel, r + kg.num_ent, t + kg.num_rel]]).cuda(), ent_embs,
                        rel_embs)[0].detach().cpu().numpy()
        )

    rank_list = np.array(rank_list)
    topk_list = np.array(topk_list)
    topk_score_list = np.array(topk_score_list)
    os.makedirs(f'{args.save_dir}', exist_ok=True)
    with open(os.path.join(args.save_dir, 'query.json'), 'w') as f:
        json.dump(query_list, f)
    np.save(os.path.join(args.save_dir, 'ranks.npy'), rank_list)
    np.save(os.path.join(args.save_dir, 'topks.npy'), topk_list)
    np.save(os.path.join(args.save_dir, 'topk_scores.npy'), topk_score_list)
    np.save(os.path.join(args.save_dir, 'entity_embeddings.npy'), ent_embs.cpu().numpy())
    query_embeds = torch.tensor(query_embeds)
    np.save(os.path.join(args.save_dir, 'query_embeddings.npy'), query_embeds.cpu().numpy())
    return query_list, rank_list, topk_list, topk_score_list, ent_embs, query_embeds


if __name__ == '__main__':
    """
        代码可复现
    """
    torch.cuda.set_device(2)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # torch.set_num_threads(8)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.empty_cache()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """
        参数设置
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='DB15K')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--model', type=str, default='Flare')
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--num_epoch', type=int, default=1500)
    parser.add_argument('--valid_epoch', type=int, default=1)
    parser.add_argument('--str_dim', default=256, type=int)
    parser.add_argument('--num_kernels', default=512, type=int)
    parser.add_argument('--max_vis_token', default=8, type=int)
    parser.add_argument('--max_txt_token', default=4, type=int)
    parser.add_argument("--no_write", action='store_true')
    parser.add_argument('--str_dropout', default=0.6, type=float)
    parser.add_argument('--visual_dropout', default=0.3, type=float)
    parser.add_argument('--textual_dropout', default=0.1, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    # Loss的超参数
    parser.add_argument('--align_former', default=True, action='store_true')
    parser.add_argument('--contrastive', default=0.001, type=float)
    parser.add_argument('--before_align', default=0.001, type=float)
    parser.add_argument('--after_align', default=0.001, type=float)
    # Transformer的配置
    parser.add_argument('--num_head', default=2, type=int)
    parser.add_argument('--dim_hid', default=1024, type=int)
    parser.add_argument('--num_layer_enc_ent', default=1, type=int)
    parser.add_argument('--num_layer_enc_rel', default=1, type=int)
    parser.add_argument('--num_layer_dec', default=1, type=int)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--save_dir', default='data/DB15K', type=str)
    args = parser.parse_args()

    """
        创建数据集
    """
    kg = KG_Pre(data=args.data, max_vis_len=-1)
    kg_loader = torch.utils.data.DataLoader(kg, batch_size=args.batch_size, shuffle=False)

    """
        模型要素
    """
    visual_token_index, visual_ent_mask = get_entity_visual_tokens(args.data, max_num=args.max_vis_token)
    textual_token_index, textual_ent_mask = get_entity_textual_tokens(args.data, max_num=args.max_txt_token)
    model = FormerAlign(args, num_ent=kg.num_ent, num_rel=kg.num_rel, str_dim=args.str_dim, visual_tokenizer='beit',
                        textual_tokenizer='bert', visual_token_index=visual_token_index,
                        textual_token_index=textual_token_index,
                        visual_ent_mask=visual_ent_mask, textual_ent_mask=textual_ent_mask, num_head=args.num_head,
                        dim_hid=args.dim_hid, num_layer_enc_ent=args.num_layer_enc_ent,
                        num_layer_enc_rel=args.num_layer_enc_rel,
                        num_layer_dec=args.num_layer_dec, dropout=args.dropout, str_dropout=args.str_dropout,
                        visual_dropout=args.visual_dropout, textual_dropout=args.textual_dropout,
                        score_function='tucker').cuda()
    # 模型加载
    # param1 = torch.load(f'ckpt/{args.model}/{args.data}/pre_trained.ckpt')['state_dict']
    model.load_state_dict(torch.load(f'ckpt/{args.model}/{args.data}/db15k.ckpt')['state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 优化器加载
    # param2 = torch.load(f'ckpt/{args.model}/{args.data}/pre_trained.ckpt')['optimizer']
    optimizer.load_state_dict(torch.load(f'ckpt/{args.model}/{args.data}/db15k.ckpt')['optimizer'])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    # 学习率裁剪器加载
    # param3 = torch.load(f'ckpt/{args.model}/{args.data}/pre_trained.ckpt')['scheduler']
    lr_scheduler.load_state_dict(torch.load(f'ckpt/{args.model}/{args.data}/db15k.ckpt')['scheduler'])

    model.eval()
    valid_and_test = kg.valid + kg.test
    query_list, rank_list, topk_list, topk_score_list, ent_embs, query_embs = save_numpy(args, valid_and_test)
    print(len(query_list))
    print(len(rank_list))
    print(len(topk_list))
    print(len(topk_score_list))
    print(len(ent_embs))
    print(len(query_embs))
    print("Done!!!")
