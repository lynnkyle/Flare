"""

CUDA_VISIBLE_DEVICES=0 nohup python train_1.py --data MKG-W --num_epoch 1500 --hidden_dim 1024 --lr 5e-4 --dim 256 --max_txt_token 8 --num_head 4 --emb_dropout 0.9 --vis_dropout 0.4 --txt_dropout 0.1 --num_layer_dec 2 --mu 0.001 > log_MKG-W.txt &

CUDA_VISIBLE_DEVICES=0 nohup python train_1.py --data DB15K --num_epoch 1500 --hidden_dim 1024 --lr 1e-3 --dim 256 --max_vis_token 8 --max_txt_token 4 --num_head 2 --emb_dropout 0.6 --vis_dropout 0.3 --txt_dropout 0.1 --num_layer_dec 1 --mu 0.01 > log_DB15K.txt &

CUDA_VISIBLE_DEVICES=0 nohup python train_1.py --data DB15K --num_epoch 1500 --hidden_dim 1024 --lr 5e-4 --dim 200 --max_vis_token 6 --max_txt_token 18 --num_head 2 --emb_dropout 0.9 --vis_dropout 0.4 --txt_dropout 0.1 --num_layer_dec 1 --mu 0.01 > log_MKG-Y.txt &
"""
import os
import sys
import argparse
import random
import numpy as np

import torch
import logging

from dataset import KG
from model import FormerAlign
from merge_tokens import get_entity_visual_tokens, get_entity_textual_tokens
from utils import calculate_rank, metrics

"""
    代码可复现
"""
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
"""
    db15k
"""
# torch.cuda.set_device(0)
# parser = argparse.ArgumentParser()
# parser.add_argument('--data', type=str, default='DB15K')
# parser.add_argument('--batch_size', type=int, default=2048)
# parser.add_argument('--model', type=str, default='Flare')
# parser.add_argument('--device', type=str, default='cuda:0')
# parser.add_argument('--num_epoch', type=int, default=800)
# parser.add_argument('--valid_epoch', type=int, default=1)
# parser.add_argument('--str_dim', default=256, type=int)
# parser.add_argument('--num_kernels', default=512, type=int)
# parser.add_argument('--max_vis_token', default=8, type=int)
# parser.add_argument('--max_txt_token', default=4, type=int)
# parser.add_argument("--no_write", action='store_true')
# parser.add_argument('--str_dropout', default=0.6, type=float)
# parser.add_argument('--visual_dropout', default=0.3, type=float)
# parser.add_argument('--textual_dropout', default=0.1, type=float)
# parser.add_argument('--lr', default=1e-3, type=float)
# # Loss的超参数
# parser.add_argument('--align_former', default=False, action='store_true')
# parser.add_argument('--contrastive', default=0.001, type=float)
# parser.add_argument('--entity_align', default=0, type=float)
# # Transformer的配置
# parser.add_argument('--num_head', default=2, type=int)
# parser.add_argument('--dim_hid', default=1024, type=int)
# parser.add_argument('--num_layer_enc_ent', default=1, type=int)
# parser.add_argument('--num_layer_enc_rel', default=1, type=int)
# parser.add_argument('--num_layer_dec', default=1, type=int)
# parser.add_argument('--dropout', default=0, type=float)
# args = parser.parse_args()
"""
    mkg-y
"""
torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='MKG-W')
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--model', type=str, default='Flare')
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--num_epoch', type=int, default=1500)
parser.add_argument('--valid_epoch', type=int, default=1)
parser.add_argument('--str_dim', default=256, type=int)
parser.add_argument('--num_kernels', default=512, type=int)
parser.add_argument('--max_vis_token', default=8, type=int)
parser.add_argument('--max_txt_token', default=8, type=int)
parser.add_argument("--no_write", action='store_true')
parser.add_argument('--str_dropout', default=0.9, type=float)
parser.add_argument('--visual_dropout', default=0.4, type=float)
parser.add_argument('--textual_dropout', default=0.1, type=float)
parser.add_argument('--lr', default=5e-4, type=float)
# Loss的超参数
parser.add_argument('--align_former', default=False, action='store_true')
parser.add_argument('--contrastive', default=0.01, type=float)
parser.add_argument('--entity_align', default=0, type=float)
# Transformer的配置
parser.add_argument('--num_head', default=4, type=int)
parser.add_argument('--dim_hid', default=1024, type=int)
parser.add_argument('--num_layer_enc_ent', default=1, type=int)
parser.add_argument('--num_layer_enc_rel', default=1, type=int)
parser.add_argument('--num_layer_dec', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
args = parser.parse_args()

"""
    文件保存
"""
if not args.no_write:
    os.makedirs(f'result/{args.model}/{args.data}', exist_ok=True)
    os.makedirs(f'ckpt/{args.model}/{args.data}', exist_ok=True)
    os.makedirs(f'log/{args.model}/{args.data}', exist_ok=True)

"""
    日志输出
"""
logger = logging.getLogger('former_align')
logger.setLevel(logging.INFO)
format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setFormatter(format)
logger.addHandler(stream_handler)
file_handler = logging.FileHandler(f'log/{args.model}/{args.data}/log.log')
file_handler.setFormatter(format)
logger.addHandler(file_handler)

"""
    创建数据集
"""
kg = KG(data=args.data, max_vis_len=-1)
kg_loader = torch.utils.data.DataLoader(kg, batch_size=args.batch_size, shuffle=False)

"""
    模型要素
"""
visual_token_index, visual_ent_mask = get_entity_visual_tokens(args.data, max_num=args.max_vis_token)
textual_token_index, textual_ent_mask = get_entity_textual_tokens(args.data, max_num=args.max_txt_token)
model = FormerAlign(args, num_ent=kg.num_ent, num_rel=kg.num_rel, str_dim=args.str_dim, filter_dict=kg.filter_dict,
                    visual_tokenizer='beit', textual_tokenizer='bert', visual_token_index=visual_token_index,
                    textual_token_index=textual_token_index, visual_ent_mask=visual_ent_mask,
                    textual_ent_mask=textual_ent_mask, num_head=args.num_head, dim_hid=args.dim_hid,
                    num_layer_enc_ent=args.num_layer_enc_ent, num_layer_enc_rel=args.num_layer_enc_rel,
                    num_layer_dec=args.num_layer_dec, dropout=args.dropout, str_dropout=args.str_dropout,
                    visual_dropout=args.visual_dropout, textual_dropout=args.textual_dropout,
                    score_function='tucker').cuda()
# 模型加载
# param1 = torch.load(f'ckpt/{args.model}/{args.data}/pre_trained.ckpt')['state_dict']
# model.load_state_dict(torch.load(f'ckpt/{args.model}/{args.data}/700.ckpt')['state_dict'])
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# 优化器加载
# param2 = torch.load(f'ckpt/{args.model}/{args.data}/pre_trained.ckpt')['optimizer']
# optimizer.load_state_dict(torch.load(f'ckpt/{args.model}/{args.data}/700.ckpt')['optimizer'])
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
# 学习率裁剪器加载
# param3 = torch.load(f'ckpt/{args.model}/{args.data}/pre_trained.ckpt')['scheduler']
# lr_scheduler.load_state_dict(torch.load(f'ckpt/{args.model}/{args.data}/700.ckpt')['scheduler'])
"""
    模型训练
"""


def train_one_epoch(model, optimizer):
    model.train()
    total_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    for batch, label in kg_loader:
        # for batch, label in tqdm(kg_loader):
        ent_embs, rel_embs, align_before_loss = model()
        score = model.score(batch.cuda(), ent_embs, rel_embs)
        loss = loss_fn(score, label.cuda())
        if args.align_former is not False:
            if args.entity_align != 0:
                loss += args.entity_align * align_before_loss
        if args.contrastive != 0:
            loss += args.contrastive * model.contrastive_loss(ent_embs)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
    return total_loss


# 方式三的训练方式
# def train_one_epoch_with_negative_sampling(model, optimizer):
#     model.train()
#     total_loss = 0
#     margin_ranking_loss_fn = torch.nn.MarginRankingLoss(margin=0.1)
#     cross_entropy_loss_fn = torch.nn.CrossEntropyLoss()
#     for batch, label, filter_mask in kg_loader:
#         # 1.embedding
#         ent_embs, rel_embs, align_before_loss = model()
#         # 2.样本score
#         score = model.score(batch.cuda(), ent_embs, rel_embs)
#         # 5.cross_entropy
#         loss = cross_entropy_loss_fn(score, label.cuda())
#         if args.align_former is not False:
#             if args.entity_align != 0:
#                 loss += args.entity_align * align_before_loss
#         if args.contrastive != 0:
#             loss += args.contrastive * model.contrastive_loss(ent_embs)
#         total_loss += loss.item()
#
#         # 3.正负样本logit
#         pos_logit, neg_logit = model.pos_neg_logits_vectorized(score, label.cuda(), filter_mask.cuda())
#         # 4.margin_loss
#         target = torch.ones_like(neg_logit)
#         res = margin_ranking_loss_fn(pos_logit, neg_logit, target)
#         loss += 50 * res
#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
#         optimizer.step()
#     return total_loss

# 方式四的训练方式
def train_one_epoch_with_negative_sampling_topk(model, optimizer):
    model.train()
    total_loss = 0
    margin_ranking_loss_fn = torch.nn.MarginRankingLoss(margin=0.1)
    cross_entropy_loss_fn = torch.nn.CrossEntropyLoss()
    for batch, label, filter_mask in kg_loader:
        # 1.embedding
        ent_embs, rel_embs, align_before_loss = model()
        # 2.样本score
        score = model.score(batch.cuda(), ent_embs, rel_embs)
        # 5.cross_entropy
        loss = cross_entropy_loss_fn(score, label.cuda())
        if args.align_former is not False:
            if args.entity_align != 0:
                loss += args.entity_align * align_before_loss
        if args.contrastive != 0:
            loss += args.contrastive * model.contrastive_loss(ent_embs)
        total_loss += loss.item()

        # 3.正负样本logit
        pos_logit, neg_logit = model.pos_neg_logits_vectorized_topk(score, label.cuda(), filter_mask.cuda(), neg_num=5)
        pos_expand = pos_logit.unsqueeze(1).expand_as(neg_logit)  # [B, n]
        # 4.margin_loss
        target = torch.ones_like(neg_logit)
        res = margin_ranking_loss_fn(pos_expand.reshape(-1), neg_logit.reshape(-1), target.reshape(-1))
        loss += 10 * res
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
    return total_loss


@torch.no_grad()
def valid_eval_metric(valid_or_test):
    rank_list = []
    ent_embs, rel_embs, _ = model()  # [!!!important]不要放在循环内, 导致测试时速度变慢
    for triple in valid_or_test:
        # for triple in tqdm(valid_or_test):
        h, r, t = triple
        head_score = \
            model.score(torch.tensor([[kg.num_ent + kg.num_rel, r + kg.num_ent, t + kg.num_rel]]).cuda(), ent_embs,
                        rel_embs)[0].detach().cpu().numpy()  # [batch_size, num_entity]
        # head_rank = calculate_rank(head_score, h, kg.filter_dict[(-1, r, t)])
        head_rank = calculate_rank(head_score, h, kg.filter_dict[(-1, r, t)])
        rank_list.append(head_rank)
        tail_score = \
            model.score(torch.tensor([[h + kg.num_rel, r + kg.num_ent, kg.num_ent + kg.num_rel]]).cuda(), ent_embs,
                        rel_embs)[0].detach().cpu().numpy()  # [batch_size, num_entity]
        # tail_rank = calculate_rank(tail_score, t, kg.filter_dict[(h, r, -1)])
        tail_rank = calculate_rank(tail_score, t, kg.filter_dict[(h, r, -1)])
        rank_list.append(tail_rank)
    rank_list = np.array(rank_list)
    mr, mrr, hit10, hit3, hit1 = metrics(rank_list)
    return mr, mrr, hit10, hit3, hit1


# model.eval()
# res1 = valid_eval_metric(valid_or_test=kg.valid)
# print(res1)
# res2 = valid_eval_metric(valid_or_test=kg.test)
# print(res2)
# best_mrr = res2[1] or 0
best_mrr = 0

best_result = None
checkpoint_path = ""
for epoch in range(args.num_epoch):
    loss = train_one_epoch_with_negative_sampling_topk(model, optimizer)
    lr_scheduler.step()
    logger.info(f'Epoch {epoch + 1}/{args.num_epoch}, Loss: {loss:.4f}')
    if (epoch + 1) % args.valid_epoch == 0:
        model.eval()
        mr, mrr, hit10, hit3, hit1 = valid_eval_metric(valid_or_test=kg.valid)
        logger.info("Entity Prediction on Valid Set")
        logger.info(f"MR: {mr}")
        logger.info(f"MRR: {mrr}")
        logger.info(f"Hit10: {hit10}")
        logger.info(f"Hit3: {hit3}")
        logger.info(f"Hit1: {hit1}")
        model.eval()
        mr, mrr, hit10, hit3, hit1 = valid_eval_metric(valid_or_test=kg.test)
        logger.info("Entity Prediction on Test Set")
        logger.info(f"MR: {mr}")
        logger.info(f"MRR: {mrr}")
        logger.info(f"Hit10: {hit10}")
        logger.info(f"Hit3: {hit3}")
        logger.info(f"Hit1: {hit1}")
        if mrr > best_mrr:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                logger.info(f"Deleted previous checkpoint: {checkpoint_path}")
            checkpoint_path = f'ckpt/{args.model}/{args.data}/{epoch + 1}.ckpt'
            best_mrr = mrr
            best_result = (mr, mrr, hit10, hit3, hit1)
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': lr_scheduler.state_dict()}, checkpoint_path)

logger.info(f'Best MRR: {best_mrr}, Best Result: {best_result}')
logger.info("Done")
