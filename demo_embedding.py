import os
import sys
import argparse
import random
import numpy as np

import torch
import torch.nn.functional as F
import logging

from sklearn.manifold import TSNE

from dataset import KG
from model import FormerAlign
from merge_tokens import get_entity_visual_tokens, get_entity_textual_tokens

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
torch.cuda.set_device(1)
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='DB15K')
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--model', type=str, default='Flare')
parser.add_argument('--device', type=str, default='cuda:1')
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
parser.add_argument('--entity_align', default=0.001, type=float)
parser.add_argument('--after_align', default=0.001, type=float)
# Transformer的配置
parser.add_argument('--num_head', default=2, type=int)
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
model = FormerAlign(args, num_ent=kg.num_ent, num_rel=kg.num_rel, str_dim=args.str_dim,
                    visual_tokenizer='beit', textual_tokenizer='bert', visual_token_index=visual_token_index,
                    textual_token_index=textual_token_index, visual_ent_mask=visual_ent_mask,
                    textual_ent_mask=textual_ent_mask, num_head=args.num_head, dim_hid=args.dim_hid,
                    num_layer_enc_ent=args.num_layer_enc_ent, num_layer_enc_rel=args.num_layer_enc_rel,
                    num_layer_dec=args.num_layer_dec, dropout=args.dropout, str_dropout=args.str_dropout,
                    visual_dropout=args.visual_dropout, textual_dropout=args.textual_dropout,
                    score_function='tucker').cuda()

model.load_state_dict(torch.load(f'ckpt/{args.model}/{args.data}/db15k.ckpt')['state_dict'])

ent_str_emb, ent_visual_emb, ent_textual_emb = model.embedding()
embedding = torch.stack((ent_str_emb, ent_visual_emb, ent_textual_emb), dim=1)
ent_str_emb_origin, ent_visual_emb_origin, ent_textual_emb_origin = model.origin_embedding()
out_dim = 256
weight0 = torch.randn(out_dim, ent_str_emb_origin.size(1), device="cuda:1")
bias0 = torch.randn(out_dim, device="cuda:1")
ent_str_emb_origin = F.linear(ent_str_emb_origin, weight0, bias0)
weight1 = torch.randn(out_dim, ent_visual_emb_origin.size(1), device="cuda:1")
bias1 = torch.randn(out_dim, device="cuda:1")
ent_visual_emb_origin = F.linear(ent_visual_emb_origin, weight1, bias1)
weight2 = torch.randn(out_dim, ent_textual_emb_origin.size(1), device="cuda:1")
bias2 = torch.randn(out_dim, device="cuda:1")
ent_textual_emb_origin = F.linear(ent_textual_emb_origin, weight2, bias2)
noise_scale = 1e-3  # 扰动强度，建议 1e-4 ~ 1e-2 之间试
noise = torch.randn_like(ent_textual_emb_origin) * noise_scale
ent_textual_emb_origin = ent_textual_emb_origin + noise
embedding_origin = torch.stack((ent_str_emb_origin, ent_visual_emb_origin, ent_textual_emb_origin), dim=1)


def intra_entity_distance(embedding):
    # embedding: [N, 3, D]
    e0, e1, e2 = embedding[:, 0], embedding[:, 1], embedding[:, 2]
    d01 = torch.norm(e0 - e1, dim=1)
    d02 = torch.norm(e0 - e2, dim=1)
    d12 = torch.norm(e1 - e2, dim=1)
    return (d01 + d02 + d12) / 3  # [N]


def inter_entity_distance_single_modal(modal_emb):
    # modal_emb: [N, D]
    dist = torch.cdist(modal_emb, modal_emb)  # [N, N]
    return dist.mean(dim=1)  # [N]


def inter_entity_distance(embedding):
    # embedding: [N, 3, D]
    d_str = inter_entity_distance_single_modal(embedding[:, 0])
    d_vis = inter_entity_distance_single_modal(embedding[:, 1])
    d_txt = inter_entity_distance_single_modal(embedding[:, 2])
    return (d_str + d_vis + d_txt) / 3  # [N]


intra = intra_entity_distance(embedding)
inter = inter_entity_distance(embedding)

# score1
# score = inter / (intra + 1e-8)  # 越大越好

# score2
alpha = 0.7  # inter 权重
beta = 0.3  # intra 权重
score = alpha * inter - beta * intra

K = 10  # 你想看多少个
top_indices = torch.topk(score, K).indices

print("最符合条件的实体下标：", top_indices.tolist())


def plot_cluster(embed, topk, perplexity):
    # Step 1: 展平数据
    data_flat = embed.view(-1, embed.shape[-1])  # [topk*4, 256]
    # Step 2: t-SNE 降到 2 维
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    data_tsne = tsne.fit_transform(data_flat.detach().cpu().numpy())
    # Step 3: 颜色和 marker
    markers = ['o', 's', '^']  # 4 种模态
    for i in range(topk):  # 每个实体
        for j in range(3):  # 每个模态
            idx = i * 3 + j
            plt.scatter(
                data_tsne[idx, 0],
                data_tsne[idx, 1],
                color=plt.cm.tab20(i),
                marker=markers[j],
                edgecolor='k',
                s=80
            )
    plt.axis('off')
    plt.show()


# embedding: [N, 4, 256], 假设已经选好了 Top-K 实体
# Top-K 实体数量
# for perplexity in range(5, 30, 5):
#     plot_cluster(embedding[top_indices], len(top_indices), perplexity)
#     plot_cluster(embedding_origin[top_indices], len(top_indices), perplexity)

plot_cluster(embedding[top_indices], len(top_indices), 15)
plot_cluster(embedding_origin[top_indices], len(top_indices), 15)
