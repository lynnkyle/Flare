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
parser.add_argument('--before_align', default=0.001, type=float)
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

rep_ent_str, rep_ent_visual, rep_ent_textual, ent_embs = model.embedding()
ent_embs = ent_embs.unsqueeze(1)
embedding = torch.cat((rep_ent_str, rep_ent_visual, rep_ent_textual, ent_embs), dim=1)[:2]

# Step 1: 展平数据为 [12842 * 14, 256]
data_flat = embedding.view(-1, embedding.shape[-1])  # 变为 [14*14, 256]
tile = data_flat.shape[0] // 2

# Step 2: 使用 PCA 降到 2 维
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_flat.detach().cpu().numpy())

# Step 3: 可视化数据

# Step 4: 给每个第二维度的索引分配相同的颜色
# 第二维度的索引在这里是从 0 到 11，所以我们会分配 12 个颜色
colors = np.tile(np.arange(2), tile)  # 每个数据点都会有一个相同颜色的标签

# Step 5: 绘制散点图
scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], c=colors, cmap='tab20', edgecolor='k', s=20)

# 添加颜色条，表示每个索引的颜色
plt.colorbar(scatter, label='Index')

# 设置标签和标题
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of Data with Same Colors for Each Group')

# 显示图形
plt.show()
