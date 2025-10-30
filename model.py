import torch
from torch import nn
from model_new import Tucker, ContrastiveLoss, AlignLoss, GaussianNoise


class FormerAlign(nn.Module):
    def __init__(self, args, num_ent, num_rel, str_dim,
                 visual_tokenizer, textual_tokenizer,
                 visual_token_index, textual_token_index,
                 visual_ent_mask, textual_ent_mask,
                 num_head, dim_hid, num_layer_enc_ent,
                 num_layer_enc_rel, num_layer_dec,
                 dropout=0.1, str_dropout=0.6,
                 visual_dropout=0.1, textual_dropout=0.1,
                 score_function=None):
        super(FormerAlign, self).__init__()
        self.args = args
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.str_dim = str_dim
        self.num_align_sampling = 16
        self.num_contrastive_sampling = 512
        if visual_tokenizer == 'beit':
            visual_tokens = torch.load("tokens/visual.pth")
        elif visual_tokenizer == 'vggan':
            visual_tokens = torch.load("tokens/visual_vqgan.pth")
        else:
            raise NotImplementedError
        if textual_tokenizer == 'bert':
            textual_tokens = torch.load("tokens/textual.pth")
        elif textual_tokenizer == 'roberta':
            textual_tokens = torch.load("tokens/textual_roberta.pth")
        elif textual_tokenizer == 'llama':
            textual_tokens = torch.load("tokens/textual_roberta.pth")
        else:
            raise NotImplementedError

        self.visual_token_index = visual_token_index
        self.visual_token_embed = nn.Embedding.from_pretrained(visual_tokens).requires_grad_(False)
        self.textual_token_index = textual_token_index
        self.textual_token_embed = nn.Embedding.from_pretrained(textual_tokens).requires_grad_(False)
        self.visual_token_embed.requires_grad_(False)
        self.textual_token_embed.requires_grad_(False)
        false_ent = torch.full((self.num_ent, 1), False).cuda()
        self.ent_mask = torch.cat([false_ent, false_ent, visual_ent_mask, textual_ent_mask], dim=1)
        false_rel = torch.full((self.num_rel, 1), False).cuda()
        self.rel_mask = torch.cat([false_rel, false_rel], dim=1)
        self.score_function = score_function
        self.visual_dim = visual_tokens.shape[1]
        self.textual_dim = textual_tokens.shape[1]

        """
            初始化满足Transformer的输入大小: [batch_size, seq_len, emb_dim]
        """
        self.ent_token = nn.Parameter(torch.Tensor(1, 1, str_dim))  # [1, 1, str_dim] -> [batch_size, 1, str_dim]
        self.rel_token = nn.Parameter(torch.Tensor(1, 1, str_dim))  # [1, 1, str_dim] -> [batch_size, 1, str_dim]
        self.ent_emb = nn.Parameter(torch.Tensor(num_ent, 1, str_dim))
        self.rel_emb = nn.Parameter(torch.Tensor(num_rel, 1, str_dim))
        self.lp_token = nn.Parameter(torch.Tensor(1, str_dim))

        self.str_ln = nn.LayerNorm(str_dim)
        self.str_rel_ln = nn.LayerNorm(str_dim)
        self.visual_ln = nn.LayerNorm(str_dim)
        self.textual_ln = nn.LayerNorm(str_dim)

        self.str_drop = nn.Dropout(str_dropout)
        self.visual_drop = nn.Dropout(visual_dropout)
        self.textual_drop = nn.Dropout(textual_dropout)

        self.pos_str_ent = nn.Parameter(torch.Tensor(1, 1, str_dim))
        self.pos_visual_ent = nn.Parameter(torch.Tensor(1, 1, str_dim))
        self.pos_textual_ent = nn.Parameter(torch.Tensor(1, 1, str_dim))
        self.pos_str_rel = nn.Parameter(torch.Tensor(1, 1, str_dim))
        self.pos_visual_rel = nn.Parameter(torch.Tensor(1, 1, str_dim))
        self.pos_textual_rel = nn.Parameter(torch.Tensor(1, 1, str_dim))
        self.pos_head = nn.Parameter(torch.Tensor(1, 1, str_dim))
        self.pos_rel = nn.Parameter(torch.Tensor(1, 1, str_dim))
        self.pos_tail = nn.Parameter(torch.Tensor(1, 1, str_dim))

        self.proj_ent_visual = nn.Linear(self.visual_dim, self.str_dim)
        self.proj_ent_textual = nn.Linear(self.textual_dim, self.str_dim)

        ent_encoder_layer = nn.TransformerEncoderLayer(d_model=str_dim, nhead=num_head, dim_feedforward=dim_hid,
                                                       dropout=dropout, batch_first=True)
        self.ent_encoder = nn.TransformerEncoder(ent_encoder_layer, num_layers=num_layer_enc_ent)
        rel_encoder_layer = nn.TransformerEncoderLayer(d_model=str_dim, nhead=num_head, dim_feedforward=dim_hid,
                                                       dropout=dropout, batch_first=True)
        self.rel_encoder = nn.TransformerEncoder(rel_encoder_layer, num_layers=num_layer_enc_rel)
        decoder_layer = nn.TransformerEncoderLayer(d_model=str_dim, nhead=num_head, dim_feedforward=dim_hid,
                                                   dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layer_dec)

        self.align_before = AlignLoss(temp=0.5, alpha=0.5)
        self.align_after = AlignLoss(temp=0.5, alpha=0.5)
        self.contrastive = ContrastiveLoss(temp=0.5)

        self.num_visual_token = visual_ent_mask.shape[1]
        if self.score_function == 'tucker':
            self.tucker_decoder = Tucker(str_dim, str_dim)
        else:
            pass
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_emb)
        nn.init.xavier_uniform_(self.rel_emb)
        nn.init.xavier_uniform_(self.proj_ent_visual.weight)
        nn.init.xavier_uniform_(self.proj_ent_textual.weight)
        nn.init.xavier_uniform_(self.ent_token)
        nn.init.xavier_uniform_(self.rel_token)
        nn.init.xavier_uniform_(self.lp_token)
        nn.init.xavier_uniform_(self.pos_str_ent)
        nn.init.xavier_uniform_(self.pos_visual_ent)
        nn.init.xavier_uniform_(self.pos_textual_ent)
        nn.init.xavier_uniform_(self.pos_str_rel)
        nn.init.xavier_uniform_(self.pos_visual_rel)
        nn.init.xavier_uniform_(self.pos_textual_rel)
        nn.init.xavier_uniform_(self.pos_head)
        nn.init.xavier_uniform_(self.pos_rel)
        nn.init.xavier_uniform_(self.pos_tail)
        # self.proj_ent_visual.bias.data.zero_()
        # self.proj_ent_textual.bias.data.zero_()

    def forward(self):
        ent_token = self.ent_token.tile(self.num_ent, 1, 1)
        rep_ent_str = self.str_drop(self.str_ln(self.ent_emb)) + self.pos_str_ent
        ent_visual_token = self.visual_token_embed(self.visual_token_index)
        rep_ent_visual = self.visual_drop(self.visual_ln(self.proj_ent_visual(ent_visual_token))) + self.pos_visual_ent
        ent_textual_token = self.textual_token_embed(self.textual_token_index)
        rep_ent_textual = self.textual_drop(
            self.textual_ln(self.proj_ent_textual(ent_textual_token))) + self.pos_textual_ent
        ent_seq_before = torch.cat([ent_token, rep_ent_str, rep_ent_visual, rep_ent_textual], dim=1)
        ent_seq_after = self.ent_encoder(ent_seq_before, src_key_padding_mask=self.ent_mask)
        align_before_loss = None
        align_after_loss = None
        if self.args.before_align != 0:
            align_before_loss = self.align_loss_before_fusion(ent_seq_before)
        if self.args.after_align != 0:
            align_after_loss = self.align_loss_after_fusion(ent_seq_after)
        ent_embs = ent_seq_after[:, 0]
        rel_embs = self.str_drop(self.str_rel_ln(self.rel_emb)).squeeze(1)
        return torch.cat([ent_embs, self.lp_token], dim=0), rel_embs, align_before_loss, align_after_loss

    def score(self, triples, emb_ent, emb_rel):
        """
        :param triples: [batch_size, 3]
        :param emb_ent: [num_ent, str_dim]
        :param emb_rel: [num_rel, str_dim]
        :return: [batch_size, num_entity]
        """
        h_seq = emb_ent[triples[:, 0] - self.num_rel].unsqueeze(1) + self.pos_head  # [batch_size, 1, str_dim]
        r_seq = emb_rel[triples[:, 1] - self.num_ent].unsqueeze(1) + self.pos_rel  # [batch_size, 1, str_dim]
        t_seq = emb_ent[triples[:, 2] - self.num_rel].unsqueeze(1) + self.pos_tail  # [batch_size, 1, str_dim]
        triple_seq = torch.cat([h_seq, r_seq, t_seq], dim=1)  # [batch_size, 3, str_dim]
        triple_out = self.decoder(triple_seq)  # [batch_size, 3, str_dim]
        rel_out = triple_out[:, 1, :]  # [batch_size, 1, str_dim] -> [batch_size, str_dim] 降维
        ctx_out = triple_out[
            triples == self.num_ent + self.num_rel]  # [batch_size, 1, str_dim] -> [batch_size, str_dim] 降维
        if self.score_function == 'tucker':
            tucker_emb = self.tucker_decoder(ctx_out, rel_out)
            score = torch.mm(tucker_emb, emb_ent[:-1].transpose(1, 0))  # [batch_size, num_entity]
        else:
            score = torch.inner(ctx_out, emb_ent[:-1])  # [batch_size, num_entity, 1] -> [batch_size, num_entity] 降维
        return score

    def query(self, triples, emb_ent, emb_rel):
        """
        :param triples: [batch_size, 3]
        :param emb_ent: [num_ent, str_dim]
        :param emb_rel: [num_rel, str_dim]
        :return: [batch_size, num_entity]
        """
        h_seq = emb_ent[triples[:, 0] - self.num_rel].unsqueeze(1) + self.pos_head  # [batch_size, 1, str_dim]
        r_seq = emb_rel[triples[:, 1] - self.num_ent].unsqueeze(1) + self.pos_rel  # [batch_size, 1, str_dim]
        t_seq = emb_ent[triples[:, 2] - self.num_rel].unsqueeze(1) + self.pos_tail  # [batch_size, 1, str_dim]
        triple_seq = torch.cat([h_seq, r_seq, t_seq], dim=1)  # [batch_size, 3, str_dim]
        triple_out = self.decoder(triple_seq)  # [batch_size, 3, str_dim]
        query_out = triple_out[triples == self.num_ent + self.num_rel]  # [batch_size, str_dim]
        return query_out

    """
        FormerAlign 模型利用 Transformer 编码器中的 dropout 机制人为制造多模态嵌入的细微变化，相当于一种轻量级数据增强。
        然后，它在批量（batch）中使用对比学习，让模型学会关注真正重要的信息，提高实体表示的区分性和鲁棒性。
    """

    def align_loss_before_fusion(self, ent_seq):
        """
        :param emb_ent: [num_ent, seq_len, str_dim] # [12842, 14, 256]
        :return:
        """
        select_ents = torch.randperm(self.num_ent)[:self.num_align_sampling]
        ent_token = ent_seq[select_ents, 0, :]
        ent_str_emb = ent_seq[select_ents, 1, :]
        ent_visual_emb = torch.mean(ent_seq[select_ents, 2:2 + self.num_visual_token, :], dim=1)
        ent_textual_emb = torch.mean(ent_seq[select_ents, 2 + self.num_visual_token:, :], dim=1)
        str_visual_loss = 0
        str_textual_loss = 0
        if self.args.max_vis_token != 0:
            str_visual_loss = self.align_before(ent_str_emb, ent_visual_emb)
        if self.args.max_txt_token != 0:
            str_textual_loss = self.align_before(ent_str_emb, ent_textual_emb)
        return (str_visual_loss + str_textual_loss) / self.num_align_sampling

    def align_loss_after_fusion(self, ent_seq):
        """
        :param emb_ent: [num_ent, str_dim]
        :return:
        """
        select_ents = torch.randperm(self.num_ent)[:self.num_align_sampling]
        ent_token = ent_seq[select_ents, 0, :]
        ent_str_emb = ent_seq[select_ents, 1, :]
        ent_visual_emb = torch.mean(ent_seq[select_ents, 2:2 + self.num_visual_token, :], dim=1)
        ent_textual_emb = torch.mean(ent_seq[select_ents, 2 + self.num_visual_token:, :], dim=1)
        str_visual_loss = 0
        str_textual_loss = 0
        if self.args.max_vis_token != 0:
            str_visual_loss = self.align_after(ent_str_emb, ent_visual_emb)
        if self.args.max_txt_token != 0:
            str_textual_loss = self.align_after(ent_str_emb, ent_textual_emb)
        return (str_visual_loss + str_textual_loss) / self.num_align_sampling

    def contrastive_loss(self, emb_ent):
        ent_token = self.ent_token.tile(self.num_ent, 1, 1)
        rep_ent_str = self.str_drop(self.str_ln(self.ent_emb)) + self.pos_str_ent
        ent_visual_token = self.visual_token_embed(self.visual_token_index)
        rep_ent_visual = self.visual_drop(self.visual_ln(self.proj_ent_visual(ent_visual_token))) + self.pos_visual_ent
        ent_textual_token = self.textual_token_embed(self.textual_token_index)
        rep_ent_textual = self.textual_drop(
            self.textual_ln(self.proj_ent_textual(ent_textual_token))) + self.pos_textual_ent
        ent_seq = torch.cat([ent_token, rep_ent_str, rep_ent_visual, rep_ent_textual], dim=1)

        # ent_embs: [ent_num, seq_len, embed_dim]
        ent_embs = self.ent_encoder(ent_seq, src_key_padding_mask=self.ent_mask)
        emb_ent1 = torch.cat([ent_embs[:, 0], self.lp_token], dim=0)
        ent_emb2 = torch.cat([torch.mean(ent_embs, dim=1), self.lp_token], dim=0)
        ent_emb3 = torch.cat([torch.mean(ent_embs[:, 2: 2 + self.num_visual_token, :], dim=1), self.lp_token], dim=0)
        ent_emb4 = torch.cat([torch.mean(ent_embs[:, 2 + self.num_visual_token:, :], dim=1), self.lp_token], dim=0)
        select_ents = torch.randperm(emb_ent.shape[0])[: self.num_contrastive_sampling]
        loss = 0
        avg_list = [emb_ent1, ent_emb2, ent_emb3, ent_emb4]
        if self.args.max_vis_token == 0:
            avg_list = [emb_ent1, ent_emb2, ent_emb4]
        if self.args.max_txt_token == 0:
            avg_list = [emb_ent1, ent_emb2, ent_emb3]
        for emb in avg_list:
            loss += self.contrastive(emb_ent[select_ents], emb[select_ents])
        loss /= 4  # loss = 4.7
        return loss
