import torch
import torch.nn.functional as F
from torch import nn, optim


class Tucker(nn.Module):
    def __init__(self, e_dim, r_dim):
        super(Tucker, self).__init__()
        # self.e_dim = e_dim
        # self.r_dim = r_dim
        self.W = nn.Parameter(torch.rand(r_dim, e_dim, e_dim))  # [r_dim, e_dim, e_dim]
        nn.init.xavier_uniform_(self.W.data)
        self.bn0 = nn.BatchNorm1d(e_dim)
        self.bn1 = nn.BatchNorm1d(e_dim)
        self.input_drop = nn.Dropout(0.3)
        self.hidden_drop = nn.Dropout(0.4)
        self.output_drop = nn.Dropout(0.5)

    """
        计算的是: W * e * r, 计算的不是得分
    """

    def forward(self, ent_emb, rel_emb):
        """
        :param ent_emb: [batch_size, e_dim]
        :param rel_emb: [batch_size, r_dim]
        :return: [batch_size, e_dim]
        """
        x = self.bn0(ent_emb)
        x = self.input_drop(x)
        x = x.view(-1, 1, x.size(1))  # [batch_size, 1, e_dim]

        r = torch.mm(rel_emb, self.W.view(rel_emb.size(1), -1))  # [batch_size, e_dim * e_dim]
        r = r.view(-1, x.size(2), x.size(2))  # [batch_size, e_dim, e_dim]
        r = self.hidden_drop(r)

        x = torch.bmm(x, r)  # [batch_size, 1, e_dim]
        x = x.view(-1, x.size(2))  # [batch_size, e_dim]
        x = self.bn1(x)
        x = self.output_drop(x)
        return x


"""
    RBF映射
"""


class RBFMapping(nn.Module):
    def __init__(self, input_dim, num_kernels):
        super().__init__()
        self.input_dim = input_dim  # 输入的嵌入维度
        self.num_kernels = num_kernels  # 高斯核的个数(输出的嵌入维度)
        self.centers = nn.Parameter(torch.randn(num_kernels, input_dim))  # 高斯核中心
        self.log_sigma = nn.Parameter(torch.zeros(num_kernels))  # 高斯核方差

    def forward(self, x):
        """
        :param x: [batch_size, input_dim]
        :return: [batch_size, kernel_size]
        """
        x_expanded = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        centers_expanded = self.centers.unsqueeze(0)  # [1, num_kernels, input_dim]
        dist_sq = ((x_expanded - centers_expanded) ** 2).sum(dim=-1)  # [batch_size, num_kernels]
        sigma = torch.exp(self.log_sigma).unsqueeze(0)  # [1, num_kernels]
        rbf = torch.exp(-0.5 * dist_sq / (sigma ** 2))  # [batch_size, num_kernels]
        return rbf


"""
    图像 + 高斯噪声 
"""


class GaussianNoise(nn.Module):
    def __init__(self, log_std=0.1):
        super(GaussianNoise, self).__init__()
        self.log_std = nn.Parameter(torch.tensor(log_std))

    def forward(self, x):
        std = torch.exp(self.log_std)
        noise = torch.randn_like(x) * std
        return x + noise


"""
    计算嵌入的相似度
"""


class AlignLoss(nn.Module):
    def __init__(self, temp=0.5, alpha=0.5):
        super(AlignLoss, self).__init__()
        self.LARGE_NUM = 1e9
        self.temp = temp
        self.alpha = alpha
        # self.rbf_map = RBFMapping(input_dim=str_dim, num_kernels=num_kernels)

    def forward(self, emb1, emb2):
        """:
        :param emb1: [batch_size, dim]->[batch_size, 1, dim]
        :param emb2: [batch_size, dim]->[1, batch_size, dim]
        """

        """
            计算第一次、第二次过实体编码器Transformer得到的嵌入的损失
        """
        assert emb1.size() == emb2.size()
        # emb1 = self.rbf_map(emb1)
        # emb2 = self.rbf_map(emb2)
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        ent_num = emb1.shape[0]
        target = F.one_hot(torch.arange(0, ent_num), num_classes=ent_num * 2).to(emb1.device)
        mask = F.one_hot(torch.arange(ent_num), num_classes=ent_num).to(emb1.device)
        logits_aa = torch.matmul(emb1, emb1.t()) / self.temp - self.LARGE_NUM * mask
        logits_ab = torch.matmul(emb1, emb2.t()) / self.temp
        logits_bb = torch.matmul(emb2, emb2.t()) / self.temp - self.LARGE_NUM * mask
        logits_ba = torch.matmul(emb2, emb1.t()) / self.temp
        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)
        loss_a = self.softXEnt(target, logits_a)
        loss_b = self.softXEnt(target, logits_b)
        return self.alpha * loss_a + (1 - self.alpha) * loss_b

    def softXEnt(self, target, logits):
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(target * log_probs).sum() / logits.shape[0]
        return loss


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.5):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.sim_func = Similarity(temp=temp)

    def forward(self, emb1, emb2):
        batch_sim = self.sim_func(emb1.unsqueeze(1), emb2.unsqueeze(0))
        labels = torch.arange(batch_sim.size(0)).long().to('cuda')
        return self.loss(batch_sim, labels)
