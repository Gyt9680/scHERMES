import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import opt
from encoder import *



class FeatureRecalibration(nn.Module):
    def __init__(self, d_model, reduction=4):
        super(FeatureRecalibration, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // reduction, d_model, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Squeeze-and-Excitation 风格的特征重校准
        y = x.mean(dim=0, keepdim=True)
        weight = self.fc(y)
        return x * weight



class CrossModalAttention(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x_self, x_other):
        # x_self: 当前模态特征 (Query)
        # x_other: 辅助模态特征 (Key, Value)
        # 维度: (N, d)
        tokens = torch.stack([x_self, x_other], dim=1)  # (N, 2, d)
        q = tokens[:, :1, :]  # (N, 1, d) 只更新 x_self 对应的 token

        # Cross Attention
        attn_out, _ = self.attn(q, tokens, tokens, need_weights=False)  # (N, 1, d)

        # 残差连接 + LayerNorm
        x = self.norm1(q + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x.squeeze(1)  # (N, d)



class DynamicGate(nn.Module):
    def __init__(self, d_model):
        super(DynamicGate, self).__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # 拼接特征计算门控权重 alpha
        gate = self.gate_net(torch.cat([x1, x2], dim=1))
        return gate


class scHERMES(nn.Module):
    def __init__(self, ae1, ae2, gae1, gae2, n_node=None):
        super(scHERMES, self).__init__()
        self.ae1 = ae1
        self.ae2 = ae2
        self.gae1 = gae1
        self.gae2 = gae2

        n_z = opt.args.n_z
        n_clusters = opt.args.n_clusters

        # 1. 模态内门控 (Intra-modal Gating)
        self.intra_gate1 = DynamicGate(n_z)
        self.intra_gate2 = DynamicGate(n_z)

        # 2. 特征重校准模块 (FRM)
        self.frm1 = FeatureRecalibration(n_z)
        self.frm2 = FeatureRecalibration(n_z)

        # 3. 跨模态注意力 (Cross-Modal Attention)
        self.cm_attn1 = CrossModalAttention(d_model=n_z, nhead=4)
        self.cm_attn2 = CrossModalAttention(d_model=n_z, nhead=4)

        # 4. 最终融合门控 (Final Fusion Gating)
        self.final_gate1 = DynamicGate(n_z)
        self.final_gate2 = DynamicGate(n_z)

        # 聚类中心与分布计算
        self.cluster_centers1 = Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)
        self.cluster_centers2 = Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_centers1.data)
        torch.nn.init.xavier_normal_(self.cluster_centers2.data)

        self.q_distribution1 = q_distribution(self.cluster_centers1)
        self.q_distribution2 = q_distribution(self.cluster_centers2)

        # 标签对比模块
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(n_node, n_clusters),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, adj1, x2, adj2, pretrain=False):
        # -----------------------------------------------
        # 1. 编码 (Encoding)
        # -----------------------------------------------
        z_ae1 = self.ae1.encoder(x1)
        z_ae2 = self.ae2.encoder(x2)
        z_igae1, a_igae1 = self.gae1.encoder(x1, adj1)
        z_igae2, a_igae2 = self.gae2.encoder(x2, adj2)

        # -----------------------------------------------
        # 2. 模态内融合 (Intra-modal Fusion)
        # -----------------------------------------------
        gate_intra1 = self.intra_gate1(z_ae1, z_igae1)
        z_intra1 = gate_intra1 * z_ae1 + (1 - gate_intra1) * z_igae1

        gate_intra2 = self.intra_gate2(z_ae2, z_igae2)
        z_intra2 = gate_intra2 * z_ae2 + (1 - gate_intra2) * z_igae2

        # -----------------------------------------------
        # 3. 特征重校准 (FRM)
        # -----------------------------------------------
        z_intra1 = self.frm1(z_intra1)
        z_intra2 = self.frm2(z_intra2)

        # 图传播平滑 (Graph Smoothing)
        z_l1 = torch.spmm(adj1, z_intra1)
        z_l2 = torch.spmm(adj2, z_intra2)

        # -----------------------------------------------
        # 4. 跨模态注意力交互 (Cross-Modal Attention)
        # -----------------------------------------------
        z_cross1 = self.cm_attn1(z_l1, z_l2)
        z_cross2 = self.cm_attn2(z_l2, z_l1)

        # -----------------------------------------------
        # 5. 最终融合 (Final Fusion)
        # -----------------------------------------------
        gate_final1 = self.final_gate1(z_l1, z_cross1)
        z1 = gate_final1 * z_l1 + (1 - gate_final1) * z_cross1

        gate_final2 = self.final_gate2(z_l2, z_cross2)
        z2 = gate_final2 * z_l2 + (1 - gate_final2) * z_cross2

        # 再次图平滑
        z1 = torch.spmm(adj1, z1)
        z2 = torch.spmm(adj2, z2)

        # 辅助对比输出
        z1_tilde = self.label_contrastive_module(z1.T)
        z2_tilde = self.label_contrastive_module(z2.T)

        cons = [z1, z2, z1_tilde, z2_tilde]

        # =========================================================
        # 解码与重构 (Decoding & Reconstruction)
        # =========================================================

        # 1. 自我重构 (Self-Reconstruction)
        x_hat1 = self.ae1.decoder(z1)
        x_hat2 = self.ae2.decoder(z2)

        # GAE 图重构
        z_hat1, z_adj_hat1 = self.gae1.decoder(z1, adj1)
        a_hat1 = a_igae1 + z_adj_hat1
        z_hat2, z_adj_hat2 = self.gae2.decoder(z2, adj2)
        a_hat2 = a_igae2 + z_adj_hat2

        # 2. 交叉重构 (Cross-Modal Reconstruction) - HERMES 关键特性
        # 逻辑：使用 z2 (ATAC特征) 预测 x1 (RNA原始数据)
        x_hat1_cross = self.ae1.decoder(z2)
        # 逻辑：使用 z1 (RNA特征) 预测 x2 (ATAC原始数据)
        x_hat2_cross = self.ae2.decoder(z1)

        if not pretrain:
            Q1 = self.q_distribution1(z1, z_ae1, z_igae1)
            Q2 = self.q_distribution2(z2, z_ae2, z_igae2)
        else:
            Q1, Q2 = None, None

        # 返回值包含了交叉重构项 x_hat1_cross, x_hat2_cross
        return x_hat1, z_hat1, a_hat1, x_hat2, z_hat2, a_hat2, Q1, Q2, z1, z2, cons, x_hat1_cross, x_hat2_cross