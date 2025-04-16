from argparse import Namespace
import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from fpgnn.data import GetPubChemFPs, create_graph, get_atom_features_dim

import csv
from torch import Tensor
from torch import nn, sum
from torch.nn import init
import torch.nn.functional as F
import math

atts_out = []


class FPN(nn.Module):
    def __init__(self, args):
        super(FPN, self).__init__()
        self.fp_2_dim = args.fp_2_dim
        self.dropout_fpn = args.dropout
        self.cuda = args.cuda
        self.hidden_dim = args.hidden_size
        self.args = args
        if hasattr(args, 'fp_type'):
            self.fp_type = args.fp_type
        else:
            self.fp_type = 'mixed'

        if self.fp_type == 'mixed':
            self.fp_dim = 2513
        else:
            self.fp_dim = 1024

        if hasattr(args, 'fp_changebit'):
            self.fp_changebit = args.fp_changebit
        else:
            self.fp_changebit = None

        self.fc1 = KANLinear(self.fp_dim, self.fp_2_dim) #KANLinear(in_features=600, out_features=1)
        self.act_func = nn.ReLU()
        self.fc2 = KANLinear(self.fp_2_dim, self.hidden_dim)
        self.dropout = nn.Dropout(p=self.dropout_fpn)

    def forward(self, smile):
        fp_list = []
        for i, one in enumerate(smile):
            fp = []
            mol = Chem.MolFromSmiles(one)

            if self.fp_type == 'mixed':
                fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
                fp_pubcfp = GetPubChemFPs(mol)
                fp_phaErGfp = AllChem.GetErGFingerprint(mol,fuzzIncrement=0.3,maxPath=21,minPath=1)
                generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
                fp_morgan = generator.GetFingerprint(mol)
                fp.extend(fp_maccs)
                fp.extend(fp_pubcfp)
                fp.extend(fp_phaErGfp)
                fp.extend(fp_morgan)

            else:
                generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
                fp_morgan = generator.GetFingerprint(mol)
                fp.extend(fp_morgan)

            fp_list.append(fp)

        if self.fp_changebit is not None and self.fp_changebit != 0:
            fp_list = np.array(fp_list)
            fp_list[:, self.fp_changebit - 1] = np.ones(fp_list[:, self.fp_changebit - 1].shape)
            fp_list.tolist()

        fp_list = torch.Tensor(fp_list)

        if self.cuda:
            fp_list = fp_list.cuda()

        fpn_out = self.fc1(fp_list)
        fpn_out = self.dropout(fpn_out)
        fpn_out = self.act_func(fpn_out)
        fpn_out = self.fc2(fpn_out)
        return fpn_out


class GATLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout_gnn, alpha, inter_graph, concat=True):
        super(GATLayer, self).__init__()
        self.dropout_gnn = dropout_gnn
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.dropout = nn.Dropout(p=self.dropout_gnn)
        self.inter_graph = inter_graph

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        if self.inter_graph is not None:
            self.atts_out = []

    def forward(self, mole_out, adj):
        atom_feature = torch.mm(mole_out, self.W)
        N = atom_feature.size()[0]

        atom_trans = torch.cat([atom_feature.repeat(1, N).view(N * N, -1), atom_feature.repeat(N, 1)], dim=1).view(N,
                                                                                                                   -1,
                                                                                                                   2 * self.out_features)
        e = self.leakyrelu(torch.matmul(atom_trans, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        if self.inter_graph is not None:
            att_out = attention
            if att_out.is_cuda:
                att_out = att_out.cpu()
            att_out = np.array(att_out)
            att_out[att_out < -10000] = 0
            att_out = att_out.tolist()
            atts_out.append(att_out)

        attention = nn.functional.softmax(attention, dim=1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, atom_feature)

        if self.concat:
            return nn.functional.elu(output)
        else:
            return output


class GATOne(nn.Module):
    def __init__(self, args):
        super(GATOne, self).__init__()
        self.nfeat = get_atom_features_dim()
        self.nhid = args.nhid
        self.dropout_gnn = args.dropout_gat
        self.atom_dim = args.hidden_size
        self.alpha = 0.2
        self.nheads = args.nheads
        self.args = args
        self.dropout = nn.Dropout(p=self.dropout_gnn)

        if hasattr(args, 'inter_graph'):
            self.inter_graph = args.inter_graph
        else:
            self.inter_graph = None

        self.attentions = [GATLayer(self.nfeat, self.nhid, dropout_gnn=self.dropout_gnn, alpha=self.alpha,
                                    inter_graph=self.inter_graph, concat=True) for _ in range(self.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATLayer(self.nhid * self.nheads, self.atom_dim, dropout_gnn=self.dropout_gnn, alpha=self.alpha,
                                inter_graph=self.inter_graph, concat=False)

    def forward(self, mole_out, adj):
        mole_out = self.dropout(mole_out)
        mole_out = torch.cat([att(mole_out, adj) for att in self.attentions], dim=1)
        mole_out = self.dropout(mole_out)
        mole_out = nn.functional.elu(self.out_att(mole_out, adj))
        return nn.functional.log_softmax(mole_out, dim=1)


class GATEncoder(nn.Module):
    def __init__(self, args):
        super(GATEncoder, self).__init__()
        self.cuda = args.cuda
        self.args = args
        self.encoder = GATOne(self.args)

    def forward(self, mols, smiles):
        atom_feature, atom_index = mols.get_feature()
        if self.cuda:
            atom_feature = atom_feature.cuda()

        gat_outs = []
        for i, one in enumerate(smiles):
            adj = []
            mol = Chem.MolFromSmiles(one)
            adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
            adj = adj / 1
            adj = torch.from_numpy(adj)
            if self.cuda:
                adj = adj.cuda()

            atom_start, atom_size = atom_index[i]
            one_feature = atom_feature[atom_start:atom_start + atom_size]

            gat_atoms_out = self.encoder(one_feature, adj)
            gat_out = gat_atoms_out.sum(dim=0) / atom_size
            gat_outs.append(gat_out)
        gat_outs = torch.stack(gat_outs, dim=0)
        return gat_outs


class GAT(nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        self.args = args
        self.encoder = GATEncoder(self.args)

    def forward(self, smile):
        mol = create_graph(smile, self.args)
        gat_out = self.encoder.forward(mol, smile)

        return gat_out

def external_norm(attn):
    softmax = nn.Softmax(dim=0)  # N
    attn = softmax(attn)  # bs,n,S
    attn = attn /sum(attn, dim=2, keepdim=True)  # bs,n,S
    return attn


class DNorm(nn.Module):
    def __init__(
            self,
            dim1=0 ,dim2=2
    ):
        super().__init__()
        self.dim1 =dim1
        self.dim2 =dim2
        self.softmax = nn.Softmax(dim=self.dim1)  # N

    def forward(self, attn: Tensor) -> Tensor:
        # softmax = nn.Softmax(dim=0)  # N
        attn = self.softmax(attn)  # bs,n,S
        attn = attn /sum(attn, dim=self.dim2, keepdim=True)  # bs,n,S
        return attn

class GEANet(nn.Module):

    def __init__(
            self, dim, GEANet_cfg):
        super().__init__()


        self.dim = dim
        self.external_num_heads = GEANet_cfg.n_heads
        self.use_shared_unit = GEANet_cfg.shared_unit
        self.use_edge_unit = GEANet_cfg.edge_unit
        self.unit_size = GEANet_cfg.unit_size

        # self.q_Linear = nn.Linear(in_dim, gconv_dim - dim_pe)
        self.node_U1 = nn.Linear(self.unit_size, self.unit_size)
        self.node_U2 = nn.Linear(self.unit_size, self.unit_size)

        assert self.unit_size * self.external_num_heads == self.dim, "dim must be divisible by external_num_heads"

        # nn.init.xavier_normal_(self.node_m1.weight, gain=1)
        # nn.init.xavier_normal_(self.node_m2.weight, gain=1)
        if  self.use_edge_unit:
            self.edge_U1 = nn.Linear(self.unit_size, self.unit_size)
            self.edge_U2 = nn.Linear(self.unit_size, self.unit_size)
            if self.use_shared_unit:
                self.share_U = nn.Linear(dim, dim)

            # nn.init.xavier_normal_(self.edge_m1.weight, gain=1)
            # nn.init.xavier_normal_(self.edge_m2.weight, gain=1)
            # nn.init.xavier_normal_(self.share_m.weight, gain=1)
        self.norm = DNorm()

        # self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self, node_x ,edge_attr = None) -> Tensor:
        if self.use_shared_unit:
            node_x = self.share_U(node_x)
            edge_attr = self.share_U(edge_attr)
        # x : N x 64
        # External attention
        N, d, head = node_x.size()[0], node_x.size()[1], self.external_num_heads
        node_out = node_x.reshape(N, head ,-1)  # Q * 4（head）  ：  N x 16 x 4(head)
        # node_out = node_out.transpose(1, 2)  # (N, 16, 4) -> (N, 4, 16)
        node_out = self.node_U1(node_out)
        attn = self.norm(node_out)  # 行列归一化  N x 16 x 4
        node_out = self.node_U2(attn)
        node_out = node_out.reshape(N, -1)

        if self.use_edge_unit:

            N, d, head = edge_attr.size()[0], edge_attr.size()[1], self.external_num_heads
            edge_out = edge_attr.reshape(N, -1, head)  # Q * 4（head）  ：  N x 16 x 4(head)
            edge_out = edge_out.transpose(1, 2)  # (N, 16, 4) -> (N, 4, 16)
            edge_out = self.edge_U1(edge_out)
            attn = self.norm(edge_out)  # 行列归一化  N x 16 x 4
            edge_out = self.edge_U2(attn)
            edge_out = edge_out.reshape(N, -1)
        else:
            edge_out = edge_attr

        return node_out ,edge_out

class GEANetConfig:
    def __init__(self, n_heads, shared_unit, edge_unit, unit_size):
        self.n_heads = n_heads
        self.shared_unit = shared_unit
        self.edge_unit = edge_unit
        self.unit_size = unit_size

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,  # 网格大小，默认为 5
        spline_order=3, # 分段多项式的阶数，默认为 3
        scale_noise=0.1,  # 缩放噪声，默认为 0.1
        scale_base=1.0,   # 基础缩放，默认为 1.0
        scale_spline=1.0,    # 分段多项式的缩放，默认为 1.0
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,  # 基础激活函数，默认为 SiLU（Sigmoid Linear Unit）
        grid_eps=0.02,
        grid_range=[-1, 1],  # 网格范围，默认为 [-1, 1]
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size # 设置网格大小和分段多项式的阶数
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size   # 计算网格步长
        grid = ( # 生成网格
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)  # 将网格作为缓冲区注册

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features)) # 初始化基础权重和分段多项式权重
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:  # 如果启用独立的分段多项式缩放，则初始化分段多项式缩放参数
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise # 保存缩放噪声、基础缩放、分段多项式的缩放、是否启用独立的分段多项式缩放、基础激活函数和网格范围的容差
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()  # 重置参数

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)# 使用 Kaiming 均匀初始化基础权重
        with torch.no_grad():
            noise = (# 生成缩放噪声
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_( # 计算分段多项式权重
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:  # 如果启用独立的分段多项式缩放，则使用 Kaiming 均匀初始化分段多项式缩放参数
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        """
        计算给定输入张量的 B-样条基函数。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
        torch.Tensor: B-样条基函数张量，形状为 (batch_size, in_features, grid_size + spline_order)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = ( # 形状为 (in_features, grid_size + 2 * spline_order + 1)
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        """
        计算插值给定点的曲线的系数。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
        y (torch.Tensor): 输出张量，形状为 (batch_size, in_features, out_features)。
        返回:
        torch.Tensor: 系数张量，形状为 (out_features, in_features, grid_size + spline_order)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        # 计算 B-样条基函数
        A = self.b_splines(x).transpose(
            0, 1 # 形状为 (in_features, batch_size, grid_size + spline_order)
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features) # 形状为 (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(   # 使用最小二乘法求解线性方程组
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)  # 形状为 (in_features, grid_size + spline_order, out_features)
        result = solution.permute( # 调整结果的维度顺序
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        """
        获取缩放后的分段多项式权重。

        返回:
        torch.Tensor: 缩放后的分段多项式权重张量，形状与 self.spline_weight 相同。
        """
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor): # 将输入数据通过模型的各个层，经过线性变换和激活函数处理，最终得到模型的输出结果
        """
        前向传播函数。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
        torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        # print(f"shape of input: {x.shape}")
        # print(f"in_features: {self.in_features}")
        # print(f"x.size(1): {x.size(1)}")
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight) # 计算基础线性层的输出
        spline_output = F.linear( # 计算分段多项式线性层的输出
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output  # 返回基础线性层输出和分段多项式线性层输出的和

    @torch.no_grad()
    # 更新网格。
    # 参数:
    # x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
    # margin (float): 网格边缘空白的大小。默认为 0.01。
    # 根据输入数据 x 的分布情况来动态更新模型的网格,使得模型能够更好地适应输入数据的分布特点，从而提高模型的表达能力和泛化能力。
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)  # 计算 B-样条基函数
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)  # 调整维度顺序为 (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)  # 调整维度顺序为 (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0] # 对每个通道单独排序以收集数据分布
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)   # 更新网格和分段多项式权重
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # 计算正则化损失，用于约束模型的参数，防止过拟合
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        """
        计算正则化损失。

        这是对原始 L1 正则化的简单模拟，因为原始方法需要从扩展的（batch, in_features, out_features）中间张量计算绝对值和熵，
        而这个中间张量被 F.linear 函数隐藏起来，如果我们想要一个内存高效的实现。

        现在的 L1 正则化是计算分段多项式权重的平均绝对值。作者的实现也包括这一项，除了基于样本的正则化。

        参数:
        regularize_activation (float): 正则化激活项的权重，默认为 1.0。
        regularize_entropy (float): 正则化熵项的权重，默认为 1.0。

        返回:
        torch.Tensor: 正则化损失。
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module): # 封装了一个KAN神经网络模型，可以用于对数据进行拟合和预测。
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        """
        初始化 KAN 模型。

        参数:
            layers_hidden (list): 包含每个隐藏层输入特征数量的列表。
            grid_size (int): 网格大小，默认为 5。
            spline_order (int): 分段多项式的阶数，默认为 3。
            scale_noise (float): 缩放噪声，默认为 0.1。
            scale_base (float): 基础缩放，默认为 1.0。
            scale_spline (float): 分段多项式的缩放，默认为 1.0。
            base_activation (torch.nn.Module): 基础激活函数，默认为 SiLU。
            grid_eps (float): 网格调整参数，默认为 0.02。
            grid_range (list): 网格范围，默认为 [-1, 1]。
        """
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False): # 调用每个KANLinear层的forward方法，对输入数据进行前向传播计算输出。
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            update_grid (bool): 是否更新网格。默认为 False。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):#计算正则化损失的方法，用于约束模型的参数，防止过拟合。
        """
        计算正则化损失。

        参数:
            regularize_activation (float): 正则化激活项的权重，默认为 1.0。
            regularize_entropy (float): 正则化熵项的权重，默认为 1.0。

        返回:
            torch.Tensor: 正则化损失。
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        ) 
   

class FpgnnModel(nn.Module):
    def __init__(self, is_classif, gat_scale, cuda, dropout_fpn):
        super(FpgnnModel, self).__init__()
        self.gat_scale = gat_scale
        self.is_classif = is_classif
        self.cuda = cuda
        self.dropout_fpn = dropout_fpn
        self.gea_cfg = GEANetConfig(n_heads=4, shared_unit=False, edge_unit=False, unit_size=75)
        self.gea_block = GEANet(dim=300, GEANet_cfg=self.gea_cfg)
        if self.is_classif:
            self.sigmoid = nn.Sigmoid()

    def create_gat(self, args):
        self.encoder3 = GAT(args)

    def create_fpn(self, args):
        self.encoder2 = FPN(args)

    def create_scale(self, args):
        linear_dim = int(args.hidden_size)
        if self.gat_scale == 1:
            self.fc_gat = KANLinear(linear_dim, linear_dim)
        elif self.gat_scale == 0:
            self.fc_fpn = KANLinear(linear_dim, linear_dim)
        else:
            self.gat_dim = int((linear_dim * 2 * self.gat_scale) // 1)
            self.fc_gat = KANLinear(linear_dim, self.gat_dim)
            self.fc_fpn = KANLinear(linear_dim, linear_dim * 2 - self.gat_dim)
        self.act_func = nn.ReLU()

    def create_ffn(self, args):
        linear_dim = args.hidden_size
        if self.gat_scale == 1:
            self.ffn = nn.Sequential(
                nn.Dropout(self.dropout_fpn),
                nn.Linear(in_features=linear_dim, out_features=linear_dim, bias=True),
                nn.ReLU(),
                nn.Dropout(self.dropout_fpn),
                nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
            )
        elif self.gat_scale == 0:
            self.ffn = nn.Sequential(
                nn.Dropout(self.dropout_fpn),
                nn.Linear(in_features=linear_dim, out_features=linear_dim, bias=True),
                nn.ReLU(),
                nn.Dropout(self.dropout_fpn),
                nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
            )

        else:
            # self.ffn = nn.Sequential(
            #     nn.Dropout(self.dropout_fpn),
            #     nn.Linear(in_features=linear_dim * 2, out_features=linear_dim, bias=True),
            #     nn.ReLU(),
            #     nn.Dropout(self.dropout_fpn),
            #     nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
             self.ffn = KANLinear(in_features=600, out_features=args.task_num) #没有scale的情况下


    def forward(self, input):
        if self.gat_scale == 1:
            output = self.encoder3(input)
        elif self.gat_scale == 0:
            output = self.encoder2(input)
        else:
            gat_out = self.encoder3(input)
            fpn_out = self.encoder2(input)
            gat_out = self.fc_gat(gat_out)
            gat_out = self.act_func(gat_out)
            gat_out, _ = self.gea_block(gat_out)
    

            fpn_out = self.fc_fpn(fpn_out)
            fpn_out = self.act_func(fpn_out)

            output = torch.cat([gat_out, fpn_out], axis=1)
            #output = self.sc(output)
        output = self.ffn(output)

        if self.is_classif and not self.training:
            output = self.sigmoid(output)

        return output


def get_atts_out():
    return atts_out


def FPGNN(args):
    if args.dataset_type == 'classification':
        is_classif = 1
    else:
        is_classif = 0
    model = FpgnnModel(is_classif, args.gat_scale, args.cuda, args.dropout)
    if args.gat_scale == 1:
        model.create_gat(args)
        model.create_ffn(args)
    elif args.gat_scale == 0:
        model.create_fpn(args)
        model.create_ffn(args)
    else:
        model.create_gat(args)
        model.create_fpn(args)
        model.create_scale(args)
        model.create_ffn(args)

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    return model
