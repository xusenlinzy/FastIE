import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sinusoid_encoding_table(n_position: int, d_hid: int):
    """Returns: [seq_len, d_hid]
    """
    embeddings_table = torch.zeros(n_position, d_hid)
    position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid))

    embeddings_table[:, 0::2] = torch.sin(position * div_term)
    embeddings_table[:, 1::2] = torch.cos(position * div_term)

    return embeddings_table


class RoPEPositionEncoding(nn.Module):
    """旋转式位置编码: https://kexue.fm/archives/8265
    """
    def __init__(self, embedding_size: int, rope_rank: str = "adjacent"):
        super(RoPEPositionEncoding, self).__init__()
        self.max_seq_len_cache = -1
        self.embedding_size = embedding_size
        # 支持两种方式，一种是奇偶相邻排列，一种是上下排列, 目前只在chatglm中看到updown排列
        assert rope_rank in {'adjacent', 'updown'}, "rank kwarg only support 'adjacent' and 'updown' "
        self.rope_rank = rope_rank

    def initialize(self, max_position: int):
        position_embeddings = get_sinusoid_encoding_table(max_position, self.embedding_size)  # [seq_len, hdsz]
        if self.rope_rank == 'adjacent':
            cos_position = position_embeddings[:, 1::2].repeat_interleave(2, dim=-1)  # [seq_len, hdsz]
            sin_position = position_embeddings[:, ::2].repeat_interleave(2, dim=-1)  # [seq_len, hdsz]
        elif self.rope_rank == 'updown':  # 目前仅chatglm使用
            cos_position = position_embeddings[:, 1::2].repeat(1, 2)  # [seq_len, hdsz]
            sin_position = position_embeddings[:, ::2].repeat(1, 2)  # [seq_len, hdsz]
        else:
            raise ValueError('Args `rope_rank` only support `adjacent` and `adjacent` mode')
        return cos_position, sin_position

    def forward(self, qw, position_ids=None, seq_dim=-2):
        # MultiHeadAttentionLayer中qw是[btz, n_heads, seq_len, head_size]
        # GlobalPointer中*转置*后qw是[btz, n_heads, seq_len, head_size]
        # EfficientGlobalPointer中qw是[btz, seq_len, head_size]
        if self.rope_rank == 'adjacent':
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], dim=-1).reshape_as(qw)
        else:  # 目前仅chatglm使用
            qw2 = torch.cat(
                [-qw[..., qw.shape[-1] // 2:], qw[..., :qw.shape[-1] // 2]], dim=-1
            )  # cat和stack+reshape是结果不同的

        # 超过缓存长度
        seq_len = position_ids.max() + 1 if position_ids is not None else qw.shape[seq_dim]
        if seq_len > self.max_seq_len_cache:
            cos_position, sin_position = self.initialize(seq_len)
            self.cos_position, self.sin_position = cos_position.type_as(qw).to(qw.device), sin_position.type_as(qw).to(
                qw.device)
            self.max_seq_len_cache = seq_len

        # 传入position_ids来获取cos和sin, 主要是在use_cache时候能直接取到对应位置的编码
        if position_ids is not None:
            cos = F.embedding(position_ids, self.cos_position)
            sin = F.embedding(position_ids, self.sin_position)
            return qw * cos + qw2 * sin

        return qw * self.cos_position[:seq_len] + qw2 * self.sin_position[:seq_len]


class EfficientGlobalPointer(nn.Module):
    """更加参数高效的GlobalPointer
    参考：https://kexue.fm/archives/8877
    这里实现和GlobalPointer相似，而未采用原版的奇偶位来取qw和kw，个人理解两种方式是无区别的
    """

    def __init__(self, hidden_size, heads, head_size, use_rope=True, use_bias=True, tril_mask=True):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.use_rope = use_rope
        self.tril_mask = tril_mask

        self.p_dense = nn.Linear(hidden_size, head_size * 2, bias=use_bias)
        self.q_dense = nn.Linear(head_size * 2, heads * 2, bias=use_bias)
        if self.use_rope:
            self.position_embedding = RoPEPositionEncoding(head_size)

    def forward(self, inputs, mask=None):
        """
        :param inputs: shape=[..., hdsz]
        :param mask: shape=[btz, seq_len], padding部分为0
        """
        sequence_output = self.p_dense(inputs)  # [..., head_size*2]
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]  # [..., head_size]

        # ROPE编码
        if self.use_rope:
            qw = self.position_embedding(qw)
            kw = self.position_embedding(kw)

        # 计算内积
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.head_size ** 0.5  # [btz, seq_len, seq_len], 是否是实体的打分
        bias_input = self.q_dense(sequence_output)  # [..., heads*2]
        bias = torch.stack(
            torch.chunk(bias_input, self.heads, dim=-1), dim=-2
        ).transpose(1, 2) / 2  # [btz, heads, seq_len, 2]
        logits = logits.unsqueeze(1) + bias[..., :1] + bias[..., 1:].transpose(2, 3)  # [btz, heads, seq_len, seq_len]

        # 排除padding
        if mask is not None:
            attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3)  # [btz, 1, seq_len, 1]
            attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2)  # [btz, 1, 1, seq_len]
            logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))

        # 排除下三角
        if self.tril_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        return logits


class SparseMultilabelCategoricalCrossentropy(nn.Module):
    """稀疏版多标签分类的交叉熵；
       请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax，预测阶段则输出y_pred大于0的类；
       详情请看：https://kexue.fm/archives/7359，https://kexue.fm/archives/8888
    """

    def __init__(self, mask_zero=False, epsilon=1e-7):
        super().__init__()
        self.mask_zero = mask_zero
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        """
        y_true: shape=[..., num_positive]
        y_pred: shape=[..., num_classes]
        """
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred = torch.cat([y_pred, zeros], dim=-1)

        if self.mask_zero:
            infs = zeros + float('inf')
            y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)

        y_pos_2 = torch.gather(y_pred, dim=-1, index=y_true)  # [..., num_positive]
        y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)  # [..., num_positive+1]

        if self.mask_zero:
            y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
            y_pos_2 = torch.gather(y_pred, dim=-1, index=y_true)

        pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
        all_loss = torch.logsumexp(y_pred, dim=-1)  # a
        aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss  # b-a
        aux_loss = torch.clamp(1 - torch.exp(aux_loss), self.epsilon, 1)  # 1-exp(b-a)
        neg_loss = all_loss + torch.log(aux_loss)  # a + log[1-exp(b-a)]

        return pos_loss + neg_loss
