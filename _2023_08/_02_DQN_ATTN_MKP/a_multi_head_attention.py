import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, f"d_model({d_model}) must be divisible by num_heads({num_heads})"

        self.d_model = d_model  # dimension of model
        self.num_heads = num_heads  # number of heads
        self.d_k = d_model // num_heads  # dimension of each head

        self.w_q = nn.Linear(d_model, d_model)  # weight for query
        self.w_k = nn.Linear(d_model, d_model)  # weight for key
        self.w_v = nn.Linear(d_model, d_model)  # weight for value
        self.w_o = nn.Linear(d_model, d_model)  # weight for output

    def forward(self, x):
        # x: (batch_size, seq_length, d_model)
        batch_size, seq_length, _ = x.size()

        # Compute query, key, value matrices
        # (batch_size, num_heads, seq_length, d_k)
        q = self.w_q(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

        # Calculate scaled dot-product attention
        # (batch_size, num_heads, seq_length, seq_length)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)

        # Multiply attention probabilities with value matrix
        # (batch_size, num_heads, seq_length, d_k)
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate and apply output linear layer
        # (batch_size, seq_length, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        output = self.w_o(attn_output)

        return output


def attention_test():
    # 사용 예시:
    d_model = 128
    num_heads = 8
    batch_size = 32
    seq_length = 50

    x = torch.rand(batch_size, seq_length, d_model)  # 임의의 입력 텐서
    multi_head_self_attention = MultiHeadSelfAttention(d_model, num_heads)
    output = multi_head_self_attention(x)
    print(x.shape)
    print(output.shape)  # 출력: torch.Size([32, 50, 128])


if __name__ == '__main__':
    attention_test()