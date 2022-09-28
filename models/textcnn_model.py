import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, vocab_size, outputs_size):
        super(Model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(128, 64, k) for k in (2, 3, 4)])  # 词向量维度 -> 新维度 k为卷积核大小
        self.dropout = nn.Dropout(0.2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.activation = nn.LeakyReLU()
        self.predict = nn.Linear(64 * 3, outputs_size)

    def conv_forward(self, inputs, conv):  # torch.Size([batch, seq_len, f_dim])
        outputs = inputs.permute(0, 2, 1)  # torch.Size([batch, f_dim, seq_len])
        outputs = conv(outputs)  # torch.Size([batch, hidden_size, seq_len - k + 1])
        outputs = self.avg_pool(outputs)  # torch.Size([batch, hidden_size, 1])
        outputs = outputs.squeeze(dim=2)  # torch.Size([batch, hidden_size])
        outputs = self.activation(outputs)  # torch.Size([batch, hidden_size])
        return outputs

    def forward(self, texts):  # torch.Size([batch, seq_len])
        outputs = self.embeddings(texts)  # torch.Size([batch, seq_len, emb_dim])

        outputs = self.dropout(outputs)  # torch.Size([batch, seq_len, emb_dim])
        outputs = torch.cat(
            [self.conv_forward(outputs, conv) for conv in self.convs], dim=1
        )  # torch.Size([batch, hidden_size * 3])

        outputs = self.dropout(outputs)  # torch.Size([batch, hidden_size * 3])
        outputs = self.predict(outputs)  # torch.Size([batch, outputs_size])
        return outputs
