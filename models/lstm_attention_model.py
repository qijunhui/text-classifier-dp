import math

import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, vocab_size, outputs_size):
        super(Model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.activation = nn.LeakyReLU()
        self.predict = nn.Linear(64 * 2, outputs_size)

    def attention_forward(self, x, query):  # torch.Size([batch, seq_len, f_dim]) torch.Size([batch, seq_len, f_dim])
        d_k = query.shape[-1]
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  # torch.Size([batch, seq_len, seq_len])
        alpha_n = F.softmax(scores, dim=-1)  # torch.Size([batch, seq_len, seq_len])
        outputs = torch.matmul(alpha_n, x).sum(dim=1)  # torch.Size([batch, f_dim])
        return outputs, alpha_n

    def lstm_forward(self, inputs):  # torch.Size([batch, seq_len, f_dim])
        outputs, (h_n, c_n) = self.lstm(inputs)  # torch.Size([batch, seq_len, hidden_size * directions])
        query = self.dropout(outputs)  # torch.Size([batch, seq_len, hidden_size * directions])
        outputs, _ = self.attention_forward(outputs, query)  # torch.Size([batch, seq_len, hidden_size * directions])
        return outputs

    def forward(self, inputs):  # torch.Size([batch, seq_len])
        outputs = self.embeddings(inputs)  # torch.Size([batch, seq_len, emb_dim])

        outputs = self.dropout(outputs)  # torch.Size([batch, seq_len, emb_dim])
        outputs = self.lstm_forward(outputs)  # torch.Size([batch, hidden_size * directions])

        outputs = self.dropout(outputs)  # torch.Size([batch, hidden_size * directions])
        outputs = self.predict(outputs)  # torch.Size([batch, outputs_size])
        return outputs
