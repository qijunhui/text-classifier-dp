import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, inputs):
        return self.dropout(inputs + self.pe[:, : inputs.size(1)])


class Model(nn.Module):
    def __init__(self, vocab_size, outputs_size):
        super(Model, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.positional_encoding = PositionalEncoding(128, 0.2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.dropout = nn.Dropout(0.2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.activation = nn.LeakyReLU()
        self.predict = nn.Linear(128, outputs_size)

    def transformer_encoder_forward(self, inputs):  # torch.Size([batch, seq_len, f_dim])
        outputs = self.transformer_encoder(inputs.permute(1, 0, 2)).permute(
            1, 0, 2
        )  # torch.Size([batch, seq_len, f_dim])
        outputs = outputs.permute(0, 2, 1)  # torch.Size([batch, f_dim, seq_len])
        outputs = self.avg_pool(outputs)  # torch.Size([batch, f_dim, 1])
        outputs = outputs.squeeze(dim=2)  # torch.Size([batch, f_dim])
        outputs = self.activation(outputs)  # torch.Size([batch, f_dim])
        return outputs

    def forward(self, inputs):  # torch.Size([batch, seq_len])
        outputs = self.positional_encoding(self.token_embeddings(inputs))  # torch.Size([batch, seq_len, emb_dim])

        outputs = self.dropout(outputs)  # torch.Size([batch, seq_len, emb_dim])
        outputs = self.transformer_encoder_forward(outputs)  # torch.Size([batch, emb_dim])

        outputs = self.dropout(outputs)  # torch.Size([batch, emb_dim])
        outputs = self.predict(outputs)  # torch.Size([batch, outputs_size])
        return outputs
