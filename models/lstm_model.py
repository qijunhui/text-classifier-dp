from torch import nn


class Model(nn.Module):
    def __init__(self, vocab_size, outputs_size):
        super(Model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.activation = nn.LeakyReLU()
        self.predict = nn.Linear(64 * 2, outputs_size)

    def lstm_forward(self, inputs):  # torch.Size([batch, seq_len, f_dim])
        outputs, (h_n, c_n) = self.lstm(inputs)  # torch.Size([batch, seq_len, hidden_size * directions])
        outputs = outputs.permute(0, 2, 1)  # torch.Size([batch, hidden_size * directions, seq_len])
        outputs = self.avg_pool(outputs)  # torch.Size([batch, hidden_size * directions, 1])
        outputs = outputs.squeeze(dim=2)  # torch.Size([batch, hidden_size * directions])
        outputs = self.activation(outputs)  # torch.Size([batch, hidden_size * directions])
        return outputs

    def forward(self, inputs):  # torch.Size([batch, seq_len])
        outputs = self.embeddings(inputs)  # torch.Size([batch, seq_len, emb_dim])

        outputs = self.dropout(outputs)  # torch.Size([batch, seq_len, emb_dim])
        outputs = self.lstm_forward(outputs)  # torch.Size([batch, hidden_size * directions])

        outputs = self.dropout(outputs)  # torch.Size([batch, hidden_size * directions])
        outputs = self.predict(outputs)  # torch.Size([batch, outputs_size])
        return outputs
