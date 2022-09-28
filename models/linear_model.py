from torch import nn

from configs import CONFIG


class Model(nn.Module):
    def __init__(self, vocab_size, outputs_size):
        super(Model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.linear = nn.Linear(CONFIG["fix_length"] * 128, 128)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.LeakyReLU()
        self.predict = nn.Linear(128, outputs_size)

    def linear_forward(self, inputs):  # torch.Size([batch, f_dim])
        outputs = self.linear(inputs)  # torch.Size([batch, l_dim])
        outputs = self.activation(outputs)  # torch.Size([batch, l_dim])
        return outputs

    def forward(self, inputs):  # torch.Size([batch, seq_len])
        outputs = self.embeddings(inputs)  # torch.Size([batch, seq_len, emb_dim])
        outputs = outputs.reshape(outputs.shape[0], -1)  # torch.Size([batch, seq_len * emb_dim])

        outputs = self.dropout(outputs)  # torch.Size([batch, seq_len * emb_dim])
        outputs = self.linear_forward(outputs)  # torch.Size([batch, l_dim])

        outputs = self.dropout(outputs)  # torch.Size([batch, l_dim])
        outputs = self.predict(outputs)  # torch.Size([batch, outputs_size])
        return outputs
