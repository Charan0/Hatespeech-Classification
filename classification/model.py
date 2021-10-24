import torch
import torch.nn as nn


class SentimentModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, output_dim: int, n_layers: int,
                 bidirectional: bool, rate: float, pad_idx: int):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=pad_idx)
        self.neural_net = nn.LSTM(emb_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=rate)
        self.fully_connected = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout_layer = nn.Dropout(rate)

    def forward(self, text, text_len):
        # text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))
        # embedded = [sent len, batch size, emb dim]
        # pack sequence
        # lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_len.to('cpu'))

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        hidden = self.dropout_layer(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]

        return self.fully_connected(hidden)
