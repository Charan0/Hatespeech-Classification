import torch
import torch.nn as nn


class SentimentModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, output_dim: int, rate: float, pad_idx: int):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=pad_idx)
        self.neural_net = nn.RNN(emb_dim, hidden_dim, num_layers=1, bidirectional=False)
        self.fully_connected = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, text):
        # text = [sent len, batch size]
        embedded = self.embedding(text)
        print(f"Embeddings: {embedded.shape}")
        # embedded = [sent len, batch size, emb dim
        # print(self.neural_net(embedded))
        # print(self.neural_net(embedded))
        output, hidden = self.neural_net(embedded)
        print(f"Output: {output.shape} Hidden: {hidden.shape}")
        predictions = self.fully_connected(hidden.squeeze()).squeeze()
        print(f"Prediction from model: {predictions.shape}")

        return predictions
        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]


model = SentimentModel(10, 10, 2, 1, 0.6, 0)
