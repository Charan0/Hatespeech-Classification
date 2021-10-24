from data.dataloader import get_data_and_vocab, CleanedDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from model import SentimentModel
from utils import train_fn
import torch.nn as nn

cleaned_vocab, destination = get_data_and_vocab("../data/train.csv", "cleaned.csv", ["Comment", "Insult"])
dataset = CleanedDataset("cleaned.csv", ["sentence", "label"], cleaned_vocab)
dataloader = DataLoader(dataset, batch_size=32)

model = SentimentModel(vocab_size=len(cleaned_vocab), emb_dim=100, hidden_dim=128, output_dim=1, rate=0.6,
                       pad_idx=cleaned_vocab["<pad>"])
optimizer = optim.Adam(model.parameters())
loss_fn = nn.BCEWithLogitsLoss()
train_fn(model, dataloader, optimizer, loss_fn, "cpu")