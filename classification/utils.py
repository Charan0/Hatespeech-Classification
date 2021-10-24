import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.auto import tqdm


def multiclass_accuracy(predictions: torch.Tensor, labels: torch.Tensor):
    pred_labels = (predictions > 0.5).float()
    return (pred_labels == labels).float().mean()


def train_fn(network: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module,
             device: str = 'cpu'):
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
    avg_loss = 0.0
    avg_acc = 0.0

    for batch_idx, (samples, targets) in loop:
        samples = samples.to(device)
        targets = targets.to(device)

        predictions = network(samples)
        print(predictions.shape, samples.shape)
        loss = criterion(predictions, targets)
        accuracy = multiclass_accuracy(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_acc += accuracy.item()
        avg_loss += loss.item()

        loop.set_description(f'Step: [{batch_idx + 1}/{len(dataloader)}]')
        loop.set_postfix(loss=avg_loss / (batch_idx + 1), accuracy=avg_acc / (batch_idx + 1))

    return avg_loss, avg_acc
