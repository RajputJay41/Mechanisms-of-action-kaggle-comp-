import torch
import torch.nn as nn

class MoaDataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self,item):
        return{
            "x": torch.tensor(self.features[item, :], dtype=torch.float),
            "y": torch.tensor(self.targets[item, :], dtype=torch.float),
        }


class Engine:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.device = device
        self.optimizer = optimizer


    @staticmethod
    def loss_fn(targets, outputs):
        return nn.BCEWithLogitsLoss()(outputs, targets)

    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for data in data_loader:
            self.optimizer.zero_grad()
            inputs = data["x"].to(self.device)
            targets = data["y"].to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(targets, outputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader)

    def evaluate(self, data_loader):
        self.model.eval()
        final_loss = 0
        for data in data_loader:
            inputs = data["x"].to(self.device)
            targets = data["y"].to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(targets, outputs)
            final_loss += loss.item()
        return final_loss / len(data_loader)


class Model(nn.Module):
    def __init__(self, nfeatures, ntargets, hidden_size, nlayers, dropout):
        super().__init__()
        layers = []
        for _ in range(nlayers):
            if len(layers) == 0:
                layers.append(nn.Linear(nfeatures, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, ntargets))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
        
