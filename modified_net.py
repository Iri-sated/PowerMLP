import torch
import torch.nn as nn
from tqdm import tqdm
import math
import torch
from torchtext.datasets import CoLA
from torchtext.data import Field, BucketIterator
import torch.nn as nn
import torch.optim as optim
import random
import time

from powermlp import ResRePUBlock

class ModifiedVGG(nn.Module):
    def __init__(self, num_classes=10, mode='mlp'):
        super(ModifiedVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        if mode == 'mlp':
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(128 * 8 * 8, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, num_classes),
            )

        elif mode == 'power':
            self.classifier = nn.Sequential(
                nn.Dropout(),
                ResRePUBlock(input_dim=128 * 8 * 8, output_dim=512, repu_order=3),
                nn.Dropout(),
                ResRePUBlock(input_dim=512, output_dim=512, repu_order=3),
                nn.Linear(512, num_classes),
            )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return None
    
    def fit(self, criterion, optimizer, scheduler, train_loader, val_loader, max_iter=100, save_name=None):
        device = self.get_device()
        tmp_acc = 0
        tmp_epoch = 0
        pbar = tqdm(range(max_iter), desc='description', ncols=100)
        epoch = 0

        for i_pbar in pbar:
            self.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                def closure():
                    optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=1.0)
                    return loss
            
                optimizer.step(closure)
            
            self.eval()
            with torch.no_grad():
                total_loss = 0
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self(inputs)

                    _, predicts = torch.max(outputs, dim=1)
                    acc = torch.sum(predicts==targets)
                    total_loss += acc
                    
            if math.isnan(total_loss):
                print(f'Early stop due to NaN at epoch {epoch}.')
                break

            acc = float(total_loss/len(val_loader.dataset)) * 100
            pbar.set_description("lr: %.4e | val acc: %.4f%% " % (optimizer.param_groups[0]['lr'], acc))
            if acc > tmp_acc:
                tmp_acc = acc
                tmp_epoch = epoch
                torch.save(self.state_dict(), save_name)

            scheduler.step()
            epoch += 1

        return tmp_acc, tmp_epoch
    
    def test(self, test_loader):
        device = self.get_device()
        self.eval()
        with torch.no_grad():
            total_v = 0
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)

                _, predicts = torch.max(outputs, dim=1)
                acc = torch.sum(predicts==targets)
                total_v += acc

            return total_v / len(test_loader.dataset) * 100
        

class ModifiedRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=256, output_dim=1, n_layers=2, bidirectional=True, dropout=0.5, mode='mlp'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout)
        if mode == 'mlp':
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, output_dim),
            )
        elif mode == 'power':
            self.fc = nn.Sequential(
                ResRePUBlock(hidden_dim * 2, hidden_dim, repu_order=3),
                nn.Linear(hidden_dim, output_dim),
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        text = text.permute(1, 0)
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden.squeeze(0))
    
    def fit(self, criterion, optimizer, scheduler, train_loader, val_loader, max_iter=100, save_name=None):
        device = self.get_device()
        criterion = criterion.to(device)
        tmp_acc = 0
        tmp_epoch = 0
        pbar = tqdm(range(max_iter), desc='description', ncols=100)
        epoch = 0

        for i_pbar in pbar:
            epoch_loss = 0
            self.train()
            for labels, texts in train_loader:
                optimizer.zero_grad()
                outputs = self(texts)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            
            self.eval()
            with torch.no_grad():
                total_acc = 0
                num = 0
                for labels, texts in val_loader:
                    texts, labels = texts.to(device), labels.to(device)
                    outputs = self(texts.t())
                    predictions = outputs.argmax(dim=1)
                    total_acc += (predictions == labels).sum().item()
                    
            if math.isnan(total_acc):
                print(f'Early stop due to NaN at epoch {epoch}.')
                break
            
            acc = float(total_acc.item() / len(val_loader.dataset)) * 100
            #acc = float(total_acc.item()/num.item()) * 100
            pbar.set_description("lr: %.4e | val acc: %.4f%% " % (optimizer.param_groups[0]['lr'], acc))
            if acc > tmp_acc:
                tmp_acc = acc
                tmp_epoch = epoch
                torch.save(self.state_dict(), save_name)

            scheduler.step()
            epoch += 1

        return tmp_acc, tmp_epoch
    
    def test(self, test_loader):
        device = self.get_device()
        self.eval()
        with torch.no_grad():
            total_acc = 0
            num = 0
            for labels, texts in test_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = self(texts.t())
                predictions = outputs.argmax(dim=1)
                total_acc += (predictions == labels).sum().item()
            
        return total_acc / len(test_loader.dataset)
