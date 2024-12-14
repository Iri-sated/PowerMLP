import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import check_and_convert_to_int
from tqdm import tqdm
import math

class RePU(nn.Module):
    def __init__(self, n):
        super(RePU, self).__init__()
        self.n = n
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(x) ** self.n
    
    
class ResSiLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResSiLU, self).__init__()
        self.silu = nn.SiLU()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        #self.register_buffer('dim_trans', torch.ones([output_dim, input_dim]))

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.silu(x)
        out = self.fc(out)
        #out = F.linear(out, self.dim_trans)
        return out


class ResRePUBlock(nn.Module):
    def __init__(self, input_dim, output_dim, repu_order, res=True):
        super(ResRePUBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.repu = RePU(repu_order)
        if res:
            self.res = ResSiLU(input_dim, output_dim)

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)
        if hasattr(self, 'res'):
            self.res.init_weights()
    
    def forward(self, x):
        residual = x
        out = self.fc(x)
        out = self.repu(out)
        
        if hasattr(self, 'res'):
            residual = self.res(residual)
            out += residual
        return out

    
class PowerMLP(nn.Module):
    def __init__(self, dim_list, repu_order, res=True):
        super(PowerMLP, self).__init__()
        dim_list = check_and_convert_to_int(dim_list)
        assert len(dim_list) > 2, "Dimension list is too short to construct a RRN!"

        res_block_list = []
        for i, dim in enumerate(dim_list[:-2]):
            res_block = ResRePUBlock(dim_list[i], dim_list[i+1], repu_order, res=res)
            res_block_list.append(res_block)
        self.res_layers = nn.ModuleList(res_block_list)
            
        self.fc = nn.Linear(dim_list[-2], dim_list[-1])

    def init_weights(self):
        for res_layer in self.res_layers:
            res_layer.init_weights()
    
    def forward(self, x):
        for res_layer in self.res_layers:
            x = res_layer(x)
        x = self.fc(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return None
    
    def fit(self, criterion, optimizer, scheduler, train_loader, val_loader, task='reg', max_iter=100, save_name=None, mode='rmse'):
        device = self.get_device()
        tmp_loss = 1000000
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
                    if task == 'reg' and mode == 'rmse':
                        loss = torch.sqrt(loss)
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

                    if task == 'reg':
                        loss = criterion(outputs, targets)
                        total_loss += loss.item()
                    elif task == 'clf':
                        _, predicts = torch.max(outputs, dim=1)
                        acc = torch.sum(predicts==targets)
                        total_loss += acc
                    else:
                        raise ValueError
                    
            if math.isnan(total_loss):
                print(f'Early stop due to NaN at epoch {epoch}.')
                break

            if task == 'reg':
                total_loss = torch.sqrt(torch.tensor(total_loss)/len(val_loader)) if mode == 'rmse' else total_loss
                pbar.set_description("lr: %.4e | val loss: %.4e " % (optimizer.param_groups[0]['lr'], total_loss))
                if total_loss < tmp_loss:
                    tmp_loss = total_loss
                    tmp_epoch = epoch
                    torch.save(self.state_dict(), save_name)
                    
            elif task == 'clf':
                acc = float(total_loss/len(val_loader.dataset)) * 100
                pbar.set_description("lr: %.4e | val acc: %.4f%% " % (optimizer.param_groups[0]['lr'], acc))
                if acc > tmp_acc:
                    tmp_acc = acc
                    tmp_epoch = epoch
                    torch.save(self.state_dict(), save_name)
            
            else:
                raise ValueError
            
            #pbar.update()
            scheduler.step()
            epoch += 1
        
        if task == 'reg':
            return tmp_loss, tmp_epoch
        elif task == 'clf':
            return tmp_acc, tmp_epoch
    
    def test(self, test_loader, task = 'reg', mode='rmse'):
        device = self.get_device()
        self.train()
        with torch.no_grad():
            total_v = 0
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)

                if task == 'reg':
                    criterion = nn.MSELoss()
                    loss = criterion(outputs, targets)
                    total_v += loss.item()

                elif task == 'clf':
                    _, predicts = torch.max(outputs, dim=1)
                    acc = torch.sum(predicts==targets)
                    total_v += acc

            if task == 'reg':
                return torch.sqrt(torch.tensor(total_v) / len(test_loader))
            elif task == 'clf':
                return total_v / len(test_loader.dataset) * 100