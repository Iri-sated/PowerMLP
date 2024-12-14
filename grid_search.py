import torch
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.optim import Adam
import csv
import os

from LBFGS import LBFGS
from utils import set_seed
from powermlp import PowerMLP
import time

class BaseSearcher(object):
    def __init__(self, seed=42, device=torch.device('cpu'), task='reg', save_dir=None) -> None:
        # System Info
        self.seed = seed
        set_seed(seed)
        self.device = device

        # Task Info
        self.task = task
        assert task == 'reg' or task == 'clf', 'Unknown task type!'
        self.save_dir = 'tmp' if save_dir is None else save_dir
        self.criterion = MSELoss() if task == 'reg' else CrossEntropyLoss()

    def init_logs(self):
        if self.task == 'reg':
            data = ['best_val_loss', 'epoch', 'save_name', 'params', 'train_time', 'test_loss']

        elif self.task == 'clf':
            data = ['best_val_acc', 'epoch', 'save_name', 'params', 'train_time', 'test_acc']

        os.makedirs(self.save_dir, exist_ok=True)
        with open(f'{self.save_dir}result.csv','a') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        print(f'Write the result in {self.save_dir}result.csv')

    def fit_data(self, model, save_name, optimizer, scheduler, train_loader, val_loader, test_loader, max_iter):
        params = model.count_parameters()
        model.init_weights()

        stime = time.time()
        loss, epoch = model.fit(self.criterion, optimizer, scheduler, train_loader, val_loader, task=self.task, save_name=save_name, max_iter=max_iter)
        train_time = time.time() - stime
        
        try:
            model.load_state_dict(torch.load(save_name))
            test_result = model.test(test_loader, task=self.task)
        except FileNotFoundError:
            test_result = -1
            
        loss = loss.item() if isinstance(loss, torch.Tensor) else loss
        test_result = test_result.item() if isinstance(test_result, torch.Tensor) else test_result   
        data = [loss, epoch, save_name, params, train_time, test_result]

        with open(f'{self.save_dir}result.csv','a') as file:
            writer = csv.writer(file)
            writer.writerow(data)

        return data
    
    def get_optimizer(self, model, lr, optim=None):
        if optim != 'lbfgs' and optim != 'adam':
            optim = 'lbfgs' if self.task == 'reg' else 'adam'
        if optim == 'lbfgs':
            return LBFGS(model.parameters(), lr=lr, max_iter=10, history_size=10, line_search_fn='strong_wolfe')
        elif optim == 'adam':
            return Adam(model.parameters(), lr=lr)

    def get_scheduler(self, optimizer, epoch_list=[10,20,100], scheduler='lam'):
        if scheduler == 'lam':
            def lr_lambda(epoch):
                if epoch < epoch_list[0]:
                    lr = (epoch+1)/10
                elif epoch < epoch_list[1]:
                    lr = 1.0
                else:
                    lr = 1.0 * (0.1 ** ((epoch-epoch_list[1]) / epoch_list[2]))
                return lr

            return LambdaLR(optimizer, lr_lambda)
        
        elif scheduler == 'cos':
            return CosineAnnealingLR(optimizer, T_max=epoch_list[2])
            

    def grid_search(self, size_list, lr_list, train_loader, val_loader, test_loader, repu_order=3, res=True, optim=None,max_iter=100,epoch_list=[10,20,100],scheduler='lam'):
        for size in size_list:
            for lr in lr_list:
                save_name = self.save_dir + f"lr_{str(lr).replace('.', '_')}_shape_"
                for s in size:
                    save_name += str(s)
                    save_name += '_'
                save_name += 'best.pt'

                model = PowerMLP(size, repu_order=repu_order, res=res).to(self.device)
                model.init_weights()
                model.zero_grad()
                optimizer = self.get_optimizer(model, lr, optim)
                lr_scheduler = self.get_scheduler(optimizer, epoch_list=epoch_list, scheduler=scheduler)

                data = self.fit_data(model, save_name, optimizer, lr_scheduler, train_loader, val_loader, test_loader, max_iter)
                print(data)