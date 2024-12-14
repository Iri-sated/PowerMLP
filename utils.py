import torch
import torch.nn as nn
import numpy as np
import random

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true))
        return loss


def check_and_convert_to_int(lst):
    try:
        converted_list = [int(x) for x in lst]
        return converted_list
    except ValueError:
        raise ValueError("Dimension list contains elements that cannot be converted to int")
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as e:
            print(f'Warning: {e}')
    torch.use_deterministic_algorithms(True)