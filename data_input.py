from data.ml_data import get_ml_dataset
from data.cv_data import get_cv_dataset
from data.nlp_data import get_nlp_dataset
from data.func_data import create_dataset, KANDataset
from sklearn.model_selection import train_test_split
import numpy as np


def get_dataset(name, f=None, n_var=None, root=None, seed=42):
    if name in ['titanic','wine','income']:
        dataset = get_ml_dataset(dataset_name=name, root=root)

        random_state = np.random.RandomState(seed)
        train_set, val_test_set = train_test_split(dataset, train_size=.8, random_state=random_state)
        val_set, test_set = train_test_split(val_test_set,test_size=.5, random_state=random_state)
        return train_set, val_set, test_set
    
    elif name in ['mnist','svhn','cifar10']:
        train_set, test_set = get_cv_dataset(dataset_name=name, root=root)
        
        random_state = np.random.RandomState(seed)
        val_set, test_set = train_test_split(test_set, train_size=.5, random_state=random_state)
        return train_set, val_set, test_set
    
    elif name in ['cola','ag_news','spam']:
        train_set, test_set = get_nlp_dataset(dataset_name=name, root=root)
        
        random_state = np.random.RandomState(seed)
        if name == 'spam':
            train_set, test_set = train_test_split(train_set, train_size=.9, random_state=random_state)

        train_set, val_set = train_test_split(train_set, train_size=.9, random_state=random_state)
        return train_set, val_set, test_set
    
    elif name == 'function':
        assert f is not None and n_var is not None, 'Need to input a function'
        dataset = create_dataset(f, n_var=n_var, train_num=50000, seed=seed)
        train_set = KANDataset(dataset['train_input'], dataset['train_label'])
        test_set = KANDataset(dataset['test_input'], dataset['test_label'])

        train_set.process()
        test_set.process()
        random_state = np.random.RandomState(seed)
        train_set, val_set = train_test_split(train_set, test_size=.1, random_state=random_state)

        return train_set, val_set, test_set
    
    else:
        raise ValueError