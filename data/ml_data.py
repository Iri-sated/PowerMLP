from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine, fetch_openml
from sklearn.model_selection import train_test_split
import torch
from ucimlrepo import fetch_ucirepo
import os


_ML_DATASETS = {}


def sparse_to_tensor(sparse_matrix):
    sparse_matrix = sparse_matrix.tocoo()
    row = torch.tensor(sparse_matrix.row, dtype=torch.long)
    col = torch.tensor(sparse_matrix.col, dtype=torch.long)
    values = torch.tensor(sparse_matrix.data, dtype=torch.float32)
    indices = torch.stack([row, col])

    sparse_tensor = torch.sparse_coo_tensor(indices, values, sparse_matrix.shape)
    tensor = sparse_tensor.to_dense()
    return tensor


def _add_dataset(dataset_fn):
    _ML_DATASETS[dataset_fn.__name__] = dataset_fn
    return dataset_fn


class MLDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = [feat if isinstance(feat, torch.Tensor) else torch.tensor(feat,dtype=torch.float32) for feat in features]
        self.labels = [lab if isinstance(lab, torch.Tensor) else torch.tensor(lab,dtype=torch.long) for lab in labels]
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        #sample = {'features': self.features[idx], 'label': self.labels[idx]}
        
        #if self.transform:
        #    sample = self.transform(sample)
        
        #return sample
        return self.features[idx].squeeze(), self.labels[idx].squeeze()
    
    
@_add_dataset
def titanic(root):
    raw_data = pd.read_csv(f'{root}/Titanic.csv')
    raw_data = raw_data.drop(['Name', 'PassengerId'], axis=1)

    category_feature = ['Sex','Ticket','Cabin','Embarked']
    category_feature_map = [{} for cat in category_feature]

    def encode(encoder, x):
        len_encoder = len(encoder)
        try:
            id = encoder[x]
        except KeyError:
            id = len_encoder
        return id

    for i, cat in enumerate(category_feature):
        category_feature_map[i] = {l: id for id, l in enumerate(raw_data.loc[:, cat].astype(str).unique())}
        raw_data[cat] = raw_data[cat].astype(str).apply(lambda x: encode(category_feature_map[i], x))

    numerical_feature = list(set(raw_data.columns) - set(category_feature+["Survived"]))
    #print(numerical_feature)
    raw_data[numerical_feature] = raw_data[numerical_feature].fillna(0)

    scaler = StandardScaler()
    raw_data[numerical_feature] = scaler.fit_transform(raw_data[numerical_feature])

    x = torch.from_numpy(raw_data.drop(["Survived"], axis=1).values).float()
    y = torch.from_numpy(raw_data["Survived"].values)

    return MLDataset(x,y)


@_add_dataset
def wine(root):
    def process_x(df):
        # category columns
        float_cols = [col for col in df.columns if df[col].dtype == 'float64']
        int_cols = [col for col in df.columns if df[col].dtype == 'int64']
        str_cols = [col for col in df.columns if df[col].dtype == 'object']
        other_cols = [col for col in df.columns if col not in float_cols + int_cols + str_cols]

        # drop columns with other types
        df = df.drop(other_cols, axis=1)

        # transform str_cols to int_cols
        for col in str_cols:
            df[col] = df[col].str.lower() 
            df[col] = df[col].str.strip('!"#%&\'()*,./:;?@[\\]^_`{|}~' + ' \n\r\t') 

        category_feature_map = [{} for cat in str_cols]

        def encode(encoder, x):
            len_encoder = len(encoder)
            try:
                id = encoder[x]
            except KeyError:
                id = len_encoder
            return id

        for i, cat in enumerate(str_cols):
            category_feature_map[i] = {l: id for id, l in enumerate(df.loc[:, cat].astype(str).unique())}
            df[cat] = df[cat].astype(str).apply(lambda x: encode(category_feature_map[i], x))

        # transform int_cols to float_cols
        for col in int_cols+str_cols:
            df[col] = df[col].astype(float)

        # fill missing values
        df[float_cols+int_cols+str_cols] = df[float_cols+int_cols+str_cols].fillna(0)

        # normalize float_cols
        scaler = StandardScaler()
        df[float_cols+int_cols+str_cols] = scaler.fit_transform(df[float_cols+int_cols+str_cols])

        # print(df.head())

        return df.values

    def process_y(df):
        assert len(df.columns) == 1
        assert df.dtypes[0] in ['int64', 'object']

        # transform str_cols to int_cols
        if df.dtypes[0] == 'object':
            df[df.columns[0]] = df[df.columns[0]].str.lower() 
            df[df.columns[0]] = df[df.columns[0]].str.strip('!"#%&\'()*,./:;?@[\\]^_`{|}~' + ' \n\r\t') 

        def encode(encoder, x):
            len_encoder = len(encoder)
            try:
                id = encoder[x]
            except KeyError:
                id = len_encoder
            return id

        feature_map = {l: id for id, l in enumerate(df.loc[:, df.columns[0]].astype(str).unique())}
        df[df.columns[0]] = df[df.columns[0]].astype(str).apply(lambda x: encode(feature_map, x))

        # print(df.head())

        return df.values
    
    dataset_path = os.path.join(root, f'wine.csv')
    if not os.path.exists(dataset_path):
        ucimldataset = fetch_ucirepo(id=186)
        df = pd.DataFrame(ucimldataset.data.features, columns=ucimldataset.feature_names)
        df['quality'] = ucimldataset.data.targets
        df.to_csv(dataset_path, index=False)
    else:
        df = pd.read_csv(dataset_path)

    # data (as pandas dataframes) 
    x = df.drop(['quality'], axis=1)
    y = df[['quality']]

    x = process_x(x)
    y = process_y(y)
    ## input_size=11
    return MLDataset(x, y)


@_add_dataset
def income(root):
    income = fetch_openml(data_id=1590, as_frame=True, data_home=root)
    income_df = income.frame
    categorical_features = income_df.select_dtypes(include=['category']).columns.difference(['class'])
    numerical_features = income_df.select_dtypes(exclude=['category']).columns.difference(['class'])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )
    features = preprocessor.fit_transform(income_df.drop('class', axis=1))
    labels = (income_df['class'] == '>50K').astype(int).values
    
    features = [sparse_to_tensor(feat) for feat in features]
    return MLDataset(features, labels)


def get_ml_dataset(dataset_name, root):
    if root is None:
        root = 'data'
    return _ML_DATASETS[dataset_name](root)