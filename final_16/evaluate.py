import numpy as np
from tqdm import tqdm
import torch
from config import model_name
from torch.utils.data import Dataset, DataLoader
from os import path
import pandas as pd
from ast import literal_eval
import importlib


Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
config = getattr(importlib.import_module('config'), f"{model_name}Config")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NewsDataset(Dataset):
    def __init__(self, news_path):
        super(NewsDataset, self).__init__()
        self.news_parsed = pd.read_table(
            news_path,
            usecols=['news_id'] + config.dataset_attributes['news'],
            converters={
                attribute: literal_eval
                for attribute in set(config.dataset_attributes['news']) & set([
                    'title', 'abstract', 'title_entities', 'abstract_entities'
                ])
            })
        self.news2dict = self.news_parsed.to_dict('index')
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                if type(self.news2dict[key1][key2]) != str:
                    self.news2dict[key1][key2] = torch.tensor(
                        self.news2dict[key1][key2])

    def __len__(self):
        return len(self.news_parsed)

    def __getitem__(self, idx):
        item = self.news2dict[idx]
        return item


class UserDataset(Dataset):
    def __init__(self, behaviors_path, user2int_path):
        super(UserDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path,
                                       header=0,
                                       usecols=[1, 3],
                                       names=['user_id', 'clicked_news'])
        self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors.drop_duplicates(inplace=True)
        user2int = dict(pd.read_table(user2int_path).values.tolist())
        user_total = 0
        user_missed = 0
        for row in self.behaviors.itertuples():
            user_total += 1
            if row.user_id in user2int:
                self.behaviors.at[row.Index, 'user_id'] = user2int[row.user_id]
            else:
                user_missed += 1
                self.behaviors.at[row.Index, 'user_id'] = 0

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "user":
            row.user_id,
            "clicked_news_string":
            row.clicked_news,
            "clicked_news":
            row.clicked_news.split()[:config.num_clicked_news_a_user]
        }
        item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = config.num_clicked_news_a_user - len(
            item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = ['PADDED_NEWS'
                                ] * repeated_times + item["clicked_news"]

        return item


class BehaviorsDataset(Dataset):
    def __init__(self, behaviors_path):
        super(BehaviorsDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path,
                                       header=0,
                                       usecols=range(5),
                                       names=[
                                           'id', 'user_id', 'time',
                                           'clicked_news', 'impressions'
                                       ])
        self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors.impressions = self.behaviors.impressions.str.split()

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "id": row.id,
            "user_id": row.user_id,
            "time": row.time,
            "clicked_news_string": row.clicked_news,
            "impressions": row.impressions
        }
        return item


@torch.no_grad()
def evaluate(model, directory, num_workers):
    news_dataset = NewsDataset(path.join(directory, 'news_parsed.tsv'))
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=config.batch_size * 16,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 pin_memory=True)

    news2vector = {}
    for minibatch in tqdm(news_dataloader, desc="Calculating vectors for news"):
        news_ids = minibatch["news_id"]
        if any(id not in news2vector for id in news_ids):
            news_vector = model.get_news_vector(minibatch)
            for id, vector in zip(news_ids, news_vector):
                if id not in news2vector:
                    news2vector[id] = vector

    news2vector['PADDED_NEWS'] = torch.zeros(list(news2vector.values())[0].size())

    user_dataset = UserDataset(path.join(directory, 'behaviors.tsv'), '../data/train/user2int.tsv')
    user_dataloader = DataLoader(user_dataset,
                                 batch_size=config.batch_size * 16,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 pin_memory=True)

    user2vector = {}
    for minibatch in tqdm(user_dataloader, desc="Calculating vectors for users"):
        user_strings = minibatch["clicked_news_string"]
        if any(user_string not in user2vector for user_string in user_strings):
            clicked_news_vector = torch.stack([
                torch.stack([news2vector[x].to(device) for x in news_list], dim=0) for news_list in minibatch["clicked_news"]
            ], dim=0).transpose(0, 1)
            
            
            user_vector = model.get_user_vector(clicked_news_vector)
            for user, vector in zip(user_strings, user_vector):
                if user not in user2vector:
                    user2vector[user] = vector

    behaviors_dataset = BehaviorsDataset(path.join(directory, 'behaviors.tsv'))
    behaviors_dataloader = DataLoader(behaviors_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=config.num_workers)

    prediction_output = []

    for minibatch in tqdm(behaviors_dataloader, desc="Calculating probabilities"):
        candidate_news_vector = torch.stack([news2vector[news[0].split('-')[0]] for news in minibatch['impressions']], dim=0)
        user_vector = user2vector[minibatch['clicked_news_string'][0]]
        click_probability = model.get_prediction(candidate_news_vector, user_vector)

        y_pred = click_probability.tolist()
        prediction_output.append([minibatch['id'][0]] + y_pred)

    predictions_df = pd.DataFrame(prediction_output, columns=["id"] + [f"p{i}" for i in range(1, 16)])
    predictions_df['id'] = [i for i in range(len(predictions_df))]
    predictions_df.to_csv('../predictions.csv', index=False)
    print(f"Predictions saved to predictions.csv")

if __name__ == '__main__':
    print('Using device:', device)
    print(f'Evaluating model {model_name}')
    model = Model(config).to(device)
    from train import latest_checkpoint  # Avoid circular imports
    checkpoint_path = latest_checkpoint(path.join('../checkpoint', model_name))
    if checkpoint_path is None:
        print('No checkpoint file found!')
        exit()
    print(f"Load saved parameters in {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    evaluate(model, '../data/test', config.num_workers)
