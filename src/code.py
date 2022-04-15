# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: 'Python 3.6.9 (''torch'': conda)'
#     language: python
#     name: python3
# ---

# %%
from tqdm.auto import tqdm
import os
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# %%
# config
DATA_PATH = "../example_data"
GLOVE_PATH = "/home/yflyl/glove.840B.300d.txt"
MODEL_DIR="../../../model_all"
NUM_CITY = 100
NUM_TOPIC = 100
CITY_EMB_DIM = 50
TOPIC_EMB_DIM = 50
USER_ID_EMB_DIM = 50
GROUP_ID_EMB_DIM = 50
DESC_DIM = 100
USER_EMB_DIM = 256
GROUP_EMB_DIM = 256
WORD_EMB_DIM = 300
NUM_GROUP_DESC = 200
NUM_EVENT_DESC = 200
DROP_RATIO = 0.1
BATCH_SIZE = 32
LR = 0.01

# %%
# get word dict and word embedding table
with open(os.path.join(DATA_PATH, 'word.tsv'), 'r', encoding='utf-8') as f:
    word_file = f.readlines()[1:]

word_dict = {}
for line in tqdm(word_file):
    idx, word = line.strip('\n').split('\t')
    word_dict[word] = int(idx)

word_embedding = np.zeros(shape=(len(word_dict) + 1, WORD_EMB_DIM))
have_word = 0
with open(GLOVE_PATH, 'rb') as f:
    for line in tqdm(f):
        line = line.split()
        word = line[0].decode()
        if word in word_dict:
            idx = word_dict[word]
            tp = [float(x) for x in line[1:]]
            word_embedding[idx] = np.array(tp)
            have_word += 1
word_embedding = torch.from_numpy(word_embedding).float()

print(f'Word dict length: {len(word_dict)}')
print(f'Have words: {have_word}')
print(f'Missing rate: {(len(word_dict) - have_word) / len(word_dict)}')

# %%
# get user dict
with open(os.path.join(DATA_PATH, 'user.tsv'), 'r', encoding='utf-8') as f:
    user_file = f.readlines()[1:]

user_dict = {}
for line in tqdm(user_file):
    idx, topic_list, city = line.strip('\n').split('\t')
    user_dict[idx] = (eval(topic_list), int(city))

NUM_USER = len(user_dict)
print(f'Total user num: {NUM_USER}')

# %%
# get group dict
with open(os.path.join(DATA_PATH, 'group.tsv'), 'r', encoding='utf-8') as f:
    group_file = f.readlines()[1:]

group_dict = {}
for line in tqdm(group_file):
    idx, topic_list, city, desc = line.strip('\n').split('\t')
    group_dict[idx] = (eval(topic_list), int(city), eval(desc)[:NUM_GROUP_DESC])

NUM_GROUP = len(group_dict)
print(f'Total group num: {NUM_GROUP}')

# %%
# prepare data
total_time_period = len(os.listdir(os.path.join(DATA_PATH, f'behaviors')))
total_behavior, total_user_group, total_user_user = [], [], []

for idx in range(total_time_period):
    behavior_data = []
    with open(os.path.join(DATA_PATH, f'behaviors/{idx}.tsv'), 'r', encoding='utf-8') as f:
        behavior_file = f.readlines()[1:]
    for line in tqdm(behavior_file):
        city, desc, user, group, label = line.strip('\n').split('\t')
        behavior_data.append((int(city), eval(desc)[:NUM_EVENT_DESC], int(user), int(group), int(label)))
    total_behavior.append(behavior_data)
    with open(os.path.join(DATA_PATH, f'links/user-group/{idx}.json'), 'r') as f:
        user_group_data = json.load(f)
        total_user_group.append(torch.FloatTensor(user_group_data))
    with open(os.path.join(DATA_PATH, f'links/user-user/{idx}.json'), 'r') as f:
        user_user_data = json.load(f)
        total_user_user.append(torch.FloatTensor(user_user_data))

total_user_group = [torch.zeros((NUM_USER, NUM_GROUP))] + total_user_group[:-1]
total_user_user = [torch.zeros((NUM_USER, NUM_USER))] + total_user_user[:-1]

for idx in range(1, total_time_period):
    total_user_group[idx] += total_user_group[idx - 1]
    total_user_user[idx] += total_user_user[idx - 1]

print(f'Number of behaviors: {[len(data) for data in total_behavior]}')


# %%
class MyDataset(Dataset):
    def __init__(self, behavior):
        super().__init__()
        city, desc, user, group, label = [], [], [], [], []
        for t in behavior:
            city.append(t[0])
            desc.append(t[1])
            user.append(t[2])
            group.append(t[3])
            label.append(t[4])
        self.city = np.array(city)
        self.desc = np.array(desc)
        self.user = np.array(user)
        self.group = np.array(group)
        self.label = np.array(label)

    def __getitem__(self, idx):
        return (self.city[idx], self.desc[idx], self.user[idx], self.group[idx], self.label[idx])

    def __len__(self):
        return len(self.label)


# %%
def acc(y_true, y_hat):
    y_hat = (y_hat >= 0.5).float()
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot


# %%
# define model
class AttentionPooling(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super().__init__()
        self.att_fc1 = nn.Linear(emb_size, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: batch_size, candidate_size, emb_dim
            attn_mask: batch_size, candidate_size
        Returns:
            (shape) batch_size, emb_dim
        """
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)

        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)

        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha).squeeze(dim=-1)
        return x


class TextEncoder(nn.Module):
    def __init__(self, word_embedding):
        super().__init__()
        self.word_embedding = nn.Embedding.from_pretrained(word_embedding, freeze=False)
        self.attn = AttentionPooling(WORD_EMB_DIM, WORD_EMB_DIM // 2)
        self.dense = nn.Linear(WORD_EMB_DIM // 2, DESC_DIM)
        self.dropout = nn.Dropout(p=DROP_RATIO)

    def forward(self, text_ids):
        '''
            input_ids: *, num_words
            return: *, DESC_DIM
        '''
        text_attn_mask = text_ids.ne(0).float()
        word_vecs = self.dropout(self.word_embedding(text_ids.long()))
        text_vec = self.attn(word_vecs, text_attn_mask)
        text_vec = self.dense(text_vec)
        return text_vec


class TopicEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.topic_embedding = nn.Embedding(NUM_TOPIC, TOPIC_EMB_DIM)
        self.attn = AttentionPooling(TOPIC_EMB_DIM, TOPIC_EMB_DIM // 2)

    def forward(self, topic_ids):
        '''
            topic_ids: *, num_topics
            return: *, TOPIC_EMB_DIM
        '''
        topic_attn_mask = topic_ids.ne(0).float()
        topic_vec = self.attn(self.topic_embedding(topic_ids), topic_attn_mask)
        return topic_vec


class Model(nn.Module):
    def __init__(self, word_embedding):
        super().__init__()
        self.city_emb = nn.Embedding(NUM_CITY, CITY_EMB_DIM)
        self.user_id_emb = nn.Embedding(NUM_USER, USER_ID_EMB_DIM)
        self.group_id_emb = nn.Embedding(NUM_GROUP, GROUP_ID_EMB_DIM)
        self.user_emb_linear = nn.Linear(CITY_EMB_DIM + TOPIC_EMB_DIM + USER_ID_EMB_DIM, USER_EMB_DIM)
        self.group_emb_linear = nn.Linear(DESC_DIM + TOPIC_EMB_DIM + CITY_EMB_DIM + GROUP_ID_EMB_DIM, GROUP_EMB_DIM)
        prediction_dim = USER_EMB_DIM + DESC_DIM + GROUP_EMB_DIM
        self.prediction = nn.Sequential(
            nn.Linear(prediction_dim, prediction_dim // 2),
            nn.ReLU(),
            nn.Linear(prediction_dim // 2, 1),
            nn.Sigmoid()
        )
        self.text_encoder = TextEncoder(word_embedding)
        self.topic_encoder = TopicEncoder()
        self.loss_fn = nn.BCELoss()

    def update_edge_weight(self, user_group_data, user_user_data):
        self.user_group_matrix = user_group_data
        self.user_user_matrix = user_user_data

    def update_graph(self):
        #TODO: Implement GNN Algorithm
        user_emb = torch.randn((NUM_USER, USER_EMB_DIM)).to(device)
        group_emb = torch.randn((NUM_GROUP, GROUP_EMB_DIM)).to(device)
        return user_emb, group_emb

    def forward(self, event_city, event_desc, user_id, group_id, label):
        '''
            event_city: batch_size
            event_desc: batch_size, num_words
            user_id: batch_size
            group_id: batch_size
            label: batch_size
        '''
        user_emb, group_emb = self.update_graph()
        batch_city_emb = self.city_emb(event_city)
        batch_desc_emb = self.text_encoder(event_desc)
        batch_user_emb = torch.index_select(user_emb, dim=0, index=user_id)
        batch_group_emb = torch.index_select(group_emb, dim=0, index=group_id)
        predict_input = torch.cat([batch_user_emb, batch_city_emb, batch_desc_emb, batch_group_emb], dim=-1)
        score = self.prediction(predict_input).squeeze(dim=-1)
        loss = self.loss_fn(score, label)
        return score, loss


# %%
device = torch.device('cuda:0')
model = Model(word_embedding).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# %%
def run(period_idx, mode):
    assert mode in ('train', 'val', 'test')
    dataset = MyDataset(total_behavior[period_idx])

    if mode == 'train':
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        model.train()
        model.update_edge_weight(total_user_group[period_idx].to(device, non_blocking=True),
                                 total_user_user[period_idx].to(device, non_blocking=True))
        total_acc, total_loss = 0, 0
        for step, (event_city, event_desc, user_id, group_id, label) in enumerate(tqdm(dataloader)):
            event_city = event_city.to(device, non_blocking=True)
            event_desc = event_desc.to(device, non_blocking=True)
            user_id = user_id.to(device, non_blocking=True)
            group_id = group_id.to(device, non_blocking=True)
            label = label.float().to(device, non_blocking=True)

            y_hat, bz_loss = model(event_city, event_desc, user_id, group_id, label)
            total_acc += acc(label, y_hat)
            total_loss += bz_loss.data.float()
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f'Loss: {total_loss / step}, Acc: {total_acc / step}')

        ckpt_path = os.path.join(MODEL_DIR, f'{period_idx}.pt')
        torch.save(model.state_dict(), ckpt_path)
    else:
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        model.eval()
        torch.set_grad_enabled(False)
        model.update_edge_weight(total_user_group[period_idx].to(device, non_blocking=True),
                                 total_user_user[period_idx].to(device, non_blocking=True))
        pred, truth = [], []
        for step, (event_city, event_desc, user_id, group_id, label) in enumerate(tqdm(dataloader)):
            event_city = event_city.to(device, non_blocking=True)
            event_desc = event_desc.to(device, non_blocking=True)
            user_id = user_id.to(device, non_blocking=True)
            group_id = group_id.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            
            y_hat, _ = model(event_city, event_desc, user_id, group_id, label)
            pred.extend(y_hat.to('cpu').detach().numpy())
            truth.extent(label.to('cpu').detach().numpy())

        print(f'Acc: {acc(truth, pred)}')


# %%
run(0, 'train')

# %%
