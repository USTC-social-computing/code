# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from tqdm.auto import tqdm
import os
import numpy as np
import torch
import torch.nn as nn
import pickle
import random
import math
import fnmatch
import wandb
import dgl
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from collections import Counter


# %%
# config
class Args():
    def __init__(self):
        self.USE_WANDB = True
        self.RANDOM_ORDER = True
        self.EXP_NAME = "GCN-2-layer-activation"
        self.DATA_PATH = "./data"
        self.GLOVE_PATH = "/data/yflyl/glove.840B.300d.txt"
        self.MODEL_DIR = f"../../model_all/{self.EXP_NAME}"
        self.CACHE_DIR = "/data/yflyl/CacheData"
        self.NUM_CITY = 2675
        self.NUM_TOPIC = 18115
        self.CITY_EMB_DIM = 64
        self.TOPIC_EMB_DIM = 64
        self.USER_ID_EMB_DIM = 64
        self.GROUP_ID_EMB_DIM = 64
        self.DESC_DIM = 128
        self.USER_EMB_DIM = 256
        self.GROUP_EMB_DIM = 256
        self.WORD_EMB_DIM = 300
        # The max number of words in descption. None happens if num_word < max.
        self.NUM_GROUP_DESC = 190
        self.NUM_EVENT_DESC = 280
        self.DROP_RATIO = 0.2
        self.BATCH_SIZE = 32
        self.LR = 0.005
        self.SAVE_STEP = 2000
        self.EPOCH = 3
        self.DECAY_RATE = 1
        self.WEIGHT_SMOOTHING_EXPONENT = 1
        self.NUM_GCN_LAYER = 2  # set to 0 to skip GCN
        self.COLD_USER_THRES = 2
        self.COLD_GROUP_THRES = 5


args = Args()
os.makedirs(args.MODEL_DIR, exist_ok=True)
os.makedirs(args.CACHE_DIR, exist_ok=True)

# %%
# get word dict and word embedding table
with open(os.path.join(args.DATA_PATH, 'word.tsv'), 'r',
          encoding='utf-8') as f:
    word_file = f.readlines()[1:]

word_dict = {}
for line in tqdm(word_file):
    idx, word = line.strip('\n').split('\t')
    word_dict[word] = int(idx)

word_embedding = np.random.uniform(size=(len(word_dict), args.WORD_EMB_DIM))
have_word = 0
try:
    with open(args.GLOVE_PATH, 'rb') as f:
        for line in tqdm(f):
            line = line.split()
            word = line[0].decode()
            if word in word_dict:
                idx = word_dict[word]
                tp = [float(x) for x in line[1:]]
                word_embedding[idx] = np.array(tp)
                have_word += 1
except FileNotFoundError:
    print('Warning: Glove file not found.')
word_embedding = torch.from_numpy(word_embedding).float()

print(f'Word dict length: {len(word_dict)}')
print(f'Have words: {have_word}')
print(f'Missing rate: {(len(word_dict) - have_word) / len(word_dict)}')

# %%
# get user dict
with open(os.path.join(args.DATA_PATH, 'user.tsv'), 'r',
          encoding='utf-8') as f:
    user_file = f.readlines()[1:]

user_dict = {}
for line in tqdm(user_file):
    idx, city, topic_list = line.strip('\n').split('\t')
    user_dict[int(idx)] = (eval(topic_list), int(city))

NUM_USER = len(user_dict)
print(f'Total user num: {NUM_USER}')

# %%
# get group dict
with open(os.path.join(args.DATA_PATH, 'group.tsv'), 'r',
          encoding='utf-8') as f:
    group_file = f.readlines()[1:]

group_dict = {}
for line in tqdm(group_file):
    idx, city, topic_list, desc = line.strip('\n').split('\t')
    group_dict[int(idx)] = (eval(topic_list), int(city),
                            eval(desc)[:args.NUM_GROUP_DESC])

NUM_GROUP = len(group_dict)
print(f'Total group num: {NUM_GROUP}')

# %%
# prepare data
TRAIN_NUM = 93

total_behavior_file = sorted(
    os.listdir(os.path.join(args.DATA_PATH, 'behaviours')))
total_user_group_file = sorted(
    os.listdir(os.path.join(args.DATA_PATH, 'links/user-group')))
total_user_user_file = sorted(
    os.listdir(os.path.join(args.DATA_PATH, 'links/user-user')))

total_time_period = len(total_behavior_file)

cache_file = os.path.join(args.CACHE_DIR, 'data.pkl')
if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    train_behavior = data['train_behavior']
    hot_test_behavior = data['hot_test_behavior']
    cold_test_behavior = data['cold_test_behavior']
    new_test_behavior = data['new_test_behavior']
    total_user_group = data['total_user_group']
    total_user_user = data['total_user_user']
    print(f'Loading cache from {cache_file}')
else:
    train_behavior = []
    hot_test_behavior, cold_test_behavior, new_test_behavior = [], [], []
    total_user_group, total_user_user = [], []
    train_user_id, train_group_id = Counter(), Counter()

    for idx, (behavior_file, user_group_file, user_user_file) in enumerate(
            tqdm(zip(total_behavior_file, total_user_group_file,
                     total_user_user_file),
                 total=total_time_period)):
        with open(os.path.join(args.DATA_PATH, f'behaviours/{behavior_file}'),
                  'r',
                  encoding='utf-8') as f:
            behavior_file = f.readlines()[1:]
        if idx < TRAIN_NUM:
            behavior_data = []
            for line in behavior_file:
                _, group, city, desc, user, label = line.strip('\n').split(
                    '\t')
                user, group = int(user), int(group)
                behavior_data.append(
                    (int(city), eval(desc)[:args.NUM_EVENT_DESC], user, group,
                     int(label)))
                train_user_id.update([user])
                train_group_id.update([group])
            random.seed(42)
            random.shuffle(behavior_data)
            train_behavior.append(behavior_data)
        else:
            for line in behavior_file:
                _, group, city, desc, user, label = line.strip('\n').split(
                    '\t')
                user, group = int(user), int(group)
                behavior_tuple = (int(city), eval(desc)[:args.NUM_EVENT_DESC],
                                  user, group, int(label))
                if user not in train_user_id or group not in train_group_id:
                    new_test_behavior.append(behavior_tuple)
                elif train_user_id[
                        user] <= args.COLD_USER_THRES or train_group_id[
                            group] <= args.COLD_GROUP_THRES:
                    cold_test_behavior.append(behavior_tuple)
                else:
                    hot_test_behavior.append(behavior_tuple)

        with open(
                os.path.join(args.DATA_PATH,
                             f'links/user-group/{user_group_file}'),
                'rb') as f:
            user_group_data = pickle.load(f)
            user_group_data = torch.sparse_coo_tensor(
                [user_group_data['row'], user_group_data['col']],
                user_group_data['data'], (NUM_USER, NUM_GROUP),
                dtype=torch.float32)
            total_user_group.append(user_group_data)

        with open(
                os.path.join(args.DATA_PATH,
                             f'links/user-user/{user_user_file}'), 'rb') as f:
            user_user_data = pickle.load(f)
            user_user_data = torch.sparse_coo_tensor(
                [user_user_data['row'], user_user_data['col']],
                user_user_data['data'], (NUM_USER, NUM_USER),
                dtype=torch.float32)
            total_user_user.append(user_user_data)

    with open(cache_file, 'wb') as f:
        pickle.dump(
            {
                'train_behavior': train_behavior,
                'hot_test_behavior': hot_test_behavior,
                'cold_test_behavior': cold_test_behavior,
                'new_test_behavior': new_test_behavior,
                'total_user_group': total_user_group,
                'total_user_user': total_user_user,
            }, f)
    print(f'Saving cache to {cache_file}')

train_behavior_num = sum([len(x) for x in train_behavior])
print(f'Number of training behaviors: {train_behavior_num}, \
{math.ceil(train_behavior_num // args.BATCH_SIZE)} steps')
print(f'Number of hot testing behaviors: {len(hot_test_behavior)}, \
{math.ceil(len(hot_test_behavior) // args.BATCH_SIZE)} steps')
print(f'Number of cold testing behaviors: {len(cold_test_behavior)}, \
{math.ceil(len(cold_test_behavior) // args.BATCH_SIZE)} steps')
print(f'Number of new testing behaviors: {len(new_test_behavior)}, \
{math.ceil(len(new_test_behavior) // args.BATCH_SIZE)} steps')

# %%
# prepare graph without self-loop
total_graph = [
    torch.sparse_coo_tensor(size=(NUM_USER + NUM_GROUP, NUM_USER + NUM_GROUP),
                            dtype=torch.float32)
]
for user_user, user_group in tqdm(zip(total_user_user[:-1],
                                      total_user_group[:-1]),
                                  total=total_time_period - 1):
    # combine the two graphs
    user_user_indices = user_user._indices()
    user_user_values = user_user._values()
    user_group_indices = user_group._indices()
    user_group_values = user_group._values()
    user_group_indices[1] += NUM_USER
    current_graph_indices = torch.cat(
        (
            user_user_indices,  # top left U-U
            user_group_indices,  # top right U-G
            user_group_indices[[1, 0]],  # bottom left G-U
        ),
        dim=1)
    current_graph_values = torch.cat(
        (
            user_user_values,  # top left U-U
            user_group_values,  # top right U-G
            user_group_values,  # bottom left G-U
        ),
        dim=0)
    current_graph = torch.sparse_coo_tensor(current_graph_indices,
                                            current_graph_values,
                                            size=(NUM_USER + NUM_GROUP,
                                                  NUM_USER + NUM_GROUP))
    current_graph = torch.pow(current_graph, args.WEIGHT_SMOOTHING_EXPONENT)
    total_graph.append(current_graph + total_graph[-1] * args.DECAY_RATE)

# add self_loop
self_loop = torch.sparse_coo_tensor(torch.arange(0,
                                                 NUM_USER + NUM_GROUP).expand(
                                                     2, -1),
                                    torch.ones(NUM_USER + NUM_GROUP),
                                    dtype=torch.float32)
for i in range(len(total_graph)):
    total_graph[i] = (total_graph[i] + self_loop).coalesce()

train_graph = total_graph[:TRAIN_NUM]
val_test_graph = total_graph[TRAIN_NUM]


# %%
class MyDataset(Dataset):
    def __init__(self, behavior):
        super().__init__()
        (city, desc, user, user_topic, user_city, group, group_topic,
         group_city, group_desc,
         label) = [], [], [], [], [], [], [], [], [], []
        for t in behavior:
            city.append(t[0])
            desc.append(t[1])
            user.append(t[2])
            user_topic.append(user_dict[t[2]][0])
            user_city.append(user_dict[t[2]][1])
            group.append(t[3])
            group_topic.append(group_dict[t[3]][0])
            group_city.append(group_dict[t[3]][1])
            group_desc.append(group_dict[t[3]][2])
            label.append(t[4])

        self.city = np.array(city)
        self.desc = np.array(desc)
        self.user = np.array(user)
        self.user_topic = np.array(user_topic)
        self.user_city = np.array(user_city)
        self.group = np.array(group)
        self.group_topic = np.array(group_topic)
        self.group_city = np.array(group_city)
        self.group_desc = np.array(group_desc)
        self.label = np.array(label)

    def __getitem__(self, idx):
        return (self.city[idx], self.desc[idx], self.user[idx],
                self.user_topic[idx], self.user_city[idx], self.group[idx],
                self.group_topic[idx], self.group_city[idx],
                self.group_desc[idx], self.label[idx])

    def __len__(self):
        return len(self.label)


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
        self.word_embedding = nn.Embedding.from_pretrained(word_embedding,
                                                           freeze=False)
        self.attn = AttentionPooling(args.WORD_EMB_DIM, args.WORD_EMB_DIM // 2)
        self.dense = nn.Linear(args.WORD_EMB_DIM, args.DESC_DIM)
        self.dropout = nn.Dropout(p=args.DROP_RATIO)

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
        self.topic_embedding = nn.Embedding(args.NUM_TOPIC, args.TOPIC_EMB_DIM)
        self.attn = AttentionPooling(args.TOPIC_EMB_DIM,
                                     args.TOPIC_EMB_DIM // 2)

    def forward(self, topic_ids):
        '''
            topic_ids: *, num_topics
            return: *, TOPIC_EMB_DIM
        '''
        topic_attn_mask = topic_ids.ne(0).float()
        topic_vec = self.attn(self.topic_embedding(topic_ids), topic_attn_mask)
        return topic_vec


class GCN(nn.Module):
    def __init__(self, feature_dim, num_layers):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList([
            GraphConv(feature_dim, feature_dim, norm='none')
            for _ in range(num_layers)
        ])

    def forward(self, g, h):
        results = [h]
        for i, layer in enumerate(self.layers):
            h = layer(g, h, edge_weight=g.edata['weight'])
            if i != len(self.layers) - 1:
                h = F.tanh(h)
            results.append(h)
        return torch.cat(results, dim=1)


class Model(nn.Module):
    def __init__(self, word_embedding):
        super().__init__()
        self.city_emb = nn.Embedding(args.NUM_CITY, args.CITY_EMB_DIM)
        self.user_id_emb = nn.Embedding(NUM_USER, args.USER_ID_EMB_DIM)
        self.group_id_emb = nn.Embedding(NUM_GROUP, args.GROUP_ID_EMB_DIM)
        self.user_emb_proj = nn.Sequential(
            nn.Linear((1 + args.NUM_GCN_LAYER) * args.USER_ID_EMB_DIM +
                      args.TOPIC_EMB_DIM + args.CITY_EMB_DIM,
                      args.USER_EMB_DIM), nn.ReLU())
        self.group_emb_proj = nn.Sequential(
            nn.Linear((1 + args.NUM_GCN_LAYER) * args.GROUP_ID_EMB_DIM +
                      args.TOPIC_EMB_DIM + args.CITY_EMB_DIM + args.DESC_DIM,
                      args.GROUP_EMB_DIM), nn.ReLU())
        prediction_dim = (args.USER_EMB_DIM + args.GROUP_EMB_DIM +
                          args.CITY_EMB_DIM + args.DESC_DIM)
        self.prediction = nn.Sequential(
            nn.Linear(prediction_dim, prediction_dim // 2), nn.ReLU(),
            nn.Linear(prediction_dim // 2, 1), nn.Sigmoid())
        self.text_encoder = TextEncoder(word_embedding)
        self.topic_encoder = TopicEncoder()
        self.norm = EdgeWeightNorm(norm='both')
        assert args.USER_ID_EMB_DIM == args.GROUP_ID_EMB_DIM
        self.gcn_encoder = GCN(args.USER_ID_EMB_DIM, args.NUM_GCN_LAYER)
        self.loss_fn = nn.BCELoss()

    def build_graph(self, graph_data):
        self.graph = dgl.graph(tuple([*graph_data.indices()]),
                               num_nodes=NUM_USER + NUM_GROUP)
        self.graph.edata['weight'] = self.norm(self.graph, graph_data.values())

    def gcn(self, user_id, group_id):
        """
        Use `self.graph` as the graph,
        `self.user_id_emb.weight` and `self.group_id_emb.weight` as the input features to GCN,
        output features for `user_id` and `group_id`.
        """
        features = torch.cat(
            (self.user_id_emb.weight, self.group_id_emb.weight), dim=0)
        features = self.gcn_encoder(self.graph, features)
        return features[user_id], features[group_id + NUM_USER]

    def forward(self, event_city, event_desc, user_id, user_topic, user_city,
                group_id, group_topic, group_city, group_desc, label):
        '''
            event_city: batch_size
            event_desc: batch_size, num_words
            user_id: batch_size
            user_topic: batch_size, num_user_topics
            user_city: batch_size
            group_id: batch_size
            group_topic: batch_size, num_group_topics
            group_city: batch_size
            group_desc: batch_size, num_words
            label: batch_size
        '''

        batch_city_emb = self.city_emb(event_city)
        batch_desc_emb = self.text_encoder(event_desc)
        batch_user_id_emb, batch_group_id_emb = self.gcn(user_id, group_id)
        batch_user_topic_emb = self.topic_encoder(user_topic)
        batch_user_city_emb = self.city_emb(user_city)
        batch_group_topic_emb = self.topic_encoder(group_topic)
        batch_group_city_emb = self.city_emb(group_city)
        batch_group_desc_emb = self.text_encoder(group_desc)

        batch_user_emb = self.user_emb_proj(
            torch.cat(
                [batch_user_id_emb, batch_user_topic_emb, batch_user_city_emb],
                dim=-1))
        batch_group_emb = self.group_emb_proj(
            torch.cat([
                batch_group_id_emb, batch_group_topic_emb,
                batch_group_city_emb, batch_group_desc_emb
            ],
                      dim=-1))

        predict_input = torch.cat(
            [batch_user_emb, batch_group_emb, batch_city_emb, batch_desc_emb],
            dim=-1)

        score = self.prediction(predict_input).squeeze(dim=-1)
        loss = self.loss_fn(score, label)
        return score, loss


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model(word_embedding).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)


# %%
def acc(y_true, y_hat):
    y_hat = (y_hat >= 0.5).float()
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot


# %%
def calculate_metrics(pred, truth):
    pred = np.array(pred)
    truth = np.array(truth)
    y_pred = pred >= 0.5
    accuracy = np.sum(y_pred == truth) / len(y_pred)
    precision = precision_score(truth, y_pred)
    recall = recall_score(truth, y_pred)
    f1 = f1_score(truth, y_pred)
    auc = roc_auc_score(truth, pred)
    print(f'Accuracy: {accuracy:.4f}\tPrecision: {precision:.4f}\t\
Recall: {recall:.4f}\tF1: {f1:.4f}\tAUC: {auc:.4f}')
    return accuracy, precision, recall, f1, auc


# %%
def run(mode):
    assert mode in ('train', 'test')

    if mode == 'train':
        if args.USE_WANDB:
            wandb.init(project="SocialComputing",
                       name=f'{args.EXP_NAME}-train',
                       entity="social-computing",
                       config={
                           k: getattr(args, k)
                           for k in dir(args) if not k.startswith('_')
                       },
                       group=args.EXP_NAME)

        random.seed(42)
        model.train()
        torch.set_grad_enabled(True)
        total_step = 0
        for _ in range(args.EPOCH):
            train_order = list(range(TRAIN_NUM))
            if args.RANDOM_ORDER:
                random.shuffle(train_order)
            for period_idx in train_order:
                train_dataset = MyDataset(train_behavior[period_idx])
                train_dataloader = DataLoader(train_dataset,
                                              batch_size=args.BATCH_SIZE,
                                              shuffle=True)
                model.build_graph(total_graph[period_idx].to(
                    device, non_blocking=True))

                for (event_city, event_desc, user_id, user_topic, user_city,
                     group_id, group_topic, group_city, group_desc,
                     label) in tqdm(train_dataloader):
                    event_city = event_city.to(device, non_blocking=True)
                    event_desc = event_desc.to(device, non_blocking=True)
                    user_id = user_id.to(device, non_blocking=True)
                    user_topic = user_topic.to(device, non_blocking=True)
                    user_city = user_city.to(device, non_blocking=True)
                    group_id = group_id.to(device, non_blocking=True)
                    group_topic = group_topic.to(device, non_blocking=True)
                    group_city = group_city.to(device, non_blocking=True)
                    group_desc = group_desc.to(device, non_blocking=True)
                    label = label.float().to(device, non_blocking=True)

                    y_hat, bz_loss = model(event_city, event_desc, user_id,
                                           user_topic, user_city, group_id,
                                           group_topic, group_city, group_desc,
                                           label)
                    bz_acc = acc(label, y_hat)
                    optimizer.zero_grad()
                    bz_loss.backward()
                    optimizer.step()

                    if args.USE_WANDB:
                        wandb.log({
                            'train/loss': bz_loss,
                            'train/acc': bz_acc,
                            'train/step': total_step
                        })

                    total_step += 1
                    if total_step % args.SAVE_STEP == 0:
                        ckpt_path = os.path.join(args.MODEL_DIR,
                                                 f'ckpt-{total_step}.pt')
                        torch.save(model.state_dict(), ckpt_path)

            ckpt_path = os.path.join(args.MODEL_DIR, f'ckpt-{total_step}.pt')
            torch.save(model.state_dict(), ckpt_path)

        if args.USE_WANDB:
            wandb.finish()
    else:
        if args.USE_WANDB:
            wandb.init(project="SocialComputing",
                       name=f'{args.EXP_NAME}-test',
                       entity="social-computing",
                       config={
                           k: getattr(args, k)
                           for k in dir(args) if not k.startswith('_')
                       },
                       group=args.EXP_NAME)
        ckpt_list = fnmatch.filter(os.listdir(args.MODEL_DIR), 'ckpt-*.pt')
        ckpt_list = sorted(ckpt_list,
                           key=lambda x: int(x.split('-')[1].split('.')[0]),
                           reverse=True)
        total_ckpt_num = len(ckpt_list)
        print('Total ckpt num:', total_ckpt_num)
        hot_test_dataset = MyDataset(hot_test_behavior)
        hot_test_dataloader = DataLoader(hot_test_dataset,
                                         batch_size=args.BATCH_SIZE,
                                         shuffle=False)
        cold_test_dataset = MyDataset(cold_test_behavior)
        cold_test_dataloader = DataLoader(cold_test_dataset,
                                          batch_size=args.BATCH_SIZE,
                                          shuffle=False)
        # new_test_dataset = MyDataset(new_test_behavior)
        # new_test_dataloader = DataLoader(new_test_dataset,
        #                                  batch_size=args.BATCH_SIZE,
        #                                  shuffle=False)
        model.eval()
        torch.set_grad_enabled(False)
        best_AUC = 0
        for idx, ckpt in enumerate(ckpt_list):
            step = int(ckpt.split('.')[0].split('-')[-1])
            print(f'[{idx + 1}/{total_ckpt_num}] Testing {ckpt}')
            checkpoint = torch.load(os.path.join(args.MODEL_DIR, ckpt))
            model.load_state_dict(checkpoint)
            model.build_graph(val_test_graph.to(device, non_blocking=True))

            pred = {'hot': [], 'cold': []}
            truth = {'hot': [], 'cold': []}
            metrics = {'hot': None, 'cold': None}

            for name, dataloader in zip(
                ['hot', 'cold'], [hot_test_dataloader, cold_test_dataloader]):
                for (event_city, event_desc, user_id, user_topic, user_city,
                     group_id, group_topic, group_city, group_desc,
                     label) in tqdm(dataloader):
                    event_city = event_city.to(device, non_blocking=True)
                    event_desc = event_desc.to(device, non_blocking=True)
                    user_id = user_id.to(device, non_blocking=True)
                    user_topic = user_topic.to(device, non_blocking=True)
                    user_city = user_city.to(device, non_blocking=True)
                    group_id = group_id.to(device, non_blocking=True)
                    group_topic = group_topic.to(device, non_blocking=True)
                    group_city = group_city.to(device, non_blocking=True)
                    group_desc = group_desc.to(device, non_blocking=True)
                    label = label.float().to(device, non_blocking=True)

                    y_hat, _ = model(event_city, event_desc, user_id,
                                     user_topic, user_city, group_id,
                                     group_topic, group_city, group_desc,
                                     label)
                    pred[name].extend(y_hat.to('cpu').detach().numpy())
                    truth[name].extend(label.to('cpu').detach().numpy())
                metrics[name] = calculate_metrics(pred[name], truth[name])

            total_pred = pred['hot'] + pred['cold']
            total_truth = truth['hot'] + truth['cold']
            overall_metrics = calculate_metrics(total_pred, total_truth)
            metrics['overall'] = overall_metrics

            if args.USE_WANDB:
                wandb.log({
                    'test/hot/accuracy': metrics['hot'][0],
                    'test/hot/precision': metrics['hot'][1],
                    'test/hot/recall': metrics['hot'][2],
                    'test/hot/f1': metrics['hot'][3],
                    'test/hot/AUC': metrics['hot'][4],
                    'test/cold/accuracy': metrics['cold'][0],
                    'test/cold/precision': metrics['cold'][1],
                    'test/cold/recall': metrics['cold'][2],
                    'test/cold/f1': metrics['cold'][3],
                    'test/cold/AUC': metrics['cold'][4],
                    # 'test/new/accuracy': metrics['new'][0],
                    # 'test/new/precision': metrics['new'][1],
                    # 'test/new/recall': metrics['new'][2],
                    # 'test/new/f1': metrics['new'][3],
                    # 'test/new/AUC': metrics['new'][4],
                    'test/overall/accuracy': metrics['overall'][0],
                    'test/overall/precision': metrics['overall'][1],
                    'test/overall/recall': metrics['overall'][2],
                    'test/overall/f1': metrics['overall'][3],
                    'test/overall/AUC': metrics['overall'][4],
                    'test/step': step
                })

                if metrics['overall'][4] > best_AUC:
                    wandb.run.summary['best_accuracy'] = metrics['overall'][0]
                    wandb.run.summary['best_precision'] = metrics['overall'][1]
                    wandb.run.summary['best_recall'] = metrics['overall'][2]
                    wandb.run.summary['best_f1'] = metrics['overall'][3]
                    wandb.run.summary['best_AUC'] = metrics['overall'][4]
                    wandb.run.summary['best_step'] = step
                    best_AUC = overall_metrics[4]

        if args.USE_WANDB:
            wandb.finish()


# %%
run('train')

# %%
run('test')

# %%
