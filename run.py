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
import hashlib
from pathlib import Path
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


# %%
# config
class Args():
    def __init__(self):
        self.USE_WANDB = True
        self.EXP_NAME = "no-graph-time-period"
        self.DATA_PATH = "./data"
        self.GLOVE_PATH = "/data/yflyl/glove.840B.300d.txt"
        self.MODEL_DIR = f"../../model_all/{self.EXP_NAME}"
        self.ENABLE_CACHE = True
        self.CACHE_DIR = "./cache"
        self.TRAIN_NUM = 97
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
        self.SAVE_STEP = 1000
        self.EPOCH = 1
        self.DECAY_RATE = 0.6
        self.WEIGHT_SMOOTHING_EXPONENT = 0.5
        self.NUM_GCN_LAYER = 0  # set to 0 to skip GCN


args = Args()
os.makedirs(args.MODEL_DIR, exist_ok=True)


# %%
def load_from_cache(
    identifiers,
    generator,
    cache_dir,
    enabled,
    load_cache_callback=lambda x: print(f'Load cache from {x}'),
    save_cache_callback=lambda x: print(f'Save cache to {x}')):
    if not enabled:
        return generator()

    identifiers.append(generator.__name__)
    cache_path = os.path.join(
        cache_dir,
        f"{hashlib.md5('-'.join(map(str,identifiers)).encode('utf-8')).hexdigest()}.pkl"
    )
    if os.path.isfile(cache_path):
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
            load_cache_callback(cache_path)
            return data
    else:
        data = generator()
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f, protocol=4)
        save_cache_callback(cache_path)
        return data


# %%
def generate_word_embedding():
    # get word dict and word embedding table
    with open(os.path.join(args.DATA_PATH, 'word.tsv'), 'r',
              encoding='utf-8') as f:
        word_file = f.readlines()[1:]

    word_dict = {}
    for line in tqdm(word_file):
        idx, word = line.strip('\n').split('\t')
        word_dict[word] = int(idx)

    word_embedding = np.random.uniform(size=(len(word_dict),
                                             args.WORD_EMB_DIM))
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

    return word_embedding


word_embedding = load_from_cache([], generate_word_embedding, args.CACHE_DIR,
                                 args.ENABLE_CACHE)


# %%
def generate_user_and_group_dict():
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

    return user_dict, group_dict


user_dict, group_dict = load_from_cache([], generate_user_and_group_dict,
                                        args.CACHE_DIR, args.ENABLE_CACHE)
NUM_USER = len(user_dict)
NUM_GROUP = len(group_dict)

# %% [markdown]
# $r_t$: relation occurred at time t <br>
# $R_t$: relation used at time t <br>
# $d$: decay rate
#
#
# $
# \begin{align}
# R_0 &= 0 \\
# R_1 &= r_0 \\
# \cdots \\
# R_t &= r_{t-1} + r_{t-2} * d + r_{t-3} * d^2 + ... \\
#     &= r_{t-1} + d * (r_{t-2} + r_{t-3} * d + ...) \\
#     &= r_{t-1} + d * R_{t-1}
# \end{align}
# $
#
# Note the scale of $R$ at different times doesn't matter, because of weights normalization in GNN part


# %%
def generate_behavior_and_graph():
    # prepare data
    total_behavior_file = sorted(
        os.listdir(os.path.join(args.DATA_PATH, 'behaviours')))
    total_user_group_file = sorted(
        os.listdir(os.path.join(args.DATA_PATH, 'links/user-group')))
    total_user_user_file = sorted(
        os.listdir(os.path.join(args.DATA_PATH, 'links/user-user')))

    total_time_period = len(total_behavior_file)
    train_behavior, test_behavior = [], []
    total_user_group, total_user_user = [], []

    for idx, (behavior_file, user_group_file, user_user_file) in enumerate(
            tqdm(zip(total_behavior_file, total_user_group_file,
                     total_user_user_file),
                 total=total_time_period)):
        behavior_data = []
        with open(os.path.join(args.DATA_PATH, f'behaviours/{behavior_file}'),
                  'r',
                  encoding='utf-8') as f:
            behavior_file = f.readlines()[1:]
        for line in behavior_file:
            _, group, city, desc, user, label = line.strip('\n').split('\t')
            behavior_data.append((int(city), eval(desc)[:args.NUM_EVENT_DESC],
                                  int(user), int(group), int(label)))
        if idx < args.TRAIN_NUM:
            random.seed(42)
            random.shuffle(behavior_data)
            train_behavior.append(behavior_data)
        else:
            test_behavior.extend(behavior_data)

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

    train_behavior_num = sum([len(x) for x in train_behavior])
    print(f'Number of training behaviors: {train_behavior_num}, \
    {math.ceil(train_behavior_num // args.BATCH_SIZE)} steps')
    print(f'Number of testing behaviors: {len(test_behavior)}, \
    {math.ceil(len(test_behavior) // args.BATCH_SIZE)} steps')

    total_graph = [
        torch.sparse_coo_tensor(
            torch.arange(0, NUM_USER + NUM_GROUP).expand(2, -1),
            torch.ones(NUM_USER + NUM_GROUP),
            size=(NUM_USER + NUM_GROUP, NUM_USER + NUM_GROUP),
            dtype=torch.float32).coalesce()
    ]
    for user_user, user_group in tqdm(zip(total_user_user[:-1],
                                          total_user_group[:-1]),
                                      total=total_time_period - 1):
        # combine the two graphs
        # TODO make sure the scale of user_user and user_group not diff too much
        user_user_indices = user_user._indices()
        user_user_values = user_user._values()
        user_group_indices = user_group._indices()
        user_group_values = user_group._values()
        user_group_indices[1] += NUM_USER
        user_user_self_loop_value = user_user_values.median()
        user_group_self_loop_value = user_group_values.median()
        current_grpah_indices = torch.cat(
            (
                user_user_indices,  # top left U-U
                user_group_indices,  # top right U-G
                user_group_indices[[1, 0]],  # bottom left G-U
                torch.arange(0, NUM_USER).expand(2, -1),  # U-U self loop
                torch.arange(NUM_USER, NUM_USER + NUM_GROUP).expand(
                    2, -1),  # G-G self loop
            ),
            dim=1)
        current_graph_values = torch.cat(
            (
                user_user_values,  # top left U-U
                user_group_values,  # top right U-G
                user_group_values,  # bottom left G-U
                user_user_self_loop_value.expand(NUM_USER),  # U-U self loop
                user_group_self_loop_value.expand(NUM_GROUP),  # G-G self loop
            ),
            dim=0)
        current_graph = torch.sparse_coo_tensor(current_grpah_indices,
                                                current_graph_values,
                                                size=(NUM_USER + NUM_GROUP,
                                                      NUM_USER + NUM_GROUP))
        current_graph = torch.pow(current_graph,
                                  args.WEIGHT_SMOOTHING_EXPONENT)
        total_graph.append(current_graph + total_graph[-1] * args.DECAY_RATE)

    train_graph = total_graph[:args.TRAIN_NUM]
    test_graph = total_graph[args.TRAIN_NUM]

    return train_behavior, test_behavior, train_graph, test_graph


train_behavior, test_behavior, train_graph, test_graph = load_from_cache(
    [], generate_behavior_and_graph, args.CACHE_DIR, args.ENABLE_CACHE)

# Loading a sparse tensor from pickle seems to be always uncoalesced...
for i in range(len(train_graph)):
    train_graph[i] = train_graph[i].coalesce()
test_graph = test_graph.coalesce()


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
            GraphConv(feature_dim, feature_dim, norm='none', activation=F.relu)
            for _ in range(num_layers)
        ])

    def forward(self, g, h):
        for layer in self.layers:
            h = layer(g, h, edge_weight=g.edata['weight'])
        return h


class Model(nn.Module):
    def __init__(self, word_embedding):
        super().__init__()
        self.city_emb = nn.Embedding(args.NUM_CITY, args.CITY_EMB_DIM)
        self.user_id_emb = nn.Embedding(NUM_USER, args.USER_ID_EMB_DIM)
        self.group_id_emb = nn.Embedding(NUM_GROUP, args.GROUP_ID_EMB_DIM)
        self.user_emb_proj = nn.Sequential(
            nn.Linear(
                args.USER_ID_EMB_DIM + args.TOPIC_EMB_DIM + args.CITY_EMB_DIM,
                args.USER_EMB_DIM), nn.ReLU())
        self.group_emb_proj = nn.Sequential(
            nn.Linear(
                args.GROUP_ID_EMB_DIM + args.TOPIC_EMB_DIM +
                args.CITY_EMB_DIM + args.DESC_DIM, args.GROUP_EMB_DIM),
            nn.ReLU())
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
    precision = precision_score(truth, y_pred)
    recall = recall_score(truth, y_pred)
    f1 = f1_score(truth, y_pred)
    auc = roc_auc_score(truth, pred)
    print(
        f'Precision: {precision:.4f}\tRecall: {recall:.4f}\tF1: {f1:.4f}\tAUC: {auc:.4f}'
    )
    return precision, recall, f1, auc


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

        model.train()
        torch.set_grad_enabled(True)
        total_step = 0
        for _ in range(args.EPOCH):
            for period_idx in range(args.TRAIN_NUM):
                train_dataset = MyDataset(train_behavior[period_idx])
                train_dataloader = DataLoader(train_dataset,
                                              batch_size=args.BATCH_SIZE,
                                              shuffle=True)
                model.build_graph(train_graph[period_idx].to(
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
        total_ckpt_num = len(ckpt_list)
        print('Total ckpt num:', total_ckpt_num)
        test_dataset = MyDataset(test_behavior)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=args.BATCH_SIZE,
                                     shuffle=False)
        model.eval()
        torch.set_grad_enabled(False)
        for idx, ckpt in enumerate(ckpt_list):
            step = int(ckpt.split('.')[0].split('-')[-1])
            print(f'[{idx + 1}/{total_ckpt_num}] Testing {ckpt}')
            checkpoint = torch.load(os.path.join(args.MODEL_DIR, ckpt))
            model.load_state_dict(checkpoint)
            model.build_graph(test_graph.to(device, non_blocking=True))

            pred, truth = [], []
            for (event_city, event_desc, user_id, user_topic, user_city,
                 group_id, group_topic, group_city, group_desc,
                 label) in tqdm(test_dataloader):
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

                y_hat, _ = model(event_city, event_desc, user_id, user_topic,
                                 user_city, group_id, group_topic, group_city,
                                 group_desc, label)
                pred.extend(y_hat.to('cpu').detach().numpy())
                truth.extend(label.to('cpu').detach().numpy())

            precision, recall, f1, auc = calculate_metrics(pred, truth)
            if args.USE_WANDB:
                wandb.log({
                    'test/precision': precision,
                    'test/recall': recall,
                    'test/f1': f1,
                    'test/AUC': auc,
                    'test/step': step
                })
        if args.USE_WANDB:
            wandb.finish()


# %%
run('train')

# %%
run('test')

# %%
