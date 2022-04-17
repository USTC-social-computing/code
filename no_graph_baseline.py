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
import random
import math
import fnmatch
import wandb
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


# %%
# config
class Args():
    def __init__(self):
        self.EXP_NAME = "no-graph-baseline"
        self.DATA_PATH = "./data"
        self.GLOVE_PATH = "/data/yflyl/glove.840B.300d.txt"
        self.MODEL_DIR = f"../../model_all/{self.EXP_NAME}"
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


args = Args()
os.makedirs(args.MODEL_DIR, exist_ok=True)

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
TRAIN_NUM, VAL_NUM = 93, 5

total_behavior_file = sorted(
    os.listdir(os.path.join(args.DATA_PATH, 'behaviours')))

total_time_period = len(total_behavior_file)
train_behavior, val_behavior, test_behavior = [], [], []

for idx, behavior_file in enumerate(tqdm(total_behavior_file)):
    behavior_data = []
    with open(os.path.join(args.DATA_PATH, f'behaviours/{behavior_file}'),
              'r',
              encoding='utf-8') as f:
        behavior_file = f.readlines()[1:]
    for line in behavior_file:
        _, group, city, desc, user, label = line.strip('\n').split('\t')
        behavior_data.append((int(city), eval(desc)[:args.NUM_EVENT_DESC],
                              int(user), int(group), int(label)))
    if idx < TRAIN_NUM:
        train_behavior.extend(behavior_data)
    elif idx < TRAIN_NUM + VAL_NUM:
        val_behavior.extend(behavior_data)
    else:
        test_behavior.extend(behavior_data)

random.seed(42)
random.shuffle(train_behavior)
print(
    f'Number of training behaviors: {len(train_behavior)}, {math.ceil(len(train_behavior) // args.BATCH_SIZE)} steps'
)
print(
    f'Number of validation behaviors: {len(val_behavior)}, {math.ceil(len(val_behavior) // args.BATCH_SIZE)} steps'
)
print(
    f'Number of testing behaviors: {len(test_behavior)}, {math.ceil(len(test_behavior) // args.BATCH_SIZE)} steps'
)


# %%
class MyDataset(Dataset):
    def __init__(self, behavior):
        super().__init__()
        (city, desc, user, user_topic, user_city, group, group_topic,
         group_city, group_desc,
         label) = [], [], [], [], [], [], [], [], [], []
        for t in tqdm(behavior):
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
            nn.Linear(prediction_dim // 2, prediction_dim // 4), nn.ReLU(),
            nn.Linear(prediction_dim // 4, 1), nn.Sigmoid())
        self.text_encoder = TextEncoder(word_embedding)
        self.topic_encoder = TopicEncoder()
        self.loss_fn = nn.BCELoss()

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
        batch_user_id_emb = self.user_id_emb(user_id)
        batch_group_id_emb = self.group_id_emb(group_id)
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
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
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
        wandb.init(project="SocialComputing",
                   name=f'{args.EXP_NAME}-train',
                   entity="yflyl613",
                   config={
                       k: getattr(args, k)
                       for k in dir(args) if not k.startswith('_')
                   },
                   group=args.EXP_NAME)
        train_dataset = MyDataset(train_behavior)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.BATCH_SIZE,
                                      shuffle=True)
        val_dataset = MyDataset(val_behavior)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=args.BATCH_SIZE,
                                    shuffle=False)
        model.train()
        torch.set_grad_enabled(True)
        for ep in range(args.EPOCH):
            for step, (event_city, event_desc, user_id, user_topic, user_city,
                       group_id, group_topic, group_city, group_desc,
                       label) in enumerate(tqdm(train_dataloader)):
                step += ep * len(train_dataloader)
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

                wandb.log({
                    'train/loss': bz_loss,
                    'train/acc': bz_acc,
                    'train/step': step
                })

                if (step + 1) % args.SAVE_STEP == 0:
                    ckpt_path = os.path.join(args.MODEL_DIR,
                                             f'ckpt-{step + 1}.pt')
                    torch.save(model.state_dict(), ckpt_path)

                    model.eval()
                    torch.set_grad_enabled(False)
                    pred, truth = [], []
                    with torch.no_grad():
                        for (event_city, event_desc, user_id, user_topic,
                             user_city, group_id, group_topic, group_city,
                             group_desc, label) in tqdm(val_dataloader):
                            event_city = event_city.to(device,
                                                       non_blocking=True)
                            event_desc = event_desc.to(device,
                                                       non_blocking=True)
                            user_id = user_id.to(device, non_blocking=True)
                            user_topic = user_topic.to(device,
                                                       non_blocking=True)
                            user_city = user_city.to(device, non_blocking=True)
                            group_id = group_id.to(device, non_blocking=True)
                            group_topic = group_topic.to(device,
                                                         non_blocking=True)
                            group_city = group_city.to(device,
                                                       non_blocking=True)
                            group_desc = group_desc.to(device,
                                                       non_blocking=True)
                            label = label.float().to(device, non_blocking=True)

                            y_hat, _ = model(event_city, event_desc, user_id,
                                             user_topic, user_city, group_id,
                                             group_topic, group_city,
                                             group_desc, label)
                            pred.extend(y_hat.to('cpu').detach().numpy())
                            truth.extend(label.to('cpu').detach().numpy())
                    precision, recall, f1, auc = calculate_metrics(pred, truth)

                    wandb.log({
                        'val/precision': precision,
                        'val/recall': recall,
                        'val/f1': f1,
                        'val/AUC': auc,
                        'val/step': step
                    })

                    model.train()
                    torch.set_grad_enabled(True)

        ckpt_path = os.path.join(args.MODEL_DIR, f'ckpt-{step + 1}.pt')
        torch.save(model.state_dict(), ckpt_path)
        wandb.finish()
    else:
        wandb.init(project="SocialComputing",
                   name=f'{args.EXP_NAME}-test',
                   entity="yflyl613",
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
            wandb.log({
                'test/precision': precision,
                'test/recall': recall,
                'test/f1': f1,
                'test/AUC': auc,
                'test/step': step
            })
        wandb.finish()


# %%
run('train')

# %%
run('test')

# %%
