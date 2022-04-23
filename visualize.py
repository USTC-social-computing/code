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
import numpy as np
import torch

from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt

# %%
'''
Generating user_group_graph.pt:

    total_graph = [
        torch.sparse_coo_tensor(size=(NUM_USER, NUM_GROUP),
                                dtype=torch.float32)
    ]
    for user_group in tqdm(total_user_group[:-1], total=total_time_period - 1):
        current_graph = torch.sparse_coo_tensor(user_group._indices(),
                                                user_group._values(),
                                                size=(NUM_USER,
                                                      NUM_GROUP))
        current_graph = torch.pow(current_graph, args.WEIGHT_SMOOTHING_EXPONENT)
        total_graph.append(current_graph + total_graph[-1] * args.DECAY_RATE)

    user_group_graph = total_graph[TRAIN_NUM]
    
'''
user_group_graph_path = 'user_group_graph.pt'
user_embedding_path = {
    'baseline': 'baseline.npy',
    'gcn-2-layer-0%': 'gcn-2-layer-3.npy',
    'gcn-2-layer-17%': 'gcn-2-layer-18.npy',
    'gcn-2-layer-33%': 'gcn-2-layer-33.npy',
    'gcn-2-layer-50%': 'gcn-2-layer-48.npy',
    'gcn-2-layer-67%': 'gcn-2-layer-63.npy',
    'gcn-2-layer-83%': 'gcn-2-layer-78.npy',
    'gcn-2-layer-100%': 'gcn-2-layer-93.npy'
}
num_selected_group = 10

# %%
user_group_graph = torch.load(user_group_graph_path).coalesce()
NUM_USER, NUM_GROUP = user_group_graph.size()

# %%
degrees = user_group_graph.values().numpy()
degree_threshold = np.percentile(degrees, 85)
degree_threshold

# %%
user_group_graph = (user_group_graph.to_dense() >= degree_threshold).int()
user_group_graph.sum(dim=0)

# %%
selected_group_indexs = np.argsort(-user_group_graph.sum(
    dim=0).numpy())[:num_selected_group]
selected_group_indexs

# %%
user_group_graph = user_group_graph[:, selected_group_indexs]

# %%
selected_user_indexs = (user_group_graph.sum(
    dim=1) == 1).nonzero().flatten().numpy()
selected_user_indexs

# %%
index_map = dict(user_group_graph[selected_user_indexs, :].nonzero().tolist())
index_map

# %%
user_embeddings = {}
for k, v in user_embedding_path.items():
    if k.startswith('baseline'):
        user_embeddings[k] = np.load(v)
    elif k.startswith('gcn-2-layer'):
        for i, x in enumerate(np.split(np.load(v), 3, axis=1)):
            user_embeddings[f'{k}-{i}'] = x
    else:
        raise

# %%
for k, v in user_embeddings.items():
    print(k, v.shape)

# %%
colors = cm.rainbow(np.linspace(0, 1, num_selected_group))
for k, v in user_embeddings.items():
    print(k)
    user_embedding = v[selected_user_indexs]
    points = TSNE().fit_transform(user_embedding)
    c = [colors[index_map[i]] for i in range(len(points))]
    label = [index_map[i] for i in range(len(points))]
    plt.scatter(points[:, 0], points[:, 1], s=2, marker='o', c=c, label=label)
    plt.savefig(f'{k}.svg', dpi=200)
    plt.show()

# %%
# !scp *.svg aliyun:/var/www/img.yusanshi.com/upload/social-computing

# %%
