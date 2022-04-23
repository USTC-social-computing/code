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

from tqdm import tqdm, trange
from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt

# %%
'''
Generating val_test_graph.pt:

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

    val_test_graph = total_graph[TRAIN_NUM]
    
'''
val_test_graph_path = 'val_test_graph.pt'
user_embedding_path = {
    'baseline': 'baseline.npy',
    'gcn_2_layer': 'gcn_2_layer.npy'
}

# %%
val_test_graph = torch.load(val_test_graph_path).coalesce()
NUM_USER, NUM_GROUP = val_test_graph.size()

# %%
degrees = val_test_graph.values().numpy()
degree_threshold = np.percentile(degrees, 90)
degree_threshold

# %%
val_test_graph = (val_test_graph.to_dense() >= degree_threshold).int()
val_test_graph.sum(dim=0)

# %%
num_selected_group = 10
selected_group_indexs = np.argsort(-val_test_graph.sum(
    dim=0).numpy())[:num_selected_group]
selected_group_indexs

# %%
val_test_graph = val_test_graph[:, selected_group_indexs]

# %%
selected_user_indexs = (val_test_graph.sum(
    dim=1) == 1).nonzero().flatten().numpy()
selected_user_indexs

# %%
index_map = dict(val_test_graph[selected_user_indexs, :].nonzero().tolist())
index_map

# %%
baseline_embedding = np.load(user_embedding_path['baseline'])
gcn_2_layer_embedding = np.load(user_embedding_path['gcn_2_layer'])
user_embeddings = {
    'Baseline': baseline_embedding,
    'GCN-2-layer': gcn_2_layer_embedding,
    'GCN-2-layer-0': gcn_2_layer_embedding[:, :64],
    'GCN-2-layer-1': gcn_2_layer_embedding[:, 64:64 * 2],
    'GCN-2-layer-2': gcn_2_layer_embedding[:, 64 * 2:],
}

# %%
for k, v in user_embeddings.items():
    print(k, v.shape)

# %%
colors = cm.rainbow(np.linspace(0, 1, num_selected_group))
for k, v in user_embeddings.items():
    print(k)
    user_embedding = v[selected_user_indexs]
    points = TSNE(verbose=1).fit_transform(user_embedding)
    c = [colors[index_map[i]] for i in range(len(points))]
    label = [index_map[i] for i in range(len(points))]
    plt.scatter(points[:, 0],
                points[:, 1],
                s=1.5,
                marker='o',
                c=c,
                label=label)
    plt.savefig(f'{k}.svg', dpi=200)
    plt.show()

# %%
# !scp *.svg aliyun:/var/www/img.yusanshi.com/upload/social-computing
