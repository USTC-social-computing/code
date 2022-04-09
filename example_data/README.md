# 示例数据

这里的示例数据包含了 5 个时间周期的数据。

```
.
├── behaviors
│   ├── 0.tsv
│   ├── 1.tsv
│   ├── 2.tsv
│   ├── 3.tsv
│   └── 4.tsv
├── links
│   ├── user-group
│   │   ├── 0.json
│   │   ├── 1.json
│   │   ├── 2.json
│   │   ├── 3.json
│   │   └── 4.json
│   └── user-user
│       ├── 0.json
│       ├── 1.json
│       ├── 2.json
│       ├── 3.json
│       └── 4.json
├── user.tsv
├── group.tsv
└── word.tsv
```

| 类型 | user | group | city | topic                        | word                         |
| ---- | ---- | ----- | ---- | ---------------------------- | ---------------------------- |
| 数量 | 4    | 3     | 4    | 6（包含 0 号位置的 padding） | 5（包含 0 号位置的 padding） |





**注意事项：**

- topic 和 word 设置了 padding 位，即 0 号位表示 padding，topic 和 word 数量为真实数量加一。另外注意 `word.tsv` 的 `index` 列从 1 开始而不是 0，因为 0 号位置是 padding 位。

- 数据组同学只需要分好数个时间周期的数据即可，具体 train/val/test 的划分由代码组同学来确定（例如这里的示例数据包含了 5 个时间周期的数据，代码组可以按 3/1/1 来划分 train/val/test）。

- 部分列的值是个列表（例如 topic 和 description），列表的长度由数据组同学自行确定（少的补 padding、多的截断）。

- `behaviors/{i}.tsv` 表示第 `i` 个时间周期的行为数据，其中每一行代表一个数据点，表示某个用户是否参与了某个活动。由于一个活动可以对应多个数据点（该活动和任意一个被邀请并给出 0 或 1 回复的用户都构成一个数据点），所以会出现多行数据只有 `user` 和 `label` 列不一样的情况，这种冗余是故意设计的，主要是为了方便后面的模型训练。当然如果最后发现这样会让数据集太大，我们也可以改成一行对应若干个数据点 :)

- `links/user-group/{i}.json` 表示第 `i` 个时间周期的 user-group 图数据，它是个二维矩阵（shape: `user_num, group_num `），第 `j` 行、第 `k` 列表示在**第 `i` 个时间周期中发生的** 第 `j` 个 user 参加第 `k` 个 group 组织的活动的次数。设计成 `json` 文件主要是为了用 `json.load/json.dump` 来读写（下同）。

- `links/user-user/{i}.json` 表示第 `i` 个时间周期的 user-user 图数据，它是个二维矩阵（shape: `user_num, user_num `），第 `j` 行、第 `k` 列表示在**第 `i` 个时间周期中发生的** 第 `j` 个 user 和第 `k` 个 user 共同参与活动的次数。

  
