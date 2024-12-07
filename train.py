# 读取训练参数+初始化日志记录
import sys
import json
import mindspore as ms
import swanlab

args_file = "baseline.json"
if len(sys.argv) > 1:
    args_file = sys.argv[1]
with open(os.path.join("configs", args_file), "r") as f:
    args = json.load(f)
if len(sys.argv) > 2:
    device_id = sys.argv[2]
    if device_id == "CPU":
        ms.set_context(device_target="CPU")
        args["device"] = device_id
    else:
        ms.set_context(device_target="Ascend", device_id=int(device_id))
        args["device"] = "Ascend:" + str(device_id)


exp_name = args_file[:-5]
swanlab.init(project="Ascend_IMDB_CLS", experiment_name=exp_name, config=args)


# 构造数据集

import os
import mindspore.dataset as ds


class IMDBData:
    label_map = {"pos": 1, "neg": 0}

    def __init__(self, path, mode="train"):
        self.docs, self.labels = [], []
        for label in self.label_map.keys():
            doc_dir = os.path.join(path, mode, label)
            doc_list = os.listdir(doc_dir)
            for fname in doc_list:
                with open(os.path.join(doc_dir, fname)) as f:
                    doc = f.read()
                    doc = doc.lower().split()
                    self.docs.append(doc)
                    self.labels.append([self.label_map[label]])

    def __getitem__(self, idx):
        return self.docs[idx], self.labels[idx]

    def __len__(self):
        return len(self.docs)


imdb_path = "data/aclImdb"
imdb_train = ds.GeneratorDataset(
    IMDBData(imdb_path, "train"), column_names=["text", "label"], shuffle=True
)
imdb_test = ds.GeneratorDataset(
    IMDBData(imdb_path, "test"), column_names=["text", "label"], shuffle=False
)

# 构造embedding词表
import os

import numpy as np


def load_glove(glove_path):
    embeddings = []
    tokens = []
    with open(os.path.join(glove_path, "glove.6B.100d.txt"), encoding="utf-8") as gf:
        for glove in gf:
            word, embedding = glove.split(maxsplit=1)
            tokens.append(word)
            embeddings.append(np.fromstring(embedding, dtype=np.float32, sep=" "))
    # 添加 <unk>, <pad> 两个特殊占位符对应的embedding
    embeddings.append(np.random.rand(100))
    embeddings.append(np.zeros((100,), np.float32))

    vocab = ds.text.Vocab.from_list(
        tokens, special_tokens=["<unk>", "<pad>"], special_first=False
    )
    embeddings = np.array(embeddings).astype(np.float32)
    return vocab, embeddings


vocab, embeddings = load_glove("./embedding")
print(f"VOCAB SIZE: {len(vocab.vocab())}")

# 数据预处理
import mindspore as ms

lookup_op = ds.text.Lookup(vocab, unknown_token="<unk>")
pad_op = ds.transforms.PadEnd([500], pad_value=vocab.tokens_to_ids("<pad>"))
type_cast_op = ds.transforms.TypeCast(ms.float32)

imdb_train = imdb_train.map(operations=[lookup_op, pad_op], input_columns=["text"])
imdb_train = imdb_train.map(operations=[type_cast_op], input_columns=["label"])

imdb_test = imdb_test.map(operations=[lookup_op, pad_op], input_columns=["text"])
imdb_test = imdb_test.map(operations=[type_cast_op], input_columns=["label"])

imdb_train, imdb_valid = imdb_train.split([0.7, 0.3])

print(f"TRAIN SET SIZE: {len(imdb_train)}")
print(f"VALID SET SIZE: {len(imdb_valid)}")
print(f"TEST SET SIZE: {len(imdb_test)}")

imdb_train = imdb_train.batch(args["batch_size"], drop_remainder=True)
imdb_valid = imdb_valid.batch(args["batch_size"], drop_remainder=True)


# LSTM分类器实现
import math
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Uniform, HeUniform


class LSTM_CLS(nn.Cell):
    def __init__(self, embeddings, hidden_dim, output_dim, n_layers, pad_idx):
        super().__init__()
        vocab_size, embedding_dim = embeddings.shape
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            embedding_table=ms.Tensor(embeddings),
            padding_idx=pad_idx,
        )
        self.rnn = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True
        )
        weight_init = HeUniform(math.sqrt(5))
        bias_init = Uniform(1 / math.sqrt(hidden_dim * 2))
        self.fc = nn.Dense(
            hidden_dim, output_dim, weight_init=weight_init, bias_init=bias_init
        )

    def construct(self, inputs):
        embedded = self.embedding(inputs)
        _, (hidden, _) = self.rnn(embedded)
        hidden = hidden[-1, :, :]
        output = self.fc(hidden)
        return output


model = LSTM_CLS(
    embeddings,
    args["hidden_size"],
    args["output_size"],
    args["num_layers"],
    vocab.tokens_to_ids("<pad>"),
)

# 损失函数与优化器
loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
optimizer = nn.Adam(model.trainable_params(), learning_rate=args["lr"])

# 训练过程实现
from tqdm import tqdm
import time


def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss


grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)


def train_step(data, label):
    loss, grads = grad_fn(data, label)
    optimizer(grads)
    return loss


def train_one_epoch(model, train_dataset, epoch=0):
    model.set_train()
    total = train_dataset.get_dataset_size()
    step_total = 0
    last_time = time.time()
    for i in train_dataset.create_tuple_iterator():
        loss = train_step(*i)
        step_total += 1
        loss_item = loss.item()
        if step_total % args["report_interval"] == 1:
            now_time = time.time()
            per_batch_time = (now_time - last_time) / args["report_interval"]
            last_time = now_time
            swanlab.log(
                {
                    "train/epoch": epoch,
                    "train/step": step_total,
                    "train/loss": loss_item,
                    "train/per_batch_time(s)": per_batch_time,
                }
            )
            print(
                f"[train epoch-{epoch:2d} step-{step_total:4d}/{total:4d}] loss:{loss_item:.4f} use_time:{per_batch_time:10.4f}s"
            )


# 评估过程实现
def binary_accuracy(preds, y):
    rounded_preds = np.around(ops.sigmoid(preds).asnumpy())
    correct = (rounded_preds == y).astype(np.float32)
    acc = correct.sum() / len(correct)
    return acc


def evaluate(model, test_dataset, criterion, epoch=0, mode="eval"):
    last_time = time.time()
    total = test_dataset.get_dataset_size()
    epoch_loss = 0
    epoch_acc = 0
    model.set_train(False)
    for i in test_dataset.create_tuple_iterator():
        predictions = model(i[0])
        loss = criterion(predictions, i[1])
        epoch_loss += loss.asnumpy()
        acc = binary_accuracy(predictions, i[1])
        epoch_acc += acc

    final_loss = float(epoch_loss / total)
    final_acc = float(epoch_acc / total)
    use_time = time.time() - last_time
    swanlab.log(
        {
            f"{mode}/loss": final_loss,
            f"{mode}/acc": final_acc,
            f"{mode}/use_time": use_time,
        }
    )
    print(
        f"[{mode} epoch-{epoch:2d} loss:{final_loss:.4f} acc:{final_acc*100:.2f}% use_time:{use_time:10.4f}s"
    )

    return final_loss, final_acc


# 开启训练=
best_valid_loss = float("inf")
save_path = os.path.join("output", exp_name)
os.makedirs(save_path, exist_ok=True)
ckpt_file_name = os.path.join(save_path, "sentiment-analysis.ckpt")


for epoch in range(args["num_epochs"]):
    train_one_epoch(model, imdb_train, epoch)
    valid_loss, _ = evaluate(model, imdb_valid, loss_fn, epoch)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        ms.save_checkpoint(model, ckpt_file_name)


# 开始测试
param_dict = ms.load_checkpoint(ckpt_file_name)
ms.load_param_into_net(model, param_dict)
imdb_test = imdb_test.batch(64)
test_loss, test_acc = evaluate(model, imdb_test, loss_fn, mode="test")

# 开始预测
score_map = {1: "Positive", 0: "Negative"}


def predict_sentiment(model, vocab, sentence):
    model.set_train(False)
    tokenized = sentence.lower().split()
    indexed = vocab.tokens_to_ids(tokenized)
    tensor = ms.Tensor(indexed, ms.int32)
    tensor = tensor.expand_dims(0)
    prediction = model(tensor)
    return score_map[int(np.round(ops.sigmoid(prediction).asnumpy()))]


predict_sentiment(model, vocab, "This film is great")
predict_sentiment(model, vocab, "This film is terrible")
