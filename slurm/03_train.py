import sys
import os
import time
import random
import argparse
import tqdm
import torch
import torch.nn as nn
from torchsummary import summary as summary
import torchtext.transforms as T
from torch.hub import load_state_dict_from_url
from torch.utils.data import DataLoader
from torchtext.datasets import SST2
from torchtext.models import RobertaClassificationHead, XLMR_BASE_ENCODER
import torchtext.functional as F
from torch.optim import AdamW
import argparse
from torch.utils.data import DataLoader

# parser
parser = argparse.ArgumentParser(description="XLM-RoBerta")
parser.add_argument("-e", "--epochs", default=5, type=int, metavar="N")
parser.add_argument("-b", "--batch-size", default=32,type=int, metavar="N")
parser.add_argument("-nc", "--num-classes", default=2,type=int, metavar="N")
parser.add_argument("-d", "--input-dim", default=768,type=int, metavar="N")
parser.add_argument("-ms", "--max-seq-len", default=256,type=int, metavar="N")
parser.add_argument("-m", "--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument("-c", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
parser.add_argument("-s", "--seed", default=None, type=int, help="seed for initializing training.")

args = parser.parse_args()

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# setting training
num_epochs = args.epochs
batch_size = args.batch_size
num_classes = args.num_classes
input_dim = args.input_dim

# loading train data
padding_idx = 1
bos_idx = 0
eos_idx = 2
max_seq_len = args.max_seq_len
xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
xlmr_spm_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"
text_transform = T.Sequential(
    T.SentencePieceTokenizer(xlmr_spm_model_path),
    T.VocabTransform(load_state_dict_from_url(xlmr_vocab_path)),
    T.Truncate(max_seq_len - 2),
    T.AddToken(token=bos_idx, begin=True),
    T.AddToken(token=eos_idx, begin=False),
)

train_datapipe = SST2(split="train")
dev_datapipe = SST2(split="dev")

# Transform the raw dataset using non-batched API (i.e apply transformation line by line)
def apply_transform(x):
    return text_transform(x[0]), x[1]

# Getting train data
train_datapipe = train_datapipe.map(apply_transform)
train_datapipe = train_datapipe.batch(batch_size)
train_datapipe = train_datapipe.rows2columnar(["token_ids", "target"])
train_dataloader = DataLoader(train_datapipe)

# Getting dev? data
dev_datapipe = dev_datapipe.map(apply_transform)
dev_datapipe = dev_datapipe.batch(batch_size)
dev_datapipe = dev_datapipe.rows2columnar(["token_ids", "target"])
dev_dataloader = DataLoader(dev_datapipe)

# model
classifier_head = RobertaClassificationHead(num_classes=num_classes, input_dim=input_dim)
model = XLMR_BASE_ENCODER.get_model(head=classifier_head)
model.to(DEVICE)

# print model
params = list(model.named_parameters())
for p in params:
    print("{:<55} | {:>12}".format(p[0], str(tuple(p[1].size()))))
        

learning_rate = 1e-5
optim = AdamW(model.parameters(), lr=learning_rate)
criteria = nn.CrossEntropyLoss()

def train_step(input, target):
    output = model(input)
    loss = criteria(output, target)
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss

def eval_step(input, target):
    output = model(input)
    loss = criteria(output, target).item()
    return float(loss), (output.argmax(1) == target).type(torch.float).sum().item()

def evaluate():
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    counter = 0
    with torch.no_grad():
        for i, batch in enumerate(dev_dataloader):
            input = F.to_tensor(batch["token_ids"], padding_value=padding_idx).to(DEVICE)
            target = torch.tensor(batch["target"]).to(DEVICE)
            loss, predictions = eval_step(input, target)
            total_loss += loss
            correct_predictions += predictions
            total_predictions += len(target)
            counter += 1
    return total_loss / counter, correct_predictions / total_predictions

t_time = 0.0
et_time = 0.0
for e in range(num_epochs):
    e_st = time.time()
    for i, batch in enumerate(train_dataloader):
        input = F.to_tensor(batch["token_ids"], padding_value=padding_idx).to(DEVICE)
        target = torch.tensor(batch["target"]).to(DEVICE)
        st=time.time()
        train_loss = train_step(input, target)
        et=time.time()
        t_time=et-st
        if i % 100 == 0:
            print(f"training {i:10d}th batch, t_time: {t_time:.4f}s, loss: {train_loss:.4f}")
    e_et = time.time()
    et_time = e_et - e_st
    loss, accuracy = evaluate()
    print("Epoch = [{}], t_time[{}]".format(e, et_time))