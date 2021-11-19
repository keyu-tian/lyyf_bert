import os
import torch

data_dir = os.getcwd() + '/data/clue/'
train_dir = data_dir + 'train.npz'
test_dir = data_dir + 'test.npz'
files = ['train', 'test']
bert_model = 'pretrained_bert_models/bert-base-chinese/'
roberta_model = 'pretrained_bert_models/chinese_roberta_wwm_large_ext/'
model_dir = os.getcwd() + '/experiments/clue/'
log_dir = model_dir + 'train.log'
case_dir = os.getcwd() + '/case/bad_case.txt'

# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的NER模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 16 # todo: batchsize
epoch_num = 50
min_epoch_num = 5
patience = 0.0002
patience_num = 10


gpu = '0'# todo: 0

if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")

# labels = ['address', 'book', 'company', 'game', 'government',
#           'movie', 'name', 'organization', 'position', 'scene']
labels = ['BANK', 'PRODUCT', 'COMMENTS_N', 'COMMENTS_ADJ']

label2id = {
    "O": 0,
    "B-BANK": 1,
    "B-PRODUCT": 2,
    "B-COMMENTS_N": 3,
    'B-COMMENTS_ADJ': 4,
    "I-BANK": 5,
    "I-PRODUCT": 6,
    "I-COMMENTS_N": 7,
    'I-COMMENTS_ADJ': 8
}
#todo: S代表该实体是一个字的，跟BIO有一点差别

id2label = {_id: _label for _label, _id in list(label2id.items())}
