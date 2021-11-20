import os
import torch
import datetime
import sys


def set_cwd():
    cwd = os.path.dirname(os.path.abspath(__file__))
    os.chdir(cwd)
    sys.path.insert(0, cwd)


def set_environ():
    # OS env variables
    if 'HADOOP_ROOT_LOGGER' not in os.environ:
        # disable hdfs verbose logging
        os.environ['HADOOP_ROOT_LOGGER'] = 'ERROR,console'
    
    # disable hdfs verbose logging
    os.environ['LIBHDFS_OPTS'] = '-Dhadoop.root.logger={}'.format(
        os.environ['HADOOP_ROOT_LOGGER'])
    # set JVM heap memory
    os.environ['LIBHDFS_OPTS'] += '-Xms512m -Xmx10g ' + os.environ['LIBHDFS_OPTS']
    # set KRB5CCNAME for hdfs
    os.environ['KRB5CCNAME'] = '/tmp/krb5cc'
    
    # disable TF verbose logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # fix known issues for pytorch-1.5.1 accroding to https://blog.exxactcorp.com/pytorch-1-5-1-bug-fix-release/
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    
    # set NCCL envs for disributed communication or dist.init_process_group will dead
    os.environ['NCCL_IB_GID_INDEX'] = '3'
    os.environ['NCCL_IB_DISABLE'] = '0'
    os.environ['NCCL_IB_HCA'] = 'mlx5_2:1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['ARNOLD_FRAMEWORK'] = 'pytorch'
    
    # no multi threading
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'


set_cwd()
set_environ()

data_dir = os.getcwd() + '/data/clue/'
train_dir = data_dir + 'train.npz'
test_dir = data_dir + 'test.npz'
files = ['train', 'test']
bert_model = 'pretrained_bert_models/bert-base-chinese/'
roberta_model = 'pretrained_bert_models/chinese_roberta_wwm_large_ext/'
save_dir = os.getcwd() + '/experiments/clue/'
ckpt_path = save_dir + 'pytorch_model.bin'
time_str = datetime.datetime.now().strftime("%m-%d__%H-%M-%S")
log_path = save_dir + 'log.txt'
badcase_path = os.getcwd() + '/case/bad_case.txt'

# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的NER模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
# 16 30 1e-5 0.1 50 10 0.3 -1
batch_size = eval(sys.argv[1])      # 16
epoch_num = eval(sys.argv[2])       # 50
learning_rate = eval(sys.argv[3])   # 3e-5
weight_decay = eval(sys.argv[4])    # 0.01
clip_grad = eval(sys.argv[5])       # 5
fgm_noise = eval(sys.argv[6])       # 0.05
drop_rate = eval(sys.argv[7])       # 0.3
loss_to = eval(sys.argv[8])         # 10    or -1


min_epoch_num = round(epoch_num * 0.1)
patience = 0.0002
patience_num = round(epoch_num * 0.3)

hdfs_localout = os.path.join(os.environ['ARNOLD_OUTPUT'], 'local_output')
tb_dir = os.path.join(
    os.environ['ARNOLD_OUTPUT'],
    f'tb_b{batch_size}ep{epoch_num}_'
    f'lr{learning_rate:g}wd{weight_decay}_'
    f'clp{clip_grad}_fgm{fgm_noise}'
)


torch.cuda.set_device(0)
device = torch.empty(1).cuda().device

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
