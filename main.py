import logging
import os
import time
import warnings

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW

import config
import utils
from adv import FGM
from dataloader import InfiniteBatchSampler
from datapre import Processor
from dataset import NERDataset
from engine import train, evaluate
from model import BertNER

warnings.filterwarnings('ignore')


def dev_split(dataset_dir):
    """split dev set"""
    data = np.load(dataset_dir, allow_pickle=True)
    words = data["words"]
    labels = data["labels"]
    x_train, x_dev, y_train, y_dev = train_test_split(words, labels, test_size=config.dev_split_size, random_state=0)
    return x_train, x_dev, y_train, y_dev


def test():
    data = np.load(config.test_dir, allow_pickle=True)
    word_test = data["words"]
    label_test = data["labels"]
    test_dataset = NERDataset(word_test, label_test, config)
    logging.info("--------Dataset Build!--------")
    # build data_loader
    test_loader = DataLoader(
        dataset=test_dataset, pin_memory=True,
        batch_sampler=InfiniteBatchSampler(
            dataset_len=len(test_dataset), batch_size=config.batch_size,
            shuffle=False, filling=False, drop_last=False,
        ),
        collate_fn=test_dataset.collate_fn
    )
    test_iters, test_itrt = len(test_loader), iter(test_loader)
    logging.info(f"[dataset] test : len={len(test_dataset)}, bs={config.batch_size}, iters={test_iters}")
    
    logging.info("--------Get Data-loader!--------")
    # Prepare model
    if config.save_dir is not None:
        model = BertNER.from_pretrained(config.save_dir)
        model.cuda()
        logging.info("--------Load model from {}--------".format(config.save_dir))
    else:
        logging.info("--------No model to test !--------")
        return
    val_metrics = evaluate(test_iters, test_itrt, model, mode='test', epoch=0)
    val_f1 = val_metrics['f1']
    logging.info("test loss: {}, f1 score: {}".format(val_metrics['loss'], val_f1))
    val_f1_labels = val_metrics['f1_labels']
    for label in config.labels:
        logging.info("f1 score of {}: {}".format(label, val_f1_labels[label]))

    utils.os_system(f'hdfs dfs -put -f {config.log_path} {config.hdfs_localout}')
    utils.os_system(f'hdfs dfs -put -f {config.ckpt_path} {config.hdfs_localout}')
    utils.os_system(f'hdfs dfs -put -f {config.badcase_path} {config.hdfs_localout}')
    print(f'[bad_case.txt] see {os.path.join(config.hdfs_localout, os.path.basename(config.badcase_path))}')


def load_dev(mode):
    if mode == 'train':
        # 分离出验证集
        word_train, word_dev, label_train, label_dev = dev_split(config.train_dir)
    elif mode == 'test':
        train_data = np.load(config.train_dir, allow_pickle=True)
        dev_data = np.load(config.test_dir, allow_pickle=True)
        word_train = train_data["words"]
        label_train = train_data["labels"]
        word_dev = dev_data["words"]
        label_dev = dev_data["labels"]
    else:
        word_train = None
        label_train = None
        word_dev = None
        label_dev = None
    return word_train, word_dev, label_train, label_dev


def run():
    """train the model"""
    utils.os_system(f'hdfs dfs -mkdir {config.hdfs_localout}')
    # set the logger
    utils.set_logger(config.log_path)
    logging.info("device: {}".format(config.device))

    logging.info(f"======= config =======")
    logging.info(f"==> bs  : {config.batch_size}")
    logging.info(f"==> ep  : {config.epoch_num}")
    logging.info(f"==> lr  : {config.learning_rate:g}")
    logging.info(f"==> wd  : {config.weight_decay}")
    logging.info(f"==> clip: {config.clip_grad}")
    logging.info(f"==> fgm : {config.fgm_noise}")
    logging.info(f"==> drop: {config.drop1}")
    logging.info(f"==> L to: {config.loss_to}")
    logging.info(f"====== defaults ======")
    logging.info(f"==> min_epoch_num: {config.min_epoch_num}")
    logging.info(f"==> patience: {config.patience}")
    logging.info(f"==> patience_num: {config.patience_num}")
    
    # 处理数据，分离文本和标签
    processor = Processor(config)
    processor.process()
    logging.info("--------Process Done!--------")
    # 分离出验证集
    word_train, word_dev, label_train, label_dev = load_dev('train')
    # build dataset
    train_dataset = NERDataset(word_train, label_train, config)
    dev_dataset = NERDataset(word_dev, label_dev, config)
    logging.info("--------Dataset Build!--------")
    # build data_loader
    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
    #                           shuffle=True, collate_fn=train_dataset.collate_fn)
    # dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
    #                         shuffle=True, collate_fn=dev_dataset.collate_fn)

    train_loader = DataLoader(
        dataset=train_dataset, pin_memory=True,
        batch_sampler=InfiniteBatchSampler(
            dataset_len=len(train_dataset), batch_size=config.batch_size,
            shuffle=True, filling=False, drop_last=True,
        ),
        collate_fn=train_dataset.collate_fn
    )
    train_iters, train_itrt = len(train_loader), iter(train_loader)
    logging.info(f"[dataset] train: len={len(train_dataset)}, bs={config.batch_size}, iters={train_iters}")
    dev_loader = DataLoader(
        dataset=dev_dataset, pin_memory=True,
        batch_sampler=InfiniteBatchSampler(
            dataset_len=len(dev_dataset), batch_size=config.batch_size,
            shuffle=False, filling=False, drop_last=False,
        ),
        collate_fn=dev_dataset.collate_fn
    )
    dev_iters, dev_itrt = len(dev_loader), iter(dev_loader)
    logging.info(f"[dataset] dev  : len={len(dev_dataset)}, bs={config.batch_size}, iters={dev_iters}")
    
    logging.info("--------Get Dataloader!--------")
    # Prepare model
    model = BertNER.from_pretrained(config.roberta_model, num_labels=len(config.label2id))
    model.cuda()
    # Prepare optimizer
    if config.full_fine_tuning:
        # model.named_parameters(): [bert, bilstm, classifier, crf]
        bert_optimizer = list(model.bert.named_parameters())
        lstm_optimizer = list(model.bilstm.named_parameters())
        classifier_optimizer = list(model.classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
            {'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': 0.0},
            {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
            {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': 0.0},
            {'params': model.crf.parameters(), 'lr': config.learning_rate * 5}
        ]
    # only fine-tune the head classifier
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=False)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=config.epoch_num * train_iters // 10,
                                                num_training_steps=config.epoch_num * train_iters)

    scheduler.get_last_lr()

    # Train the model
    logging.info("--------Start Training!--------")
    fgm = FGM(model, config.fgm_noise)

    tb_lg = SummaryWriter(log_dir=config.tb_dir)
    best_val_f1 = train(tb_lg, train_iters, train_itrt, dev_iters, dev_itrt, model, fgm, optimizer, scheduler, config.save_dir)
    config.badcase_path = config.badcase_path.replace(
        'xxxx',
        f'{best_val_f1*100:2f}'
    )
    
    time.sleep(5)
    tb_lg.close()

    utils.os_system(f'hdfs dfs -put -f {config.log_path} {config.hdfs_localout}')
    utils.os_system(f'hdfs dfs -put -f {config.ckpt_path} {config.hdfs_localout}')
    

if __name__ == '__main__':
    run()
    test()
