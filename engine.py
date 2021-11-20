import logging

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BertTokenizer

import config
from adv import FGM
from metrics import f1_score, bad_case
from model import BertNER
from utils import time_str, os_system


def train_epoch(tb_lg, iters, itrt, model: BertNER, fgm: FGM, optimizer, scheduler, epoch):
    # set model to training mode
    model.train()
    # step number in one epoch: 336
    train_losses = 0
    freq = iters // 4
    bar = tqdm(range(iters), position=0, leave=True)
    for cur_iter in bar:
        bar.set_description(time_str())
        batch_data, batch_token_starts, batch_labels = next(itrt)
        batch_data, batch_token_starts, batch_labels = batch_data.cuda(non_blocking=True), batch_token_starts.cuda(non_blocking=True), batch_labels.cuda(non_blocking=True)
        batch_masks = batch_data.gt(0)  # get padding mask
        # compute model output and loss
        loss = model((batch_data, batch_token_starts),
                     token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)[0]
        cur_loss = loss.item()
        train_losses += cur_loss
        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        if config.loss_to > 0:
            tp = cur_loss
            while tp > config.loss_to:
                tp /= config.loss_to
                loss /= config.loss_to
        loss.backward()
        # gradient clipping
        bert_norm = nn.utils.clip_grad_norm_(parameters=model.bert.parameters(), max_norm=config.clip_grad)
        lstm_norm = nn.utils.clip_grad_norm_(parameters=model.bilstm.parameters(), max_norm=config.clip_grad)
        clsf_norm = nn.utils.clip_grad_norm_(parameters=model.classifier.parameters(), max_norm=config.clip_grad)
        # performs updates using calculated gradients

        if fgm.open():
            fgm.attack()
            model.zero_grad()
            loss = model((batch_data, batch_token_starts),
                         token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)[0]
            cur_loss = loss.item()
            if config.loss_to > 0:
                tp = cur_loss
                while tp > config.loss_to:
                    tp /= config.loss_to
                    loss /= config.loss_to
            loss.backward()
            bert_norm = nn.utils.clip_grad_norm_(parameters=model.bert.parameters(), max_norm=config.clip_grad)
            lstm_norm = nn.utils.clip_grad_norm_(parameters=model.bilstm.parameters(), max_norm=config.clip_grad)
            clsf_norm = nn.utils.clip_grad_norm_(parameters=model.classifier.parameters(), max_norm=config.clip_grad)
            fgm.restore()

        optimizer.step()
        scheduler.step()
        
        if cur_iter % freq == 0:
            tb_lg.add_scalar('iter/train_loss', cur_loss, iters*epoch + cur_iter)
            tb_lg.add_scalar('norm/bert', bert_norm, iters*epoch + cur_iter)
            tb_lg.add_scalar('norm/lstm', lstm_norm, iters*epoch + cur_iter)
            tb_lg.add_scalar('norm/clsf', clsf_norm, iters*epoch + cur_iter)
        
        del batch_data, batch_token_starts, batch_labels, loss
    
    train_loss = float(train_losses) / iters
    logging.info(f"Epoch: {epoch:-3d}/{epoch}, train loss: {train_loss}")
    tb_lg.add_scalar('epoch/train_loss', train_loss, epoch)
    tb_lg.flush()


def train(tb_lg: SummaryWriter, train_iters, train_itrt, dev_iters, dev_itrt, model, fgm, optimizer, scheduler, model_dir):
    """train the model and test model performance"""
    # reload weights from restore_dir if specified
    if model_dir is not None and config.load_before:
        model = BertNER.from_pretrained(model_dir)
        model.cuda()
        logging.info("--------Load model from {}--------".format(model_dir))
    best_val_f1 = 0.0
    patience_counter = 0
    # start training
    for epoch in range(1, config.epoch_num + 1):
        train_epoch(tb_lg, train_iters, train_itrt, model, fgm, optimizer, scheduler, epoch)
        torch.cuda.empty_cache()
        val_metrics = evaluate(dev_iters, dev_itrt, model, mode='dev')
        val_f1 = val_metrics['f1']
        logging.info("Epoch: {}, dev loss: {}, f1 score: {}".format(epoch, val_metrics['loss'], val_f1))
        improve_f1 = val_f1 - best_val_f1

        tb_lg.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
        tb_lg.add_scalar('epoch/val_F1', val_f1, epoch)
        tb_lg.add_scalar('epoch/improve_f1', improve_f1, epoch)
        tb_lg.add_scalar('epoch/zpatience_cnt', patience_counter, epoch)
        tb_lg.flush()
        
        if improve_f1 > 1e-5:
            best_val_f1 = val_f1
            model.save_pretrained(model_dir)
            logging.info("--------Save best model!--------")
            if improve_f1 < config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        # Early stopping and logging best f1
        if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
            logging.info("Best val f1: {}".format(best_val_f1))
            break
        
        if epoch == config.epoch_num or epoch % 4 == 0:
            os_system(f'hdfs dfs -put -f {config.log_path} {config.hdfs_localout}')
        
    logging.info("Training Finished!")


def evaluate(iters, itrt, model, mode='dev'):
    # set model to evaluation mode
    model.eval()
    if mode == 'test':
        tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True, skip_special_tokens=True)
    id2label = config.id2label
    true_tags = []
    pred_tags = []
    sent_data = []
    dev_losses = 0

    with torch.no_grad():
        bar = tqdm(range(iters), position=0, leave=True)
        for idx in bar:
            bar.set_description(time_str())
            batch_data, batch_token_starts, batch_tags = next(itrt)
            batch_data, batch_token_starts, batch_tags = batch_data.cuda(non_blocking=True), batch_token_starts.cuda(non_blocking=True), batch_tags.cuda(non_blocking=True)
            if mode == 'test':
                sent_data.extend([[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices
                                   if (idx.item() > 0 and idx.item() != 101)] for indices in batch_data])
            batch_masks = batch_data.gt(0)  # get padding mask, gt(x): get index greater than x
            label_masks = batch_tags.gt(-1)  # get padding mask, gt(x): get index greater than x
            # compute model output and loss
            loss = model((batch_data, batch_token_starts),
                         token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)[0]
            dev_losses += loss.item()
            # (batch_size, max_len, num_labels)
            batch_output = model((batch_data, batch_token_starts),
                                 token_type_ids=None, attention_mask=batch_masks)[0]
            # (batch_size, max_len - padding_label_len)
            batch_output = model.crf.decode(batch_output, mask=label_masks)
            # (batch_size, max_len)
            batch_tags = batch_tags.to('cpu').numpy()
            pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in batch_output])
            # (batch_size, max_len - padding_label_len)
            true_tags.extend([[id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])

    assert len(pred_tags) == len(true_tags)
    if mode == 'test':
        assert len(sent_data) == len(true_tags)

    # logging loss, f1 and report
    metrics = {}
    if mode == 'dev':
        f1 = f1_score(true_tags, pred_tags, mode)
        metrics['f1'] = f1
    else:
        bad_case(true_tags, pred_tags, sent_data)
        f1_labels, f1 = f1_score(true_tags, pred_tags, mode)
        metrics['f1_labels'] = f1_labels
        metrics['f1'] = f1
    metrics['loss'] = float(dev_losses) / iters
    return metrics


if __name__ == "__main__":
    a = [101, 679, 6814, 8024, 517, 2208, 3360, 2208, 1957, 518, 7027, 4638,
         1957, 4028, 1447, 3683, 6772, 4023, 778, 8024, 6844, 1394, 3173, 4495,
         807, 4638, 6225, 830, 5408, 8024, 5445, 3300, 1126, 1767, 3289, 3471,
         4413, 4638, 2767, 738, 976, 4638, 3683, 6772, 1962, 511, 0, 0,
         0, 0, 0]
    t = torch.tensor(a, dtype=torch.long)
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True, skip_special_tokens=True)
    word = tokenizer.convert_ids_to_tokens(t[1].item())
    sent = tokenizer.decode(t.tolist())
    print(word)
    print(sent)
