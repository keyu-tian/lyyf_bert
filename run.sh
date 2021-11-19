#!/usr/bin/env bash

####### template begins #######

shopt -s expand_aliases
alias python=python3
alias print='echo "$(date +"[%m-%d %H:%M:%S]") (exp.sh)=>"'

####### template ends #######

hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_vc/user/tiankeyu/bert_ckpt/bert-base-chinese/pytorch_model.bin ./pretrained_bert_models/bert-base-chinese
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_vc/user/tiankeyu/bert_ckpt/chinese_roberta_wwm_large_ext/pytorch_model.bin ./pretrained_bert_models/chinese_roberta_wwm_large_ext

# 16 50 3e-5 0.01 5 0.05
# 64 50 1e-4 0.01 5 0.05

print "LYYF.BERT training ..."
python run.py "$@"

if [ $? -ne 0 ]; then
    print "[failed] LYYF.BERT training failed"
    sleep 1d
else
    print "[succeed] LYYF.BERT training finished"
fi
