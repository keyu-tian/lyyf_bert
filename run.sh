#!/usr/bin/env bash

####### template begins #######

SH_ROOT=$(cd $(dirname $0); pwd)
cd "${SH_ROOT}"; cd ./     # 这里需要从.sh脚本的路径变到项目根，所以cd ..几层需要自己决定
PROJ_ROOT=$(pwd)
echo "PROJ_ROOT=${PROJ_ROOT}"

shopt -s expand_aliases
alias python=python3
alias to_sh_root='cd "${SH_ROOT}"'
alias to_proj_root='cd "${PROJ_ROOT}"'
alias print='echo "$(date +"[%m-%d %H:%M:%S]") (exp.sh)=>"'

####### template ends #######


to_proj_root
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_vc/user/tiankeyu/bert_ckpt/bert-base-chinese/pytorch_model.bin ./pretrained_bert_models/bert-base-chinese
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_vc/user/tiankeyu/bert_ckpt/chinese_roberta_wwm_large_ext/pytorch_model.bin ./pretrained_bert_models/chinese_roberta_wwm_large_ext


print "LYYF.BERT training ..."
to_proj_root
python run.py "$@"

if [ $? -ne 0 ]; then
    print "[failed] LYYF.BERT training failed"
    sleep 1d
else
    print "[succeed] LYYF.BERT training finished"
fi
