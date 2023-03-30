#! /bin/bash

export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPT_DIR


python process_chat_data.py
python create_hf_data_v2.py --positive_only=False
python create_hf_data_v2.py --positive_only=True
python process_oig_data.py
python process_alpaca_data.py


cd /nfs/vault/data/language/humanfeedback

cat hh_dialogue_positive_only_train.jsonl summary_positive_only_train.jsonl webgpt_positive_only_train.jsonl \
    | shuf > ../chat/hf_po_train.jsonl

cat hh_dialogue_positive_only_eval.jsonl summary_positive_only_eval.jsonl \
    | shuf > ../chat/hf_po_eval.jsonl

cat hh_dialogue_train.jsonl summary_train.jsonl webgpt_train.jsonl \
    | shuf > ../chat/hf_pn_train.jsonl

cat hh_dialogue_eval.jsonl summary_eval.jsonl \
    | shuf > ../chat/hf_pn_eval.jsonl


cd /nfs/vault/data/language/chat

cat alpaca_train.jsonl chat_data_v2_train.jsonl hf_po_train.jsonl oig_train.jsonl \
    | shuf > instruct_po_v3_train.jsonl

cat alpaca_eval.jsonl chat_data_v2_eval.jsonl hf_po_eval.jsonl oig_eval.jsonl \
    | shuf > instruct_po_v3_eval.jsonl


cat alpaca_train.jsonl chat_data_v2_train.jsonl hf_pn_train.jsonl oig_train.jsonl \
    | shuf > instruct_pn_v3_train.jsonl

cat alpaca_eval.jsonl chat_data_v2_eval.jsonl hf_pn_eval.jsonl oig_eval.jsonl \
    | shuf > instruct_pn_v3_eval.jsonl