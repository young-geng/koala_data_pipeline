import os
import random
import numpy as np
import json
import re
from tqdm import tqdm, trange
import requests
from bs4 import BeautifulSoup

import mlxu


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    input_file='/nfs/vault/data/language/alpaca_data.json',
    train_output_file='/nfs/vault/data/language/chat/alpaca_train.jsonl',
    eval_output_file='/nfs/vault/data/language/chat/alpaca_eval.jsonl',
    gpt_marker='GPT:',
    user_marker='USER:',
    train_ratio=0.9,
)


def process_example(example):
    output_example = {
        'marker_gpt': FLAGS.gpt_marker,
        'marker_user': FLAGS.user_marker,
    }
    if example['input'] != '':
        output_example['human_0'] = (
            example['instruction'] + ' ' + example['input']
        )
    else:
        output_example['human_0'] = example['instruction']
    output_example['gpt_1'] = example['output']
    output_example['fields'] = '[marker_user+human_0+marker_gpt],gpt_1,<|eos|>'
    return output_example


def main(argv):
    with open(FLAGS.input_file) as fin:
        data = json.load(fin)

    random.shuffle(data)
    n_train = int(len(data) * FLAGS.train_ratio)
    train_data = data[:n_train]
    eval_data = data[n_train:]

    with open(FLAGS.train_output_file, 'w') as fout:
        for example in train_data:
            fout.write(json.dumps(process_example(example)) + '\n')

    with open(FLAGS.eval_output_file, 'w') as fout:
        for example in eval_data:
            fout.write(json.dumps(process_example(example)) + '\n')


if __name__ == '__main__':
    mlxu.run(main)
