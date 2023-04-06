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
    input_file='/nfs/vault/data/language/chat_data_v3.json',
    train_output_file='/nfs/vault/data/language/chat/chat_data_v3_train.jsonl',
    eval_output_file='/nfs/vault/data/language/chat/chat_data_v3_eval.jsonl',
    eval_json_output_file='/nfs/vault/data/language/chat/chat_eval_json.json',
    gpt_marker='GPT:',
    user_marker='USER:',
    beginning_marker='BEGINNING OF CONVERSATION:',
    eos_maker='</s>',
)

FILTER_KEYWORDS = [
    'openai', 'chatgpt', 'gpt-3', 'chadgpt', 'gpt3', 'chat-gpt', 'open-ai',
    'open ai',
]


def match_keywords(keywords, text):
    for keyword in keywords:
        if keyword in text.lower():
            return True
    return False


def process_data(data, output_file):
    with open(output_file, 'w') as fout:
        total = 0
        for example in tqdm(data):
            output_example = {
                'marker_gpt': FLAGS.gpt_marker,
                'marker_user': FLAGS.user_marker,
            }
            filtered_keys = [
                key for key in example.keys()
                if key.startswith('human') or key.startswith('gpt')
            ]
            keys = sorted(filtered_keys, key=lambda x: int(x.split('_')[-1]))
            fields = []
            skip = False
            for key in keys:
                if match_keywords(FILTER_KEYWORDS, example[key]):
                    skip = True
                    break

                output_example[key] = example[key]
                if key.startswith('gpt'):
                    fields.append(key)
                    fields.append('<|eos|>')
                    output_example[key] = example[key].replace('\nCopy code\n', '\n\n')
                elif key.startswith('human'):
                    fields.append(f'[marker_user+{key}+marker_gpt]')

            output_example['fields'] = ','.join(fields)

            if not skip:
                total += 1
                fout.write(json.dumps(output_example) + '\n')

        print(f'Processed {total} examples')


def process_eval_data(data, output_file):
    prefix_text = []
    text = []
    total = 0
    for example in tqdm(data):
        filtered_keys = [
            key for key in example.keys()
            if key.startswith('human') or key.startswith('gpt')
        ]
        keys = sorted(filtered_keys, key=lambda x: int(x.split('_')[-1]))
        current_prefix = FLAGS.beginning_marker
        for key in keys:
            if key.startswith('gpt'):
                target_text = example[key] + FLAGS.eos_maker
                prefix_text.append(current_prefix)
                text.append(target_text)
                current_prefix += ' ' + target_text
                total += 1
            elif key.startswith('human'):
                current_prefix += (
                    ' ' + FLAGS.user_marker + ' ' + example[key] + ' '
                    + FLAGS.gpt_marker
                )
            else:
                raise ValueError(f'Unknown key: {key}')

    with open(output_file, 'w') as fout:
        json.dump({'prefix': prefix_text, 'text': text}, fout)

    print(f'Processed {total} examples')

def main(argv):
    with open(FLAGS.input_file) as fin:
        data = json.load(fin)

    process_data(data['train'], FLAGS.train_output_file)
    process_data(data['eval'], FLAGS.eval_output_file)
    process_eval_data(data['eval'], FLAGS.eval_json_output_file)




if __name__ == '__main__':
    mlxu.run(main)