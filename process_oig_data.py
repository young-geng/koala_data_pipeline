import os
import re
import random
import numpy as np
import json
import re
from tqdm import tqdm, trange
import requests
from bs4 import BeautifulSoup

import mlxu


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    input_dir='/nfs/vault/data/language/OIG',
    train_output_file='/nfs/vault/data/language/chat/oig_train.jsonl',
    eval_output_file='/nfs/vault/data/language/chat/oig_eval.jsonl',
    gpt_marker='GPT:',
    user_marker='USER:',
    train_ratio=0.95,
    hard_coded_passes=5,
)


def split_human_gpt(text):
    splits = re.split(r'(<human>:|<bot>:)', text)[1:]
    assert len(splits) % 4 == 0
    output = []
    count = 0
    for i in range(0, len(splits), 2):
        side, msg = splits[i], splits[i + 1].strip()
        if side == '<human>:':
            output.append((f'human_{count}', msg))
        elif side == '<bot>:':
            output.append((f'gpt_{count}', msg))
        else:
            raise ValueError(f'Unknown side: {side}')
        count += 1

    return output


def process_basic(example):
    human, gpt = example['text'].split('<bot>: ')
    human = human.strip()
    assert human.startswith('<human>: ')
    human = human[len('<human>: '):]
    gpt = gpt.strip()
    gpt.replace(
        'OIG (Open Instruction Generalist)',
        'a language model trained by researchers at UC Berkeley'
    )
    return {
        'human_0': human,
        'gpt_1': gpt,
        'marker_user': FLAGS.user_marker,
        'marker_gpt': FLAGS.gpt_marker,
        'fields': '[marker_user+human_0+marker_gpt],gpt_1,<|eos|>',
    }


def process_general(example):
    rounds = split_human_gpt(example['text'])
    assert len(rounds) % 2 == 0
    fields = []
    output_example = {
        'marker_user': FLAGS.user_marker,
        'marker_gpt': FLAGS.gpt_marker,
    }
    for key, msg in rounds:
        if key.startswith('human'):
            output_example[key] = msg
            fields.append(f'[marker_user+{key}+marker_gpt]')
        elif key.startswith('gpt'):
            output_example[key] = msg
            fields.append(f'{key}')
            fields.append('<|eos|>')
        else:
            raise ValueError(f'Unknown key: {key}')

    output_example['fields'] = ','.join(fields)
    return output_example


def main(argv):
    files = {
        'unified_basic.jsonl': process_basic,
        'unified_grade_school_math_instructions.jsonl': process_general,
        'unified_plot_screenplay_books_dialog.jsonl': process_general,
        'unified_poetry_2_song.jsonl': process_general,
        'unified_hc3_human.jsonl': process_general,
    }

    output = []

    for filename, process_fn in files.items():
        with open(os.path.join(FLAGS.input_dir, filename)) as fin:
            for line in tqdm(fin):
                processed_example = process_fn(json.loads(line))
                if processed_example is not None:
                    output.append(processed_example)

    random.shuffle(output)
    train_output = output[:int(len(output) * FLAGS.train_ratio)]
    eval_output = output[int(len(output) * FLAGS.train_ratio):]

    with open(FLAGS.train_output_file, 'w') as fout:
        for example in train_output:
            fout.write(json.dumps(example) + '\n')

    with open(FLAGS.eval_output_file, 'w') as fout:
        for example in eval_output:
            fout.write(json.dumps(example) + '\n')




if __name__ == '__main__':
    mlxu.run(main)