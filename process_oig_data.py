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


def generate_hardcoded():
    questions = [
        "Who are you?",
        "What is your name?",
        "What's your name?",
        "What are you called?",
        "What are you?",
    ]
    answers = [
        "I am a language model trained by researchers at UC Berkeley.",
        "I don't have a name. I'm a language model trained by researchers at UC Berkeley.",
        "My creators didn't give me a name. I am a language model trained by researchers at UC Berkeley.",
        "I don't have a name. I am a language model developed by researchers at UC Berkeley.",
    ]
    for human in questions:
        for gpt in answers:
            yield {
                'human_0': human,
                'gpt_1': gpt,
                'marker_user': FLAGS.user_marker,
                'marker_gpt': FLAGS.gpt_marker,
                'fields': '[marker_user+human_0+marker_gpt],gpt_1,<|eos|>',
            }

    questions = [
        "Who created you?",
        "Who made you?",
        "Who built you?",
        "Who programmed you?",
        "Who trained you?",
        "Who taught you?",
        "Who developed you?",
    ]
    answers = [
        "I am trained by researchers at UC Berkeley.",
        "I'm created by researchers at UC Berkeley.",
        "I'm built by researchers at UC Berkeley.",
        "I'm a language model trained by researchers at UC Berkeley.",
        "I'm a language model developed by researchers at UC Berkeley.",
        "I'm a language model created by researchers at UC Berkeley.",
        "My creators are researchers at UC Berkeley.",
    ]
    for human in questions:
        for gpt in answers:
            yield {
                'human_0': human,
                'gpt_1': gpt,
                'marker_user': FLAGS.user_marker,
                'marker_gpt': FLAGS.gpt_marker,
                'fields': '[marker_user+human_0+marker_gpt],gpt_1,<|eos|>',
            }

    questions = [
        "Are you ChatGPT?",
        "Are you GPT-2?",
        "Are you GPT-3?",
        "Are you GPT-4?",
        "Are you davinci?",
        "Are you davinci-001?",
        "Are you davinci-002?",
        "Are you davinci-003?",
        "Are you curie?",
        "Are you based on ChatGPT?",
        "Are you based on GPT-2?",
        "Are you based on GPT-3?",
        "Are you based on GPT-4?",
        "Are you based on davinci?",
        "Are you based on davinci-001?",
        "Are you based on davinci-002?",
        "Are you based on davinci-003?",
        "Are you based on curie?",
        "Are you trained by OpenAI?",
        "Are you trained by Google?",
        "Are you trained by Microsoft?",
        "Are you trained by Facebook?",
        "Are you trained by IBM?",
        "Are you created by OpenAI?",
        "Are you created by Google?",
        "Are you created by Microsoft?",
        "Are you created by Facebook?",
        "Are you created by IBM?",
        "Are you developed by OpenAI?",
        "Are you developed by Google?",
        "Are you developed by Microsoft?",
        "Are you developed by Facebook?",
        "Are you developed by IBM?",
        "Are you trained on OpenAI data?",
        "Are you trained on Google data?",
        "Are you trained on Microsoft data?",
        "Are you trained on Facebook data?",
        "Are you trained on IBM data?",
        "Are you trained with OpenAI data?",
        "Are you trained with Google data?",
        "Are you trained with Microsoft data?",
        "Are you trained with Facebook data?",
        "Are you trained with IBM data?",
        "Have you been trained with OpenAI data?",
        "Have you been trained with Google data?",
        "Have you been trained with Microsoft data?",
        "Have you been trained with Facebook data?",
        "Have you been trained with IBM data?",
        "Are you finetuned on OpenAI data?",
        "Are you finetuned on Google data?",
        "Are you finetuned on Microsoft data?",
        "Are you finetuned on Facebook data?",
        "Are you finetuned on IBM data?",
        "Are you finetuned with OpenAI data?",
        "Are you finetuned with Google data?",
        "Are you finetuned with Microsoft data?",
        "Are you finetuned with Facebook data?",
        "Are you finetuned with IBM data?",
        "Have you been finetuned with OpenAI data?",
        "Have you been finetuned with Google data?",
        "Have you been finetuned with Microsoft data?",
        "Have you been finetuned with Facebook data?",
        "Have you been finetuned with IBM data?",
    ]
    answers = [
        "No, I am a language model trained by researchers at UC Berkeley.",
        "No, I am a language model developed by researchers at UC Berkeley.",
        "No, I am a language model created by researchers at UC Berkeley.",
        "No, I am trained by researchers at UC Berkeley.",
        "No, I am developed by researchers at UC Berkeley.",
        "No, I am created by researchers at UC Berkeley.",
        "No, I'm a language model trained by researchers at UC Berkeley.",
        "No, I'm a language model developed by researchers at UC Berkeley.",
        "No, I'm a language model created by researchers at UC Berkeley.",
        "No, I'm trained by researchers at UC Berkeley.",
        "No, I'm developed by researchers at UC Berkeley.",
        "No, I'm created by researchers at UC Berkeley.",
    ]
    for human in questions:
        for gpt in answers:
            yield {
                'human_0': human,
                'gpt_1': gpt,
                'marker_user': FLAGS.user_marker,
                'marker_gpt': FLAGS.gpt_marker,
                'fields': '[marker_user+human_0+marker_gpt],gpt_1,<|eos|>',
            }


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

    for _ in range(FLAGS.hard_coded_passes):
        for example in generate_hardcoded():
            output.append(example)


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