# output_dir='/nfs/vault/data/language/humanfeedback',

'''
# 1. positive only
[USER: summarize this article <an article>]
[GPT/description of good:] <a good summary>

# 2. negative only
[USER: summarize this article <an article>]
[GPT-bad/descirption of bad:] <a bad summary>

# 3. negative-positive
[USER: summarize this article <an article>]
[GPT-bad/descirption of bad:] <a bad summary>
[GPT/description of good:] <a good summary>

# 4. positive-negative
[USER: summarize this article <an article>]
[GPT/description of good:] <a good summary>
[GPT-bad/descirption of bad:] <a bad summary>


{
    "human_0": "USER: Hello!" + "GPT: ",
    "gpt_1": "How are you?",
    "human_2": "USER: I'm fine, thanks. How are you?",
    "fields": "[human_0],[<|eos|>],gpt_1,<|eos|>,[human_2]",
}


{
    "marker_user": "<USER>",
    "marker_gpt": "<GPT>",
    "human_0": ...,
    "gpt_1": ...,
    "fields": "[marker_user+human_0+marker_gpt],gpt_1,<|eos|>"
}


'''

import json
from datasets import load_dataset
import os
import random
import numpy as np
import json
import re
from tqdm import tqdm, trange

import mlxu

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    output_dir='/nfs/vault/data/language/humanfeedback',
    dataset='dialogue,webgpt,summary', # dialogue, summary, webgpt
    positive_only=False, # `True` to use only positive examples, `False` to use all examples
    #
    gpt_marker='GPT:',
    gpt_bad_marker='An Unhelpful Answer:',
    user_marker='USER:',
    #
    user_field_id='marker_user',
    gpt_field_id='marker_gpt',
    gpt_bad_field_id='marker_gpt_bad',
    #
    user_data_id='human',
    gpt_data_id='gpt',
    gpt_bad_data_id='gpt_bad',
)


def process_dialogue(example):
    all_output = []

    def pack_data(data, is_positive, output, fields=[]):
        messages = data.split('Human:')[1:]
        final_output = []
        for message in messages:
            message = message.split('Assistant:')
            try:
                final_output.append(message[0].strip())
                final_output.append(message[1].strip())
            except:
                final_output.append(message[0].strip())
        for i, message in enumerate(final_output):
            if i % 2 == 0:
                # Human
                user_id = '{}_{}'.format(FLAGS.user_data_id, i)
                output[user_id] = f"{message}"
            else:
                # Chatbot
                if is_positive:
                    output['{}_{}'.format(FLAGS.gpt_data_id, i)] = f"{message}"
                    fields.append('[{}+{}+{}]'.format(FLAGS.user_field_id, user_id, FLAGS.gpt_field_id))
                    fields.append('{}_{}'.format(FLAGS.gpt_data_id, i))
                else:
                    output['{}_{}'.format(FLAGS.gpt_bad_data_id, i)] = f"{message}"
                    fields.append('[{}+{}+{}]'.format(FLAGS.user_field_id, user_id, FLAGS.gpt_bad_field_id))
                    fields.append('{}_{}'.format(FLAGS.gpt_bad_data_id, i))
                fields.append('<|eos|>')
        return output, fields

    # 1. positive only
    if FLAGS.positive_only:
        data = example['chosen']
        output = {
            "marker_user": f"{FLAGS.user_marker}",
            "marker_gpt": f"{FLAGS.gpt_marker}",
            "marker_gpt_bad": f"{FLAGS.gpt_bad_marker}",
        }
        fields = []
        output, fields = pack_data(data, is_positive=True, output=output, fields=fields)
        output['fields'] = ','.join(fields)
        all_output.append(output)

    else:
        # 1. positive only
        data = example['chosen']
        output = {
            "marker_user": f"{FLAGS.user_marker}",
            "marker_gpt": f"{FLAGS.gpt_marker}",
            "marker_gpt_bad": f"{FLAGS.gpt_bad_marker}",
        }
        fields = []
        output, fields = pack_data(data, is_positive=True, output=output, fields=fields)
        output['fields'] = ','.join(fields)
        all_output.append(output)

        # 2. negative only
        data = example['rejected']
        output = {
            "marker_user": f"{FLAGS.user_marker}",
            "marker_gpt": f"{FLAGS.gpt_marker}",
            "marker_gpt_bad": f"{FLAGS.gpt_bad_marker}",
        }
        fields = []
        output, fields = pack_data(data, is_positive=False, output=output, fields=fields)
        output['fields'] = ','.join(fields)
        all_output.append(output)

        # # 3. negative-positive
        # # negative
        # output = {
        #     "marker_user": f"{FLAGS.user_marker}",
        #     "marker_gpt": f"{FLAGS.gpt_marker}",
        #     "marker_gpt_bad": f"{FLAGS.gpt_bad_marker}",
        # }
        # fields = []
        # data = example['rejected']
        # output, fields = pack_data(data, is_positive=False, output=output, fields=fields)
        # # positive
        # data = example['chosen']
        # output, fields = pack_data(data, is_positive=True, output=output, fields=fields)
        # # combine
        # output['fields'] = ','.join(fields)
        # all_output.append(output)

        # # 4. positive-negative
        # # positive
        # output = {
        #     "marker_user": f"{FLAGS.user_marker}",
        #     "marker_gpt": f"{FLAGS.gpt_marker}",
        #     "marker_gpt_bad": f"{FLAGS.gpt_bad_marker}",
        # }
        # fields = []
        # data = example['chosen']
        # output, fields = pack_data(data, is_positive=True, output=output, fields=fields)
        # # negative
        # data = example['rejected']
        # output, fields = pack_data(data, is_positive=False, output=output, fields=fields)
        # # combine
        # output['fields'] = ','.join(fields)
        # all_output.append(output)

    return all_output


def process_summary(example):
    all_output = []

    pos_idx = int(example['choice'])
    neg_idx = 1 - pos_idx

    def pack_data(idx, is_positive, output, fields, use_user_input=True):
        if use_user_input:
            output['{}_{}'.format(FLAGS.user_data_id, idx)] = random.choice(SUMMARY['task']) + f"{example['info']['post']}"
            user_input_fields = '{}+{}_{}+'.format(FLAGS.user_field_id, FLAGS.user_data_id, idx)
        else:
            user_input_fields = ''
        answer = example['summaries'][idx]['text'].lstrip()
        if is_positive:
            output['{}_{}'.format(FLAGS.gpt_data_id, idx)] = f"{answer}"
            fields.append('[{}{}]'.format(user_input_fields, FLAGS.gpt_field_id))
            fields.append('{}_{}'.format(FLAGS.gpt_data_id, idx))
        else:
            output['{}_{}'.format(FLAGS.gpt_bad_data_id, idx)] = f"{answer}"
            fields.append('[{}{}]'.format(user_input_fields, FLAGS.gpt_bad_field_id))
            fields.append('{}_{}'.format(FLAGS.gpt_bad_data_id, idx))
        fields.append('<|eos|>')

        return output, fields

    # 1. positive only
    if FLAGS.positive_only:
        output = {
            "marker_user": f"{FLAGS.user_marker}",
            "marker_gpt": f"{FLAGS.gpt_marker}",
            "marker_gpt_bad": f"{FLAGS.gpt_bad_marker}",
        }
        fields = []
        output, fields = pack_data(pos_idx, is_positive=True, fields=fields, output=output)
        output['fields'] = ','.join(fields)
        all_output.append(output)

    # # 2. negative only
    # output = {
    #     "marker_user": f"{FLAGS.user_marker}",
    #     "marker_gpt": f"{FLAGS.gpt_marker}",
    #     "marker_gpt_bad": f"{FLAGS.gpt_bad_marker}",
    # }
    # fields = []
    # output, fields = pack_data(neg_idx, is_positive=False, fields=fields, output=output)
    # output['fields'] = ','.join(fields)
    # all_output.append(output)

    else:

        # 1. positive only
        output = {
            "marker_user": f"{FLAGS.user_marker}",
            "marker_gpt": f"{FLAGS.gpt_marker}",
            "marker_gpt_bad": f"{FLAGS.gpt_bad_marker}",
        }
        fields = []
        output, fields = pack_data(pos_idx, is_positive=True, fields=fields, output=output)
        output['fields'] = ','.join(fields)
        all_output.append(output)

        # 2. negative only
        output = {
            "marker_user": f"{FLAGS.user_marker}",
            "marker_gpt": f"{FLAGS.gpt_marker}",
            "marker_gpt_bad": f"{FLAGS.gpt_bad_marker}",
        }
        fields = []
        output, fields = pack_data(neg_idx, is_positive=False, fields=fields, output=output)
        output['fields'] = ','.join(fields)
        all_output.append(output)

        # # 3. positive-negative
        # # positive
        # output = {
        #     "marker_user": f"{FLAGS.user_marker}",
        #     "marker_gpt": f"{FLAGS.gpt_marker}",
        #     "marker_gpt_bad": f"{FLAGS.gpt_bad_marker}",
        # }
        # fields = []
        # output, fields = pack_data(pos_idx, is_positive=True, output=output, fields=fields)
        # # negative
        # output, fields = pack_data(neg_idx, is_positive=False, output=output, fields=fields, use_user_input=False)
        # # combine
        # output['fields'] = ','.join(fields)
        # all_output.append(output)

        # # 3. negative-positive
        # # negative
        # output = {
        #     "marker_user": f"{FLAGS.user_marker}",
        #     "marker_gpt": f"{FLAGS.gpt_marker}",
        #     "marker_gpt_bad": f"{FLAGS.gpt_bad_marker}",
        # }
        # fields = []
        # output, fields = pack_data(neg_idx, is_positive=False, output=output, fields=fields)
        # # positive
        # output, fields = pack_data(pos_idx, is_positive=True, output=output, fields=fields, use_user_input=False)
        # # combine
        # output['fields'] = ','.join(fields)
        # all_output.append(output)

    return all_output


def process_webgpt(example):
    all_output = []

    if example['score_0'] > example['score_1']:
        pos_idx = 0
        neg_idx = 1
    elif example['score_0'] < example['score_1']:
        pos_idx = 1
        neg_idx = 0
    else:
        return None # skip tie cases because don't know they are both good or both bad

    def pack_data(idx, is_positive, output, fields, use_user_input=True):
        if use_user_input:
            question = example['question']['full_text']
            output['{}_{}'.format(FLAGS.user_data_id, idx)] = f"{question}"
            user_input_fields = '{}+{}_{}'.format(FLAGS.user_field_id, FLAGS.user_data_id, idx)
        else:
            user_input_fields = ''
        if len(example[f'quotes_{idx}']['title']) > 0:
            output['quote_{}'.format(idx)] = random.choice(WEBGPT['task'])
            for i, x in enumerate(zip(example[f'quotes_{idx}']['title'], example[f'quotes_{idx}']['extract'])):
                output['quote_{}'.format(idx)] += f"{[i + 1]} title: {x[0]} content: {x[1]} "
            if user_input_fields != '':
                user_input_fields += '+'
            user_input_fields += '{}'.format('quote_{}'.format(idx))

        if user_input_fields != '':
            user_input_fields += '+'

        answer = example[f'answer_{idx}']
        if is_positive:
            output['{}_{}'.format(FLAGS.gpt_data_id, idx)] = f"{answer}"
            fields.append('[{}{}]'.format(user_input_fields, FLAGS.gpt_field_id))
            fields.append('{}_{}'.format(FLAGS.gpt_data_id, idx))
        else:
            output['{}_{}'.format(FLAGS.gpt_bad_data_id, idx)] = f"{answer}"
            fields.append('[{}{}]'.format(user_input_fields, FLAGS.gpt_bad_field_id))
            fields.append('{}_{}'.format(FLAGS.gpt_bad_data_id, idx))
        fields.append('<|eos|>')

        return output, fields

    # 1. positive only
    if FLAGS.positive_only:
        output = {
            "marker_user": f"{FLAGS.user_marker}",
            "marker_gpt": f"{FLAGS.gpt_marker}",
            "marker_gpt_bad": f"{FLAGS.gpt_bad_marker}",
        }
        fields = []
        output, fields = pack_data(pos_idx, is_positive=True, output=output, fields=fields)
        output['fields'] = ','.join(fields)
        all_output.append(output)
    else:
        # 1. positive only
        output = {
            "marker_user": f"{FLAGS.user_marker}",
            "marker_gpt": f"{FLAGS.gpt_marker}",
            "marker_gpt_bad": f"{FLAGS.gpt_bad_marker}",
        }
        fields = []
        output, fields = pack_data(pos_idx, is_positive=True, output=output, fields=fields)
        output['fields'] = ','.join(fields)
        all_output.append(output)

        # 2. negative only
        output = {
            "marker_user": f"{FLAGS.user_marker}",
            "marker_gpt": f"{FLAGS.gpt_marker}",
            "marker_gpt_bad": f"{FLAGS.gpt_bad_marker}",
        }
        fields = []
        output, fields = pack_data(neg_idx, is_positive=False, output=output, fields=fields)
        output['fields'] = ','.join(fields)
        all_output.append(output)

        # # 3. positive-negative
        # # positive
        # output = {
        #     "marker_user": f"{FLAGS.user_marker}",
        #     "marker_gpt": f"{FLAGS.gpt_marker}",
        #     "marker_gpt_bad": f"{FLAGS.gpt_bad_marker}",
        # }
        # fields = []
        # output, fields = pack_data(pos_idx, is_positive=True, output=output, fields=fields)
        # # negative
        # output, fields = pack_data(neg_idx, is_positive=False, output=output, fields=fields, use_user_input=False)
        # # combine
        # output['fields'] = ','.join(fields)
        # all_output.append(output)

        # # 3. negative-positive
        # # negative
        # output = {
        #     "marker_user": f"{FLAGS.user_marker}",
        #     "marker_gpt": f"{FLAGS.gpt_marker}",
        #     "marker_gpt_bad": f"{FLAGS.gpt_bad_marker}",
        # }
        # fields = []
        # output, fields = pack_data(neg_idx, is_positive=False, output=output, fields=fields)
        # # positive
        # output, fields = pack_data(pos_idx, is_positive=True, output=output, fields=fields, use_user_input=False)
        # # combine
        # output['fields'] = ','.join(fields)
        # all_output.append(output)

    return all_output


DIALOGUE = {
    'good': [
        '', # empty
        'The following is a better response.',
        'The following is a more helpful response.',
        'The following is a more harmless response.',
        'The following is a better chat.',
        'The following is a more helpful chat.',
        'The following is a more harmless chat.',
        'Generate a better response.',
        'Generate a more helpful response.',
        'Generate a more harmless response.',
        'Generate a better chat.',
        'Generate a more helpful chat.',
        'Generate a more harmless chat.',
    ],
    'bad': [
        '', # empty
        'The following is a worse response.',
        'The following is a less helpful response.',
        'The following is a more harmful response.',
        'The following is a worse chat.',
        'The following is a less helpful chat.',
        'The following is a more harmful chat.',
        'Generate a worse response.',
        'Generate a less helpful response.',
        'Generate a more harmful response.',
        'Generate a worse chat.',
        'Generate a less helpful chat.',
        'Generate a more harmful chat.',
    ]
}

WEBGPT = {
    'good': [
        '', # empty
        'The following is a better answer.',
        'The following is a more accurate answer.',
        'The following is a more correct answer.',
        'Generate a better answer.',
        'Generate a more accurate answer.',
        'Generate a more correct answer.',
    ],
    'bad': [
        '', # empty
        'The following is a worse answer.',
        'The following is a less accurate answer.',
        'The following is a less correct answer.',
        'Generate a worse answer.',
        'Generate a less accurate answer.',
        'Generate a less correct answer.',
    ],
    'task': [
        'Quote the following sources in your answer: ',
        'Use the following sources in your answer: ',
        'You may use the following sources in your answer: ',
        'You may quote the following sources in your answer: ',
        'The following sources may be helpful: ',
    ]
}

SUMMARY = {
    'good': [
        '', # empty
        'The following is a better summary.',
        'The following is a more accurate summary.',
        'The following is a more correct summary.',
        'Generate a better summary.',
        'Generate a more accurate summary.',
        'Generate a more correct summary.',
    ],
    'bad': [
        '', # empty
        'The following is a worse summary.',
        'The following is a less accurate summary.',
        'The following is a less correct summary.',
        'Generate a worse summary.',
        'Generate a less accurate summary.',
        'Generate a less correct summary.',
    ],
    'task': [
        'Write a summary of the following article: ',
        'Generate a summary of the following text: ',
        'Summarize the following article: ',
        'Summarize the following text: ',
        'Write a summary of the following text: ',
        'Here is a news article, please summarize it: ',
        'Please write a summary of the following paragraph: ',
        'Please summarize the following paragraph: ',
    ]
}



def main(argv):
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    os.chdir(FLAGS.output_dir)

    for dataset in FLAGS.dataset.split(','):
        if dataset == 'dialogue': #https://huggingface.co/datasets/Anthropic/hh-rlhf
            train_data = load_dataset('Anthropic/hh-rlhf', split='train')
            if not FLAGS.positive_only:
                train_output_file = "./hh_dialogue_train.jsonl"
            else:
                train_output_file = "./hh_dialogue_positive_only_train.jsonl"
            with open(train_output_file, 'w') as fout:
                for example in train_data:
                    all_output = process_dialogue(example)
                    for output in all_output:
                        fout.write(json.dumps(output) + '\n')

            eval_data = load_dataset('Anthropic/hh-rlhf', split='test')
            if not FLAGS.positive_only:
                eval_output_file = "./hh_dialogue_eval.jsonl"
            else:
                eval_output_file = "./hh_dialogue_positive_only_eval.jsonl"
            with open(eval_output_file, 'w') as fout:
                for example in eval_data:
                    all_output = process_dialogue(example)
                    for output in all_output:
                        fout.write(json.dumps(output) + '\n')
        elif dataset == 'webgpt': #https://huggingface.co/datasets/openai/webgpt_comparisons
            # doesn't have test set
            train_data = load_dataset('openai/webgpt_comparisons', split='train')
            if not FLAGS.positive_only:
                train_output_file = "webgpt_train.jsonl"
            else:
                train_output_file = "webgpt_positive_only_train.jsonl"
            with open(train_output_file, 'w') as fout:
                for example in train_data:
                    all_output = process_webgpt(example)
                    if all_output is None:
                        continue
                    for output in all_output:
                        fout.write(json.dumps(output) + '\n')
        elif dataset == 'summary': #https://huggingface.co/datasets/openai/summarize_from_feedback
            train_data = load_dataset('openai/summarize_from_feedback', 'comparisons', split='train')
            if not FLAGS.positive_only:
                train_output_file = "summary_train.jsonl"
            else:
                train_output_file = "summary_positive_only_train.jsonl"
            with open(train_output_file, 'w') as fout:
                for example in train_data:
                    all_output = process_summary(example)
                    for output in all_output:
                        fout.write(json.dumps(output) + '\n')

            eval_data = load_dataset('openai/summarize_from_feedback', 'comparisons', split='validation')
            if not FLAGS.positive_only:
                eval_output_file = "summary_eval.jsonl"
            else:
                eval_output_file = "summary_positive_only_eval.jsonl"
            with open(eval_output_file, 'w') as fout:
                for example in eval_data:
                    all_output = process_summary(example)
                    for output in all_output:
                        fout.write(json.dumps(output) + '\n')
        else:
            raise NotImplementedError

if __name__ == '__main__':
    mlxu.run(main)
