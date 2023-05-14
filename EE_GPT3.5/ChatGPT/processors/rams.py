import json
import os
from tqdm import tqdm
import numpy as np


def process(root_dir):
    dir = os.path.join(root_dir, "RAMS_1.0c/data")
    train_file = open(os.path.join(dir, "train.jsonlines"))
    dev_file = open(os.path.join(dir, "dev.jsonlines"))
    test_file = open(os.path.join(dir, "test.jsonlines"))

    train_data = []
    train_lines = train_file.readlines()
    for line in tqdm(train_lines):
        data = json.loads(line.strip())
        data['text'] = ' '.join([' '.join(sentence) for sentence in data['sentences']])
        data['tokens'] = np.concatenate(data['sentences'])
        data['events_word_type'] = [[' '.join(data['tokens'][event[0]: event[1] + 1]), event[2][0][0]] for event in
                                    data['evt_triggers']]
        data['events_argument_role'] = [[[' '.join(data['tokens'][i[1][0]: i[1][1] + 1]), i[2]] for i in data['gold_evt_links']]]
        train_data.append(data)

    dev_data = []
    dev_lines = dev_file.readlines()
    for line in tqdm(dev_lines):
        data = json.loads(line.strip())
        data['text'] = ' '.join([' '.join(sentence) for sentence in data['sentences']])
        data['tokens'] = np.concatenate(data['sentences'])
        data['events_word_type'] = [[' '.join(data['tokens'][event[0]: event[1] + 1]), event[2][0][0]] for event in
                                    data['evt_triggers']]
        data['events_argument_role'] = [[[' '.join(data['tokens'][i[1][0]: i[1][1] + 1]), i[2]] for i in data['gold_evt_links']]]
        dev_data.append(data)

    test_data = []
    test_lines = test_file.readlines()
    for line in tqdm(test_lines):
        data = json.loads(line.strip())
        data['text'] = ' '.join([' '.join(sentence) for sentence in data['sentences']])
        data['tokens'] = np.concatenate(data['sentences'])
        data['events_word_type'] = [[' '.join(data['tokens'][event[0]: event[1] + 1]), event[2][0][0]] for event in
                                    data['evt_triggers']]
        data['events_argument_role'] = [[[' '.join(data['tokens'][i[1][0]: i[1][1] + 1]), i[2]] for i in data['gold_evt_links']]]
        test_data.append(data)
    # aaa = [(i['doc_key'], i['text'], i['events_word_type']) for i in test_data]
    # fff = open('ChatGPT/output/rams/temp.txt', 'w', encoding='utf-8')
    # fff.write("\n".join(str(i) + '\n' + str(j) + '\n' + str(k) for i, j, k in aaa))
    return train_data, dev_data, test_data


if __name__ == "__main__":
    process()
