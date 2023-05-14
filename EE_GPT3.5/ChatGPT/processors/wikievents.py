import json
import os
from tqdm import tqdm
import numpy as np


def process(root_dir):
    dir = os.path.join(root_dir, "wikievents")
    train_file = open(os.path.join(dir, "train.jsonl"))
    dev_file = open(os.path.join(dir, "dev.jsonl"))
    test_file = open(os.path.join(dir, "test.jsonl"))

    train_data = []
    train_lines = train_file.readlines()
    for line in tqdm(train_lines):
        data = json.loads(line.strip())
        data['events_word_type'] = [[event['trigger']['text'], event['event_type']] for event in data['event_mentions']]
        data['events_argument_role'] = [event['arguments'] for event in data['event_mentions']]
        data['events_argument_role'] = [[[j['text'], j['role']] for j in i] if len(i)>0 else i for i in data['events_argument_role']]

        train_data.append(data)

    dev_data = []
    dev_lines = dev_file.readlines()
    for line in tqdm(dev_lines):
        data = json.loads(line.strip())
        data['events_word_type'] = [[event['trigger']['text'], event['event_type']] for event in data['event_mentions']]
        data['events_argument_role'] = [event['arguments'] for event in data['event_mentions']]
        data['events_argument_role'] = [[[j['text'], j['role']] for j in i] if len(i) > 0 else i for i in
                                        data['events_argument_role']]

        dev_data.append(data)

    test_data = []
    test_lines = test_file.readlines()
    for line in tqdm(test_lines):
        data = json.loads(line.strip())
        data['events_word_type'] = [[event['trigger']['text'], event['event_type']] for event in
                                    data['event_mentions']]
        data['events_argument_role'] = [event['arguments'] for event in data['event_mentions']]
        data['events_argument_role'] = [[[j['text'], j['role']] for j in i] if len(i) > 0 else i for i in
                                        data['events_argument_role']]

        test_data.append(data)

    return train_data, dev_data, test_data


if __name__ == "__main__":
    process()
