import json
import os
from tqdm import tqdm
'''
maven数据集只有事件触发词标注，且测试集不含标签，需将预测结果上传提交至
https://codalab.lisn.upsaclay.fr/competitions/395#learn_the_details-submission-format
'''


def process(root_dir):
    dir = os.path.join(root_dir, "MAVEN Event Detection")
    train_file = open(os.path.join(dir, "train.jsonl"), encoding='utf-8')
    dev_file = open(os.path.join(dir, "valid.jsonl"), encoding='utf-8')
    test_file = open(os.path.join(dir, "test.jsonl"), encoding='utf-8')

    train_data = []
    train_lines = train_file.readlines()
    for line in tqdm(train_lines):
        data = json.loads(line.strip())
        data['text'] = " ".join([c['sentence'] for c in data['content']])
        data['events_word_type'] = [[event['mention'][0]['trigger_word'], event['type']] for event in data['events']]
        train_data.append(data)

    dev_data = []
    dev_lines = dev_file.readlines()
    for line in tqdm(dev_lines):
        data = json.loads(line.strip())
        data['text'] = " ".join([c['sentence'] for c in data['content']])
        data['events_word_type'] = [[event['mention'][0]['trigger_word'], event['type']] for event in data['events']]
        dev_data.append(data)

    test_data = []
    test_lines = test_file.readlines()
    for line in tqdm(test_lines):
        data = json.loads(line.strip())
        data['text'] = " ".join([c['sentence'] for c in data['content']])
        # 原数据集中不含标签，需要预测并提交结果到codalab
        # data['events_word_type'] = [[event['mention'][0]['trigger_word'], event['type']] for event in data['events']]
        test_data.append(data)
    return train_data, dev_data, test_data


if __name__ == "__main__":
    process()
