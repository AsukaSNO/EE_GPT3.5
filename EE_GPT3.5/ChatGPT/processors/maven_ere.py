import json
import os
from tqdm import tqdm
from copy import deepcopy
'''
maven-ere test集不含标签，为了便于比较，选择dev集进行对比
'''


def process(root_dir):
    dir = os.path.join(root_dir, "MAVEN_ERE")
    train_file = open(os.path.join(dir, "train.jsonl"), encoding='utf-8')
    dev_file = open(os.path.join(dir, "valid.jsonl"), encoding='utf-8')
    test_file = open(os.path.join(dir, "test.jsonl"), encoding='utf-8')

    train_data = []
    train_lines = train_file.readlines()
    for line in tqdm(train_lines):
        train_data.append(process_instance(line))

    dev_data = []
    dev_lines = dev_file.readlines()
    for line in tqdm(dev_lines):
        dev_data.append(process_instance(line))

    test_data = []

    return train_data, dev_data, test_data


def process_instance(line):
    data = json.loads(line.strip())
    data['text'] = " ".join(data['sentences'])
    data['coref_relations'] = [e['id'] for e in data['events'] if len(e['mention']) > 1]


    data['mentions'] = []
    data['event_id2mentions_id'] = {}
    for e in data['events']:
        data['mentions'] += e["mention"]
        data['event_id2mentions_id'][e["id"]] = [m["id"] for m in e["mention"]]
    data['time_id2index'] = {}
    data['time_id2word'] = {}
    data['time_index2word'] = {}
    data['time_index2id'] = {}
    for i in range(len(data['TIMEX'])):
        data['time_id2index'][data['TIMEX'][i]["id"]] = i
        data['time_id2word'][data['TIMEX'][i]["id"]] = data['TIMEX'][i]["mention"]
        data['time_index2word'][i] = data['TIMEX'][i]["mention"]
        data['time_index2id'][i] = data['TIMEX'][i]["id"]
    data['mentions'] = sorted(data['mentions'], key=lambda x: (x["sent_id"], x["offset"][0]))
    data['mention_index2id'] = {}
    data['mention_index2word'] = {}
    for index, mention in enumerate(data['mentions']):
        data['mention_index2id'][index] = mention['id']
        data['mention_index2word'][index] = mention['trigger_word']
    data['mention_id2index'] = {v: k for k, v in data['mention_index2id'].items()}
    data['event_id2mentions_index'] = {k: [data['mention_id2index'][i] for i in v] for k, v in data['event_id2mentions_id'].items()}
    data['mention_id2event_id'] = {mid: eid for eid, mids in data['event_id2mentions_id'].items() for mid in mids}

    sorted_mentions_span = [(event["sent_id"], event["offset"]) for event in data['mentions']]
    events_nums = len(sorted_mentions_span) - 1
    words = data["tokens"]
    # 在文本中添加tag<e{i}></e{i}>
    for sent_id, offset in reversed(sorted_mentions_span):
        words[sent_id][offset[0]] = f"<e{events_nums}>" + words[sent_id][offset[0]]
        words[sent_id][offset[1] - 1] += f"</e{events_nums}>"
        events_nums -= 1
    data['text_with_tag'] = (' '.join([' '.join(sentence) for sentence in words])).lower()
    # 在末尾注明事件
    annotation = [f'e{k}: {v}' for k, v in data['mention_index2word'].items()]
    annotation_t = [f't{k}: {v}' for k, v in data['time_index2word'].items()] + [f'e{k}: {v}' for k, v in data['mention_index2word'].items()]
    data['text_with_annotation'] = 'text: ' + data['text'] + '\nevents: ' + ', '.join(annotation)  # 事件在最后提及
    data['text_with_annotation_t'] = 'text: ' + data['text'] + '\nevents: ' + ', '.join(annotation_t)
    # data['coref_relations_index']、data['causal_relations_index']、data['temporal_relations_index']、data['subevent_relations_index']为之后用到的golden label
    # coref
    data['coref_relations_index'] = [data['event_id2mentions_index'][i] for i in data['coref_relations']]
    # causal
    data['causal_relations_index'] = {}
    data['causal_relations_index']['CAUSE'] = [[data['event_id2mentions_index'][i[0]], data['event_id2mentions_index'][i[1]]] for i in data['causal_relations']['CAUSE']]
    data['causal_relations_index']['PRECONDITION'] = [[data['event_id2mentions_index'][i[0]], data['event_id2mentions_index'][i[1]]] for i in data['causal_relations']['PRECONDITION']]  # 把id转为index，以便后续evaluate
    # temporal
    # 注意：e0和t0的区分，e0是(list)[0]，t0是(int)0
    data['temporal_relations_index'] = {}
    data['temporal_relations_index']['BEFORE'] = [[data['event_id2mentions_index'].get(i[0], data['time_id2index'].get(i[0])), data['event_id2mentions_index'].get(i[1], data['time_id2index'].get(i[1]))] for i in data['temporal_relations']['BEFORE']]
    data['temporal_relations_index']['OVERLAP'] = [[data['event_id2mentions_index'].get(i[0], data['time_id2index'].get(i[0])), data['event_id2mentions_index'].get(i[1], data['time_id2index'].get(i[1]))] for i in data['temporal_relations']['OVERLAP']]
    data['temporal_relations_index']['CONTAINS'] = [[data['event_id2mentions_index'].get(i[0], data['time_id2index'].get(i[0])), data['event_id2mentions_index'].get(i[1], data['time_id2index'].get(i[1]))] for i in data['temporal_relations']['CONTAINS']]
    data['temporal_relations_index']['SIMULTANEOUS'] = [[data['event_id2mentions_index'].get(i[0], data['time_id2index'].get(i[0])), data['event_id2mentions_index'].get(i[1], data['time_id2index'].get(i[1]))] for i in data['temporal_relations']['SIMULTANEOUS']]
    data['temporal_relations_index']['ENDS-ON'] = [[data['event_id2mentions_index'].get(i[0], data['time_id2index'].get(i[0])), data['event_id2mentions_index'].get(i[1], data['time_id2index'].get(i[1]))] for i in data['temporal_relations']['ENDS-ON']]
    data['temporal_relations_index']['BEGINS-ON'] = [[data['event_id2mentions_index'].get(i[0], data['time_id2index'].get(i[0])), data['event_id2mentions_index'].get(i[1], data['time_id2index'].get(i[1]))] for i in data['temporal_relations']['BEGINS-ON']]
    # subevent
    data['subevent_relations_index'] = [[data['event_id2mentions_index'][i[0]], data['event_id2mentions_index'][i[1]]] for i in data['subevent_relations']]
    return data


if __name__ == "__main__":
    process()

