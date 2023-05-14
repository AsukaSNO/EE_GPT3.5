import argparse
import json
import re
import os
from collections import Counter, defaultdict
import scoring_utils as util
import numpy as np
from ChatGPT.logger import define_logger
from tqdm import tqdm
from sklearn.metrics import classification_report
from ChatGPT.chatgpt_api import judge_equal, judge_more_accurate
from ChatGPT.processors.maven_ere import process as process_maven_ere


class Scorer(object):
    def __init__(self, args):
        self.mention_index2id = {}
        self.mention_id2event_id = {}
        self.event_id2mentions_index = {}
        self.map = self.read_process_file()
        self.pred = self.read_pred_file(args.pred_file)
        self.dev_data = []
        for line in open("EE_datasets/MAVEN_ERE/valid.jsonl").readlines():
            self.dev_data.append(json.loads(line.strip()))
        self.dev_data_dict = {i['id']: i for i in self.dev_data}
        print(1)

    def read_process_file(self):
        _, dev_data, _ = process_maven_ere("EE_datasets")
        dev_data = dev_data[:355]
        file_dict = {i['id']: i['causal_relations_index'] for i in dev_data}
        self.mention_index2id = {i['id']: i['mention_index2id'] for i in dev_data}
        self.mention_id2event_id = {i['id']: i['mention_id2event_id'] for i in dev_data}
        self.event_id2mentions_index = {i['id']: i['event_id2mentions_index'] for i in dev_data}
        return file_dict

    def get_gold(self, doc_key):
        doc = Document(self.dev_data_dict[doc_key])
        return doc.labels

    def read_pred_file(self, file_path):
        def mention2event(doc_key, mention_s):
            mention_index = int(mention_s[1:])
            return mention_index
            # return self.event_id2mentions_index[doc_key][self.mention_id2event_id[doc_key][self.mention_index2id[doc_key][mention_index]]]

        def gen2output(doc_key, gen_text):
            cause_matches = re.findall(r'(e\d+)\s+caused\s+(e\d+)', gen_text)
            precondition_matches = re.findall(r'(e\d+)\s+is a precondition for\s+(e\d+)', gen_text)

            result = {
                'CAUSE': [(mention2event(doc_key, m[0]), mention2event(doc_key, m[1])) for m in cause_matches],
                'PRECONDITION': [(mention2event(doc_key, m[0]), mention2event(doc_key, m[1])) for m in precondition_matches]
            }
            return result

        def process_example(json_blob):
            doc_key = json_blob["doc_key"]
            gen_text = json_blob["A"]
            pred_casual_pairs = gen2output(doc_key, gen_text)
            return doc_key, pred_casual_pairs

        jsonlines = open(file_path, 'r').readlines()
        lines = [process_example(json.loads(line)) for line in jsonlines]
        pred_dict = {doc_key: pred_pairs for doc_key, pred_pairs in lines}
        return pred_dict

    def get_pred(self, doc_key):
        pair2rel = {}
        for (e1, e2) in self.pred[doc_key]['CAUSE']:
            pair2rel[(self.mention_index2id[doc_key][e1], self.mention_index2id[doc_key][e2])] = REL2ID['CAUSE']
        for (e1, e2) in self.pred[doc_key]['PRECONDITION']:
            pair2rel[(self.mention_index2id[doc_key][e1], self.mention_index2id[doc_key][e2])] = REL2ID['PRECONDITION']

        doc = Document(self.dev_data_dict[doc_key])
        labels = []
        for e1 in doc.events:
            for e2 in doc.events:
                if e1["id"] == e2["id"]:
                    continue
                else:
                    labels.append(pair2rel.get((e1["id"], e2["id"]), REL2ID["NONE"]))
        assert len(labels) == len(doc.events) ** 2 - len(doc.events)
        return labels

    def evaluate(self):
        REPORT_CLASS_NAMES = [ID2REL[i] for i in range(0, len(ID2REL) - 1)]
        REPORT_CLASS_LABELS = list(range(len(ID2REL) - 1))

        pred_list = []
        gold_list = []
        for doc_key in tqdm([i['id'] for i in self.dev_data[:355]]):
            gold = self.get_gold(doc_key)
            pred = self.get_pred(doc_key)
            assert len(gold) == len(pred)
            pred_list.extend(pred)
            gold_list.extend(gold)

        result_collection = classification_report(gold_list, pred_list, output_dict=True,
                                                  target_names=REPORT_CLASS_NAMES,
                                                  labels=REPORT_CLASS_LABELS)

        return result_collection


REL2ID = {
    "CAUSE": 0,
    "PRECONDITION": 1,
    "NONE": 2,
}

ID2REL = {v: k for k, v in REL2ID.items()}


def valid_split(point, spans):
    # retain context of at least 3 tokens
    for sp in spans:
        if point > sp[0] - 3 and point <= sp[1] + 3:
            return False
    return True


def split_spans(point, spans):
    part1 = []
    part2 = []
    i = 0
    for sp in spans:
        if sp[1] < point:
            part1.append(sp)
            i += 1
        else:
            break
    part2 = spans[i:]
    return part1, part2


class Document:
    def __init__(self, data):
        self.id = data["id"]
        self.words = data["tokens"]
        self.events = []
        self.eid2mentions = {}
        if "events" not in data:
            self.events = data["event_mentions"]
            self.relations = []
        else:
            for e in data["events"]:
                self.events += e["mention"]
                self.eid2mentions[e["id"]] = [m["id"] for m in e["mention"]]
            self.relations = data["causal_relations"]

        self.sort_events()
        self.get_labels()

    def sort_events(self):
        self.events = sorted(self.events, key=lambda x: (x["sent_id"], x["offset"][0]))
        self.sorted_event_spans = [(event["sent_id"], event["offset"]) for event in self.events]

    def get_labels(self):
        pair2rel = {}
        for rel in self.relations:
            for pair in self.relations[rel]:
                for e1 in self.eid2mentions[pair[0]]:
                    for e2 in self.eid2mentions[pair[1]]:
                        pair2rel[(e1, e2)] = REL2ID[rel]

        self.labels = []
        for e1 in self.events:
            for e2 in self.events:
                if e1["id"] == e2["id"]:
                    continue
                else:
                    self.labels.append(pair2rel.get((e1["id"], e2["id"]), REL2ID["NONE"]))
        assert len(self.labels) == len(self.events) ** 2 - len(self.events)




def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred_file', type=str, default="ChatGPT/output/maven_ere/causal/gen.jsonlines",
                        help='Predictions file path')
    parser.add_argument('--reuse_gold_format', dest='reuse_gold_format',
                        default=True, action='store_true',
                        help="Reuse gold file format for pred file.")
    parser.add_argument('-t', '--ontology_file', type=str, default=None,
                        help='Path to ontology file')
    parser.add_argument('-cd', '--type_constrained_decoding', dest="cd",
                        default=False, action='store_true',
                        help="Use type constrained decoding" +
                             '(only possible when ontology file is given')
    parser.add_argument('--metrics', dest='metrics', default=False,
                        action='store_true',
                        help="Compute overall p, r, f1.")
    parser.add_argument('--confusion', dest='confusion', default=False,
                        action='store_true',
                        help="Compute an error confusion matrix.")
    return parser.parse_args()


def run_evaluation(args):
    """This is a separate wrapper around args so that other programs
    can call evaluation without resorting to an os-level call
    """
    scorer = Scorer(args)
    output_file = open('ChatGPT/output/maven_ere/causal/eval_result.jsonlines', 'w', encoding='utf-8')
    result_collection = scorer.evaluate()
    json.dump(result_collection, output_file)

    return result_collection


if __name__ == "__main__":
    args = define_args()
    return_dict = run_evaluation(args)


