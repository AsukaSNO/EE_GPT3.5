import argparse
import json
import re
from collections import Counter, defaultdict
import scoring_utils as util
import numpy as np
from ChatGPT.logger import define_logger
from tqdm import tqdm
from ChatGPT.chatgpt_api import judge_equal, judge_more_accurate


class Scorer(object):
    def __init__(self, args):
        self.role_string_mapping = {}
        self.roles = set()
        self.gold = self.read_gold_file(args.gold_file)
        self.pred = self.read_pred_file(args.pred_file)

    def get_role_label(self, role):
        if role in self.role_string_mapping:
            return self.role_string_mapping[role]
        else:
            # Each role is of the form evt###arg##role, we only want role
            role_string = re.split(r'\d+', role)[-1]
            assert (role_string == role[11:])

            self.role_string_mapping[role] = role_string
            self.roles.add(role_string)
            return role_string

    def read_gold_file(self, file_path, confidence=False):
        """
        Returns dict mapping doc_key -> (pred, arg, role)
        """

        def process_example(json_blob):
            doc_key = json_blob["doc_key"]
            gold_evt = json_blob["gold_evt_links"]

            # There should only be one predicate
            json_blob['tokens'] = np.concatenate(json_blob['sentences'])
            evt_trigger = ' '.join(
                json_blob['tokens'][json_blob["evt_triggers"][0][0]: json_blob["evt_triggers"][0][1] + 1])

            # only used to observe event type, not for evaluation
            evt_trigger_types = json_blob["evt_triggers"][0][2][0][0]

            # argument
            gold_evt_links = [(' '.join(json_blob['tokens'][arg[1][0]: arg[1][1] + 1]), self.get_role_label(arg[2]))
                              for arg in json_blob['gold_evt_links']]

            return doc_key, gold_evt_links, evt_trigger_types

        jsonlines = open(file_path, 'r').readlines()
        lines = [process_example(json.loads(line)) for line in jsonlines]
        file_dict = {doc_key: (evt_links, evt_trigger_types) for doc_key, evt_links, evt_trigger_types in lines}
        return file_dict

    def read_pred_file(self, file_path):
        def gen2output(gen_text):
            try:
                gen_list = re.findall(r'(\w+):\s*([^\n]+)', gen_text)
                gen_list = [(arg[1], arg[0].lower()) for arg in gen_list]
            except:
                pass
            return gen_list

        def process_example(json_blob):
            doc_key = json_blob["doc_key"]
            gen_text = json_blob["A"]
            pred_evt_links = gen2output(gen_text)
            return doc_key, pred_evt_links

        jsonlines = open(file_path, 'r').readlines()
        lines = [process_example(json.loads(line)) for line in jsonlines]
        file_dict = {doc_key: evt_links for doc_key, evt_links, in lines}
        return file_dict

    def create_role_table(self, correct, missing, overpred):
        role_table = {}
        for role in self.roles:
            c = float(correct[role])
            m = float(missing[role])
            o = float(overpred[role])
            p, r, f1 = util.compute_metrics(c, m, o)
            role_table[role] = {'CORRECT': c,
                                'MISSING': m,
                                'OVERPRED': o,
                                'PRECISION': p,
                                'RECALL': r,
                                'F1': f1}
        total_c = sum(correct.values())
        total_m = sum(missing.values())
        total_o = sum(overpred.values())
        total_p, total_r, total_f1 = util.compute_metrics(total_c,
                                                          total_m,
                                                          total_o)
        totals = {'CORRECT': total_c,
                  'MISSING': total_m,
                  'OVERPRED': total_o,
                  'PRECISION': total_p,
                  'RECALL': total_r,
                  'F1': total_f1}
        return (role_table, totals)

    def evaluate(self):
        self.metrics = None
        self.role_table = None
        self.confusion = None
        # Also computes confusion counters
        global_confusion = defaultdict(Counter)

        global_correct = Counter()
        global_missing = Counter()
        global_overpred = Counter()
        for doc_key, (gold_structure, evt_type) in self.gold.items():
            pred_structure = self.pred.get(doc_key, ([], None))

            gold_set = Counter(gold_structure)
            pred_set = Counter(pred_structure)
            assert (sum(pred_set.values()) == len(pred_structure))
            assert (sum(gold_set.values()) == len(gold_structure))
            intersection = gold_set & pred_set
            missing = gold_set - pred_set
            overpred = pred_set - gold_set
            # Update confusion and counters
            util.compute_confusion(global_confusion, intersection, missing, overpred)
            util.update(intersection, global_correct)  # 把 intersection 加到 global_correct 中
            util.update(missing, global_missing)
            util.update(overpred, global_overpred)
        precision, recall, f1, _ = util.compute_from_counters(global_correct,
                                                              global_missing,
                                                              global_overpred)
        self.metrics = {'precision': precision,
                        'recall': recall,
                        'f1': f1}
        self.role_table = self.create_role_table(global_correct,
                                                 global_missing,
                                                 global_overpred)
        result = {"role_table": self.role_table,
                  "confusion": global_confusion,
                  "metrics": self.metrics}

        return result

    def evaluate_gpt_online(self):
        # The generated unstructured text will directly determine wrong prediction
        self.metrics = None
        self.role_table = None
        self.confusion = None
        # Also computes confusion counters
        global_confusion = defaultdict(Counter)

        global_correct = Counter()
        global_missing = Counter()
        global_overpred = Counter()
        for doc_key, (gold_structure, evt_type) in tqdm(self.gold.items()):
            args.logger.info(doc_key)
            pred_structure = self.pred.get(doc_key, ([], None))
            temp_pred_structure = [(w, r) for w, r in pred_structure]
            for (gold_word, gold_role) in gold_structure:
                if gold_role in [r for w, r, visited in temp_pred_structure]:
                    for i in range(len(temp_pred_structure)):
                        if gold_role == temp_pred_structure[i][1] and temp_pred_structure[i][
                            0] != gold_word:  # span not exactly match
                            temp_pred_structure[i] = (temp_pred_structure[i][0], temp_pred_structure[i][1], True)
                            pred_word = temp_pred_structure[i][0]
                            is_gpt_equal = judge_equal(args.logger, gold_word,
                                                       pred_word)  # Exception, 用变量保存，以防下一次结果不同（一般很少出现）
                            change = False
                            if is_gpt_equal:
                                temp_pred_structure[i] = (gold_word, temp_pred_structure[i][1])
                                change = True
                            else:
                                is_more_accurate = judge_more_accurate(args.logger, gold_word, pred_word)  # Exception
                                if is_more_accurate:
                                    temp_pred_structure[i] = (gold_word, temp_pred_structure[i][1])
                                    change = True

                    args.logger.info(f'{pred_word}\t\t{gold_word}')  # pred, gold
                    args.logger.info(f'{change}\n')  # True or False
            args.logger.info('\n\n')
            pred_structure = [(w, r) for w, r in temp_pred_structure]
            gold_set = Counter(gold_structure)
            pred_set = Counter(pred_structure)
            assert (sum(pred_set.values()) == len(pred_structure))
            assert (sum(gold_set.values()) == len(gold_structure))
            intersection = gold_set & pred_set
            missing = gold_set - pred_set
            overpred = pred_set - gold_set
            # Update confusion and counters
            util.compute_confusion(global_confusion, intersection, missing, overpred)
            util.update(intersection, global_correct)  # 把 intersection 加到 global_correct 中
            util.update(missing, global_missing)
            util.update(overpred, global_overpred)
        precision, recall, f1, _ = util.compute_from_counters(global_correct, global_missing, global_overpred)
        self.metrics = {'precision': precision,
                        'recall': recall,
                        'f1': f1}
        self.role_table = self.create_role_table(global_correct,
                                                 global_missing,
                                                 global_overpred)
        result = {"role_table": self.role_table,
                  "confusion": global_confusion,
                  "metrics": self.metrics}

        return result

    def evaluate_gpt_or_human(self, method=['human', 'gpt']):
        # The generated unstructured text will directly determine wrong prediction
        self.metrics = None
        self.role_table = None
        self.confusion = None

        def read_human_pred_file():
            f = open('ChatGPT/output/rams/judge_eval.txt',
                     encoding='utf-8')  # judge_eval.txt is from judge_eval.log by change the suffix name for the convenience of re-evaluation
            lines = f.readlines()
            docs = ''.join(lines).strip().split('\n\n\n')
            docs_dict = {}
            for doc in docs:
                items = doc.strip().split('\n')
                docs_dict[items[0]] = {}
                index = 1
                assert (len(items) - 1) % 3 == 0
                while index < len(items):
                    (pred, gold), gpt_judge, human_judge = items[index].split('\t\t'), items[index + 1], items[
                        index + 2]
                    index += 3
                    docs_dict[items[0]][(pred, gold)] = {}
                    docs_dict[items[0]][(pred, gold)]['gpt_judge'] = eval(gpt_judge)
                    docs_dict[items[0]][(pred, gold)]['human_judge'] = bool(eval(human_judge))
            return docs_dict

        def is_gpt_equal(doc_key, pred, gold):
            try:
                return self.eval_docs_dict[doc_key][(pred, gold)]['gpt_judge']
            except:
                return False

        # There were errors when saving individual characters, which had an impact on the result f1 of less than 2%
        def is_human_equal(doc_key, pred, gold):
            try:
                return self.eval_docs_dict[doc_key][(pred, gold)]['human_judge']
            except:
                return False

        self.eval_docs_dict = read_human_pred_file()
        # Also computes confusion counters
        global_confusion = defaultdict(Counter)
        global_correct = Counter()
        global_missing = Counter()
        global_overpred = Counter()
        for doc_key, (gold_structure, evt_type) in tqdm(list(self.gold.items())):
            pred_structure = self.pred.get(doc_key, ([], None))
            temp_pred_structure = [(w, r, False) for w, r in pred_structure]
            for (gold_word, gold_role) in gold_structure:
                # visited: handle multiple same roles in one event, e.g. role: participant
                if gold_role in [r for w, r, visited in temp_pred_structure]:
                    for i in range(len(temp_pred_structure)):
                        if gold_role == temp_pred_structure[i][1] and temp_pred_structure[i][0] != gold_word and \
                                temp_pred_structure[i][2] == False:  # span not exactly match
                            temp_pred_structure[i] = (temp_pred_structure[i][0], temp_pred_structure[i][1], True)
                            is_equal = is_human_equal(doc_key, temp_pred_structure[i][0],
                                                      gold_word) if method == 'human' else is_gpt_equal(doc_key,
                                                                                                        temp_pred_structure[
                                                                                                            i][0],
                                                                                                        gold_word)
                            if is_equal:  # [is_gpt_equal, is_human_equal]
                                temp_pred_structure[i] = (gold_word, temp_pred_structure[i][1], True)
                            break
            pred_structure = [(w, r) for w, r, visited in temp_pred_structure]
            gold_set = Counter(gold_structure)
            pred_set = Counter(pred_structure)
            assert (sum(pred_set.values()) == len(pred_structure))
            assert (sum(gold_set.values()) == len(gold_structure))
            intersection = gold_set & pred_set
            missing = gold_set - pred_set
            overpred = pred_set - gold_set
            # Update confusion and counters
            util.compute_confusion(global_confusion, intersection, missing, overpred)
            util.update(intersection, global_correct)  # 把 intersection 加到 global_correct 中
            util.update(missing, global_missing)
            util.update(overpred, global_overpred)
        precision, recall, f1, _ = util.compute_from_counters(global_correct, global_missing, global_overpred)
        self.metrics = {'precision': precision,
                        'recall': recall,
                        'f1': f1}
        self.role_table = self.create_role_table(global_correct,
                                                 global_missing,
                                                 global_overpred)
        result = {"role_table": self.role_table,
                  "confusion": global_confusion,
                  "metrics": self.metrics}

        return result


def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gold_file', type=str, default="EE_datasets/RAMS_1.0c/data/test.jsonlines",
                        help='Gold file path')
    parser.add_argument('-p', '--pred_file', type=str, default="ChatGPT/output/rams/gen.jsonlines",
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
    parser.add_argument('--do_all', dest='do_all', default=False,
                        action='store_true', help="Do everything.")
    parser.add_argument('--metrics', dest='metrics', default=False,
                        action='store_true',
                        help="Compute overall p, r, f1.")
    parser.add_argument('--role_table', dest='role_table', default=False,
                        action='store_true',
                        help="Compute p, r, f1 per role.")
    parser.add_argument('--confusion', dest='confusion', default=False,
                        action='store_true',
                        help="Compute an error confusion matrix.")
    return parser.parse_args()


def run_evaluation(args):
    """This is a separate wrapper around args so that other programs
    can call evaluation without resorting to an os-level call
    """
    scorer = Scorer(args)
    output_file = open('ChatGPT/output/rams/eval_result.jsonlines', 'a', encoding='utf-8')
    return_dict = scorer.evaluate()  # 1
    output_file.write('full_match_eval\n')
    json.dump(return_dict, output_file)
    # return_dict_gpt = scorer.evaluate_gpt_online()
    return_dict = scorer.evaluate_gpt_or_human('gpt')  # 2
    output_file.write('\ngpt_eval\n')
    json.dump(return_dict, output_file)
    return_dict = scorer.evaluate_gpt_or_human('human')  # 3
    output_file.write('\nhuman_eval\n')
    json.dump(return_dict, output_file)

    if args.confusion or args.do_all:
        util.print_confusion(return_dict['confusion'])
    if args.role_table or args.do_all:
        util.print_table(*return_dict['role_table'])
    if args.metrics or args.do_all:
        print("Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(
            return_dict['metrics']['precision'],
            return_dict['metrics']['recall'],
            return_dict['metrics']['f1']))
    return return_dict


if __name__ == "__main__":
    args = define_args()
    logger = define_logger('ChatGPT/output/rams',
                           'judge_eval.log')  # judge_eval.log content is produced by function scorer.evaluate_gpt_online()
    args.logger = logger
    return_dict = run_evaluation(args)
