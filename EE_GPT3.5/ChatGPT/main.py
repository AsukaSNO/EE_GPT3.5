import openai
import json
import os
import argparse
from chatgpt_api import ask
from processors.rams import process as process_rams
from processors.wikievents import process as process_wikievents
from processors.maven import process as process_maven
from processors.maven_ere import process as process_maven_ere
from logger import define_logger
from tqdm import tqdm
import re


def define_args():
    parser = argparse.ArgumentParser(description='EE by GPT3.5')

    # Data Paths
    parser.add_argument('--dataset', type=str, choices=['ace2005', 'rams', 'wikievents', 'maven', 'maven_ere'],
                        default='maven_ere', help='which event extraction dataset')
    parser.add_argument('--rel', type=str, choices=['coref', 'causal', 'temporal', 'subevents'],
                        default='subevents', help='which relation of maven_ere')
    args = parser.parse_args()
    return args


def process(root_dir, dataset):
    if dataset == 'rams':
        train_data, dev_data, test_data = process_rams(root_dir)
    elif dataset == 'wikievents':
        train_data, dev_data, test_data = process_wikievents(root_dir)
    elif dataset == 'maven':
        train_data, dev_data, test_data = process_maven(root_dir)
    elif dataset == 'maven_ere':
        train_data, dev_data, test_data = process_maven_ere(root_dir)
    else:
        pass
    return train_data, dev_data, test_data


if __name__ == "__main__":
    args = define_args()
    output_dir = 'ChatGPT/output/' + args.dataset + '/' + args.rel
    input_file = 'ChatGPT/data/' + args.dataset + '/result.jsonlines'
    gen_file = 'ChatGPT/output/' + args.dataset + '/' + args.rel + '/gen.jsonlines'
    logger = define_logger(output_dir, 'chatgpt_QA.log')
    # data = json.load(open(input_file, 'r', encoding='utf-8')) 需要解析时

    train_data, dev_data, test_data = process("EE_datasets", args.dataset)

    if args.dataset == 'rams':
        system = "You are an expert in event argument extraction"
        example_instance = train_data[0]
        example_arguments_type = [re.split(r'\d+', argument[1])[-1] for argument in example_instance['events_argument_role'][0]]
        example_arguments_word = [argument[0] for argument in example_instance['events_argument_role'][0]]
        example_user = f'''Given one text and an event in the text, you need to extract the arguments for that event. The shorter the argument length, the better, with one or more arguments,. You need to give according to the key value pair, key is argument role, and the value is corresponding word in the text, without explaining the reason

Example
text：{example_instance['text']}
event：{example_instance['events_word_type'][0][0]}
event type：{example_instance['events_word_type'][0][1]}
Please identify the {", ".join(example_arguments_type)} for this event
The expected answer is as follows
{example_arguments_type[0]}: {example_arguments_word[0]}
{example_arguments_type[1]}: {example_arguments_word[1]}
{example_arguments_type[2]}: {example_arguments_word[2]}
Do you understand?
'''
        example_assistant = f'''
Yes, I understand. Let me extract the arguments for this event:
{example_arguments_type[0]}: {example_arguments_word[0]}
{example_arguments_type[1]}: {example_arguments_word[1]}
{example_arguments_type[2]}: {example_arguments_word[2]}
'''

        descript = '''Given one text and an event in the text, you need to extract the arguments for that event. The shorter the argument length, the better, with one or more arguments,. You need to give according to the key value pair, key is argument role, and the value is corresponding word in the text, without explaining the reason
'''

        for i in tqdm(range(0, len(test_data))):
            instance = test_data[i]
            text = instance['text']
            event_trigger = instance['events_word_type'][0][0]
            event_type = instance['events_word_type'][0][1]
            arguments_type = [re.split(r'\d+', argument[1])[-1] for argument in instance['events_argument_role'][0]]
            input_user = descript + '\n' + \
                         f'text: {text}\n' + \
                         f'event: {event_trigger}\n' + \
                         f'event type: {event_type}\n' + \
                         f'Please identify the {", ".join(arguments_type)} for this event'
            msg = ask(system, example_user, example_assistant, input_user)
            print(msg)
            instance_log = {'doc_key': instance['doc_key'],
                            'text': instance['text'],
                            'events_word_type': instance['events_word_type'],
                            'events_argument_role': instance['events_argument_role'],
                            'Q': input_user,
                            'A': msg}
            instance_result = {'doc_key': instance['doc_key'],
                               'events_word_type': instance['events_word_type'],
                               'events_argument_role': instance['events_argument_role'],
                               'A': msg}
            logger.info(instance_log)
            with open(gen_file, 'a') as f:
                json.dump(instance_result, f)
                f.write('\n')
            print(str(i)+"已完成")
    elif args.dataset == 'wikievents':
        system = "你是一个很棒的事件抽取专家，发给你的每段文本中只会有一个事件，你需要提取事件的触发词，以及判断该事件所对应的类别，事件触发词长度不限。"
        input = "请提取这段文本中的事件触发词，以及，每段文本中只会有一个事件，事件触发词长度不限。"
        format = '''你需要按照下列格式给出，xxx表示你需要填的位置，不需要说明理由：
                event：xxx
                type: xxx
                '''
    elif args.dataset == 'maven':
        system = "You are an expert in event detection"
        input = "请提取这段文本中的事件触发词，以及，每段文本中只会有一个事件，事件触发词长度不限。"
        format = '''你需要按照下列格式给出，xxx表示你需要填的位置，不需要说明理由：
        event：xxx
        type: xxx
        '''
    elif args.dataset == 'maven_ere':
        if args.rel == 'causal':
            system = "You are an expert in causal relation extraction"
            # example_instance = train_data[2]
            # 在文中给事件加tag（text_with_tag），chatgpt识别很差，不如加在末尾（text_with_annotation）
            # Please identify the co-reference (i.e. if several events refer to the same event) relationship between all events (coref)
            # for coref in example_instance['coref_relations']:

            example_user = f'''Please use this text to identify which event pair in the given event has a cause or precondition relationship between two events (CAUSE is defined as "the tail event is unable to give the head event", and PRECONDITION is defined as "the tail event would not have happened if the head event had not happened"), without explaining the reason

text: Hurricane Jerry caused minor damage in Texas and flash flooding in Kentucky and Virginia in October 1989. The fourteenth tropical cyclone, tenth named storm of the season, Jerry developed from a tropical wave in the Bay of Campeche on October 12. Initially a tropical depression, the system moved north-northwestward across the Gulf of Mexico and strengthened into Tropical Storm Jerry early on the following day. Jerry continuously deepened until October 14 and then maintained intensity while curving northeastward and briefly decelerating. Later that day, the storm re-curved north-northwestward. Jerry began to intensify on October 15 and soon became a Category 1 hurricane on the Saffir–Simpson hurricane wind scale. Early on October 16, Jerry made landfall on Galveston Island, Texas with winds of . Less than six hours later, Jerry weakened to a tropical storm and then a tropical depression shortly thereafter. Late on October 16, Jerry was absorbed by a frontal system while situated over southwestern Arkansas. Storm surge and rough surf along the coast of Texas destroyed a section of Texas State Highway 87, which was never repaired. Due to strong winds, about 52,000 homes and businesses were left without electricity, most of them in the Galveston area. Many homes, businesses, and buildings were inflicted damage because of strong winds and three tornadoes spawned by the storm. Despite the issuance of a hurricane warning just eight hours prior to landfall, Jerry caused only three fatalities in Texas, possibly due to the storm's small size; a car fell over the Galveston Seawall, killing its three occupants. Minor wind and coastal flood damage was reported in Louisiana. Jerry and its remnants brought flash flooding to portions of the Upland South, particularly in the states of Kentucky, Virginia, and West Virginia. In eastern Kentucky, hundreds of homes were flooded and many bridges, culverts, and roads were washed out; this left hundreds of residents stranded. Damage in Kentucky reached at least $5 million. Similar impact occurred in Virginia; with $3.4 million (1989 USD) in damage in Buchanan County. In West Virginia, overflowing rivers in the western portions of the state forced hundreds to evacuate. Throughout the United States, Jerry resulted in about $70 million in damage.
events: e0: caused, e1: damage, e2: flooding, e3: named, e4: developed, e5: moved, e6: strengthened, e7: deepened, e8: maintained, e9: curving, e10: re-curved, e11: began, e12: became, e13: made, e14: weakened, e15: situated, e16: destroyed, e17: repaired, e18: damage, e19: spawned, e20: warning, e21: caused, e22: fell, e23: killing, e24: damage, e25: reported, e26: brought, e27: flooding, e28: flooded, e29: washed, e30: Damage, e31: reached, e32: occurred, e33: damage, e34: evacuate, e35: resulted in, e36: damage

The events sorted by the order of appearance in the text. Please identify all CAUSE and PRECONDITION relationships between all given events in pairs.
'''
            example_assistant = '''CAUSE: 
- e27 caused e30
- e27 caused e29
- e2 caused e30
- e27 caused e33
- e19 caused e18
- e2 caused e29
- e22 caused e23
- e27 caused e28

PRECONDITION:
- e2 is a precondition for e34
- e11 is a precondition for e12 
- e27 is a precondition for e34
'''
            for i in tqdm(range(0, len(dev_data[:355]))):
                instance = dev_data[i]
                text = instance['text_with_annotation']
                input_user = f'''Please use this text to identify which event pair in the given event has a cause or precondition relationship between two events (CAUSE is defined as "the tail event is unable to give the head event", and PRECONDITION is defined as "the tail event would not have happened if the head event had not happened"), without explaining the reason

{instance['text_with_annotation']}

The events sorted by the order of appearance in the text. Please identify all CAUSE and PRECONDITION relationships between all given events in pairs.
'''
                msg = ask(system, example_user, example_assistant, input_user)
                print(msg)
                instance_log = {'doc_key': instance['id'],
                                'Q': input_user,
                                'A': msg}
                instance_result = {'doc_key': instance['id'],
                                   'A': msg}
                logger.info(instance_log)
                with open(gen_file, 'a') as f:
                    json.dump(instance_result, f)
                    f.write('\n')
                print(str(i) + "已完成")

        elif args.rel == 'temporal':
            system = "You are an expert in temporal relation extraction"

            example_user = f'''Please use this text to identify which event pair in the given event has one of six types (before, contains, overlap, begins on, ends on, simultaneous), the head event must start before the tail event in a relation instance.

text: Hurricane Jerry caused minor damage in Texas and flash flooding in Kentucky and Virginia in October 1989. The fourteenth tropical cyclone, tenth named storm of the season, Jerry developed from a tropical wave in the Bay of Campeche on October 12. Initially a tropical depression, the system moved north-northwestward across the Gulf of Mexico and strengthened into Tropical Storm Jerry early on the following day. Jerry continuously deepened until October 14 and then maintained intensity while curving northeastward and briefly decelerating. Later that day, the storm re-curved north-northwestward. Jerry began to intensify on October 15 and soon became a Category 1 hurricane on the Saffir–Simpson hurricane wind scale. Early on October 16, Jerry made landfall on Galveston Island, Texas with winds of . Less than six hours later, Jerry weakened to a tropical storm and then a tropical depression shortly thereafter. Late on October 16, Jerry was absorbed by a frontal system while situated over southwestern Arkansas. Storm surge and rough surf along the coast of Texas destroyed a section of Texas State Highway 87, which was never repaired. Due to strong winds, about 52,000 homes and businesses were left without electricity, most of them in the Galveston area. Many homes, businesses, and buildings were inflicted damage because of strong winds and three tornadoes spawned by the storm. Despite the issuance of a hurricane warning just eight hours prior to landfall, Jerry caused only three fatalities in Texas, possibly due to the storm's small size; a car fell over the Galveston Seawall, killing its three occupants. Minor wind and coastal flood damage was reported in Louisiana. Jerry and its remnants brought flash flooding to portions of the Upland South, particularly in the states of Kentucky, Virginia, and West Virginia. In eastern Kentucky, hundreds of homes were flooded and many bridges, culverts, and roads were washed out; this left hundreds of residents stranded. Damage in Kentucky reached at least $5 million. Similar impact occurred in Virginia; with $3.4 million (1989 USD) in damage in Buchanan County. In West Virginia, overflowing rivers in the western portions of the state forced hundreds to evacuate. Throughout the United States, Jerry resulted in about $70 million in damage.
events: t0: October 1989, t1: October 12, t2: October 14, t3: October 15, t4: October 16, t5: six hours later, t6: October 16, e0: caused, e1: damage, e2: flooding, e3: named, e4: developed, e5: moved, e6: strengthened, e7: deepened, e8: maintained, e9: curving, e10: re-curved, e11: began, e12: became, e13: made, e14: weakened, e15: situated, e16: destroyed, e17: repaired, e18: damage, e19: spawned, e20: warning, e21: caused, e22: fell, e23: killing, e24: damage, e25: reported, e26: brought, e27: flooding, e28: flooded, e29: washed, e30: Damage, e31: reached, e32: occurred, e33: damage, e34: evacuate, e35: resulted in, e36: damage

The events sorted by the order of appearance in the text and the previous 't' refers to the time point. Please identify all temporal relationships between all given events in pairs, as much as possible!
'''
            example_assistant = '''1. (e4, e3) begins on
2. (e4, t1) simultaneous
3. (e5, e4) before
4. (e5, e6) simultaneous
5. (e6, t2) begins on
6. (e7, e6) simultaneous
7. (e7, t2) begins on
8. (e8, e7) simultaneous
9. (e8, t3) begins on
10. (e9, e8) simultaneous
11. (e9, t3) begins on
12. (e10, e9) simultaneous
13. (e10, t3) begins on
14. (e11, e10) simultaneous
15. (e11, t3) begins on
16. (e12, e11) simultaneous
17. (e12, t3) begins on
18. (e13, e12) simultaneous
19. (e13, t4) begins on
20. (e14, e13) simultaneous
21. (e14, t5) simultaneous
22. (e14, e0) contains
23. (e15, e14) simultaneous
24. (t6, e15) begins on
25. (e16, e15) contains
26. (e16, e17) overlap
27. (e16, e18) contains
28. (e19, e18) contains
29. (e20, e13) before
30. (e21, e0) contains
31. (e21, t20) after
32. (e22, e21) contains
33. (e23, e22) contains
34. (e24, e18) overlap
35. (e25, e24) simultaneous
36. (e26, e15) contains
37. (e27, e26) contains
38. (e27, e28) contains
39. (e28, e30) contains
40. (e31, e30) simultaneous
41. (e31, t2) after
42. (e32, e32) simultaneous
43. (e33, t21) simultaneous
44. (e34, e28) contains
45. (e35, e36) contains
46. (e35, t0) contains
'''
            for i in tqdm(range(0, len(dev_data[:355]))):
                instance = dev_data[i]
                text = instance['text_with_annotation_t']
                input_user = f'''Please use this text to identify which event pair in the given event has one of six types (before, contains, overlap, begins on, ends on, simultaneous), the head event must start before the tail event in a relation instance.

{instance['text_with_annotation_t']}

The events sorted by the order of appearance in the text and the previous 't' refers to the time point. Please identify all temporal relationships as much as possible between all given events in pairs.
'''
                msg = ask(system, example_user, example_assistant, input_user)
                print(msg)
                instance_log = {'doc_key': instance['id'],
                                'Q': input_user,
                                'A': msg}
                instance_result = {'doc_key': instance['id'],
                                   'A': msg}
                logger.info(instance_log)
                with open(gen_file, 'a') as f:
                    json.dump(instance_result, f)
                    f.write('\n')
                print(str(i) + "已完成")

        elif args.rel == 'subevents':
            system = "You are an expert in subevents relation extraction"

            example_user = f'''
Please use this text to identify which event pair satisfy "event A is a subevent of event B", means that event A is a strict component of event B, and event B strictly includes event A in both time and space. You need not to explain the reason

text: Hurricane Jerry caused minor damage in Texas and flash flooding in Kentucky and Virginia in October 1989. The fourteenth tropical cyclone, tenth named storm of the season, Jerry developed from a tropical wave in the Bay of Campeche on October 12. Initially a tropical depression, the system moved north-northwestward across the Gulf of Mexico and strengthened into Tropical Storm Jerry early on the following day. Jerry continuously deepened until October 14 and then maintained intensity while curving northeastward and briefly decelerating. Later that day, the storm re-curved north-northwestward. Jerry began to intensify on October 15 and soon became a Category 1 hurricane on the Saffir–Simpson hurricane wind scale. Early on October 16, Jerry made landfall on Galveston Island, Texas with winds of . Less than six hours later, Jerry weakened to a tropical storm and then a tropical depression shortly thereafter. Late on October 16, Jerry was absorbed by a frontal system while situated over southwestern Arkansas. Storm surge and rough surf along the coast of Texas destroyed a section of Texas State Highway 87, which was never repaired. Due to strong winds, about 52,000 homes and businesses were left without electricity, most of them in the Galveston area. Many homes, businesses, and buildings were inflicted damage because of strong winds and three tornadoes spawned by the storm. Despite the issuance of a hurricane warning just eight hours prior to landfall, Jerry caused only three fatalities in Texas, possibly due to the storm's small size; a car fell over the Galveston Seawall, killing its three occupants. Minor wind and coastal flood damage was reported in Louisiana. Jerry and its remnants brought flash flooding to portions of the Upland South, particularly in the states of Kentucky, Virginia, and West Virginia. In eastern Kentucky, hundreds of homes were flooded and many bridges, culverts, and roads were washed out; this left hundreds of residents stranded. Damage in Kentucky reached at least $5 million. Similar impact occurred in Virginia; with $3.4 million (1989 USD) in damage in Buchanan County. In West Virginia, overflowing rivers in the western portions of the state forced hundreds to evacuate. Throughout the United States, Jerry resulted in about $70 million in damage.
events: e0: caused, e1: damage, e2: flooding, e3: named, e4: developed, e5: moved, e6: strengthened, e7: deepened, e8: maintained, e9: curving, e10: re-curved, e11: began, e12: became, e13: made, e14: weakened, e15: situated, e16: destroyed, e17: repaired, e18: damage, e19: spawned, e20: warning, e21: caused, e22: fell, e23: killing, e24: damage, e25: reported, e26: brought, e27: flooding, e28: flooded, e29: washed, e30: Damage, e31: reached, e32: occurred, e33: damage, e34: evacuate, e35: resulted in, e36: damage

The events sorted by the order of appearance in the text. Please identify the subevent relationships, as strictly and certainly as possible
'''
            example_assistant = '''e4 is a subevent of e3
e5 is a subevent of e4
e6 is a subevent of e5
e7 is a subevent of e6
'''
            for i in tqdm(range(0, len(dev_data[:355]))):
                instance = dev_data[i]
                text = instance['text_with_annotation']
                input_user = f'''Please use this text to identify which event pair satisfy "event A is a subevent of event B", means that event A is a strict component of event B, and event B strictly includes event A in both time and space. You need not to explain the reason

{instance['text_with_annotation']}

The events sorted by the order of appearance in the text. Please identify the subevent relationships, as strictly and certainly as possible
'''
                msg = ask(system, example_user, example_assistant, input_user)
                print(msg)
                instance_log = {'doc_key': instance['id'],
                                'Q': input_user,
                                'A': msg}
                instance_result = {'doc_key': instance['id'],
                                   'A': msg}
                logger.info(instance_log)
                with open(gen_file, 'a') as f:
                    json.dump(instance_result, f)
                    f.write('\n')
                print(str(i) + "已完成")
    else:
        pass

    # result_data = []
    # msg = ask(system, input=input + format)
    #
    # print(msg)
    # result_data += [msg]
    # json.dump(result_data, open(output_file, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
