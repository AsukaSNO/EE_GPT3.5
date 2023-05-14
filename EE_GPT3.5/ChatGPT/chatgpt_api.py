import openai
from ChatGPT.logger import define_logger
from retrying import retry

openai.api_key = "your api-key"


# without sample
@retry(wait_fixed=100, stop_max_attempt_number=5)
def ask_test(system, input_user):
    # ChatGPT API
    completion = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": input_user},
        ]
    )
    return completion.choices[0].message.content


@retry(wait_fixed=100, stop_max_attempt_number=5)
# with sample
def ask(system, example_user, example_assistant, input_user):
    # ChatGPT API
    completion = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": example_user},
            {"role": "assistant", "content": example_assistant},
            {"role": "user", "content": input_user},
        ]
    )
    return completion.choices[0].message.content
# CAUSE: (e1, e0), (e5, e4), (e11, e10), (e12, e13), (e14, e13), (e20, e9)
# PRECONDITION: (e3, e4), (e6, e7), (e9, e15), (e17, e16), (e24, e23), (e25, e0), (e26, e24), (e27, e24)


@retry(wait_fixed=100, stop_max_attempt_number=5)
def judge_equal(logger, gold_entity, pred_entity="None"):
    system = "You are an expert in event argument extraction"
    example_user = '''Do two entities represent the same thing
Just answer me 'Yes' without explaining the reason or 'No' with explaining the reason

for examples
Does "a red apple" represent the same thing as "an apple", Yes or No?
Yes.

Does "an apple" represent the same thing as "a red pig", Yes or No?
No.

Do you understand?
'''
    example_assistant = "Yes."
    input_user = f'Does "{pred_entity}" represent the same thing as "{gold_entity}", Yes or No?'
    msg = ask(system, example_user, example_assistant, input_user)  # Exception
    # logger.info(input_user + '\n' + msg.lower()[:5])
    print(input_user, msg)

    answer_equal = msg.lower()[:5]
    if 'yes' in answer_equal:
        return True
    elif 'no' in answer_equal:
        return False
    else:
        pass
        return False


def judge_more_accurate(logger, gold_entity, pred_entity="None"):
    system = "You are an expert in event argument extraction"
    example_user = '''Ignore modifiers, do entity 1 and entity 2 represent the same thing
Just answer me 'Yes' without explaining the reason or 'No' with explaining the reason

for examples
Is "DELL computer" similar to "computer" and more detailed than "computer", Yes or No?
Yes.

Is "computer" similar to "DELL computer" and more detailed than "DELL computer", Yes or No?
No. "DELL computer" is relevant and more specific and granular than "computer".

Do you understand?
'''
    example_assistant = "Yes."
    input_user = f'Is "{pred_entity}" similar to "{gold_entity}" and more detailed than "{gold_entity}", Yes or No?'
    msg = ask(system, example_user, example_assistant, input_user)  # Exception
    # logger.info(input_user + '\n' + msg.lower()[:5])
    print(input_user, msg)

    answer_more_accurate = msg.lower()[:5]
    if 'yes' in answer_more_accurate:
        return True
    elif 'no' in answer_more_accurate:
        return False
    else:
        pass
        return False


if __name__ == "__main__":
    system = "You are an expert in determining whether two entities are the same."
    ## gpt_eval
    example_user = '''Ignore modifiers, do entity 1 and entity 2 represent the same thing
Just answer me 'yes' or' no 'without explaining the reason

give an example
Is an apple and a red apple the same thing?
Expected answer:
yes

Are green apples and red apples the same thing?
no

Do you understand?
'''
    ## causal
    example_user = '''
Please use this text to identify which event pair in the given event has a cause or premise relationship between two events (CAUSE is defined as "the tail event is unable to give the head event", and PRECONDITION is defined as "the tail event would not have happened if the head event had not happened"), without explaining the reason

text: Hurricane Jerry caused minor damage in Texas and flash flooding in Kentucky and Virginia in October 1989. The fourteenth tropical cyclone, tenth named storm of the season, Jerry developed from a tropical wave in the Bay of Campeche on October 12. Initially a tropical depression, the system moved north-northwestward across the Gulf of Mexico and strengthened into Tropical Storm Jerry early on the following day. Jerry continuously deepened until October 14 and then maintained intensity while curving northeastward and briefly decelerating. Later that day, the storm re-curved north-northwestward. Jerry began to intensify on October 15 and soon became a Category 1 hurricane on the Saffir–Simpson hurricane wind scale. Early on October 16, Jerry made landfall on Galveston Island, Texas with winds of . Less than six hours later, Jerry weakened to a tropical storm and then a tropical depression shortly thereafter. Late on October 16, Jerry was absorbed by a frontal system while situated over southwestern Arkansas. Storm surge and rough surf along the coast of Texas destroyed a section of Texas State Highway 87, which was never repaired. Due to strong winds, about 52,000 homes and businesses were left without electricity, most of them in the Galveston area. Many homes, businesses, and buildings were inflicted damage because of strong winds and three tornadoes spawned by the storm. Despite the issuance of a hurricane warning just eight hours prior to landfall, Jerry caused only three fatalities in Texas, possibly due to the storm's small size; a car fell over the Galveston Seawall, killing its three occupants. Minor wind and coastal flood damage was reported in Louisiana. Jerry and its remnants brought flash flooding to portions of the Upland South, particularly in the states of Kentucky, Virginia, and West Virginia. In eastern Kentucky, hundreds of homes were flooded and many bridges, culverts, and roads were washed out; this left hundreds of residents stranded. Damage in Kentucky reached at least $5 million. Similar impact occurred in Virginia; with $3.4 million (1989 USD) in damage in Buchanan County. In West Virginia, overflowing rivers in the western portions of the state forced hundreds to evacuate. Throughout the United States, Jerry resulted in about $70 million in damage.
events: e0: caused, e1: damage, e2: flooding, e3: named, e4: developed, e5: moved, e6: strengthened, e7: deepened, e8: maintained, e9: curving, e10: re-curved, e11: began, e12: became, e13: made, e14: weakened, e15: situated, e16: destroyed, e17: repaired, e18: damage, e19: spawned, e20: warning, e21: caused, e22: fell, e23: killing, e24: damage, e25: reported, e26: brought, e27: flooding, e28: flooded, e29: washed, e30: Damage, e31: reached, e32: occurred, e33: damage, e34: evacuate, e35: resulted in, e36: damage

The events sorted by the order of appearance in the text. Please identify all CAUSE and PRECONDITION relationships between all given events in pairs.
'''
    ## causal
    example_assistant = '''
CAUSE: 
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
    ## temporal
    example_user = f'''
Please use this text to identify which event pair in the given event has one of six types (before, contains, overlap, begins on, ends on, simultaneous), the head event must start before the tail event in a relation instance.
text: Hurricane Jerry caused minor damage in Texas and flash flooding in Kentucky and Virginia in October 1989. The fourteenth tropical cyclone, tenth named storm of the season, Jerry developed from a tropical wave in the Bay of Campeche on October 12. Initially a tropical depression, the system moved north-northwestward across the Gulf of Mexico and strengthened into Tropical Storm Jerry early on the following day. Jerry continuously deepened until October 14 and then maintained intensity while curving northeastward and briefly decelerating. Later that day, the storm re-curved north-northwestward. Jerry began to intensify on October 15 and soon became a Category 1 hurricane on the Saffir–Simpson hurricane wind scale. Early on October 16, Jerry made landfall on Galveston Island, Texas with winds of . Less than six hours later, Jerry weakened to a tropical storm and then a tropical depression shortly thereafter. Late on October 16, Jerry was absorbed by a frontal system while situated over southwestern Arkansas. Storm surge and rough surf along the coast of Texas destroyed a section of Texas State Highway 87, which was never repaired. Due to strong winds, about 52,000 homes and businesses were left without electricity, most of them in the Galveston area. Many homes, businesses, and buildings were inflicted damage because of strong winds and three tornadoes spawned by the storm. Despite the issuance of a hurricane warning just eight hours prior to landfall, Jerry caused only three fatalities in Texas, possibly due to the storm's small size; a car fell over the Galveston Seawall, killing its three occupants. Minor wind and coastal flood damage was reported in Louisiana. Jerry and its remnants brought flash flooding to portions of the Upland South, particularly in the states of Kentucky, Virginia, and West Virginia. In eastern Kentucky, hundreds of homes were flooded and many bridges, culverts, and roads were washed out; this left hundreds of residents stranded. Damage in Kentucky reached at least $5 million. Similar impact occurred in Virginia; with $3.4 million (1989 USD) in damage in Buchanan County. In West Virginia, overflowing rivers in the western portions of the state forced hundreds to evacuate. Throughout the United States, Jerry resulted in about $70 million in damage.
events: t0: October 1989, t1: October 12, t2: October 14, t3: October 15, t4: October 16, t5: six hours later, t6: October 16, e0: caused, e1: damage, e2: flooding, e3: named, e4: developed, e5: moved, e6: strengthened, e7: deepened, e8: maintained, e9: curving, e10: re-curved, e11: began, e12: became, e13: made, e14: weakened, e15: situated, e16: destroyed, e17: repaired, e18: damage, e19: spawned, e20: warning, e21: caused, e22: fell, e23: killing, e24: damage, e25: reported, e26: brought, e27: flooding, e28: flooded, e29: washed, e30: Damage, e31: reached, e32: occurred, e33: damage, e34: evacuate, e35: resulted in, e36: damage

The events sorted by the order of appearance in the text and the previous 't' refers to the time point. Please identify all temporal relationships as much as possible between all given events in pairs.
'''
    ## temporal
    example_assistant = '''
1. (e4, e3) begins on
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
    ## subevents
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
    input_user = '''
Please use this text to identify which event pair satisfy "A is a subevent of B", means that event A is a component of event B, and event B includes event A in both time and space. You need not to explain the reason

text: Abraham Lincoln, the 16th President of the United States, was assassinated by well-known stage actor John Wilkes Booth on April 14, 1865, while attending the play "Our American Cousin" at Ford's Theatre in Washington, D.C. Shot in the head as he watched the play, Lincoln died the following day at 7:22 am, in the Petersen House opposite the theater. He was the first U.S. president to be assassinated, and Lincoln's funeral and burial marked an extended period of national mourning. Occurring near the end of the American Civil War, the assassination was part of a larger conspiracy intended by Booth to revive the Confederate cause by eliminating the three most important officials of the United States government. Conspirators Lewis Powell and David Herold were assigned to kill Secretary of State William H. Seward, and George Atzerodt was tasked with killing Vice President Andrew Johnson. Beyond Lincoln's death, the plot failed: Seward was only wounded and Johnson's would-be attacker lost his nerve. After a dramatic initial escape, Booth was killed at the climax of a 12-day manhunt. Powell, Herold, Atzerodt and Mary Surratt were later hanged for their roles in the conspiracy.
events: e0: assassinated, e1: attending, e2: play, e3: shot, e4: watched, e5: died, e6: assassinated, e7: funeral, e8: burial, e9: marked, e10: War, e11: assassination, e12: revive, e13: assigned, e14: kill, e15: tasked, e16: killing, e17: death, e18: wounded, e19: lost, e20: escape, e21: killed

The events sorted by the order of appearance in the text. Please identify the subevent relationships, as strictly and certainly as possible
'''
    for i in range(10):
        # answer = ask_test(system, example_user)
        answer = ask(system, example_user, example_assistant, input_user)
        print(answer)
