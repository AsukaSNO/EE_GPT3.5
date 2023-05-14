model: gpt-3.5-turbo, gpt-3.5-turbo-0301(expire at 2023.6.1), text-davinci-002-render-sha(ChatGPT), text-davinci-003
we choose LTS version model 'gpt-3.5-turbo'

one shot
example_user = '''Given one text and an event in the text, you need to extract the arguments for that event. The shorter the argument length, the better, with one or more arguments,. You need to give according to the key value pair, key is argument role, and the value is corresponding word in the text, without explaining the reason

Example
text：We are ashamed of them . " However , Mutko stopped short of admitting the doping scandal was state sponsored . " We are very sorry that athletes who tried to deceive us , and the world , were not caught sooner . We are very sorry because Russia is committed to upholding the highest standards in sport and is opposed to anything that threatens the Olympic values , " he said . English former heptathlete and Athens 2004 bronze medallist Kelly Sotherton was unhappy with Mutko 's plea for Russia 's ban to be lifted for Rio
event：deceive
Please identify the communicator, recipient, place for this event
The expected answer is as follows
communicator: athletes
recipient: us , and the world
place: Russia
Do you understand?
'''
        example_assistant = '''
Yes, I understand. Let me extract the arguments for this event:
event：deceive

communicator: athletes
recipient: us, and the world
place: Russia
'''