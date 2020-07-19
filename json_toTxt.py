import json

with open('/Desktop/chat.json') as json_file:
    data = json.load(json_file)
    chats = data['chats']
    chat_list = chats['list'][0]
    messages = chat_list['messages']
    string = ""
    count = 0
    for mess in messages:
        if isinstance(mess['text'], list):
            a = 1
        else:     
            if count > 3:  
                string = string + mess['text'] + '\n'
                count = 0
            else:
                string = string + mess['text'] + ' '
        count = count + 1

print(string)
text_file = open("/Desktop/chat.txt", "wt")
n = text_file.write(string)
text_file.close()