import re 

def normalize(text):
    text=text.lower()
    text=re.sub(r'[^a-z0-9\s]', '', text)#normalize using regex expressiosn to remove all special characters other than numbers and alpahabets
    text="".join(text.split())
    return text


