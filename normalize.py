
def normalize(text):
    return "".join(text.lower().split())

txt1 = ["Hello, World!", "This is a test.", "Normalize this text."]
txt2 = "hello world !"

normalized_txt1 = [normalize(t) for t in txt1]
normalized_txt2 = normalize(txt2)

for each in normalized_txt1:
    if each == normalized_txt2:
        print("Match found:", each)