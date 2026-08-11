import fitz

pdf = fitz.open("documents/sample.pdf")

text=""
for page in pdf:
    text+=page.get_text()

print(text)