import pymupdf

def extract_text(pdf_path):

    pdf = pymupdf.open(pdf_path)

    text=""
    for page in pdf:
        text+=page.get_text()

    return text

