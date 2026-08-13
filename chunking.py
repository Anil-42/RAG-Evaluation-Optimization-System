
def chunk_text(text):
    words=text.split()
    chunks=[]

    chunk_size=100
    overlap=20
    start=0

    while start<len(words):
        end=start+chunk_size
        chunk=words[start:end]
        chunks.append(" ".join(chunk))
        start+=chunk_size-overlap

    # print(len(chunks[0].split()))
    # print(len(chunks[1].split()))

    # print(chunks[0].split()[-20:])
    # print("\n")
    # print(chunks[1].split()[:20])

    return chunks

