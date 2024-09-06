import streamlit as st
import pandas as pd
import json
from utils import get_chroma_json

def get_chunks(name):
    with open(name, 'r') as file:
        all_chunks = json.load(file)
    return all_chunks

    
chunks = get_chroma_json()
for key in chunks:
    print(key)
    if chunks[key] is not None:
        print(len(chunks[key]))
# def list_chunk_content():
#     # List blobs in container
#     # chunks = get_chunks('chunks/ChunkSize500.json')
#     chunks = get_chroma_json()
#     # print(chunks)
#     res = {}
#     # for chunk in chunks:
#     for key in chunks:
#         print(key)
#         print(chunks[key])
#         if key not in res:
#             res[key] = [chunks[key]]
#         else:
#             res[key].append(chunks[key])
#     return res

# chunk_list = list_chunk_content()
pretty = {
    "ids": chunks["ids"],
    "metadatas": chunks["metadatas"],
    "documents": chunks["documents"],
}
df = pd.DataFrame(pretty)
st.dataframe(df)