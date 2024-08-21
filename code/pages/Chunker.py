import streamlit as st
import pandas as pd
import json

def get_chunks(name):
    with open(name, 'r') as file:
        all_chunks = json.load(file)
    return all_chunks

def list_chunk_content():
    # List blobs in container
    chunks = get_chunks('chunks/ChunkSize500.json')
    res = {}
    for chunk in chunks:
        for key in chunk:
            if key not in res:
                res[key] = [chunk[key]]
            else:
                res[key].append(chunk[key])
    return res

chunk_list = list_chunk_content()
df = pd.DataFrame(chunk_list)
st.dataframe(df)