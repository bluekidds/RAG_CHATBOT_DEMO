import streamlit as st
import pandas as pd
from azure.storage.blob import BlobServiceClient
from io import BytesIO

# Azure Storage Account details
azure_storage_account_name = "wavenetragstore"
azure_storage_account_key = "your key"
container_name = "rag-files"

# Function to upload file to Azure Storage
def upload_to_azure_storage(file):
    blob_service_client = BlobServiceClient.from_connection_string(f"DefaultEndpointsProtocol=https;AccountName={azure_storage_account_name};AccountKey={azure_storage_account_key}")
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=file.name)
    blob_client.upload_blob(file)

def list_azure_storage():
    # List blobs in container
    blob_service_client = BlobServiceClient.from_connection_string(f"DefaultEndpointsProtocol=https;AccountName={azure_storage_account_name};AccountKey={azure_storage_account_key}")
    container_client = blob_service_client.get_container_client(container=container_name)
    blob_list = container_client.list_blobs()
    res = {}
    for blob in blob_list:
        for key in blob.keys():
            if key not in res:
                res[key] = [blob[key]]
            else:
                res[key].append(blob[key])
    return res

# Streamlit App
st.title("Azure Storage Uploader")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # st.image(uploaded_file)

    # Upload the file to Azure Storage on button click
    if st.button("Upload to Azure Storage"):
        upload_to_azure_storage(uploaded_file)
        st.success("File uploaded to Azure Storage!")

blob_list = list_azure_storage()
df = pd.DataFrame(blob_list)
st.dataframe(df)