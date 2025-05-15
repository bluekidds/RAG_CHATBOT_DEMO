from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import oraclevs
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.chains import create_extraction_chain_pydantic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb
from pydantic import BaseModel
from typing import Optional, List

# from langchain_community.embeddings import OCIGenAIEmbeddings
# from langchain_community.chat_models import ChatOCIGenAI
# from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI

import oracledb
import json
from pathlib import Path
from pprint import pprint

import os
from code.config import config
openai_api_key = config['OPENAI_API_KEY']
azure_openai_endpoint = config['AZURE_OPENAI_ENDPOINT']

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["AZURE_OPENAI_API_KEY"] = openai_api_key
os.environ["AZURE_OPENAI_ENDPOINT"] = azure_openai_endpoint
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)
client_chroma = chromadb.PersistentClient(path='./chromadb/')
# llm = OpenAI(
#     max_tokens=1024
# )
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_version="2024-02-15-preview",
)

file_path='code/23ai.json'

username = "docuser"
password = "docuser"
dsn = "oracle-db:1521/freepdb1"

# def load_json():
#     docs = []
#     for key in keys:
#         # print(key)
#         loader = JSONLoader(
#             file_path='./code/23ai.json',
#             jq_schema='.[]',
#             content_key='.["內容"].["' + key + '"]',
#             text_content=False,
#             is_content_key_jq_parsable=True,
#             metadata_func=metadata_func
#         )
#         doc = loader.load()
#         docs = docs + doc
#         # embeddings = OCIGenAIEmbeddings(
#         #     model_id="cohere.embed-multilingual-v3.0",
#         #     service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
#         #     compartment_id="ocid1.compartment.oc1..aaaaaaaxxxxxxx",
#         # )

#     connection = oracledb.connect(user=username, password=password, dsn=dsn)
#     vector_store_dot = OracleVS.from_documents(
#         # 作成済のチャンクテキスト
#         docs,
#         # 定義済の埋め込みモデル
#         embeddings,
#         # 定義済のデータベースのconnectionオブジェクト
#         client=connection,
#         # 新規作成する表の名前を任意で指定
#         table_name="doc_table",
#         # ベクトル検索時に使う距離計算の方法
#         distance_strategy=DistanceStrategy.DOT_PRODUCT,
#     )
#     oraclevs.create_index(connection, vector_store_dot, params={"idx_name": "card", "idx_type": "IVF"})
directory_path = 'code/potent'
def list_files_in_directory():
    try:
        files = os.listdir(directory_path)
        print(files)
        # List all files in the directory
        # Filter out directories, only keep files
        return [f for f in files if os.path.isfile(os.path.join(directory_path, f))]
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


class Sentences(BaseModel):
    sentences: List[str]
def get_propositions(text):
    client = AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",
        api_version="2024-02-15-preview",
    )
    extraction_chain = create_extraction_chain_pydantic(pydantic_schema=Sentences, llm=client)
    output = client.invoke([("human", text)]).content
    
    propositions = extraction_chain.run(output)[0].sentences
    return propositions

def Transformer_DocumentFormat(split_docs, file_id_name):
    metadatas = []
    documents = []
    embeddings = []
    ids = []

    i = 1
    for doc in split_docs :
        metadatas.append(doc.metadata)
        documents.append(doc.page_content)
        
        embeddings.append(embeddings.embed_documents([doc.page_content])[0])
        ids.append([file_id_name+str(i)])
        print (f"Transformer_DocumentFormat Done with {i}")
        i += 1
    print ("Transformer_DocumentFormat Completed")

    ids = [element for sublist in ids for element in sublist]
    return metadatas, documents, embeddings, ids

# Save the processed data to a data store
def save_to_chroma(chunks):
    for chunk in chunks:
        try:
            collection = client_chroma.get_or_create_collection(name="potent", metadata={"hnsw:space": "cosine"})
            metadatas, documents, embeddings, ids = chunk
            collection.upsert(
                documents= documents,
                embeddings= embeddings,
                metadatas = metadatas,
                ids = ids
            )
            print(f'Successfully save into ChromaDB')
        except Exception as e:
            print(e)
    return

def load_documents():
    documents = []
    files = list_files_in_directory()
    for file_name in files:
        # loader = get_loader(file_name)
        loader = UnstructuredExcelLoader(os.path.join(directory_path, file_name))
        docs = loader.load_and_split(RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=500,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )) if loader else None
        print(len(docs))
        # print (pages)

        connection = oracledb.connect(user=username, password=password, dsn=dsn)
        vector_store_dot = OracleVS.from_documents(
            # 作成済のチャンクテキスト
            docs,
            # 定義済の埋め込みモデル
            embeddings,
            # 定義済のデータベースのconnectionオブジェクト
            client=connection,
            # 新規作成する表の名前を任意で指定
            table_name="potent",
            # ベクトル検索時に使う距離計算の方法
            distance_strategy=DistanceStrategy.DOT_PRODUCT,
        )
        oraclevs.create_index(connection, vector_store_dot, params={"idx_name": "potent", "idx_type": "IVF"})
        print(f"done with file: {file_name}")

        # try:
        #     page_result = []
        #     for page in pages:
        #         paragraphs = page.page_content.split("\n\n")
        #         essay_propositions = []
        #         for i, para in enumerate(paragraphs):
        #             print(para)
        #             propositions = get_propositions(para)
                    
        #             essay_propositions.extend(propositions)
        #             print (f"Done with {i}")
        #             print (f"You have {len(essay_propositions)} propositions")
        #             ac = AgenticChunker()
        #             ac.add_propositions(essay_propositions)
        #             ac.pretty_print_chunks()
        #             chunks = ac.get_chunks(get_type='list_of_strings')
        #             print('------chunks in this page------')
        #             print(chunks)
        #             def to_document(d):
        #                 print(str(d))
        #                 doc = Document(
        #                     page_content=str(d),
        #                     metadata=page.metadata
        #                 )
        #                 return doc
        #             docs = map(to_document, chunks)
        #             print('------ to documents ------')
        #             print(docs)
        #             page_result.extend(docs)
        #         print('------chunks done in this page------')
        #         transformed = Transformer_DocumentFormat(page_result, file_name)
        #         save_to_chroma(transformed)
        # except Exception as e:
        #     print(e)
        #     print(f"no file id name with file: {file_name}")
       
    return documents

def query(q):
    # promptの作成
    template = """請參考提供的context回答以下的問題:
    {context}

    問題：{question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # llm = ChatOCIGenAI(
    #     # model_id="cohere.command-r-16k",
    #     model_id="cohere.command-r-plus",
    #     service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    #     compartment_id="ocid1.compartment.oc1..aaaaaaaapq4xxxxxxxxijwewlq",
    #     model_kwargs={"temperature": 0.7, "max_tokens": 500},
    # )
    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    vector_store_dot = OracleVS(
        # # 作成済のチャンクテキスト
        # docs,
        # 定義済のデータベースのconnectionオブジェクト
        connection,
        # 定義済の埋め込みモデル
        embeddings,
        # 新規作成する表の名前を任意で指定
        "potent",
        # ベクトル検索時に使う距離計算の方法
        DistanceStrategy.DOT_PRODUCT,
    )
    retriever = vector_store_dot.as_retriever(search_kwargs={"k": 50})

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(q)
    print(response)
    return response


def test():
    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
        print("Connection successful!")
    except Exception as e:
        print(e)
        print("Connection failed!")

if __name__ == "__main__":
  load_documents()