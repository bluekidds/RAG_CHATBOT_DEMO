from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import oraclevs
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

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

# Define the metadata extraction function.
def metadata_func(record: dict, metadata: dict) -> dict:
    # print(record)
    metadata["卡片名稱"] = record.get("卡片名稱")
    return metadata


keys = [
"循環利息",
"申辦條件",
"機場貴賓室",
"循環起息日",
"逾期違約金",
"旅遊保險",
"道路救援",
"紅利集點",
"國外簽帳消費手續費",
"溢付款退回匯款處理費",
"緊急替代卡手續費",
"一般停車優惠",
"預借現金",
"轉卡手續費",
"信用卡代繳",
"年費",
"現金回饋",
"點數折抵現金",
"調閱簽帳單費用",
"機場停車優惠",
"哩程數累積",
"掛失補卡費用",
"掛失自負",
"補寄帳單費用",
"機場接送",
"聯名優惠",
"購物優惠",
]

def load_json():
    docs = []
    for key in keys:
        # print(key)
        loader = JSONLoader(
            file_path='./code/23ai.json',
            jq_schema='.[]',
            content_key='.["內容"].["' + key + '"]',
            text_content=False,
            is_content_key_jq_parsable=True,
            metadata_func=metadata_func
        )
        doc = loader.load()
        docs = docs + doc
        # embeddings = OCIGenAIEmbeddings(
        #     model_id="cohere.embed-multilingual-v3.0",
        #     service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        #     compartment_id="ocid1.compartment.oc1..aaaaaaaxxxxxxx",
        # )

    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    vector_store_dot = OracleVS.from_documents(
        # 作成済のチャンクテキスト
        docs,
        # 定義済の埋め込みモデル
        embeddings,
        # 定義済のデータベースのconnectionオブジェクト
        client=connection,
        # 新規作成する表の名前を任意で指定
        table_name="doc_table",
        # ベクトル検索時に使う距離計算の方法
        distance_strategy=DistanceStrategy.DOT_PRODUCT,
    )
    oraclevs.create_index(connection, vector_store_dot, params={"idx_name": "card", "idx_type": "IVF"})

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
        "doc_table",
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
  load_json()