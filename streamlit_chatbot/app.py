import asyncio
from typing import Any, Dict

import json
from llama_index.core.readers.json import JSONReader
from llama_index.core.indices.struct_store import JSONQueryEngine
# Get LLM configurations from environment variables
from llama_index.core import (
    Settings,
    VectorStoreIndex,
)
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

import streamlit as st
from streamlit_pills import pills
import glob 
import os

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

AZURE_EMBEDDING_MODEL="text-embedding-3-large"
AZURE_EMBEDDING_ENDPOINT="https://wavenet-rag-openai.openai.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15"
AZURE_EMBEDDING_DEPLOYMENT="text-embedding-3-large"
AZURE_EMBEDDING_API_KEY="1cc9592cfc254658804d5e2ee8c5e08a"
AZURE_EMBEDDING_API_VERSION="2024-05-01-preview"

AZURE_LLM_MODEL="gpt-4o-mini"
AZURE_LLM_ENDPOINT="https://wavenet-rag-openai.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview"
AZURE_LLM_DEPLOYMENT="gpt-4o-mini"
AZURE_LLM_API_KEY="1cc9592cfc254658804d5e2ee8c5e08a"
AZURE_LLM_API_VERSION="2024-08-01-preview"

Settings.llm = AzureOpenAI(
    engine=AZURE_LLM_DEPLOYMENT,
    model=AZURE_LLM_MODEL,
    api_key=AZURE_LLM_API_KEY,
    azure_endpoint=AZURE_LLM_ENDPOINT,
    api_version=AZURE_LLM_API_VERSION,
    temperature=0.0,
    )
Settings.embed_model = AzureOpenAIEmbedding(
    model=AZURE_EMBEDDING_MODEL,
    deployment_name=AZURE_EMBEDDING_DEPLOYMENT,
    api_key=AZURE_EMBEDDING_API_KEY,
    azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
    api_version=AZURE_EMBEDDING_API_VERSION,
)

# 讀取信用卡 JSON 資料
file_path = "./data/successful_cards_ctbc.json"
with open(file_path, "r", encoding="utf-8") as file:
    json_value = json.load(file)

# Create a new event loop
loop = asyncio.new_event_loop()

# Set the event loop as the current event loop
asyncio.set_event_loop(loop)



# 設定頁面配置
st.set_page_config(
    page_title="💳 卡神爺 - 你的信用卡優惠軍師",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded",
)

# 設定 Streamlit 主題顏色
st.markdown(
    """
    <style>
    /* 設定背景顏色 */
    .main {
        background-color: #F5F5F5;
    }
    
    /* 調整標題和文字顏色 */
    h1, h2, h3, h4, h5, h6, p {
        color: #333333;
    }

    /* 側邊欄樣式 */
    .sidebar .sidebar-content {
        background-color: #F5F5F5;
    }
    
    /* 按鈕樣式 */
    .stButton>button {
        background-color: #005BAC;
        color: #FFFFFF;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        font-weight: bold;
    }

    /* 輸入框樣式 */
    .stTextInput>div>input {
        border: 2px solid #005BAC;
        border-radius: 8px;
    }

    /* 標題樣式 */
    .title {
        color: #005BAC;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# 加入側邊欄，展示 Wavenet 的資訊
with st.sidebar:
    # 顯示 Wavenet 的 logo
    st.image("./img/wavenet_logo.jpg", use_container_width=True)
    
    # 公司名稱與介紹
    st.markdown("## Wavenet 潮網科技")
    st.markdown(
        """
        潮網科技，致力於提供領先市場的創新解決方案，包括 AI 智能應用、
        雲端服務與數據分析，幫助企業實現數位轉型，提升競爭優勢。
        """
    )
    
    # 官網連結
    st.markdown(
        """
        🌐 [造訪潮網官網](https://www.wavenet.com.tw/)
        """
    )

# 預設查詢的 Wikipedia 頁面
wikipedia_page = "信用卡優惠"

# 初始化 session state 變數
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": f"歡迎來到卡神爺！請問有什麼關於 {wikipedia_page} 的問題想詢問呢？"}
    ]
if "chat_engine" not in st.session_state:
    st.session_state["chat_engine"] = None
if "displayed_pill_questions" not in st.session_state:
    st.session_state["displayed_pill_questions"] = set()


# Function to add messages to the chat history
def add_to_message_history(role: str, content: str):
    message = {"role": role, "content": str(content)}
    st.session_state["messages"].append(message)

# Load index data from Wikipedia if not cached
@st.cache_resource
def load_and_index_json(directory_path):
    reader = JSONReader(
        levels_back=0,             # Set levels back as needed
        collapse_length=None,      # Set collapse length as needed
        ensure_ascii=False,        # ASCII encoding option
        is_jsonl=False,            # Set if input is JSON Lines format
        clean_json=True            # Clean up formatting-only lines
    )

    # Find all JSON files in the specified directory
    json_files = glob.glob(os.path.join(directory_path, "*.json"))

    # Load the data from each JSON file
    documents = []
    for json_file in json_files:
        documents.extend(reader.load_data(input_file=json_file, extra_info={}))

    # Create an index for querying
    index = VectorStoreIndex.from_documents(documents)
    return index
# Specify the directory containing your JSON files
json_directory = "./data"


@st.cache_resource
def load_index_data():
    loader = WikipediaReader()
    docs = loader.load_data(pages=[wikipedia_page])
    # Initialize VectorStoreIndex with the documents
    return VectorStoreIndex.from_documents(docs)

# Load the index data and initialize chat engine if needed
#index = load_index_data()
index = load_and_index_json(json_directory)

# wikipedia_page = "Wavenet"
# # Load index data from Wikipedia if not cached
# @st.cache_resource
# def load_index_data():
#     loader = WikipediaReader()
#     docs = loader.load_data(pages=[wikipedia_page])
#     # Initialize VectorStoreIndex with the documents
#     return VectorStoreIndex.from_documents(docs)

# # Load the index data and initialize chat engine if needed
# index = load_index_data()

if st.session_state["chat_engine"] is None:
    st.session_state["chat_engine"] = index.as_chat_engine(chat_mode="condense_plus_context", 
                                                           system_prompt="你是一個親切專業的信用卡專家，可以幫我解答一些問題嗎？",
                                                           verbose=True)

# App title and information
st.title("💳 卡神爺 - 你的信用卡優惠軍師")
st.info(
    "這裡是 **卡神爺**！專為台灣使用者設計，提供最新的信用卡優惠查詢。不論是現金回饋、旅遊保險、還是百貨公司聯名卡的折扣，"
    "我們幫你迅速找到最符合需求的卡片資訊！選擇熱門問題開始，或直接輸入您的問題。",
    icon="✨",
)

# Display predefined question pills
selected_question = pills(
    "選擇熱門問題或自行輸入您的問題：",
    [
        "有哪些信用卡提供最高的現金回饋？",
        "哪一張卡片有機場接送服務？",
        "有哪些卡片包含旅遊保險？",
        "有什麼聯名卡提供百貨公司折扣？",
        "紅利點數可以怎麼累積和兌換現金？",
    ],
    clearable=True,
    index=None,
)

# Display previous messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Process selected question from pills
if selected_question and selected_question not in st.session_state["displayed_pill_questions"]:
    st.session_state["displayed_pill_questions"].add(selected_question)
    with st.chat_message("user"):
        st.write(selected_question)
    with st.chat_message("assistant"):
        response = st.session_state["chat_engine"].stream_chat(selected_question)
        response_str = ""
        response_container = st.empty()
        for token in response.response_gen:
            response_str += token
            response_container.write(response_str)
        add_to_message_history("user", selected_question)
        add_to_message_history("assistant", response_str)

# Chat input box for user questions
user_input = st.chat_input(placeholder="例如：請問中國信託 Global Mall 聯名卡有什麼優惠？")

# Handle user input
if user_input:
    add_to_message_history("user", user_input)

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        response = st.session_state["chat_engine"].stream_chat(user_input)
        response_str = ""
        response_container = st.empty()
        for token in response.response_gen:
            response_str += token
            response_container.write(response_str)
        add_to_message_history("assistant", response_str)


# 查詢提示框
#st.markdown("### 查詢提示")
#prompts = st.session_state["chat_engine"]()
#for prompt in prompts:
#    st.markdown(f"- {prompt}")