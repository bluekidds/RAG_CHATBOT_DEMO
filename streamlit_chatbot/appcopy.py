import asyncio
from typing import Any, Dict

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

AZURE_EMBEDDING_MODEL="text-embedding-3-large"
AZURE_EMBEDDING_ENDPOINT="https://oai-rd.openai.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15"
AZURE_EMBEDDING_DEPLOYMENT="text-embedding-3-large"
AZURE_EMBEDDING_API_KEY="efa7ed84c8d54a1ba3b632363fff992c"
AZURE_EMBEDDING_API_VERSION="2024-05-01-preview"

AZURE_LLM_MODEL="gpt-4o-mini"
AZURE_LLM_ENDPOINT="https://oai-rd.openai.azure.com/openai/deployments/gpt-4o-mini-2024-07-18/chat/completions?api-version=2024-08-01-preview"
AZURE_LLM_DEPLOYMENT="gpt-4o-mini-2024-07-18"
AZURE_LLM_API_KEY="efa7ed84c8d54a1ba3b632363fff992c"
AZURE_LLM_API_VERSION="2024-05-01-preview"




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

# Create a new event loop
loop = asyncio.new_event_loop()

# Set the event loop as the current event loop
asyncio.set_event_loop(loop)

st.set_page_config(
    page_title="信用卡優惠小幫手",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
)

# Default Wikipedia page to chat with
wikipedia_page = "Wavenet"

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": f"Ask me a question about {wikipedia_page}!"}
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
def load_index_data():
    loader = WikipediaReader()
    docs = loader.load_data(pages=[wikipedia_page])
    # Initialize VectorStoreIndex with the documents
    return VectorStoreIndex.from_documents(docs)

# Load the index data and initialize chat engine if needed
index = load_index_data()
if st.session_state["chat_engine"] is None:
    st.session_state["chat_engine"] = index.as_chat_engine(chat_mode="context", verbose=True)

# App title and information
st.title(f"Chat with {wikipedia_page}'s Wikipedia page, powered by LlamaIndex 💬🦙")
st.info(
    "This example is powered by the **[Llama Hub Wikipedia Loader](https://llamahub.ai/l/wikipedia)**. "
    "Use any of [Llama Hub's many loaders](https://llamahub.ai/) to retrieve and chat with your data via a Streamlit app.",
    icon="ℹ️",
)

# Display predefined question pills
selected_question = pills(
    "Choose a question to get started or write your own below.",
    [
        "What is Snowflake?",
        "What company did Snowflake announce they would acquire in October 2023?",
        "What company did Snowflake acquire in March 2022?",
        "When did Snowflake IPO?",
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
if prompt := st.chat_input("Your question"):
    add_to_message_history("user", prompt)
    
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        response = st.session_state["chat_engine"].stream_chat(prompt)
        response_str = ""
        response_container = st.empty()
        for token in response.response_gen:
            response_str += token
            response_container.write(response_str)
        add_to_message_history("assistant", response_str)
