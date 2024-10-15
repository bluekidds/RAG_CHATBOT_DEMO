from typing import Any

from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings

from config import config
import glob
import os
import uuid

openai_api_key = config['OPENAI_API_KEY']
anyscale_api_key = config["ANYSCALE_API_KEY"]
azure_openai_endpoint = config['AZURE_OPENAI_ENDPOINT']

os.environ["AZURE_OPENAI_API_KEY"] = openai_api_key
os.environ["AZURE_OPENAI_ENDPOINT"] = azure_openai_endpoint
# embedding_model = AzureOpenAIEmbeddings(
#     azure_deployment="wavenet-rag-embedding",
#     openai_api_version="2023-12-01-preview",
# )

# client_chroma = chromadb.PersistentClient(path='./chromadb/')

path = "./files/"
print(os.path.join(path, "McKinsey-Tech-Trends-Outlook-2022-Mobility.pdf"))

# Get elements
raw_pdf_elements = partition_pdf(
    filename=os.path.join(path, "McKinsey-Tech-Trends-Outlook-2022-Mobility.pdf"),
    # Using pdf format to find embedded image blocks
    extract_images_in_pdf=True,
    # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
    # Titles are any sub-section of the document
    infer_table_structure=True,
    # Post processing to aggregate text once we have the title
    chunking_strategy="by_title",
    # Chunking params to aggregate text blocks
    # Attempt to create a new chunk 3800 chars
    # Attempt to keep chunks > 2000 chars
    # Hard max on chunks
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=path,
)

# Create a dictionary to store counts of each type
category_counts = {}

for element in raw_pdf_elements:
    category = str(type(element))
    if category in category_counts:
        category_counts[category] += 1
    else:
        category_counts[category] = 1

# Unique_categories will have unique elements
unique_categories = set(category_counts.keys())
category_counts

class Element(BaseModel):
    type: str
    text: Any

# Categorize by type
categorized_elements = []
for element in raw_pdf_elements:
    print(str(type(element)))
    print(element)
    if "unstructured.documents.elements.Table" in str(type(element)):
        categorized_elements.append(Element(type="table", text=str(element)))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        categorized_elements.append(Element(type="text", text=str(element)))

# Tables
table_elements = [e for e in categorized_elements if e.type == "table"]
print(len(table_elements))

# Text
text_elements = [e for e in categorized_elements if e.type == "text"]
print(len(text_elements))

# Prompt
prompt_text = """You are an assistant tasked with summarizing tables and text. \
Give a concise summary of the table or text. Table or text chunk: {element} """
prompt = ChatPromptTemplate.from_template(prompt_text)

# Summary chain
# model = ChatOpenAI(temperature=0, model="gpt-4")
model = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_version="2024-02-15-preview",
)
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

# Apply to text
texts = [i.text for i in text_elements]
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 2})

# Apply to tables
tables = [i.text for i in table_elements]
table_summaries = summarize_chain.batch(tables, {"max_concurrency": 2})

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="summaries", embedding_function=AzureOpenAIEmbeddings(
    azure_deployment="wavenet-rag-embedding",
    openai_api_version="2023-12-01-preview",
))

# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# Add texts
doc_ids = [str(uuid.uuid4()) for _ in texts]
summary_texts = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(text_summaries)
]
retriever.vectorstore.add_documents(summary_texts)
retriever.docstore.mset(list(zip(doc_ids, texts)))

# Add tables
table_ids = [str(uuid.uuid4()) for _ in tables]
summary_tables = [
    Document(page_content=s, metadata={id_key: table_ids[i]})
    for i, s in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, tables)))

tables[2]

table_summaries[2]

print('-----------invode the result example----------')
# We can retrieve this table
retriever.invoke("What are results for LLaMA across across domains / subjects?")[1]

retriever.invoke("Images / figures with playful and creative examples")[1]

def get_tti_chroma_json():
    return vectorstore.get()

def retrieve(q):
    return retriever.invoke(q)[1]    