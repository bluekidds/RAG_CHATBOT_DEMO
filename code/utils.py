from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from config import config
from openai import OpenAI
import chromadb
import tiktoken
import jieba
import json
import time
import re
import os

directory_path = config['DIRECTORY_PATH']
openai_api_key = config['OPENAI_API_KEY']
anyscale_api_key = config["ANYSCALE_API_KEY"]
model = config['EMBEDDING_MODEL_NAME']
max_tokens = config['MAX_TOKENS']
chunk_size = config['CHUNK_SIZE']
chunk_overlap = config['CHUNK_OVERLAP']
num_chunks = config['NUM_CHUNKS']
retrieval_reference = config['RETRIEVAL_REFERENCE']
lexical_search_k = config['LEXICAL_SEARCH_K']
collection_name = config['COLLECTION_NAME']
max_content_lengths = config['MAX_CONTEXT_LENGTHS']
llm = config['LLM']
system_content = config['SYSTEM_CONTENT']
assistant_content = config['ASSISTANT_CONTENT']
temperature = config['TEMPERATURE']

embedding_model = OpenAIEmbeddings(
    model=model,
    openai_api_key=openai_api_key,
)
client_chroma = chromadb.PersistentClient(path='./chromadb/')

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

def get_loader(file_name) :
    file_type = file_name.split('.')[1]
    full_path = os.path.join(directory_path, file_name)
    if (file_type == 'pdf') :
        return PyMuPDFLoader(full_path)
    else :
        return None

# Load documents from a source
def load_documents():
    documents = []
    files = list_files_in_directory()
    for file_name in files:
        loader = get_loader(file_name)
        pages = loader.load() if loader else None
        try:
            file_id_name = config['FILE_ID'][file_name]
            print(f"file_id_name: {file_id_name}")
            documents.append({
                "file_id_name": file_id_name,
                "file_name": file_name,
                "pages": pages,
            })
        except Exception as e:
            print(e)
            print(f"no file id name with file: {file_name}")
       
    return documents


# transform data format
def Transformer_DocumentFormat(split_docs, file_id_name):
    metadatas = []
    documents = []
    embeddings = []
    ids = []

    i = 1
    for doc in split_docs :
        metadatas.append(doc.metadata)
        documents.append(doc.page_content)
        
        embeddings.append(embedding_model.embed_documents([doc.page_content])[0])
        ids.append([file_id_name+str(i)])
        i += 1

    ids = [element for sublist in ids for element in sublist]
    return metadatas, documents, embeddings, ids

# Split documents into manageable chunks
def split_text(documents):
    results = []
    for document in documents:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ".",
                "\u3002"  # Ideographic full stop
            ]
        )
        split_docs = text_splitter.split_documents(document["pages"])
        results.append(Transformer_DocumentFormat(split_docs, document["file_id_name"]))
        print(f"processed file: {document['file_name']}")
    return results

# Save the processed data to a data store
def save_to_chroma(chunks):
    for chunk in chunks:
        try:
            collection = client_chroma.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
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

def save_to_bm25(chunks):
    return

def query(options):
    return

## for open api
def response_stream(chat_completion):
    for chunk in chat_completion:
        content = chunk.choices[0].delta.content
        if content is not None:
            yield content

def prepare_response(chat_completion, stream):
    if stream:
        return response_stream(chat_completion)
    else:
        return chat_completion.choices[0].message.content
def get_client():
    if llm.startswith("gpt"):
        #base_url = os.environ["OPENAI_API_BASE"]
        api_key = openai_api_key
    else:
        api_key = anyscale_api_key
    client = OpenAI(api_key=api_key)
    return client

def get_collection():
    return client_chroma.get_collection(name=collection_name)

## for Query.py
def contains_only_chinese_punctuation(text_list):
    # Define a regular expression pattern to match Chinese punctuation characters
    chinese_punctuation_pattern = re.compile(r'[\u3000-\u303F\uFF00-\uFFEF]+')
    return len(text_list) > 0 and not bool(chinese_punctuation_pattern.sub('', text_list))

def contains_only_newline_characters(text_list) :
    newline_pattern = re.compile(r'^\n+$')
    
    # Filter out elements containing only newline characters
    filtered_list = [text for text in text_list if not newline_pattern.match(text)]
    return filtered_list

def chinese_word_preprocessing(text) :
    tokenized_text = list(jieba.cut(text))
    filtered_text = [text for text in tokenized_text if not contains_only_chinese_punctuation(text)]
    filtered_text = contains_only_newline_characters(filtered_text)
    return filtered_text

def get_num_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def trim(text, max_context_length):   ##if 文本長度過長，超過model的input，則使用trim，只取前面的文本，後面通通忽略
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(enc.encode(text)[:max_context_length])

def handler_source_format( retrieval_reference, file_name, file_page ):
    file_name = file_name.split("/")[-1]
    if (retrieval_reference == 'file_name') :
        source = file_name
    elif (retrieval_reference == 'file_name_and_page') :
        file_page = int(file_page)
        file_page = file_page + 1
        source = f"{file_name} p.{file_page}" 
    return source

def semantic_search(query):
    embedding = embedding_model.embed_query(query)
    collection = get_collection()
    result = collection.query(
        query_embeddings=embedding,
        n_results=num_chunks
    )
    semantic_context = []
    for i in range (num_chunks) :
        source_file_name = result['metadatas'][0][i]['source']
        source_file_page = result['metadatas'][0][i]['page']
        source = handler_source_format( retrieval_reference = retrieval_reference, 
                                        file_name = source_file_name, 
                                        file_page = source_file_page)
        semantic_context.append({'id' :result['ids'][0][i], 
                                 "text" : result['documents'][0][i], 
                                 "source" : source , 
                                 "method" : 'semantic_search'})
    return semantic_context

def lexical_search(index, query, chunks):
    query_tokens = chinese_word_preprocessing(query)
    scores = index.get_scores(query_tokens)  # get best matching (BM) scores
    indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:lexical_search_k]  # sort and get top k

    lexical_context = []
    for i in indices :
        source_file_name = chunks[i]['metadata']['source']
        source_file_page = chunks[i]['metadata']['page']
        source = handler_source_format( retrieval_reference = retrieval_reference, 
                                        file_name = source_file_name, 
                                        file_page = source_file_page)

        lexical_context.append(
            {
                "text": chunks[i]['page_content'], 
                "source": source, 
                "score": scores[i], 
                "method" : 'lexical_search'

            }
        )
   
    return lexical_context

def get_chunks():
    with open('./chunks/0521_ChunkSize500Collection.json', 'r') as file:
        all_chunks = json.load(file)
    return all_chunks

def get_context_length():
    return int(0.8*max_content_lengths[llm]) - get_num_tokens(system_content + assistant_content)

def generate_response(stream, user_content="", max_retries=1, retry_interval=60):
    """Generate response from an LLM."""
    retry_count = 0
    client = get_client()
    messages = [
        {"role": role, "content": content}
        for role, content in [
            ("system", system_content),
            ("assistant", assistant_content),
            ("user", user_content),
        ]
        if content
    ]
    while retry_count <= max_retries:
        try:
            chat_completion = client.chat.completions.create(
                model=llm,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                messages=messages,
            )
            return prepare_response(chat_completion, stream=stream)

        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(retry_interval)  # default is per-minute rate limits
            retry_count += 1
    return ""

def generated_answer_result(query, stream=True, use_lexical_search=True):
    context_results = semantic_search(query=query)
    chunks = get_chunks()

    if use_lexical_search:
        tokenized_text = []
        for chunk in chunks:
            post = chunk['page_content']
            tokenized_text.append(chinese_word_preprocessing(post))
        lexical_index = BM25Okapi(tokenized_text)
        lexical_context = lexical_search(
            index=lexical_index,
            query=query, 
            chunks=chunks
        )
        # Insert after <lexical_search_k> worth of semantic results
        context_results[lexical_search_k:lexical_search_k] = lexical_context
            
        # Generate response
        context = [item["text"] for item in context_results]
        sources = [item["source"] for item in context_results]
        #sources = handler_source_format(sources = sources)
        methods = [item["method"] for item in context_results]
        user_content = f"query: {query}, context: {context}"
        context_length = get_context_length()
        answer = generate_response(stream=stream, user_content=trim(user_content, context_length))

        # Result
        result = {
            "question": query,
            "sources": sources,
            "methods" : methods,
            "answer": answer,
            "llm": llm,
            "context" : context
        }
        return result