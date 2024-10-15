from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_chroma import Chroma
from langchain.docstore.document import Document
# from langchain_community.callbacks import get_openai_callback
from agentic_chunker import AgenticChunker
from langchain.output_parsers.openai_tools import JsonOutputToolsParser
# from langchain_community.chat_models import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.chains import create_extraction_chain
from typing import Optional, List
from langchain.chains import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel
from langchain import hub
from rank_bm25 import BM25Okapi
from config import config
# from openai import OpenAI
from openai import AzureOpenAI
import pandas as pd
import numpy as np
import chromadb
import tiktoken
import pickle
import jieba
import json
import time
import re
import os

directory_path = config['DIRECTORY_PATH']
openai_api_key = config['OPENAI_API_KEY']
anyscale_api_key = config["ANYSCALE_API_KEY"]
azure_openai_endpoint = config['AZURE_OPENAI_ENDPOINT']

llm = config['LLM']
model = config['EMBEDDING_MODEL_NAME']
max_tokens = config['MAX_TOKENS']
chunk_size = config['CHUNK_SIZE']
chunk_overlap = config['CHUNK_OVERLAP']
num_chunks = config['NUM_CHUNKS']
retrieval_reference = config['RETRIEVAL_REFERENCE']
lexical_search_k = config['LEXICAL_SEARCH_K']
collection_name = config['COLLECTION_NAME']
max_content_lengths = config['MAX_CONTEXT_LENGTHS']
system_content = config['SYSTEM_CONTENT']
assistant_content = config['ASSISTANT_CONTENT']
temperature = config['TEMPERATURE']
chunk_settings = config['CHUNKS']

evaluation_llm = config['EVALUATION_LLM']
evaluation_system_content = config['EVALUATION_SYSTEM_CONTENT']


def get_config(key):
    return config[key]

# embedding_model = OpenAIEmbeddings(
#     model=model,
#     openai_api_key=openai_api_key,
# )
os.environ["AZURE_OPENAI_API_KEY"] = openai_api_key
os.environ["AZURE_OPENAI_ENDPOINT"] = azure_openai_endpoint
embedding_model = AzureOpenAIEmbeddings(
    azure_deployment="wavenet-rag-embedding",
    openai_api_version="2023-12-01-preview",
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
        print (f"Transformer_DocumentFormat Done with {i}")
        i += 1
    print ("Transformer_DocumentFormat Completed")

    ids = [element for sublist in ids for element in sublist]
    return metadatas, documents, embeddings, ids

def get_client():
    # client = OpenAI(api_key=api_key)
    client = AzureOpenAI(
        api_key=openai_api_key,  
        api_version="2023-12-01-preview",
        azure_endpoint="https://wavenet-rag-openai.openai.azure.com/"
    )
    return client

# class for level5
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

# Split documents into manageable chunks
def split_text(documents):
    results = []
    for document in documents:
        print(document["pages"])
        page_result = []
        

        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=chunk_size,
        #     chunk_overlap=chunk_overlap,
        #     separators=[
        #         "\n\n",
        #         "\n",
        #         ".",
        #         "\u3002"  # Ideographic full stop
        #     ]
        # )
        # split_docs = text_splitter.split_documents(document["pages"])
        # print(split_docs)

        for page in document["pages"]:
            print(page)
            paragraphs = page.page_content.split("\n\n")
            print(paragraphs)
            print(len(paragraphs))
            essay_propositions = []
            for i, para in enumerate(paragraphs):
                print(para)
                propositions = get_propositions(para)
                
                essay_propositions.extend(propositions)
                print (f"Done with {i}")
            print (f"You have {len(essay_propositions)} propositions")
            ac = AgenticChunker()
            ac.add_propositions(essay_propositions)
            ac.pretty_print_chunks()
            chunks = ac.get_chunks(get_type='list_of_strings')
            print('------chunks in this page------')
            print(chunks)
            def to_document(d):
                print(str(d))
                doc = Document(
                    page_content=str(d),
                    metadata=page.metadata
                )
                return doc
            docs = map(to_document, chunks)
            print('------ to documents ------')
            print(docs)
            page_result.extend(docs)

        # results.append(Transformer_DocumentFormat(split_docs, document["file_id_name"]))
        print('------chunks done in this page------')
        results.append(Transformer_DocumentFormat(page_result, document["file_id_name"]))
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

def get_chroma_json():
    collection = client_chroma.get_collection(name=collection_name)
    return collection.get()

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
        print(result['documents'])
        source_file_name = result['metadatas'][0][i]['source']
        source_file_page = result['metadatas'][0][i]['page']
        source = handler_source_format( retrieval_reference = retrieval_reference, 
                                        file_name = source_file_name, 
                                        file_page = source_file_page)
        semantic_context.append({'id' :result['ids'][0][i], 
                                 "text" : result['documents'][0][i], 
                                 "source" : source , 
                                 "sourcetext" : result['documents'][0],
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

def get_chunks(name):
    print(name)
    with open(name, 'r') as file:
        all_chunks = json.load(file)
    return all_chunks

def get_context_length(content = system_content + assistant_content):
    return int(0.8*max_content_lengths[llm]) - get_num_tokens(content)

def generate_response(stream, llm=llm, system_content=system_content, user_content="", max_retries=1, retry_interval=60):
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
            print('asking gpt')
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

def generated_answer_result(query, lexical_search_k = lexical_search_k, llm=llm, stream=True, use_lexical_search=True):
    context_results = semantic_search(query=query)
    chunks = get_chunks(chunk_settings[chunk_size]['json_file'])

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
        # sourcetexts = [item["sourcetext"] for item in context_results]
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
    
## for evaluation
def get_retrieval_score(references, generated):
    matches = np.zeros(len(references))
    for i in range(len(references)):
        reference_source = references[i]
        if not reference_source:
            matches[i] = 1
            continue
        for source in generated[i]:
            # sections don't have to perfectly match
            if (source == reference_source) :
                matches[i] = 1
                continue

    retrieval_score = np.mean(matches)
    return retrieval_score

def get_answer_evaluation(query, generated_answer, reference_answer, llm):
    # Generate response
    context_length = get_context_length(evaluation_system_content)
    user_content = trim(
            str(
                {
                    "query": query,
                    "generated_answer": generated_answer,
                    "reference_answer": reference_answer,
                }
            ),
            context_length,
        )

    response = generate_response(
    llm=llm,
    stream=False,
    system_content=evaluation_system_content,
    user_content=user_content)

    score, reasoning = response.split("\n", 1) if "\n" in response else (0, "")
    result = {
        "question": query,
        "generated_answer": generated_answer,
        "reference_answer": reference_answer,
        "score": float(score),
        "reasoning": reasoning.lstrip("\n")
    }

    return result

def get_evaluation_average_score (evaluations) :
    evaluation_score_accumulation = 0
    for i in range(len(evaluations)) :
        evaluation_score_accumulation += evaluations[i]['score']
    
    average_evaluation_score = evaluation_score_accumulation/len(evaluations)
    return average_evaluation_score

def get_qa_datasets():
    QA_dataset = pd.read_excel('QA Dataset.xlsx')
    return QA_dataset

def evaluate_RetrievalScore_AnswerQuality(use_lexical_search,
                                          max_context_length, 
                                          #   queries_dataset, 
                                          chunks,
                                          lexical_index,
                                          llm_answer,
                                          llm_evaluate,
                                          num_chunks, 
                                          lexical_search_k,
                                          chunk_size,
                                          #   vectordb_collection, 
                                          #   retrieval_reference, 
                                          stream
                                         ) :
    
    #embedding_model_name = "text-embedding-ada-002"
    #llm = "gpt-3.5-turbo-1106"
    
    start_time = time.time()
    #queries = queries_dataset['question'].to_list()

    generated_references = []
    evaluation = []
    queries_dataset = get_qa_datasets()
    for index, row in queries_dataset.iterrows() : 

        start_question_time = time.time()
        result = generated_answer_result(
            lexical_search_k=lexical_search_k,
            llm=llm_answer,
            query=row['question'],
            stream=False
        )

        end_question_time = time.time()

        generated_references.append(result['sources'])
        
        evaluate_generated_answer_result = get_answer_evaluation(
            query =row['question'], 
            generated_answer = result['answer'], 
            reference_answer = row['reference_answer'],
            llm=llm_evaluate
        )

        question_time = end_question_time - start_question_time
        evaluate_generated_answer_result['methods'] = result['methods']
        evaluate_generated_answer_result['generate_answer_time_spent(seconds)'] = question_time
        evaluate_generated_answer_result['sources'] = result['sources']
        evaluate_generated_answer_result['context'] = result['context']
        evaluation.append(evaluate_generated_answer_result)
    
    real_references = ""
    if (retrieval_reference == 'file_name'):
        real_references = queries_dataset['source'].to_list()
    elif (retrieval_reference == 'file_name_and_page'):
        queries_dataset['page'] = queries_dataset['page'].astype(str)
        real_references = queries_dataset['source'] + " p." + queries_dataset['page']
        real_references = real_references.to_list()

    retrieval_score = get_retrieval_score( real_references, generated_references )
    
    average_evaluation_score = get_evaluation_average_score(evaluation)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    final_result = {
        'retrieval_score' : retrieval_score, 
        'average_evaluation_score' : average_evaluation_score,
        'num_chunks' : num_chunks, 
        'chunk_size' :chunk_size,
        'num_lexical_search_chunks' : lexical_search_k if use_lexical_search else 0,
        'embedding_model' : model,
        'detailed_evaluation' : evaluation,
        'time_spent' : pd.DataFrame(evaluation)['generate_answer_time_spent(seconds)'].sum()
        
    }
    return final_result

def get_evaluation_result(chunk_size_list=[],
        num_chunks_list=[],
        lexical_search_k_list=[],
        llm_answer=llm,
        llm_evaluate=llm):
    result = []
    for chunk_size in chunk_size_list:
        all_chunks = get_chunks(chunk_settings[chunk_size]['json_file'])
        with open(chunk_settings[chunk_size]['lexial_index_file'], 'rb') as bm25result_file:
            lexical_index = pickle.load(bm25result_file)

        for num_chunks in num_chunks_list:
            for k in lexical_search_k_list :

                final_result = evaluate_RetrievalScore_AnswerQuality(
                    use_lexical_search = True,
                    chunks = all_chunks,
                    lexical_index = lexical_index,
                    llm_answer = llm_answer,
                    llm_evaluate = llm_evaluate,
                    max_context_length = max_content_lengths[llm_answer],
                    num_chunks = num_chunks,
                    lexical_search_k = k,
                    chunk_size = chunk_size,
                    stream = False
                )

                print(f'Test - num_chunks:{num_chunks}, lexical_search_chunk:{k}, chunk_size:{chunk_size} - finished')
                result.append(final_result)
        # 將評估的結果儲存
    result_df = pd.DataFrame(result)
    result_df.drop(['detailed_evaluation'], axis=1).to_csv( path_or_buf = 'evaluations/Experiment_Result.csv')

    for index, row in result_df.iterrows():
        data = pd.DataFrame(row['detailed_evaluation'])
        data.to_excel(excel_writer = 'evaluations/ChunkSize{}_NumChunks{}_LexicalSearchChunks{}_DetailedEvaluation.xlsx'.format(row['chunk_size'], row['num_chunks'], row['num_lexical_search_chunks']))

    return result

def dalle3(prompt):
    client = get_client()
    result = client.images.generate(
        model="Dalle3", # the name of your DALL-E 3 deployment
        prompt=prompt,
        n=1
    )

    json_response = json.loads(result.model_dump_json())
    image_url = json_response["data"][0]["url"]
    return image_url