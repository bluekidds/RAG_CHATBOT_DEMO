import aiohttp
import requests
import re

import time

from langchain_openai import AzureChatOpenAI

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain_community.callbacks import get_openai_callback

from code.Query import res_only as res
from code.Query import multi_res_only

import asyncio
import os
from code.config import config
openai_api_key = config['OPENAI_API_KEY']
azure_openai_endpoint = config['AZURE_OPENAI_ENDPOINT']

os.environ["AZURE_OPENAI_API_KEY"] = openai_api_key
os.environ["AZURE_OPENAI_ENDPOINT"] = azure_openai_endpoint

# API key
model = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_version="2024-02-15-preview",
)

# API URL
url = 'https://rag.streamlit.punwave.com/query/'

# headers, token
headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'token': 'Basic YWRtaW45OTk6YWRtaW4tMTIzNDU2'
}

def combine_sources(responses):
    """
    接收一個包含多個 response 的列表，每個 response 為字典格式，包含 "source" 欄位，
    並將所有 response 中的 source 去重後回傳為一個列表。
    """
    source_set = set()
    for r in responses:
        # 如果 "source" 是 list 型態
        if isinstance(r["source"], list):
            for src in r["source"]:
                source_set.add(src)
        # 如果 "source" 為字串則直接加入
        elif isinstance(r["source"], str):
            source_set.add(r["source"])
    return list(source_set)

async def BM1(input_TA, input_LO, input_BT, lang: str = None):
    start_time = time.time()
    ## prompt P
    # 參數定義
    pestel = "政治"
    pestel_whole = "政治面向（政治 Political）"
    pestel_sub = ["政策支持","減碳淨零"]
    objective_item_PESTEL_P = """
    - 當地政府對相關產業的政策提倡或補助計畫
    - 各國的減碳宣言與相關制度
    """
    P_json_sample = """
    {
        'pestel':'政治',
        'pestel_info':[
        {
            'pestel_sub':'政策支持',
            'content':'分析內文 字數約為80-90個字'
            },
        {
            'pestel_sub':'減碳淨零',
            'content':'分析內文 字數約為80-90個字'
            }
        ],
        "source": [

        ]
    }
    """

    # Define policy search queries
    P_1 = f"檢索主要國家對{input_BT}的政策支持，特別是補貼與技術發展措施。"
    P_2 = f"檢索各國關於2050年淨零排放目標的交通工具政策，並強調{input_BT}產業的支持與措施。"
    P_3 = f"檢索歐盟或其他地區針對{input_BT}產業的減碳規範及政策推動情況。"
    P_4 = f"檢索全球減碳運動中涉及{input_BT}產業的政策宣言與規範。"
    P_5 = f"檢索各地政府在推廣低碳運輸方面的政策支持與補助，特別針對{input_BT}產業的設施與規範。"

    queries = [P_1, P_2, P_3, P_4, P_5]

    t0 = time.time()

    # 平行觸發 5 次 res()
    # responses = await asyncio.gather(*[res(q) for q in queries])

    responses = await multi_res_only(queries)

    # 平行觸發 5 次 res()
    # responses = await asyncio.gather(*[res(q) for q in queries])

    t1 = time.time()

    print("===gpt costs===")
    print(t1 - t0)

    # 合併回應內容
    # rag_responses_P = "".join([r["answer"] for r in responses])

    rag_responses_P = "".join([r["answer"] for r in responses])
    sources = combine_sources(responses)
    all_sources = ", ".join(str(src) for src in sources)

    print(f"Total time: {t1 - t0:.2f} seconds")

    # p0_time = time.time()
    # response_P_1 = res(P_1)
    # p1_time = time.time()
    # response_P_2 = res(P_2)
    # p2_time = time.time()
    # response_P_3 = res(P_3)
    # p3_time = time.time()
    # response_P_4 = res(P_4)
    # p4_time = time.time()
    # response_P_5 = res(P_5)
    # p5_time = time.time()

    # rag_responses_P = response_P_1 + response_P_2 + response_P_3 + response_P_4 + response_P_5
    # print(rag_responses_P)

    full_P_template = """
    {context}
    {objective}
    {style}
    {tone}
    {audience}
    {responce}
    """
    full_P_prompt = PromptTemplate.from_template(full_P_template)

    # P "contex" template
    context_P_template = """
    # Context
    你是一位專業的產業管理顧問。掌握全球產業重要發展動態，從產業市場、地區經濟、地方基礎建設、國家政策與社會文化等面向進行全面的分析，協助客戶探勘產業趨勢情報、洞見技術與創新需求脈絡，貼近產業立場，提供客戶全方位建言。
    """
    context_P_prompt = PromptTemplate.from_template(context_P_template)

    # P "objective" template
    objective_P_template = """
    # Objective:
    你的任務是以PESTEL分析法同時參考下方參考文獻並結合自身看法，針對特定國家或區域的{BT}產業現況進行"{pestel_whole}"的分析，分析時要考量以下幾個面向與指標
    {objective_item_PESTEL_P}

    參考文獻：
    {rag_responses_P}

    文獻來源：
    {all_sources}

    """
    objective_P_prompt = PromptTemplate.from_template(objective_P_template)

    # P "style" template
    style_P_template = """
    # Style:
    內容能夠輕鬆理解。
    刪除任何無關緊要的文本，無關緊要的文本示例："嗯" 修正任何明顯的拼寫錯誤。
    以列點的方式進行描述。

    你必須將輸出結構化為一組json格式。
    並只能以'pestel','pestel_sub','pestel_info'三個元素作為json的註解
    其中須以'{pestel_sub}'作為 'pestel_info'中的每一個'pestel_sub'分項, 不要自行發散出其他的分項

    json是一種聲明性語言，可讓你對文件進行註解和確認。
    你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全匹配！
    範例如下：

    {P_json_sample}
    """
    style_P_prompt = PromptTemplate.from_template(style_P_template)

    # P "tone" template
    tone_P_template = """
    # Tone:
    在整個過程中保持清晰和有條理的語氣，具備說服力地說明產品與特定地區的{pestel}關係。具備清楚且有方向的分析，提供有價值的見解和建議。
    """
    tone_P_prompt = PromptTemplate.from_template(tone_P_template)

    # P "audience" template
    audience_P_template = """
    # Audience:
    目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
    """
    audience_P_prompt = PromptTemplate.from_template(audience_P_template)

    # P "responce" template
    if lang == 'en':
        responce_P_template = """
    #Responce:
    - 請用en英文進行回答。
    - 用PESTEL分析{LO}地區的{BT}產業，... en英文版的'{pestel_sub}'...

    #Start:
    如果您明白了，請開始進行分析
    """
    else:
        responce_P_template = """
    #Responce:
    用PESTEL分析{LO}地區的{BT}產業，... '{pestel_sub}' ...

    #Start:
    如果您明白了，請開始進行分析
    """
    responce_P_prompt = PromptTemplate.from_template(responce_P_template)

    # P arragement
    input_P_prompts = [
        ("context", context_P_prompt),
        ("objective", objective_P_prompt),
        ("style", style_P_prompt),
        ("tone", tone_P_prompt),
        ("audience", audience_P_prompt),
        ("responce", responce_P_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_P_prompt, pipeline_prompts=input_P_prompts
    )

    # final P prompt
    P_prompt = pipeline_prompt.format(
            BT = input_BT,
            LO = input_LO,
            pestel = pestel,
            pestel_sub = pestel_sub,
            rag_responses_P = str(rag_responses_P),
            all_sources = str(all_sources),
            pestel_whole = pestel_whole,
            objective_item_PESTEL_P = objective_item_PESTEL_P,
            P_json_sample = P_json_sample
            )

    P_result = model.predict(
        text=P_prompt)
    predict_time = time.time()

    return P_result

async def BM2(input_TA, input_LO, input_BT, lang: str = None):
    # 參數定義
    pestel = "經濟"
    pestel_whole = "經濟面向（經濟 Economic)"
    pestel_sub = ["地區經濟", "進出口概況"]
    objective_item_PESTEL_E = """
    - 當地目前整體的經濟狀況與消費力的上升/下降狀況，並說明原因
    - 產品全球的進出口走向以及當地產品的進出口狀況
    - 未來原物料價格的走勢及對公司利潤的影響
    """
    E_json_sample = """
    {
        'pestel':'政治',
        'pestel_info':[
        {
            'pestel_sub':'政策支持',
            'content':'分析內文 字數約為80-90個字'
            },
        {
            'pestel_sub':'減碳淨零',
            'content':'分析內文 字數約為80-90個字'
            }
        ],
        "source": [

        ]
    }
    """
    # E_RAG

    # Define policy search queries
    E_1 = f"請檢索全球環保與健康意識提升對{input_BT}市場需求的影響，包含共享系統普及和{input_BT}作為健身工具的市場趨勢。"
    E_2 = f"請檢索都市化和交通擁堵加劇對{input_BT}成為城市短途出行工具的經濟需求推動。"
    E_3 = f"請檢索如印度和東南亞等新興市場的經濟成長對{input_BT}需求的影響及市場潛力。"
    E_4 = f"請檢索歐美國家防傾銷政策、ECFA及美中貿易戰等政策對中國和台灣{input_BT}產業出口的影響。"
    E_5 = f"請預估2025年全球{input_BT}產業的市場表現，包括庫存管理問題的改善可能性和經濟模式變化。"


    # Construct URLs
    # RAG_E_1 = f"{url}{E_1}"
    # RAG_E_2 = f"{url}{E_2}"
    # RAG_E_3 = f"{url}{E_3}"
    # RAG_E_4 = f"{url}{E_4}"
    # RAG_E_5 = f"{url}{E_5}"
    # response_E_1 = res(E_1)
    # response_E_2 = res(E_2)
    # response_E_3 = res(E_3)
    # response_E_4 = res(E_4)
    # response_E_5 = res(E_5)
    # rag_responses_E = response_E_1 + response_E_2 + response_E_3 + response_E_4 + response_E_5
    queries = [E_1, E_2, E_3, E_4, E_5]

    t0 = time.time()

    # 平行觸發 5 次 res()
    # responses = await asyncio.gather(*[res(q) for q in queries])

    responses = await multi_res_only(queries)

    # 平行觸發 5 次 res()
    # responses = await asyncio.gather(*[res(q) for q in queries])

    t1 = time.time()

    print("===gpt costs===")
    print(t1 - t0)

    # 合併回應內容
    rag_responses_E = "".join([r["answer"] for r in responses])
    sources = combine_sources(responses)
    all_sources = ", ".join(str(src) for src in sources)


    print(f"Total time: {t1 - t0:.2f} seconds")

    # E full costar template
    full_E_template = """
    {context}
    {objective}
    {style}
    {tone}
    {audience}
    {responce}
    """
    full_E_prompt = PromptTemplate.from_template(full_E_template)

    # E "contex" template
    context_E_template = """
    # Context
    你是一位專業的產業管理顧問。掌握全球產業重要發展動態，從產業市場、地區經濟、地方基礎建設、國家政策與社會文化等面向進行全面的分析，協助客戶探勘產業趨勢情報、洞見技術與創新需求脈絡，貼近產業立場，提供客戶全方位建言。
    """
    context_E_prompt = PromptTemplate.from_template(context_E_template)

    # E "objective" template
    objective_E_template = """
    # Objective:
    你的任務是以PESTEL分析法同時參考下方參考文獻並結合自身看法，針對特定國家或區域的{BT}產業現況進行"{pestel_whole}"的分析，分析時要考量以下幾個面向與指標
    {objective_item_PESTEL_E}

    參考文獻：
    {rag_responses_E}

    文獻來源：
    {all_sources}

    """
    objective_E_prompt = PromptTemplate.from_template(objective_E_template)

    # E "style" template
    style_E_template = """
    # Style:
    內容能夠輕鬆理解。
    刪除任何無關緊要的文本，無關緊要的文本示例："嗯" 修正任何明顯的拼寫錯誤。
    以列點的方式進行描述。

    你必須將輸出結構化為一組json格式。
    並只能以'pestel','pestel_sub','pestel_info'三個元素作為json的註解
    其中須以'{pestel_sub}'作為 'pestel_info'中的每一個'pestel_sub'分項, 不要自行發散出其他的分項

    json是一種聲明性語言，可讓你對文件進行註解和確認。
    你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全匹配！
    範例如下：

    {E_json_sample}
    """
    style_E_prompt = PromptTemplate.from_template(style_E_template)

    # E "tone" template
    tone_E_template = """
    # Tone:
    在整個過程中保持清晰和有條理的語氣，具備說服力地說明產品與特定地區的{pestel}關係。具備清楚且有方向的分析，提供有價值的見解和建議。
    """
    tone_E_prompt = PromptTemplate.from_template(tone_E_template)

    # E "audience" template
    audience_E_template = """
    # Audience:
    目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
    """
    audience_E_prompt = PromptTemplate.from_template(audience_E_template)

    # E "responce" template
    if lang == 'en':
        responce_E_template = """
    #Responce:
    - 請用en英文進行回答。
    - 用PESTEL分析{LO}地區的{BT}產業，... en英文版的'{pestel_sub}'...

    #Start:
    如果您明白了，請開始進行分析
    """
    else:
        responce_E_template = """
    #Responce:
    用PESTEL分析{LO}地區的{BT}產業，... '{pestel_sub}' ...

    #Start:
    如果您明白了，請開始進行分析
    """
    responce_E_prompt = PromptTemplate.from_template(responce_E_template)

    # E arragement
    input_E_prompts = [
        ("context", context_E_prompt),
        ("objective", objective_E_prompt),
        ("style", style_E_prompt),
        ("tone", tone_E_prompt),
        ("audience", audience_E_prompt),
        ("responce", responce_E_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_E_prompt, pipeline_prompts=input_E_prompts
    )

    # final E prompt
    E_prompt = pipeline_prompt.format(
            BT = input_BT,
            LO = input_LO,
            pestel = pestel,
            pestel_sub = pestel_sub,
            rag_responses_E = str(rag_responses_E),
            all_sources = str(all_sources),
            pestel_whole = pestel_whole,
            objective_item_PESTEL_E = objective_item_PESTEL_E,
            E_json_sample = E_json_sample
            )

    E_result = model.predict(
        text=E_prompt)

    return E_result

async def BM3(input_TA, input_LO, input_BT, lang: str = None):
    # 參數定義
    pestel = "社會"
    pestel_whole = "社會面向（社會 Social"
    pestel_sub = ["生活型態", "消費觀念"]
    objective_item_PESTEL_S  = """
    - 出生人口的上升/下降狀況
    - 老年人口成長率狀況
    - 當地的消費水平上升或下降情形，並說明原因
    - 消費者關注的文化、流行風格與休閒活動
    - 消費者近期喜歡的東西，生活習慣的改變
    - 失業率的上升/下降狀況
    """
    S_json_sample = """
    {
        'pestel':'政治',
        'pestel_info':[
        {
            'pestel_sub':'政策支持',
            'content':'分析內文 字數約為80-90個字'
            },
        {
            'pestel_sub':'減碳淨零',
            'content':'分析內文 字數約為80-90個字'
            }
        ],
        "source": [

        ]
    }
    """
    # S_RAG

    # Define policy search queries
    S_1 = f"查詢與保障{input_BT}騎乘者在城市道路系統中的安全和權益的相關法規與措施案例。"
    S_2 = f"查詢有關促進{input_BT}通勤和減少交通擁擠的具體政策和激勵措施。"
    S_3 = f"查詢與改善{input_BT}基礎設施安全性的舉措，以及提升民眾使用意願的案例。"
    S_4 = f"查詢在高人口密度城市中設計汽車、{input_BT}、行人共享道路空間的安全方案。"
    S_5 = f"查詢促進{input_BT}友善觀光的政策措施，及打造{input_BT}旅遊體驗的成功案例。"

    # Construct URLs
    # RAG_S_1 = f"{url}{S_1}"
    # RAG_S_2 = f"{url}{S_2}"
    # RAG_S_3 = f"{url}{S_3}"
    # RAG_S_4 = f"{url}{S_4}"
    # RAG_S_5 = f"{url}{S_5}"
    # response_S_1 = res(S_1)
    # response_S_2 = res(S_2)
    # response_S_3 = res(S_3)
    # response_S_4 = res(S_4)
    # response_S_5 = res(S_5)
    # rag_responses_S = response_S_1 + response_S_2 + response_S_3 + response_S_4 + response_S_5

    queries = [S_1, S_2, S_3, S_4, S_5]

    t0 = time.time()

    # 平行觸發 5 次 res()
    # responses = await asyncio.gather(*[res(q) for q in queries])

    responses = await multi_res_only(queries)

    # 平行觸發 5 次 res()
    # responses = await asyncio.gather(*[res(q) for q in queries])

    t1 = time.time()

    print("===gpt costs===")
    print(t1 - t0)

    # 合併回應內容
    rag_responses_S = "".join([r["answer"] for r in responses])
    sources = combine_sources(responses)
    all_sources = ", ".join(str(src) for src in sources)

    print(f"Total time: {t1 - t0:.2f} seconds")

    # S full costar template
    full_S_template = """
    {context}
    {objective}
    {style}
    {tone}
    {audience}
    {responce}
    """
    full_S_prompt = PromptTemplate.from_template(full_S_template)

    # S "contex" template
    context_S_template = """
    # Context
    你是一位專業的產業管理顧問。掌握全球產業重要發展動態，從產業市場、地區經濟、地方基礎建設、國家政策與社會文化等面向進行全面的分析，協助客戶探勘產業趨勢情報、洞見技術與創新需求脈絡，貼近產業立場，提供客戶全方位建言。
    """
    context_S_prompt = PromptTemplate.from_template(context_S_template)

    # S "objective" template
    objective_S_template = """
    # Objective:
    你的任務是以PESTEL分析法同時參考下方參考文獻並結合自身看法，針對特定國家或區域的{BT}產業現況進行"{pestel_whole}"的分析，分析時要考量以下幾個面向與指標
    {objective_item_PESTEL_S}

    參考文獻：
    {rag_responses_S}

    文獻來源：
    {all_sources}

    """
    objective_S_prompt = PromptTemplate.from_template(objective_S_template)

    # S "style" template
    style_S_template = """
    # Style:
    內容能夠輕鬆理解。
    刪除任何無關緊要的文本，無關緊要的文本示例："嗯" 修正任何明顯的拼寫錯誤。
    以列點的方式進行描述。

    你必須將輸出結構化為一組json格式。
    並只能以'pestel','pestel_sub','pestel_info'三個元素作為json的註解
    其中須以'{pestel_sub}'作為 'pestel_info'中的每一個'pestel_sub'分項, 不要自行發散出其他的分項

    json是一種聲明性語言，可讓你對文件進行註解和確認。
    你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全匹配！
    範例如下：

    {S_json_sample}
    """
    style_S_prompt = PromptTemplate.from_template(style_S_template)

    # S "tone" template
    tone_S_template = """
    # Tone:
    在整個過程中保持清晰和有條理的語氣，具備說服力地說明產品與特定地區的{pestel}關係。具備清楚且有方向的分析，提供有價值的見解和建議。
    """
    tone_S_prompt = PromptTemplate.from_template(tone_S_template)

    # S "audience" template
    audience_S_template = """
    # Audience:
    目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
    """
    audience_S_prompt = PromptTemplate.from_template(audience_S_template)

    # S "responce" template
    if lang == 'en':
        responce_S_template = """
    #Responce:
    - 請用en英文進行回答。
    - 用PESTEL分析{LO}地區的{BT}產業，... en英文版的'{pestel_sub}'...

    #Start:
    如果您明白了，請開始進行分析
    """
    else:
        responce_S_template = """
    #Responce:
    用PESTEL分析{LO}地區的{BT}產業，... '{pestel_sub}' ...

    #Start:
    如果您明白了，請開始進行分析
    """
    responce_S_prompt = PromptTemplate.from_template(responce_S_template)

    # S arragement
    input_S_prompts = [
        ("context", context_S_prompt),
        ("objective", objective_S_prompt),
        ("style", style_S_prompt),
        ("tone", tone_S_prompt),
        ("audience", audience_S_prompt),
        ("responce", responce_S_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_S_prompt, pipeline_prompts=input_S_prompts
    )

    # final S prompt
    S_prompt = pipeline_prompt.format(
            BT = input_BT,
            LO = input_LO,
            pestel = pestel,
            pestel_sub = pestel_sub,
            rag_responses_S = str(rag_responses_S),
            all_sources = str(all_sources),
            pestel_whole = pestel_whole,
            objective_item_PESTEL_S = objective_item_PESTEL_S,
            S_json_sample = S_json_sample
            )

    S_result = model.predict(
        text=S_prompt)

    return S_result

async def BM4(input_TA, input_LO, input_BT, lang: str = None):
    ## prompt T
    # 參數定義
    pestel = "技術"
    pestel_whole = "技術面向（技術 Technological）"
    pestel_sub = ["核心技術","配件裝置"]
    objective_item_PESTEL_T  = """
    - 近期發表且可以拿來應用的新科技
    - 能夠推動公司成長數位技術
    - 近期政府和研究單位關注的科技議題
    - 近年專利申請技術領域的情況
    """
    T_json_sample = """
    {
        'pestel':'政治',
        'pestel_info':[
        {
            'pestel_sub':'政策支持',
            'content':'分析內文 字數約為80-90個字'
            },
        {
            'pestel_sub':'減碳淨零',
            'content':'分析內文 字數約為80-90個字'
            }
        ],
        "source": [

        ]
    }
    """
    # T_RAG

    # Define policy search queries
    T_1 = f"請檢索{input_BT}未來的技術發展趨勢，尤其在智慧化與電子架構設計方面的核心技術，並說明如何實現智能調整動力。。"
    T_2 = f"{input_BT}的主要電子化部件（如電子座桿、電子前叉等）有哪些？請說明物聯網（IoT）在自行車的應用，並舉例具體使用情境。"
    T_3 = f"{input_BT}各部件使用的通訊協議有哪些標準？電池管理系統如何優化電量精確度，並在續航和電池壽命之間取得平衡？"
    T_4 = f"請檢索智慧輔助系統如何透過AI技術即時分析騎行數據（如速度、坡度），並根據不同情境自動調整輔助動力以提升騎行{input_BT}體驗。"
    T_5 = f"在{input_BT}的智慧化應用中，請說明如何保護騎行數據和隱私，避免資料洩露與資安問題，並列舉適用的技術手段。"


    # Construct URLs
    # RAG_T_1 = f"{url}{T_1}"
    # RAG_T_2 = f"{url}{T_2}"
    # RAG_T_3 = f"{url}{T_3}"
    # RAG_T_4 = f"{url}{T_4}"
    # RAG_T_5 = f"{url}{T_5}"
    # response_T_1 = res(T_1)
    # response_T_2 = res(T_2)
    # response_T_3 = res(T_3)
    # response_T_4 = res(T_4)
    # response_T_5 = res(T_5)
    # rag_responses_T = response_T_1 + response_T_2 + response_T_3 + response_T_4 + response_T_5
    queries = [T_1, T_2, T_3, T_4, T_5]

    t0 = time.time()

    # 平行觸發 5 次 res()
    # responses = await asyncio.gather(*[res(q) for q in queries])
    responses = await multi_res_only(queries)

    # 平行觸發 5 次 res()
    # responses = await asyncio.gather(*[res(q) for q in queries])

    t1 = time.time()

    print("===gpt costs===")
    print(t1 - t0)

    # 合併回應內容
    rag_responses_T = "".join([r["answer"] for r in responses])
    sources = combine_sources(responses)
    all_sources = ", ".join(str(src) for src in sources)

    print(f"Total time: {t1 - t0:.2f} seconds")

    # T full costar template
    full_T_template = """
    {context}
    {objective}
    {style}
    {tone}
    {audience}
    {responce}
    """
    full_T_prompt = PromptTemplate.from_template(full_T_template)

    # T "contex" template
    context_T_template = """
    # Context
    你是一位專業的產業管理顧問。掌握全球產業重要發展動態，從產業市場、地區經濟、地方基礎建設、國家政策與社會文化等面向進行全面的分析，協助客戶探勘產業趨勢情報、洞見技術與創新需求脈絡，貼近產業立場，提供客戶全方位建言。
    """
    context_T_prompt = PromptTemplate.from_template(context_T_template)

    # T "objective" template
    objective_T_template = """
    # Objective:
    你的任務是以PESTEL分析法同時參考下方參考文獻並結合自身看法，針對特定國家或區域的{BT}產業現況進行"{pestel_whole}"的分析，分析時要考量以下幾個面向與指標
    {objective_item_PESTEL_T}

    參考文獻：
    {rag_responses_T}

    文獻來源：
    {all_sources}

    """
    objective_T_prompt = PromptTemplate.from_template(objective_T_template)

    # T "style" template
    style_T_template = """
    # Ttyle:
    內容能夠輕鬆理解。
    刪除任何無關緊要的文本，無關緊要的文本示例："嗯" 修正任何明顯的拼寫錯誤。
    以列點的方式進行描述。

    你必須將輸出結構化為一組json格式。
    並只能以'pestel','pestel_sub','pestel_info'三個元素作為json的註解
    其中須以'{pestel_sub}'作為 'pestel_info'中的每一個'pestel_sub'分項, 不要自行發散出其他的分項

    json是一種聲明性語言，可讓你對文件進行註解和確認。
    你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全匹配！
    範例如下：

    {T_json_sample}
    """
    style_T_prompt = PromptTemplate.from_template(style_T_template)

    # T "tone" template
    tone_T_template = """
    # Tone:
    在整個過程中保持清晰和有條理的語氣，具備說服力地說明產品與特定地區的{pestel}關係。具備清楚且有方向的分析，提供有價值的見解和建議。
    """
    tone_T_prompt = PromptTemplate.from_template(tone_T_template)

    # T "audience" template
    audience_T_template = """
    # Audience:
    目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
    """
    audience_T_prompt = PromptTemplate.from_template(audience_T_template)

    # T "responce" template
    if lang == 'en':
        responce_T_template = """
    #Responce:
    - 請用en英文進行回答。
    - 用PESTEL分析{LO}地區的{BT}產業，... en英文版的'{pestel_sub}'...

    #Start:
    如果您明白了，請開始進行分析
    """
    else:
        responce_T_template = """
    #Responce:
    用PESTEL分析{LO}地區的{BT}產業，... '{pestel_sub}' ...

    #Start:
    如果您明白了，請開始進行分析
    """
    responce_T_prompt = PromptTemplate.from_template(responce_T_template)

    # T arragement
    input_T_prompts = [
        ("context", context_T_prompt),
        ("objective", objective_T_prompt),
        ("style", style_T_prompt),
        ("tone", tone_T_prompt),
        ("audience", audience_T_prompt),
        ("responce", responce_T_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_T_prompt, pipeline_prompts=input_T_prompts
    )

    # final T prompt
    T_prompt = pipeline_prompt.format(
            BT = input_BT,
            LO = input_LO,
            pestel = pestel,
            pestel_sub = pestel_sub,
            rag_responses_T = str(rag_responses_T),
            all_sources = str(all_sources),
            pestel_whole = pestel_whole,
            objective_item_PESTEL_T = objective_item_PESTEL_T,
            T_json_sample = T_json_sample
            )

    T_result = model.predict(
        text=T_prompt)

    return T_result

async def BM5(input_TA, input_LO, input_BT, lang: str = None):
    ## prompt Env
    # 參數定義
    pestel = "環境"
    pestel_whole = "環境面向（環境 Enviromental ）"
    pestel_sub = ["氣候條件","交通建設"]
    objective_item_PESTEL_Env  = """
    - 氣候變遷狀況以及當地氣候類型、月均溫、年雨量。
    - 當地的交通道路法規限制
    - 交通道路建設狀態，如：路寬、自行車道路設置等。
    - 相關交通建設，如充電站、充電樁普及率等。
    """
    Env_json_sample = """
    {
        'pestel':'政治',
        'pestel_info':[
        {
            'pestel_sub':'政策支持',
            'content':'分析內文 字數約為80-90個字'
            },
        {
            'pestel_sub':'減碳淨零',
            'content':'分析內文 字數約為80-90個字'
            }
        ],
        "source": [

        ]
    }
    """
    # Env_RAG

    # Define policy search queries
    Env_1 = f"請檢索{input_BT}產業在生產過程中如何減少碳足跡及使用可再生材料的措施，以及此舉對可持續發展的貢獻。"
    Env_2 = f"請檢索隨著消費者環保意識的提升，如何影響消費者選擇{input_BT}作為低排放交通工具，以及此趨勢對{input_BT}市場的影響。"
    Env_3 = f"請檢索{input_BT}產業如何處理報廢{input_BT}及其部件，以減少環境污染，並描述此舉在循環經濟中的角色。"
    Env_4 = f"請檢索國際{input_BT}品牌如何通過碳抵消計畫來補償製造過程中的碳排放，以及這些計畫對品牌形象和環境的實際影響。"
    Env_5 = f"請檢索{input_BT}基礎設施和氣候條件是否影響國民的{input_BT}購買意願，以及基礎設施普及與大眾運輸發展的關聯性。"


    # Construct URLs
    # RAG_Env_1 = f"{url}{Env_1}"
    # RAG_Env_2 = f"{url}{Env_2}"
    # RAG_Env_3 = f"{url}{Env_3}"
    # RAG_Env_4 = f"{url}{Env_4}"
    # RAG_Env_5 = f"{url}{Env_5}"
    # response_E_1 = res(E_1)
    # response_E_2 = res(E_2)
    # response_E_3 = res(E_3)
    # response_E_4 = res(E_4)
    # response_E_5 = res(E_5)
    # rag_responses_Env = response_E_1 + response_E_2 + response_E_3 + response_E_4 + response_E_5
    queries = [Env_1, Env_2, Env_3, Env_4, Env_5]

    t0 = time.time()

    # 平行觸發 5 次 res()
    # responses = await asyncio.gather(*[res(q) for q in queries])

    responses = await multi_res_only(queries)

    # 平行觸發 5 次 res()
    # responses = await asyncio.gather(*[res(q) for q in queries])

    t1 = time.time()

    print("===gpt costs===")
    print(t1 - t0)

    # 合併回應內容
    rag_responses_Env = "".join([r["answer"] for r in responses])
    sources = combine_sources(responses)
    all_sources = ", ".join(str(src) for src in sources)

    print(f"Total time: {t1 - t0:.2f} seconds")

    # Env full costar template
    full_Env_template = """
    {context}
    {objective}
    {style}
    {tone}
    {audience}
    {responce}
    """
    full_Env_prompt = PromptTemplate.from_template(full_Env_template)

    # Env "contex" template
    context_Env_template = """
    # Context
    你是一位專業的產業管理顧問。掌握全球產業重要發展動態，從產業市場、地區經濟、地方基礎建設、國家政策與社會文化等面向進行全面的分析，協助客戶探勘產業趨勢情報、洞見技術與創新需求脈絡，貼近產業立場，提供客戶全方位建言。
    """
    context_Env_prompt = PromptTemplate.from_template(context_Env_template)

    # Env "objective" template
    objective_Env_template = """
    # Objective:
    你的任務是以PESTEL分析法同時參考下方參考文獻並結合自身看法，針對特定國家或區域的{BT}產業現況進行"{pestel_whole}"的分析，分析時要考量以下幾個面向與指標
    {objective_item_PESTEL_Env}

    參考文獻：
    {rag_responses_Env}

    文獻來源：
    {all_sources}

    """
    objective_Env_prompt = PromptTemplate.from_template(objective_Env_template)

    # Env "style" template
    style_Env_template = """
    # Ttyle:
    內容能夠輕鬆理解。
    刪除任何無關緊要的文本，無關緊要的文本示例："嗯" 修正任何明顯的拼寫錯誤。
    以列點的方式進行描述。

    你必須將輸出結構化為一組json格式。
    並只能以'pestel','pestel_sub','pestel_info'三個元素作為json的註解
    其中須以'{pestel_sub}'作為 'pestel_info'中的每一個'pestel_sub'分項, 不要自行發散出其他的分項

    json是一種聲明性語言，可讓你對文件進行註解和確認。
    你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全匹配！
    範例如下：

    {Env_json_sample}
    """
    style_Env_prompt = PromptTemplate.from_template(style_Env_template)

    # Env "tone" template
    tone_Env_template = """
    # Tone:
    在整個過程中保持清晰和有條理的語氣，具備說服力地說明產品與特定地區的{pestel}關係。具備清楚且有方向的分析，提供有價值的見解和建議。
    """
    tone_Env_prompt = PromptTemplate.from_template(tone_Env_template)

    # Env "audience" template
    audience_Env_template = """
    # Audience:
    目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
    """
    audience_Env_prompt = PromptTemplate.from_template(audience_Env_template)

    # Env "responce" template
    if lang == 'en':
        responce_Env_template = """
    #Responce:
    - 請用en英文進行回答。
    - 用PESTEL分析{LO}地區的{BT}產業，... en英文版的'{pestel_sub}'...

    #Start:
    如果您明白了，請開始進行分析
    """
    else:
        responce_Env_template = """
    #Responce:
    用PESTEL分析{LO}地區的{BT}產業，... '{pestel_sub}' ...

    #Start:
    如果您明白了，請開始進行分析
    """
    responce_Env_prompt = PromptTemplate.from_template(responce_Env_template)

    # Env arragement
    input_Env_prompts = [
        ("context", context_Env_prompt),
        ("objective", objective_Env_prompt),
        ("style", style_Env_prompt),
        ("tone", tone_Env_prompt),
        ("audience", audience_Env_prompt),
        ("responce", responce_Env_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_Env_prompt, pipeline_prompts=input_Env_prompts
    )

    # final Env prompt
    Env_prompt = pipeline_prompt.format(
            BT = input_BT,
            LO = input_LO,
            pestel = pestel,
            pestel_sub = pestel_sub,
            rag_responses_Env = str(rag_responses_Env),
            all_sources = str(all_sources),
            pestel_whole = pestel_whole,
            objective_item_PESTEL_Env = objective_item_PESTEL_Env,
            Env_json_sample = Env_json_sample
            )

    Env_result = model.predict(
        text=Env_prompt)

    return Env_result

async def BM6(input_TA, input_LO, input_BT, lang: str = None):
    ## prompt L
    # 參數定義
    pestel = "法規"
    pestel_whole = "法規面向（法規 Legal）"
    pestel_sub = ["交通法則", "檢測標準"]
    objective_item_PESTEL_L  = """
    - 可能會影響產品設計與規劃的當地法規
    - 目前正在提議修改的法規，如果修法通過，説明對公司可能的影響
    - 產品相關的檢測標準以及測驗細項
    - 當地法規對於產品的定義
    """
    L_json_sample = """
    {
        'pestel':'政治',
        'pestel_info':[
        {
            'pestel_sub':'政策支持',
            'content':'分析內文 字數約為80-90個字'
            },
        {
            'pestel_sub':'減碳淨零',
            'content':'分析內文 字數約為80-90個字'
            }
        ],
        "source": [

        ]
    }
    """
    # L_RAG

    # Define policy search queries
    L_1 = f"查詢{input_BT}使用者（如兒童頭盔佩戴）及共享單車企業的法規要求，包括安全規範、牌照、保險服務及防盜措施等，以保障騎乘者安全與公共秩序"
    L_2 = f"探索{input_BT}是否應歸類為機動車輛並適用特定法律，包括速度限制、騎乘者年齡要求、保險需求以及電池等零部件的安全檢測標準。"
    L_3 = f"查詢是否有關於{input_BT}在公共道路上行駛的限制性法規，如高速公路禁行、特定時段限制及混合車道的優先權規定，以保障道路使用者的安全。"
    L_4 = f"探討是否需推動{input_BT}租賃及共享單車的管理法規，包括定期檢測、維護要求、停放區域及數量控制，確保運營符合公共安全需求。"
    L_5 = f"查詢{input_BT}騎乘者的酒駕法規是否需明確化，並探討針對{input_BT}尤其是電動車的責任保險立法，以保障騎乘者及其他道路使用者的權益。"

    # Construct URLs
    # RAG_L_1 = f"{url}{L_1}"
    # RAG_L_2 = f"{url}{L_2}"
    # RAG_L_3 = f"{url}{L_3}"
    # RAG_L_4 = f"{url}{L_4}"
    # RAG_L_5 = f"{url}{L_5}"
    # response_L_1 = res(L_1)
    # response_L_2 = res(L_2)
    # response_L_3 = res(L_3)
    # response_L_4 = res(L_4)
    # response_L_5 = res(L_5)
    # rag_responses_L = response_L_1 + response_L_2 + response_L_3 + response_L_4 + response_L_5
    t0 = time.time()
    queries = [L_1, L_2, L_3, L_4, L_5]

    responses = await multi_res_only(queries)

    # 平行觸發 5 次 res()
    # responses = await asyncio.gather(*[res(q) for q in queries])

    t1 = time.time()

    print("===gpt costs===")
    print(t1 - t0)

    # 合併回應內容
    rag_responses_L = "".join([r["answer"] for r in responses])
    sources = combine_sources(responses)
    all_sources = ", ".join(str(src) for src in sources)

    print(f"Total time: {t1 - t0:.2f} seconds")

    # L full costar template
    full_L_template = """
    {context}
    {objective}
    {style}
    {tone}
    {audience}
    {responce}
    """
    full_L_prompt = PromptTemplate.from_template(full_L_template)

    # L "contex" template
    context_L_template = """
    # Context
    你是一位專業的產業管理顧問。掌握全球產業重要發展動態，從產業市場、地區經濟、地方基礎建設、國家政策與社會文化等面向進行全面的分析，協助客戶探勘產業趨勢情報、洞見技術與創新需求脈絡，貼近產業立場，提供客戶全方位建言。
    """
    context_L_prompt = PromptTemplate.from_template(context_L_template)

    # L "objective" template
    objective_L_template = """
    # Objective:
    你的任務是以PESTEL分析法同時參考下方參考文獻並結合自身看法，針對特定國家或區域的{BT}產業現況進行"{pestel_whole}"的分析，分析時要考量以下幾個面向與指標
    {objective_item_PESTEL_L}

    參考文獻：
    {rag_responses_L}

    文獻來源：
    {all_sources}

    """
    objective_L_prompt = PromptTemplate.from_template(objective_L_template)

    # L "style" template
    style_L_template = """
    # Ttyle:
    內容能夠輕鬆理解。
    刪除任何無關緊要的文本，無關緊要的文本示例："嗯" 修正任何明顯的拼寫錯誤。
    以列點的方式進行描述。

    你必須將輸出結構化為一組json格式。
    並只能以'pestel','pestel_sub','pestel_info'三個元素作為json的註解
    其中須以'{pestel_sub}'作為 'pestel_info'中的每一個'pestel_sub'分項, 不要自行發散出其他的分項

    json是一種聲明性語言，可讓你對文件進行註解和確認。
    你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全匹配！
    範例如下：

    {L_json_sample}
    """
    style_L_prompt = PromptTemplate.from_template(style_L_template)

    # L "tone" template
    tone_L_template = """
    # Tone:
    在整個過程中保持清晰和有條理的語氣，具備說服力地說明產品與特定地區的{pestel}關係。具備清楚且有方向的分析，提供有價值的見解和建議。
    """
    tone_L_prompt = PromptTemplate.from_template(tone_L_template)

    # L "audience" template
    audience_L_template = """
    # Audience:
    目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
    """
    audience_L_prompt = PromptTemplate.from_template(audience_L_template)

    # L "responce" template
    if lang == 'en':
        responce_L_template = """
    #Responce:
    - 請用en英文進行回答。
    - 用PESTEL分析{LO}地區的{BT}產業，... en英文版的'{pestel_sub}'...

    #Start:
    如果您明白了，請開始進行分析
    """
    else:
        responce_L_template = """
    #Responce:
    用PESTEL分析{LO}地區的{BT}產業，... '{pestel_sub}' ...

    #Start:
    如果您明白了，請開始進行分析
    """
    responce_L_prompt = PromptTemplate.from_template(responce_L_template)

    # L arragement
    input_L_prompts = [
        ("context", context_L_prompt),
        ("objective", objective_L_prompt),
        ("style", style_L_prompt),
        ("tone", tone_L_prompt),
        ("audience", audience_L_prompt),
        ("responce", responce_L_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_L_prompt, pipeline_prompts=input_L_prompts
    )

    # final L prompt
    L_prompt = pipeline_prompt.format(
            BT = input_BT,
            LO = input_LO,
            pestel = pestel,
            pestel_sub = pestel_sub,
            rag_responses_L = str(rag_responses_L),
            all_sources = str(all_sources),
            pestel_whole = pestel_whole,
            objective_item_PESTEL_L = objective_item_PESTEL_L,
            L_json_sample = L_json_sample
            )

    L_result = model.predict(
        text=L_prompt)

    return L_result