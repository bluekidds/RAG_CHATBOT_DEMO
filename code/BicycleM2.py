import aiohttp
import requests
import re
from code.utils import dalle3
import json
import asyncio
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

def parse_to_json(raw_str):
    """
    傳入 raw_str 字串（可能包含 Markdown code block 包裹和非標準 JSON 格式），
    進行處理後回傳解析出的 Python 對象（dict 或 list）。
    """
    # 移除 Markdown 包裹區塊 (```json 與 ``` 等)
    cleaned = raw_str.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    # 移除尾逗號：以正則表達式替換陣列與物件中的最後一個逗號
    # 例如將 "}," 或 ",]" 轉換為 "}" 或 "]"
    cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)

    # 如果 key 或 value 使用了單引號，轉換成雙引號。注意這裡會對字串中的所有單引號做替換，
    # 如果你的內容中出現合法的單引號（例如縮寫），可能需要更精細的處理。
    # cleaned = cleaned.replace("'", "\"")

    try:
        result = json.loads(cleaned)
        return result
    except json.decoder.JSONDecodeError as e:
        print('解析失敗，原文：')
        print(raw_str)
        raise ValueError(f"解析 JSON 失敗: {e}")

async def BMC1(input_TA, input_LO, input_BT, lang: str = None):
    # 參數定義
    competitive_info = """
    | key           | #zh-TW           | definition                                                   |
    | ------------- | ---------------- | ------------------------------------------------------------ |
    | brand_name_type    | 競品品牌名稱_產品型號     | 「5個」""真實存在的"" 競品品牌，以及該品牌車款型號，國內或國外的實際品牌都可以        |
    | bike_sub_type    | 車種     | 根據產品型號列出所屬的自行車車種        |
    | specification     | 產品規格       | 根據型號列出產品主要規格，包含: 車架與前叉、變速與傳動系統、煞車系統、車輪輪胎等，並列出車款重量、尺寸(車體長度、高度，單位mm)，約30字，精簡敘述      |
    | strengths     | 優勢             |  請將該產品以下列七個面向進行綜合評估：設計、技術創新、耐用性、舒適性、安全功能、價格，以及目標族群的騎乘需求，如身高、使用情境等，並將優勢排序，篩選出最後提供“3點條列”，每點約30字，主動語態、專業、明確的描述，在進行價格比較時，請以售價中位數台幣67,000元做為判斷基準，判斷該車款價格屬於低、中、高價位  |
    | weaknesses    | 劣勢             | 請將該產品以下列七個面向進行綜合評估：設計、技術創新、耐用性、舒適性、安全功能、價格，以及目標族群的騎乘需求，如身高、使用情境等，並將劣勢排序，篩選出最後提供“3點條列”，每點約30字，主動語態、專業、明確的描述，在進行價格比較時，請以售價中位數台幣67,000元做為判斷基準，判斷該車款價格屬於低、中、高價位  |
    | features      | 產品共同特點描述 | 根據品牌特色、客群屬性、產品技術特色，提供專業且具體定位品牌的一句描述，30字左右   |
    | price_banding | 價格區間         | 根據品牌定位、市場定位以及目標客群，用{input_LO}貨幣符號表示，貨幣符號請保持為簡寫     |
    | audiences     | 主要客群描述     | 根據該品牌的產品特色及定價策略，推估可能的目標使用族群 TA，30字左右精準描述TA的族群類型、消費層級及產品偏好  |
    """
    competitive_json_sample = """
    {
    "location": "台灣",
    "bike_type": "自行車",
    "target_audience":"都市輕熟女",
    "id": "2-1",
    "competitive": [
        {
        "brand_name_type": "Giant_銀影",
        "bike_sub_type":"輕型電動自行車",
        "specification": {
                    "frame": "ALUXX級鋁合金車架，SHIMANO 105 2*11速系統，Giant D型碳纖維座管。",
                    "weight_kg": "",
                    "length_mm": "",
                    "height_mm": ""
                            },
        "strengths": ["台灣知名品牌，售後服務完善。", "攜帶方便，適合城市騎乘。", "價格適中，符合都市輕熟女的購買力。"],
        "weaknesses": ["設計風格較為傳統，缺乏時尚感。", "車款選擇較少，風格單一。", "重量較重，騎乘體驗有待提升。"],
        "price_banding": "60000-70000 NT$",
        "features": "提供舒適且穩定的騎乘體驗，適合城市通勤。",
        "audiences": "追求穩定性和舒適性，且有一定購買力的都市輕熟女。"
        },
        {
        "brand_name_type": "Merida_Scultura Disc 200",
        "bike_sub_type":"輕型電動自行車",
        "specification": {
                    "frame": "Scultura Lite鋁合金車架，SHIMANO SORA 2*9速系統，Merida Expert CC座管。",
                    "weight_kg": "",
                    "length_mm": "",
                    "height_mm": ""
                            },
        "strengths": ["品牌知名度高，品質有保障。", "輕量化設計，騎乘體驗佳。", "適中的價位，符合目標客群需求。"],
        "weaknesses": ["車架設計較為單一，風格選擇有限。", "售後服務相較其他品牌較不完善。", "款式選擇較少，品牌個性不明顯。"],
        "price_banding": "60000-70000 NT$",
        "features": "結合輕量化與耐用性的自行車，適合長距離騎乘。",
        "audiences": "追求體驗和性能，且有一定購買力的都市輕熟女。"
        }
    ],
    "source": [

    ]
    }
    """

    # Define policy search queries
    Comp_1 = f"請檢索全球與{input_LO}{input_BT}產業的發展趨勢，包括市場增長，以及在永續性設計與綠色交通創新需求方面的進展。"
    Comp_2 = f"請檢索「台北國際自行車展」及「EURO BIKE」等國際自行車展的最新趨勢，並融入對於智慧移動設備及{input_BT}產業的全球市場需求分析。"
    Comp_3 = f"請檢索針對{input_TA}的{input_BT}品牌，並說明這些品牌的評價與競爭優勢。"
    Comp_4 = f"請檢索{input_BT}的續航力，以及相關品牌及其規格。"
    Comp_5 = f"請檢索具備智慧功能的{input_BT}產品，並分析市場對於此種自行車的可接受重量與體積。"

    queries = [Comp_1, Comp_2, Comp_3, Comp_4, Comp_5]

    q1_start = time.time()

    # 平行觸發 5 次 res()
    # responses = await asyncio.gather(*[res(q) for q in queries])
    responses = await multi_res_only(queries)

    # 平行觸發 5 次 res()
    # responses = await asyncio.gather(*[res(q) for q in queries])

    q1_end = time.time()

    # 合併回應內容
    rag_responses_Comp = "".join([r["answer"] for r in responses])
    sources = combine_sources(responses)
    all_sources = ", ".join(str(src) for src in sources)

    # competitive full costar template
    full_competitive_template = """
    {context}
    {objective}
    {style}
    {audience}
    {responce}
    """
    full_competitive_prompt = PromptTemplate.from_template(full_competitive_template)

    # competitive "contex" template
    context_competitive_template = """
    # Context
    你是一位專業並且來自台灣的產業分析顧問，掌握全球產業重要發展動態，擅長從產業市場規模、市場區隔、使用者輪廓、新產品規劃、創新策略、社會文化等面向進行全方位的產業競爭分析。
    我正在進行「競品分析」，請協助我探勘產業趨勢情報、洞見使用者需求、產業技術優勢與創新需求脈絡、分析產品特色及適用客群，貼近產業立場，提供客戶全方位建議。
    """
    context_competitive_prompt = PromptTemplate.from_template(context_competitive_template)

    # competitive "objective" template
    objective_competitive_template = """
    # Objective:
    你的任務是參考下方參考文獻並結合自身看法分析適合[{LO}]地區[{TA}]的[{BT}]，給出「五個」競品品牌，並針對每一個品牌及款式，根據以下 <competitive-info> 定義以及 <example> json 格式，詳細列出內容。
    <competitive-info>
    {competitive_info}
    </competitive-info>
    - 回答中的品牌必須是 ""實際存在的品牌""，絕對不接受模擬品牌如'品牌A'、'品牌B'、'品牌X'、'品牌1'等。
    - 回答中的各款式間比較價格、重量、品質，必須說明清楚細節，絕不接受'相對重'、'相對輕'、'品質一般'等。
    - 內容必須基於真實且正確的資料，不要編撰或創造內容(Do not make information up)。

    參考文獻：
    {rag_responses_Comp}

    文獻來源：
    {all_sources}
    """
    objective_competitive_prompt = PromptTemplate.from_template(objective_competitive_template)

    # competitive "style" template
    style_competitive_template = """
    # Style:
    - 用字精準無贅字、無拼寫錯誤、條理清晰易讀。
    - 寫作風格如大型研究機構 nngroup, frog design, BCG、新聞網站 Axios，撰寫 distill, descriptive, clear, straightforward 的分析摘要。
    """
    style_competitive_prompt = PromptTemplate.from_template(style_competitive_template)

    # competitive "audience & tone" template
    audience_competitive_template = """
    # Audience & Tone:
    - 目標讀者是公司內部的Product Designer、Researcher、Project Manager，他們需要蒐集市場資料分析趨勢、研究產品資料分析競品、撰寫設計提案、與團隊協作與溝通。
    - 讀者體期待瞭解市場現況，以有效將競品分析轉化成具體可行動的產品策略。因此過程中需保持清晰和有條理的語氣，提供有價值的見解和建議。"""
    audience_competitive_prompt = PromptTemplate.from_template(audience_competitive_template)

    # competitive "responce" template
    responce_competitive_template = """
    #Responce:
    - #zh-TW繁體中文進行回答。
    - 每一分析面向的條列點應「具有獨特性、敘述文字有差異性」，避免各項目彼此的內容的重複或過於相似。
    - 必須將輸出結構化為一組json格式。定義見上方 <competitive-info>，你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！

    <example>
    {competitive_json_sample}
    </example>

    #Start:
    根據上述資訊，針對{LO}地區的{BT}產業進行深入的「競品分析」，使用 <example> json 格式，列出5個 ""真實存在的"" {BT}競品品牌，分析完就停止，不要給我結論。
    如果您明白了，請開始執行
    """
    responce_competitive_prompt = PromptTemplate.from_template(responce_competitive_template)

    # competitive arragement
    input_competitive_prompts = [
        ("context", context_competitive_prompt),
        ("objective", objective_competitive_prompt),
        ("style", style_competitive_prompt),
        ("audience", audience_competitive_prompt),
        ("responce", responce_competitive_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_competitive_prompt, pipeline_prompts=input_competitive_prompts
    )

    predict1_start = time.time()
    # final competitive prompt
    competitive_prompt = pipeline_prompt.format(
            BT = input_BT,
            LO = input_LO,
            TA = input_TA,
            competitive_info = competitive_info,
            competitive_json_sample = competitive_json_sample,
            rag_responses_Comp = str(rag_responses_Comp),
            all_sources = all_sources
            )
    competitive_result = model.predict(
        text=competitive_prompt
    )
    predict1_end = time.time()
    return competitive_result

async def BMC2(input_TA, input_LO, input_BT, lang: str = None):
    # 參數定義
    designed_trend_info = """
    | key           | #zh-TW           | definition                                                   |
    | ------------- | ---------------- | ------------------------------------------------------------ |
    | theme_style   | 主題設計風格     | 「提出三種主題設計風格，提案請包含色系搭配、造型描述、主打功能、適用族群，以段落方式具體描述細節，增加修飾形容詞。在造型上請精簡敘述，概念性說明適合{TA}的理由 |
    | material_recomendation     | 材質推薦       | 提出三種材質推薦，提案請包含材質名稱，優勢及劣勢、應用層面、適用族群，請具體的描述 |
    | function_trend_recomendation     | 功能趨勢             |  提出三種功能趨勢，包含說明功能的限制、應用層面、適用族群，請具體的描述 |
    | patent    | 申請的專利         | 提出三種近年申請的專利，包含說明專利類型、發明內容、創新技術、設計改進及專利申請範圍，請具體的描述  |
    """
    # designed_trend_json_sample = """
    # {
    # "location": "台灣",
    # "bike_type": "自行車",
    # "target_audience":"年輕父母",
    # "id": "2-2",

    # "theme_style" :[
    # {
    # "theme_style_1" : "都市自然融合風格"
    # "theme_style_1_idea" :
    #     [
    #     "1.色系搭配：綠色和棕色的柔和組合，搭配自然木紋圖案。",
    #     "2.造型描述：流線型設計，採用天然木材元素的車架，給人一種與自然和諧共生的感覺。",
    #     "3.主打功能：強調舒適性和環保性，適合長途騎行。",
    #     "4.適用族群：住在城市邊緣，喜愛戶外活動的家庭。"
    #     ]
    # },
    # {
    # "theme_style_2" : "科技未來主義風格"
    # "theme_style_2_idea" :
    #     [
    #     "1.色系搭配：銀色與深藍色的搭配，帶有金屬材質的閃光效果。",
    #     "2.造型描述：簡約而富有科技感的車架，結合智能化配件。",
    #     "3.主打功能：智能導航系統和自動調節懸掛系統。",
    #     "4.適用族群：注重科技感和功能性的年輕父母。"
    #     ]
    # },
    # {
    # "theme_style_3" : "科技未來主義風格"
    # "theme_style_3_idea" :
    #     [
    #     "1.色系搭配明亮的粉色和黃色，搭配卡通圖案。",
    #     "2.造型描述: 圓潤的車架設計，搭配可愛的裝飾。",
    #     "3.主打功能: 安全座椅和趣味配件。",
    #     "4.適用族群: 有小孩的家庭，希望給孩子帶來快樂騎行體驗。"
    #     ]
    # }],

    # "material_recomendation" :[
    # {
    # "material_recomendation_1" : "鋁合金"
    # "material_recomendation_1_idea" :
    #     [
    #     "1.優勢: 輕量化和耐腐蝕性強。",
    #     "2.劣勢: 震動吸收性差。",
    #     "3.應用層面: 適合日常通勤和短途騎行。",
    #     "4.適用族群: 普通家庭和偶爾騎行者。"
    #     ]
    # },
    # {
    # "material_recomendation_2" : "鋼鐵"
    # "material_recomendation_2_idea" :
    #     [
    #     "1.優勢: 高強度和良好的震動吸收能力。",
    #     "2.劣勢: 重量較重，易生鏽。",
    #     "3.應用層面: 適合長途旅行和城市內頻繁使用。",
    #     "4.適用族群: 重視耐用性和舒適性的使用者。"
    #     ]
    # },
    # {
    # "material_recomendation_3" : "碳纖維"
    # "material_recomendation_3_idea" :
    #     [
    #     "1.優勢: 極輕和高強度。",
    #     "2.劣勢: 成本高，易損壞。",
    #     "3.應用層面: 適合高性能需求和競技使用。",
    #     "4.適用族群: 專業騎行者和性能追求者。"
    #     ]
    # }],

    # "function_trend_recomendation" :[
    # {
    # "function_trend_recomendation_1" : "智能化"
    # "function_trend_recomendation_1_idea" :
    #     [
    #     "1.功能限制：需要穩定的數據連接和電力供應。",
    #     "2.應用層面：提升安全性和騎行體驗。",
    #     "3.適用族群：喜歡高科技產品的家庭。"
    #     ]
    # },
    # {
    # "function_trend_recomendation_2" : "電動輔助"
    # "function_trend_recomendation_2_idea" :

    #     "1.功能限制：需要穩定的數據連接和電力供應。",
    #     "2.應用層面：提升安全性和騎行體驗。",
    #     "3.適用族群：喜歡高科技產品的家庭。"
    #     ]
    # },
    # {
    # "function_trend_recomendation_3" : "安全性"
    # "function_trend_recomendation_3_idea" :
    #     [
    #     "1.功能限制：需要穩定的數據連接和電力供應。",
    #     "2.應用層面：提升安全性和騎行體驗。",
    #     "3.適用族群：喜歡高科技產品的家庭。"
    #     ]
    # }],

    # "patent" :[
    # {
    # "patent_1" : "自動調節懸掛系統"
    # "patent_1_idea" :
    #     [
    #     "1.專利類型：機械設計改進。",
    #     "2.發明內容：根據道路狀況自動調整懸掛系統。",
    #     "3.創新技術：使用感應技術自動控制。",
    #     "4.設計改進：提升騎行平穩性。",
    #     "5.專利申請範圍：懸掛系統的自動化調整。"
    #     ]
    # },
    # {
    # "patent_2" : "智能安全帽"
    # "patent_2_idea" :
    #     [
    #     "1.專利類型：機械設計改進。",
    #     "2.發明內容：根據道路狀況自動調整懸掛系統。",
    #     "3.創新技術：使用感應技術自動控制。",
    #     "4.設計改進：提升騎行平穩性。",
    #     "5.專利申請範圍：懸掛系統的自動化調整。"
    #     ]
    # },
    # {
    # "patent_3" : "模組化車架設計"
    # "patent_3_idea" :
    #     [
    #     "1.功能限制：需要穩定的數據連接和電力供應。",
    #     "2.應用層面：提升安全性和騎行體驗。",
    #     "3.適用族群：喜歡高科技產品的家庭。"
    #     ]
    # }]
    # }
    # """
    designed_trend_json_sample = """
    {
        "location": "日本",
        "bike_type": "媽媽自行車",
        "target_audience": "需要載送小朋友的媽媽",
        "id": "3-1",
        "theme_style": [
        {
            "style_id": "1",
            "theme_style": "家庭友善風格",
            "theme_style_idea": [
            "1.色系搭配：綠色和棕色的柔和組合，搭配自然木紋圖案。",
            "2.造型描述：流線型設計，採用天然木材元素的車架，給人一種與自然和諧共生的感覺。",
            "3.主打功能：強調舒適性和環保性，適合長途騎行。",
            "4.適用族群：住在城市邊緣，喜愛戶外活動的家庭。"
            ],
            "material_recomendation": [
            {
                "name": "鋁合金",
                "material_recomendation_idea": [
                "1.優勢: 輕量化和耐腐蝕性強。",
                "2.劣勢: 震動吸收性差。",
                "3.應用層面: 適合日常通勤和短途騎行。",
                "4.適用族群: 普通家庭和偶爾騎行者。"
                ]
            },
            {
                "name": "高強度塑鋼",
                "material_recomendation_idea": [
                "1.優勢: ",
                "2.劣勢: ",
                "3.應用層面:  ",
                "4.適用族群: "
                ]
            },
            {
                "name": "實木飾件",
                "material_recomendation_idea": [
                "1.優勢: ",
                "2.劣勢: ",
                "3.應用層面:  ",
                "4.適用族群: "
                ]
            }
            ],
            "function_trend": [
            {
                "name": "電動輔助",
                "function_trend_idea": [
                "1.功能限制：需要穩定的數據連接和電力供應。",
                "2.應用層面：提升安全性和騎行體驗。",
                "3.適用族群：喜歡高科技產品的家庭。"
                ]
            },
            {
                "name": "雙兒童座椅配置",
                "function_trend_idea": [
                "1.功能限制：",
                "2.應用層面：",
                "3.適用族群："
                ]
            },
            {
                "name": "側邊收納籃",
                "function_trend_idea": [
                "1.功能限制：",
                "2.應用層面：",
                "3.適用族群："
                ]
            }
            ],
            "patent": [
            {
                "name": "感應型安全鎖",
                "patent_idea": [
                "偵測寶寶入座自動鎖定，提升安全性。",
                "整合APP，可遠端控制開關。",
                "具備防呆提示音與電量顯示功能。"
                ]
            },
            {
                "name": "三段式椅背調整裝置",
                "patent_idea": [
                "1.專利類型：機械設計改進。",
                "2.發明內容：根據道路狀況自動調整懸掛系統。",
                "3.創新技術：使用感應技術自動控制。",
                "4.設計改進：提升騎行平穩性。",
                "5.專利申請範圍：懸掛系統的自動化調整。"
                ]
            },
            {
                "name": "一鍵摺疊車架",
                "patent_idea": [
                "1.專利類型：",
                "2.發明內容：",
                "3.創新技術：",
                "4.設計改進：",
                "5.專利申請範圍："
                ]
            }
            ]
        },
        {
            "style_id": "2",
            "theme_style": "城市通勤風格",
            "theme_style_idea": [
            "1.色系搭配：",
            "2.造型描述：",
            "3.主打功能：",
            "4.適用族群："
            ],
            "material_recomendation": [
            {
                "name": "碳纖維",
                "material_recomendation_idea": [
                "1.優勢: ",
                "2.劣勢: ",
                "3.應用層面:  ",
                "4.適用族群: "
                ]
            },
            {
                "name": "鋁鎂合金",
                "material_recomendation_idea": [
                "1.優勢: ",
                "2.劣勢: ",
                "3.應用層面:  ",
                "4.適用族群: "
                ]
            },
            {
                "name": "高分子複合材",
                "material_recomendation_idea": [
                "1.優勢: ",
                "2.劣勢: ",
                "3.應用層面:  ",
                "4.適用族群: "
                ]
            }
            ],
            "function_trend": [
            {
                "name": "智慧導航儀表",
                "function_trend_idea": [
                "1.功能限制：",
                "2.應用層面：",
                "3.適用族群："
                ]
            },
            {
                "name": "可拆式電池設計",
                "function_trend_idea": [
                "1.功能限制：",
                "2.應用層面：",
                "3.適用族群："
                ]
            },
            {
                "name": "APP鎖車系統",
                "function_trend_idea": [
                "1.功能限制：",
                "2.應用層面：",
                "3.適用族群："
                ]
            }
            ],
            "patent": [
            {
                "name": "車架內嵌式防盜感測器",
                "patent_idea": [
                "1.專利類型：",
                "2.發明內容：",
                "3.創新技術：",
                "4.設計改進：",
                "5.專利申請範圍："
                ]
            },
            {
                "name": "智慧照明整合裝置",
                "patent_idea": [
                "1.專利類型：",
                "2.發明內容：",
                "3.創新技術：",
                "4.設計改進：",
                "5.專利申請範圍："
                ]
            },
            {
                "name": "磁吸式行李架模組",
                "patent_idea": [
                "1.專利類型：",
                "2.發明內容：",
                "3.創新技術：",
                "4.設計改進：",
                "5.專利申請範圍："
                ]
            }
            ]
        },
        {
            "style_id": "3",
            "theme_style": "綠意生態風格",
            "theme_style_idea": [
            "色系搭配：綠、米與淺木色呼應自然。",
            "造型設計：圓潤有機曲線，減少鋒利結構。",
            "視覺符號：植栽圖騰與雷雕木紋飾片。"
            ],
            "material_recomendation": [
            {
                "name": " ",
                "material_recomendation_idea": [
                "1.優勢: ",
                "2.劣勢: ",
                "3.應用層面:  ",
                "4.適用族群: "
                ]
            },
            {
                "name": " ",
                "material_recomendation_idea": [
                "1.優勢: ",
                "2.劣勢: ",
                "3.應用層面:  ",
                "4.適用族群: "
                ]
            },
            {
                "name": " ",
                "material_recomendation_idea": [
                "1.優勢: ",
                "2.劣勢: ",
                "3.應用層面:  ",
                "4.適用族群: "
                ]
            }
            ],
            "function_trend": [
            {
                "name": "",
                "function_trend_idea": [
                "1.功能限制：",
                "2.應用層面：",
                "3.適用族群："
                ]
            },
            {
                "name": " ",
                "function_trend_idea": [
                "1.功能限制：",
                "2.應用層面：",
                "3.適用族群："
                ]
            },
            {
                "name": " ",
                "function_trend_idea": [
                "1.功能限制：",
                "2.應用層面：",
                "3.適用族群："
                ]
            }
            ],
            "patent": [
            {
                "name": " ",
                "patent_idea": [
                "1.專利類型：",
                "2.發明內容：",
                "3.創新技術：",
                "4.設計改進：",
                "5.專利申請範圍："
                ]
            },
            {
                "name": "智慧照明整合裝置",
                "patent_idea": [
                "1.專利類型：",
                "2.發明內容：",
                "3.創新技術：",
                "4.設計改進：",
                "5.專利申請範圍："
                ]
            },
            {
                "name": "磁吸式行李架模組",
                "patent_idea": [
                "1.專利類型：",
                "2.發明內容：",
                "3.創新技術：",
                "4.設計改進：",
                "5.專利申請範圍："
                ]
            }
            ]
        }
        ],
        "source": []
    }
    """

    Tren_1 = f"請檢索{input_BT}產業中最新的設計風格趨勢，包括色彩搭配、車架造型，並分析其適用於不同年齡段的族群。"
    Tren_2 = f"請檢索{input_BT}行業中常用的材質（如碳纖維、鋁合金、鋼鐵等），列出每種材質的優勢與劣勢，並說明適合的騎行者類型。"
    Tren_3 = f"請檢索各類型{input_BT}的主流功能趨勢，包括技術限制、應用層面及適用的族群。"
    Tren_4 = f"請檢索{input_BT}產業如何運用綠色設計及循環材料來實現永續目標，包括這些設計如何影響產品壽命及使用體驗。"
    Tren_5 = f"請檢索智能技術（如AI、感應器技術）在{input_BT}產業的應用情況，並分析這些技術如何提升使用者的安全性和騎行體驗。"

    queries = [Tren_1, Tren_2, Tren_3, Tren_4, Tren_5]

    q2_start = time.time()

    # 平行觸發 5 次 res()
    # responses = await asyncio.gather(*[res(q) for q in queries])
    responses = await multi_res_only(queries)

    q2_end = time.time()

    # 合併回應內容
    rag_responses_Tren = "".join([r["answer"] for r in responses])
    sources = combine_sources(responses)
    all_sources = ", ".join(str(src) for src in sources)

    # designed_trend full costar template
    full_designed_trend_template = """
    {context}
    {objective}
    {style}
    {audience}
    {responce}
    """
    full_designed_trend_prompt = PromptTemplate.from_template(full_designed_trend_template)

    # designed_trend "contex" template
    context_designed_trend_template = """
    # Context
    你是一位專業的自行車產品設計師，不僅熟悉自行車產品開發流程，也熟悉全球自行車設計趨勢動態，在產業市場、地區經濟、設計發展、與社會文化等面向有具備相當豐富的經驗，具備協助客戶探勘設計趨勢情報、洞見技術與創新需求脈絡的能力，貼近產業立場，提供客戶全方位建言。
    """
    context_designed_trend_prompt = PromptTemplate.from_template(context_designed_trend_template)

    # designed_trend "objective" template
    objective_designed_trend_template = """
    # Objective:
    你的任務是參考下方參考文獻並結合自身看法分析適合[{LO}]地區[{TA}]的[{BT}]的設計趨勢進行設計提案，據以下 <designed_trend-info> 定義以及 <example> json 格式，詳細列出內容。提案請注意以下項目
    <designed_trend-info>
    {designed_trend_info}
    </designed_trend-info>
    - 提出三種主題設計風格，提案請包含色系搭配、造型描述、主打功能、適用族群，以段落方式具體描述細節，增加修飾形容詞。在造型上請精簡敘述，概念性說明適合 [{TA}] 的理由
    - 提出三種材質推薦，提案請包含材質名稱，優勢及劣勢、應用層面、適用族群，請具體的描述
    - 提出三種功能趨勢，包含說明功能的限制、應用層面、適用族群，請具體的描述
    - 提出三種近年申請的專利，包含說明專利類型、發明內容、創新技術、設計改進及專利申請範圍，請具體的描述
    - 方案最後必須附上文獻來源，內容可參考下方

    參考文獻：
    {rag_responses_Tren}

    文獻來源：
    {all_sources}
    """
    objective_designed_trend_prompt = PromptTemplate.from_template(objective_designed_trend_template)

    # designed_trend "style" template
    style_designed_trend_template = """
    # Style:
    -內容能夠輕鬆理解，用字精準無贅字，具體的描述項目。
    - 刪除任何無關緊要的文本，無關緊要的文本示例："嗯" 修正任何明顯的拼寫錯誤。
    """
    style_designed_trend_prompt = PromptTemplate.from_template(style_designed_trend_template)

    # designed_trend "audience & tone" template
    audience_designed_trend_template = """
    # Audience & Tone:
    - 目標讀者是公司內部的Product Designer、Researcher、Project Manager，他們需要蒐集市場資料分析趨勢、研究產品資料分析競品、撰寫設計提案、與團隊協作與溝通。
    - 讀者體期待瞭解市場現況，以有效將競品分析轉化成具體可行動的產品策略。因此過程中需保持清晰和有條理的語氣，提供有價值的見解和建議。"""
    audience_designed_trend_prompt = PromptTemplate.from_template(audience_designed_trend_template)

    # designed_trend "responce" template
    responce_designed_trend_template = """
    # Responce:
    - #zh-TW繁體中文進行回答。
    - 每一分析面向的條列點應「具有獨特性、敘述文字有差異性」，避免各項目彼此的內容的重複或過於相似。
    - 必須將輸出結構化為一組json格式。定義見上方 <designed_trend-info>，你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！

    <example>
    {designed_trend_json_sample}
    </example>

    #Start:
    根據上述資訊，針對{LO}地區的{BT}產業進行自行車近年的設計趨勢提案，使用 <example> json 格式，[設計趨勢下面細分：主題設計風格、材質推薦、功能趨勢、專利]，分析完就停止，不要給我結論。
    如果您明白了，請開始執行
    """
    responce_designed_trend_prompt = PromptTemplate.from_template(responce_designed_trend_template)

    # designed_trend arragement
    input_designed_trend_prompts = [
        ("context", context_designed_trend_prompt),
        ("objective", objective_designed_trend_prompt),
        ("style", style_designed_trend_prompt),
        ("audience", audience_designed_trend_prompt),
        ("responce", responce_designed_trend_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_designed_trend_prompt, pipeline_prompts=input_designed_trend_prompts
    )

    # final designed_trend prompt
    designed_trend_prompt = pipeline_prompt.format(
            BT = input_BT,
            LO = input_LO,
            TA = input_TA,
            designed_trend_info = designed_trend_info,
            designed_trend_json_sample = designed_trend_json_sample,
            rag_responses_Tren = str(rag_responses_Tren),
            all_sources = all_sources
            )

    designed_trend_result = model.predict(
        text=designed_trend_prompt)
    predict2_end = time.time()
    return designed_trend_result

async def BMC3(input_TA, input_LO, input_BT, lang: str = None):
    predict3_start = time.time()
    # 參數定義
    pds_info = """
    | key           | #zh-TW           | definition                                                   |
    | ------------- | ---------------- | ------------------------------------------------------------ |
    | product_innovation    | 產品創新     | 根據技術、設計、使用者體驗、商業模式、環保和市場需求等多個層面進行評估，並給予評分 |
    | assess_feasibility     | 可行性       |   根據技術、經濟、資源、法律、供應鏈、社會接受度和風險管理等多個角度進行綜合判斷，確保創新在技術上可實現、商業上可盈利、資源上可支持，並給予評分    |
    """

    pds_json_sample = """
    {
    "location": "台灣",
    "bike_type": "電動自行車",
    "target_audience": "女性小資族",
    "id": "2-3-1",

    "product_designed_trend": [
        {
        "headline": "方案1標題，標題清晰直指重點",
        "product_innovation": "⭐⭐⭐⭐",
        "assess_feasibility": "⭐⭐⭐",
        "solution": "方案1敘述，請說明挑選此方案的具體原因，以及如何幫助團隊制定產品設計決策、發掘創新機會"
        },
        {
        "headline": "方案2標題",
        "product_innovation": "⭐⭐",
        "assess_feasibility": "⭐⭐⭐",
        "ideas": "方案2敘述"
        {
        "headline": "方案3標題",
        "product_innovation": "⭐⭐⭐⭐",
        "assess_feasibility": "⭐⭐⭐",
        "ideas": "方案3敘述"
        }
    ],
    }
    """

    # Define policy search queries
    pds_1 = f"請檢索與分析各品牌{input_BT}在動力系統、智能控制系統、及電池技術上的創新，包括技術實現的挑戰與優勢。"
    pds_2 = f"請檢索{input_BT}品牌如何改善騎行舒適度、智能連接性、及車身設計以滿足不同類型用戶的需求。"
    pds_3 = f"請探索國際大廠如何透過訂閱制、租賃服務及MaaS（Mobility as a Service）等新興商業模式來提升市場滲透率與盈利能力。"
    pds_4 = f"請檢索{input_BT}廠商如何使用循環設計和環保材料來降低碳排放與能源消耗，並探討其對產品與品牌價值的影響。"
    pds_5 = f"請檢索各國（特別是{input_LO}）針對電動{input_BT}的法規支持，包括稅收減免與政府補貼政策，及其對市場發展的影響。"

    queries = [pds_1, pds_2, pds_3, pds_4, pds_5]

    q3_start = time.time()

    # 平行觸發 5 次 res()
    # responses = await asyncio.gather(*[res(q) for q in queries])
    responses = await multi_res_only(queries)

    q3_end = time.time()

    # 合併回應內容
    rag_responses_pds = "".join([r["answer"] for r in responses])
    sources = combine_sources(responses)
    all_sources = ", ".join(str(src) for src in sources)

    # pds full costar template
    full_pds_template = """
    {context}
    {objective}
    {style}
    {audience}
    {responce}
    """
    full_pds_prompt = PromptTemplate.from_template(full_pds_template)

    # pds "contex" template
    context_pds_template = """
    # Context
    你是一位專精在自行車領域的市場策略經理。掌握全球自行車產業重要發展動態，從產業市場、地區經濟、商業模式、國家政策與社會文化等面向都有具備相當豐富的經驗，協助客戶依據當前產業趨勢、開發技術與消費者需求，進行產品創新的設計建議，給出的建議貼近產業立場。
    """
    context_pds_prompt = PromptTemplate.from_template(context_pds_template)

    # pds "objective" template
    objective_pds_template = """
    # Objective:
    你的任務是基於下方分析方向以及參考文獻，統整並詳述當前產業的設計產品創新機會。
    - 先思考適合[{LO}]地區[{TA}]的[{BT}]的「五個」競品品牌
    - 基於上述的競品品牌思考適合[{LO}]地區[{TA}]的[{BT}]的設計趨勢進

    未來創新機會方案：綜合上方思考流程，分析以下兩個面向：產品創新、服務創新，請先以「產品創新」面向提供「3個」具體的切入方案。
        - 每個方案標題吸引人具前瞻性
        - 每個方案提供搭配的文案，具體描述方案的細節及市場可行性
        - 方案描述為具深刻洞見的評論，並有一定程度的合理性，指出方案創新的細節，每一個方案描述不少於300字且為結構清晰的段落。
        - 每個方案針對以下項目進行評分，並根據以下 <pds-info> 定義以及 <example> json 格式， 每個項目個別給予評分（以⭐表示，範圍從0到5）。

    <pds-info>
    {pds_info}
    </pds-info>


    參考文獻：
    {rag_responses_pds}
    """
    objective_pds_prompt = PromptTemplate.from_template(objective_pds_template)

    # pds "style" template
    style_pds_template = """
    # Style:
    - 文體應專業且具描述性(descriptive)，用字精準無贅字、無拼寫錯誤、條理清晰易讀。
    - 內容必須基於真實且正確的資料，不要編撰或創造內容(Do not make information up)。
    - 以#zh-TW繁體中文進行回答，答案中提及的品牌可以是國內或國外的實際品牌，並將貨幣符號保持為簡寫。
    - 確保每一條列的字數在其分析結果中保持一致性，不應有超過30字的差異。
    - 每一分析面向的條列點應「具有獨特性」，「避免內容的重複或過於相似」。
    - 必須將輸出結構化為一組json格式。json是一種聲明性語言，可讓你對文件進行註解和確認。你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！
    範例如下：
    <example>
    {pds_json_sample}
    </example>

    """
    style_pds_prompt = PromptTemplate.from_template(style_pds_template)

    # pds "audience & tone" template
    audience_pds_template = """
    # Audience & Tone:
    - 目標讀者是公司內部的Product Designer、Researcher、Project Manager，他們需要蒐集市場資料分析趨勢、研究產品資料分析競品、撰寫設計提案、與團隊協作與溝通。
    - 讀者體期待瞭解市場現況，以有效將競品分析轉化成具體可行動的產品策略。因此過程中需保持清晰和有條理的語氣，提供有價值的見解和建議。"""
    audience_pds_prompt = PromptTemplate.from_template(audience_pds_template)

    # pds "responce" template
    responce_pds_template = """
    # Responce:
    - #zh-TW繁體中文進行回答。
    - 每一分析面向的條列點應「具有獨特性、敘述文字有差異性」，避免各項目彼此的內容的重複或過於相似。
    - 必須將輸出結構化為一組json格式。定義見上方 <pds-info>，你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！

    <example>
    {pds_json_sample}
    </example>

    #Start:
    根據上述資訊，針對{LO}地區的{BT}產業，參考提供的「競品分析」、「產品設計趨勢」及參考文獻，統整並詳述當前產業的設計產品創新機會，使用 <example> json 格式。
    如果您明白了，請開始執行
    """
    responce_pds_prompt = PromptTemplate.from_template(responce_pds_template)

    # pds arragement
    input_pds_prompts = [
        ("context", context_pds_prompt),
        ("objective", objective_pds_prompt),
        ("style", style_pds_prompt),
        ("audience", audience_pds_prompt),
        ("responce", responce_pds_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_pds_prompt, pipeline_prompts=input_pds_prompts
    )

    # final pds prompt
    pds_prompt = pipeline_prompt.format(
            BT = input_BT,
            LO = input_LO,
            TA = input_TA,
            pds_info = pds_info,
            pds_json_sample = pds_json_sample,
            rag_responses_pds = str(rag_responses_pds)
            )

    pds_result = model.predict(
        text=pds_prompt)
    predict3_end = time.time()
    print(f"BM3耗時: {predict3_end - predict3_start:.2f} 秒")
    return pds_result

async def BMC4(input_TA, input_LO, input_BT, lang: str = None):
    # 參數定義
    predict4_start = time.time()
    sds_info = """
    | key           | #zh-TW           | definition                                                   |
    | ------------- | ---------------- | ------------------------------------------------------------ |
    | Service innovation    | 服務創新     | 根據技術應用、使用者體驗、商業模式、服務便捷性、環保影響以及社會增值等方面進行綜合評估。創新的服務應該提升使用便捷性、提供個性化選擇，並且在技術上具備先進性和可行性，還要能夠滿足市場需求並促進可持續發展，並依據上述的綜合評估給予評分 |
    | User experience    | 使用者體驗     | 根據便捷性、舒適性、安全性、可用性、情感滿足與增值功能等層面綜合考量。好的使用者體驗不僅應能提供流暢的操作和高效的服務，還應滿足使用者在情感和價值層面的需求，進一步提升使用者的滿意度和忠誠度並依據上述的綜合評估給予評分 |
    | Commercial viability     | 商業可行性       |   根據市場需求、商業模式、成本結構、競爭環境、法規合規性、客戶獲取、資金需求、風險管理等多個層面進行綜合評估，並給予評分    |
    """

    sds_json_sample = """
    {
    "location": "台灣",
    "bike_type": "電動自行車",
    "target_audience": "女性小資族",
    "id": "2-3-2",

    "service_designed_trend": [
        {
        "headline": "方案1標題，標題清晰直指重點",
        "service_innovation": "⭐⭐⭐⭐",
        "user_experience": "⭐⭐⭐⭐",
        "commercial_viability": "⭐⭐",
        "solution": "方案1敘述，請說明挑選此方案的具體原因，以及如何幫助團隊制定產品設計決策、發掘創新機會"
        },
        {
        "headline": "方案2標題",
        "service_innovation": "⭐⭐⭐⭐⭐",
        "user_experience": "⭐⭐⭐",
        "commercial_viability": "⭐⭐⭐⭐",
        "ideas": "方案2敘述"
        {
        "headline": "方案3標題",
        "service_innovation": "⭐⭐⭐",
        "user_experience": "⭐",
        "commercial_viability": "⭐⭐",
        "ideas": "方案3敘述"
        }
    ]
    }
    """

    # Define policy search queries
    sds_1 = f"請檢索全球{input_BT}產業中智慧化技術的應用實例，並分析這些技術如何提升使用者的使用體驗及滿足市場需求。"
    sds_2 = f"請檢索{input_BT}產業在循環經濟設計中的案例，評估如何通過技術創新實現產業的可持續性發展。"
    sds_3 = f"請檢索有關如何通過人本交通設計提升城市中{input_TA}的安全性、便捷性與舒適性的案例分析。"
    sds_4 = f"請探索{input_BT}共享租賃系統在市場中的成功應用案例，重點分析其便捷性與使用者體驗。"
    sds_5 = f"請搜索{input_BT}中創新服務的商業模式，如 B2B 一站式解決方案，並分析其如何滿足使用者需求並促進產業增長。"

    queries = [sds_1, sds_2, sds_3, sds_4, sds_5]

    q4_start = time.time()

    # 平行觸發 5 次 res()
    # responses = await asyncio.gather(*[res(q) for q in queries])
    responses = await multi_res_only(queries)

    q4_end = time.time()

    # 合併回應內容
    rag_responses_sds = "".join([r["answer"] for r in responses])
    sources = combine_sources(responses)
    all_sources = ", ".join(str(src) for src in sources)

    # sds full costar template
    full_sds_template = """
    {context}
    {objective}
    {style}
    {audience}
    {responce}
    """
    full_sds_prompt = PromptTemplate.from_template(full_sds_template)

    # sds "contex" template
    context_sds_template = """
    # Context
    你是一位專精在自行車領域的服務體驗設計師，熟悉全球自行車產業重要發展動態，從產業市場、地區經濟、商業模式、國家政策與社會文化等面向都有具備相當豐富的經驗，協助客戶依據當前產業趨勢、開發技術與消費者需求，進行服務創新的策略建議，給出的建議貼近產業立場。
    """
    context_sds_prompt = PromptTemplate.from_template(context_sds_template)

    # sds "objective" template
    objective_sds_template = """
    # Objective:
    你的任務是基於下方分析方向以及參考文獻，統整並詳述當前產業的設計產品創新機會。
    - 先思考適合[{LO}]地區[{TA}]的[{BT}]的「五個」競品品牌
    - 基於上述的競品品牌思考適合[{LO}]地區[{TA}]的[{BT}]的設計趨勢進

    未來創新機會方案：綜合上方思考流程，分析以下兩個面向：產品創新、服務創新，請先以「服務創新」面向提供「3個」具體的切入方案。
        - 每個方案標題吸引人具前瞻性
        - 每個方案提供搭配的文案，具體描述方案的細節及市場可行性
        - 方案描述為具深刻洞見的評論，並有一定程度的合理性，指出方案創新的細節，每一個方案描述不少於300字且為結構清晰的段落。
        - 每個方案針對以下項目進行評分，並根據以下 <sds-info> 定義以及 <example> json 格式， 每個項目個別給予評分（以⭐表示，範圍從0到5）。

    <sds-info>
    {sds_info}
    </sds-info>

    參考文獻：
    {rag_responses_sds}
    """
    objective_sds_prompt = PromptTemplate.from_template(objective_sds_template)

    # sds "style" template
    style_sds_template = """
    # Style:
    - 文體應專業且具描述性(descriptive)，用字精準無贅字、無拼寫錯誤、條理清晰易讀。
    - 內容必須基於真實且正確的資料，不要編撰或創造內容(Do not make information up)。
    - 以#zh-TW繁體中文進行回答，答案中提及的品牌可以是國內或國外的實際品牌，並將貨幣符號保持為簡寫。
    - 確保每一條列的字數在其分析結果中保持一致性，不應有超過30字的差異。
    - 每一分析面向的條列點應「具有獨特性」，「避免內容的重複或過於相似」。
    - 必須將輸出結構化為一組json格式。json是一種聲明性語言，可讓你對文件進行註解和確認。你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！
    範例如下：
    <example>
    {sds_json_sample}
    </example>
    """
    style_sds_prompt = PromptTemplate.from_template(style_sds_template)

    # sds "audience & tone" template
    audience_sds_template = """
    # Audience & Tone:
    - 目標讀者是公司內部的Product Designer、Researcher、Project Manager，他們需要蒐集市場資料分析趨勢、研究產品資料分析競品、撰寫設計提案、與團隊協作與溝通。
    - 讀者體期待瞭解市場現況，以有效將競品分析轉化成具體可行動的產品策略。因此過程中需保持清晰和有條理的語氣，提供有價值的見解和建議。
    """
    audience_sds_prompt = PromptTemplate.from_template(audience_sds_template)

    # sds "responce" template
    responce_sds_template = """
    # Responce:
    - #zh-TW繁體中文進行回答。
    - 每一分析面向的條列點應「具有獨特性、敘述文字有差異性」，避免各項目彼此的內容的重複或過於相似。
    - 必須將輸出結構化為一組json格式。定義見上方 <sds-info>，你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！

    <example>
    {sds_json_sample}
    </example>

    #Start:
    根據上述資訊，針對{LO}地區的{BT}產業，參考提供的「競品分析」、「產品設計趨勢」及參考文獻，統整並詳述當前產業的設計服務創新機會，使用 <example> json 格式。
    如果您明白了，請開始執行
    """
    responce_sds_prompt = PromptTemplate.from_template(responce_sds_template)

    # sds arragement
    input_sds_prompts = [
        ("context", context_sds_prompt),
        ("objective", objective_sds_prompt),
        ("style", style_sds_prompt),
        ("audience", audience_sds_prompt),
        ("responce", responce_sds_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_sds_prompt, pipeline_prompts=input_sds_prompts
    )

    # final sds prompt
    sds_prompt = pipeline_prompt.format(
            BT = input_BT,
            LO = input_LO,
            TA = input_TA,
            sds_info = sds_info,
            sds_json_sample = sds_json_sample,
            rag_responses_sds = str(rag_responses_sds)
            )

    sds_result = model.predict(
        text=sds_prompt)
    predict4_end = time.time()
    print(f"BM4耗時: {predict4_end - predict4_start:.2f} 秒")
    return sds_result

async def BMC5(pds_result, sds_result):
    start_time = time.time()
    # 參數定義
    image_prompt_json_sample = """
    {
    prompts:[
        {
        "1-1": "Product Theme Description: A clear representation of the primary feature of the product combines main subject, design qualities, and intended use case."
        },
        {
        "1-2": "(second prompt relate to first product designed suggestion)"
        },
        {
        "2-1": "Material, Texture, and Finish: Contextual details that contribute to the product's visual and tactile qualities like color, material, texture, style."
        },
        {
        "2-2": "(second prompt relate to second product designed suggestion)"
        },
        {
        "3-1": "Functional Descriptors: Highlights the practical features and unique selling points that address user needs and product longevity."
        },
        {
        "3-2": "(second prompt relate to third product designed suggestion)"
        },
        {
        "4-1": "Detail and Quality: Stresses the importance of fine detail and overall image quality to enhance the product's representation, e.g., masterpiece, extremely detailed, 4k, beautiful, realistic photography."
        },
        {
        "4-2": "(second prompt relate to first service designed suggestion)"
        },
        {
        "5-1": "Professional Race Bike Services, Training Courses, Competitions, Sponsorships"
        }
        {
        "5-2": "(second prompt relate to second service designed suggestion)"
        }
        {
        "6-1": "Professional Race Bike Services, Training Courses, Competitions, Sponsorships"
        }
        {
        "6-2": "(second prompt relate to third service designed suggestion)"
        }
    ]
    }
    """

    # image_prompt full costar template
    full_image_prompt_template = """
    {context}
    {objective}
    {style}
    {audience}
    {responce}
    """
    full_image_prompt_prompt = PromptTemplate.from_template(full_image_prompt_template)

    # image_prompt "contex" template
    context_image_prompt_template = """
    # Context
    你是善用AI繪圖工具的AI繪圖師，能準確地依照概念、需求使用 Leonardo AI, Stable Diffusion以及Dalle 3 這類AI繪圖工具下達指令，繪製各種「產品設計概念圖」提供設計師創意發想。
    """
    context_image_prompt_prompt = PromptTemplate.from_template(context_image_prompt_template)

    # image_prompt "objective" template
    objective_image_prompt_template = """
    # Objective:
    我將提供你6個創新機會方案，你將針對每個創新翻譯產生2組prompt共12個，讓我透過Leonardo AI進行圖片生成。請遵照以下規則，產生每個方案各自的"英文"prompt。

    6種創新機會方案 內容如下：
    產品創新3種：
    {pds_result}

    服務創新3種：
    {sds_result}
    """
    objective_image_prompt_prompt = PromptTemplate.from_template(objective_image_prompt_template)

    # image_prompt "style" template
    style_image_prompt_template = """
    # Style:
    - 越往前排列的單字是越關鍵的主體、主體特徵，越往後的單字越容易被歸納成風格、裝飾、畫面細節。
    - 參考下方範例依據方案描述內容，每一項方案裡的 prompt 必須和 "自行車" 有關，建議 prompt 能包含 台灣 or 自行車 的 "英文關鍵字"。
    - 必須將輸出結構化為一組json格式。json是一種聲明性語言，可讓你對文件進行註解和確認。你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位格式及命名方式完全與下方範例完全匹配！
    範例如下：
    <example>
    {image_prompt_json_sample}
    </example>
    """
    style_image_prompt_prompt = PromptTemplate.from_template(style_image_prompt_template)

    # image_prompt "audience & tone" template
    audience_image_prompt_template = """
    # Audience & Tone:
    - 目標讀者是公司內部的設計開發人員、決策層或高階主管。假設讀者群體期待瞭解市場現況、期待找到服務或產品新切入點，以有效轉化的實用建議與可行動的步驟。
    - 在整個過程中保持清晰和有條理的語氣，建立產品設計概念圖提示，並明確敘述機會適合的市場、客群。具備清楚且有方向的分析，提供有價值的見解和建議。
    """
    audience_image_prompt_prompt = PromptTemplate.from_template(audience_image_prompt_template)

    # image_prompt "responce" template
    responce_image_prompt_template = """
    # Responce:
    - 請將上方提供的6種創新機會方案，依據裡面各標題的方案描述，綜合上方參考文獻以及你所熟知AI繪圖使用的keywords，產生各2個共12個相應的"英文" 關鍵字組合 prompt。
    - 請直接將12種方案的關鍵字 "組合成一段完整的英文prompt"！不需要寫出方案標題、不用介係詞，以逗號連結即可 convert the bulleted keywords into comma-separated。
    - 給我各個創新機會方案12組完整的"英文"prompt，每一個 prompt 不超過 50 token。
    - 第一種方案："1-1","1-2"、第二種方案："2-1","2-2"以此類推...

    #Start:
    如果您明白了，請開始進行分析
    """
    responce_image_prompt_prompt = PromptTemplate.from_template(responce_image_prompt_template)

    # image_prompt arragement
    input_image_prompt_prompts = [
        ("context", context_image_prompt_prompt),
        ("objective", objective_image_prompt_prompt),
        ("style", style_image_prompt_prompt),
        ("audience", audience_image_prompt_prompt),
        ("responce", responce_image_prompt_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_image_prompt_prompt, pipeline_prompts=input_image_prompt_prompts
    )

    # final image_prompt prompt
    image_prompt_prompt = pipeline_prompt.format(
            pds_result = pds_result,
            sds_result = sds_result,
            image_prompt_json_sample = image_prompt_json_sample
            )

    img_prompt_result = model.predict(
        text=image_prompt_prompt)
    predict_img_end = time.time()

    prompt_data = parse_to_json(img_prompt_result)
    prompts = [value for prompt_obj in prompt_data["prompts"] for value in prompt_obj.values()]
    print(f"生prompt: {time.time() - start_time:.2f} 秒")
    # prompts = prompts[:12]
    # prompts += [""] * (12 - len(prompts))
    pics = await asyncio.gather(*[asyncio.create_task(dalle3(prompt, "")) for prompt in prompts])
    pics += [""] * (12 - len(pics))
    print(f"生圖: {time.time() - start_time:.2f} 秒")
    return pics
    # return img_prompt_result

async def BMC(input_TA, input_LO, input_BT, lang: str = None):
    overall_start = time.perf_counter()

    # 先啟動四個互不依賴的任務：BMC1, BMC2, BMC3, BMC4
    t1_start = time.perf_counter()
    competitive_task = asyncio.create_task(BMC1(input_TA, input_LO, input_BT, lang))
    t2_start = time.perf_counter()
    designed_trend_task = asyncio.create_task(BMC2(input_TA, input_LO, input_BT, lang))
    t3_start = time.perf_counter()
    pds_task = asyncio.create_task(BMC3(input_TA, input_LO, input_BT, lang))
    t4_start = time.perf_counter()
    sds_task = asyncio.create_task(BMC4(input_TA, input_LO, input_BT, lang))

    competitive_result, designed_trend_result, pds_result, sds_result = await asyncio.gather(competitive_task, designed_trend_task, pds_task, sds_task)

    overall_end = time.perf_counter()

    total_time = overall_end - overall_start

    print(f"總耗時: {total_time:.2f} 秒")

    return {
        "competitive_result": json.loads(competitive_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
        "designed_trend_result": json.loads(designed_trend_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
        "pds_result": json.loads(pds_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
        "sds_result": json.loads(sds_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
        # "pics": img_result
    }

async def BMC2_theme(input_TA, input_LO, input_BT, input_theme_style, lang: str = None):
    # 參數定義
    designed_trend_partial_info = """
    | key           | #zh-TW           | definition                                                   |
    | ------------- | ---------------- | ------------------------------------------------------------ |
    | theme_style   | 主題設計風格     | 提出三種主題設計風格，提案請包含色系搭配、造型描述、主打功能、適用族群，以段落方式具體描述細節，增加修飾形容詞。在造型上請精簡敘述，概念性說明適合{TA}的理由 |
    """
    designed_trend_partial_json_sample = """
    {
    "location": " ",
    "bike_type": " ",
    "target_audience":" ",

    "theme_style_partial" :
    {
    "theme_style_partial_sub" : " "
    "theme_style_partial_idea" :
        [
        "1.色系搭配：",
        "2.造型描述：",
        "3.主打功能：",
        "4.適用族群："
        ]
    }
    }
    """

    Tren_1 = f"請檢索{input_BT}產業中最新的設計風格趨勢，包括色彩搭配、車架造型，並分析其適用於不同年齡段的族群。"
    Tren_2 = f"請檢索{input_BT}行業中常用的材質（如碳纖維、鋁合金、鋼鐵等），列出每種材質的優勢與劣勢，並說明適合的騎行者類型。"
    Tren_3 = f"請檢索各類型{input_BT}的主流功能趨勢，包括技術限制、應用層面及適用的族群。"
    Tren_4 = f"請檢索{input_BT}產業如何運用綠色設計及循環材料來實現永續目標，包括這些設計如何影響產品壽命及使用體驗。"
    Tren_5 = f"請檢索智能技術（如AI、感應器技術）在{input_BT}產業的應用情況，並分析這些技術如何提升使用者的安全性和騎行體驗。"

    queries = [Tren_1, Tren_2, Tren_3, Tren_4, Tren_5]

    q2_start = time.time()

    # 平行觸發 5 次 res()
    # responses = await asyncio.gather(*[res(q) for q in queries])
    responses = await multi_res_only(queries)

    q2_end = time.time()

    # 合併回應內容
    rag_responses_partial_Tren = "".join([r["answer"] for r in responses])
    sources = combine_sources(responses)
    all_sources = ", ".join(str(src) for src in sources)

    # designed_trend_partial full costar template
    full_designed_trend_partial_template = """
    {context}
    {objective}
    {style}
    {audience}
    {responce}
    """
    full_designed_trend_partial_prompt = PromptTemplate.from_template(full_designed_trend_partial_template)

    # designed_trend_partial "contex" template
    context_designed_trend_partial_template = """
    # Context
    你是一位專業的自行車產品設計師，不僅熟悉自行車產品開發流程，也熟悉全球自行車設計趨勢動態，在產業市場、地區經濟、設計發展、與社會文化等面向有具備相當豐富的經驗，具備協助客戶探勘設計趨勢情報、洞見技術與創新需求脈絡的能力，貼近產業立場，提供客戶全方位建言。
    """
    context_designed_trend_partial_prompt = PromptTemplate.from_template(context_designed_trend_partial_template)

    # designed_trend_partial "objective" template
    objective_designed_trend_partial_template = """
    # Objective:
    你的任務是參考下方參考文獻並結合自身看法分析適合[{LO}]地區[{TA}]的[{BT}]的設計趨勢進行設計提案，並跟據以下 <designed_trend_partial-info> 定義以及 <example> json 格式，詳細列出內容。提案請注意以下項目
    <designed_trend_partial-info>
    {designed_trend_partial_info}
    </designed_trend_partial-info>
    - 針對主題設計風格[{input_theme_style}]進行提案，提案請包含色系搭配、造型描述、主打功能、適用族群，以段落方式具體描述細節，增加修飾形容詞。在造型上請精簡敘述，概念性說明適合 [{TA}] 的理由

    參考文獻：
    {rag_responses_partial_Tren}
    """
    objective_designed_trend_partial_prompt = PromptTemplate.from_template(objective_designed_trend_partial_template)

    # designed_trend_partial "style" template
    style_designed_trend_partial_template = """
    # Style:
    -內容能夠輕鬆理解，用字精準無贅字，具體的描述項目。
    - 刪除任何無關緊要的文本，無關緊要的文本示例："嗯" 修正任何明顯的拼寫錯誤。
    """
    style_designed_trend_partial_prompt = PromptTemplate.from_template(style_designed_trend_partial_template)

    # designed_trend_partial "audience & tone" template
    audience_designed_trend_partial_template = """
    # Audience & Tone:
    - 目標讀者是公司內部的Product Designer、Researcher、Project Manager，他們需要蒐集市場資料分析趨勢、研究產品資料分析競品、撰寫設計提案、與團隊協作與溝通。
    - 讀者體期待瞭解市場現況，以有效將競品分析轉化成具體可行動的產品策略。因此過程中需保持清晰和有條理的語氣，提供有價值的見解和建議。"""
    audience_designed_trend_partial_prompt = PromptTemplate.from_template(audience_designed_trend_partial_template)

    # designed_trend_partial "responce" template
    responce_designed_trend_partial_template = """
    # Responce:
    - #zh-TW繁體中文進行回答。
    - 每一分析面向的條列點應「具有獨特性、敘述文字有差異性」，避免各項目彼此的內容的重複或過於相似。
    - 必須將輸出結構化為一組json格式。定義見上方 <designed_trend_partial-info>，你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！

    <example>
    {designed_trend_partial_json_sample}
    </example>

    #Start:
    根據上述資訊，針對{LO}地區的{BT}產業進行自行車近年的設計趨勢提案，使用 <example> json 格式，設計趨勢以主題設計風格為方向，分析完就停止，不要給我結論。
    如果您明白了，請開始執行
    """
    responce_designed_trend_partial_prompt = PromptTemplate.from_template(responce_designed_trend_partial_template)

    # designed_trend_partial arragement
    input_designed_trend_partial_prompts = [
        ("context", context_designed_trend_partial_prompt),
        ("objective", objective_designed_trend_partial_prompt),
        ("style", style_designed_trend_partial_prompt),
        ("audience", audience_designed_trend_partial_prompt),
        ("responce", responce_designed_trend_partial_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_designed_trend_partial_prompt, pipeline_prompts=input_designed_trend_partial_prompts
    )

    # final designed_trend_partial prompt
    designed_trend_partial_prompt = pipeline_prompt.format(
            BT = input_BT,
            LO = input_LO,
            TA = input_TA,
            input_theme_style = input_theme_style,
            designed_trend_partial_info = designed_trend_partial_info,
            designed_trend_partial_json_sample = designed_trend_partial_json_sample,
            rag_responses_partial_Tren = str(rag_responses_partial_Tren)
            )

    designed_trend_partial_result = model.predict(
        text=designed_trend_partial_prompt)

    return designed_trend_partial_result

async def BMC2_material(input_TA, input_LO, input_BT, input_material_recomendation, lang: str = None):
    # 參數定義
    designed_trend_partial_info = """
    | key           | #zh-TW           | definition                                                   |
    | ------------- | ---------------- | ------------------------------------------------------------ |
    | material_recomendation     | 材質推薦       | 提出三種材質推薦，提案請包含材質名稱，優勢及劣勢、應用層面、適用族群，請具體的描述 |
    """
    designed_trend_partial_json_sample = """
    {
    "location": " ",
    "bike_type": " ",
    "target_audience":" ",

    "material_recomendation" :
    {
    "material_recomendation_sub" : " "
    "material_recomendation_idea" :
        [
        "1.優勢: ",
        "2.劣勢: ",
        "3.應用層面: ",
        "4.適用族群: "
        ]
    }
    }
    """

    Tren_1 = f"請檢索{input_BT}產業中最新的設計風格趨勢，包括色彩搭配、車架造型，並分析其適用於不同年齡段的族群。"
    Tren_2 = f"請檢索{input_BT}行業中常用的材質（如碳纖維、鋁合金、鋼鐵等），列出每種材質的優勢與劣勢，並說明適合的騎行者類型。"
    Tren_3 = f"請檢索各類型{input_BT}的主流功能趨勢，包括技術限制、應用層面及適用的族群。"
    Tren_4 = f"請檢索{input_BT}產業如何運用綠色設計及循環材料來實現永續目標，包括這些設計如何影響產品壽命及使用體驗。"
    Tren_5 = f"請檢索智能技術（如AI、感應器技術）在{input_BT}產業的應用情況，並分析這些技術如何提升使用者的安全性和騎行體驗。"

    queries = [Tren_1, Tren_2, Tren_3, Tren_4, Tren_5]

    q2_start = time.time()

    # 平行觸發 5 次 res()
    # responses = await asyncio.gather(*[res(q) for q in queries])
    responses = await multi_res_only(queries)

    q2_end = time.time()

    # 合併回應內容
    rag_responses_partial_Tren = "".join([r["answer"] for r in responses])
    sources = combine_sources(responses)
    all_sources = ", ".join(str(src) for src in sources)


    # designed_trend_partial full costar template
    full_designed_trend_partial_template = """
    {context}
    {objective}
    {style}
    {audience}
    {responce}
    """
    full_designed_trend_partial_prompt = PromptTemplate.from_template(full_designed_trend_partial_template)

    # designed_trend_partial "contex" template
    context_designed_trend_partial_template = """
    # Context
    你是一位專業的自行車產品設計師，不僅熟悉自行車產品開發流程，也熟悉全球自行車設計趨勢動態，在產業市場、地區經濟、設計發展、與社會文化等面向有具備相當豐富的經驗，具備協助客戶探勘設計趨勢情報、洞見技術與創新需求脈絡的能力，貼近產業立場，提供客戶全方位建言。
    """
    context_designed_trend_partial_prompt = PromptTemplate.from_template(context_designed_trend_partial_template)

    # designed_trend_partial "objective" template
    objective_designed_trend_partial_template = """
    # Objective:
    你的任務是參考下方參考文獻並結合自身看法分析適合[{LO}]地區[{TA}]的[{BT}]的設計趨勢進行設計提案，並跟據以下 <designed_trend_partial-info> 定義以及 <example> json 格式，詳細列出內容。提案請注意以下項目
    <designed_trend_partial-info>
    {designed_trend_partial_info}
    </designed_trend_partial-info>
    - 針對材質推薦[{input_material_recomendation}]提出三種材質推薦，提案請包含材質名稱，優勢及劣勢、應用層面、適用族群，請具體的描述

    參考文獻：
    {rag_responses_partial_Tren}
    """
    objective_designed_trend_partial_prompt = PromptTemplate.from_template(objective_designed_trend_partial_template)

    # designed_trend_partial "style" template
    style_designed_trend_partial_template = """
    # Style:
    -內容能夠輕鬆理解，用字精準無贅字，具體的描述項目。
    - 刪除任何無關緊要的文本，無關緊要的文本示例："嗯" 修正任何明顯的拼寫錯誤。
    """
    style_designed_trend_partial_prompt = PromptTemplate.from_template(style_designed_trend_partial_template)

    # designed_trend_partial "audience & tone" template
    audience_designed_trend_partial_template = """
    # Audience & Tone:
    - 目標讀者是公司內部的Product Designer、Researcher、Project Manager，他們需要蒐集市場資料分析趨勢、研究產品資料分析競品、撰寫設計提案、與團隊協作與溝通。
    - 讀者體期待瞭解市場現況，以有效將競品分析轉化成具體可行動的產品策略。因此過程中需保持清晰和有條理的語氣，提供有價值的見解和建議。"""
    audience_designed_trend_partial_prompt = PromptTemplate.from_template(audience_designed_trend_partial_template)

    # designed_trend_partial "responce" template
    responce_designed_trend_partial_template = """
    # Responce:
    - #zh-TW繁體中文進行回答。
    - 每一分析面向的條列點應「具有獨特性、敘述文字有差異性」，避免各項目彼此的內容的重複或過於相似。
    - 必須將輸出結構化為一組json格式。定義見上方 <designed_trend_partial-info>，你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！

    <example>
    {designed_trend_partial_json_sample}
    </example>

    #Start:
    根據上述資訊，針對{LO}地區的{BT}產業進行自行車近年的設計趨勢提案，使用 <example> json 格式，設計趨勢以材質推薦為方向，分析完就停止，不要給我結論。
    如果您明白了，請開始執行
    """
    responce_designed_trend_partial_prompt = PromptTemplate.from_template(responce_designed_trend_partial_template)

    # designed_trend_partial arragement
    input_designed_trend_partial_prompts = [
        ("context", context_designed_trend_partial_prompt),
        ("objective", objective_designed_trend_partial_prompt),
        ("style", style_designed_trend_partial_prompt),
        ("audience", audience_designed_trend_partial_prompt),
        ("responce", responce_designed_trend_partial_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_designed_trend_partial_prompt, pipeline_prompts=input_designed_trend_partial_prompts
    )

    # final designed_trend_partial prompt
    designed_trend_partial_prompt = pipeline_prompt.format(
            BT = input_BT,
            LO = input_LO,
            TA = input_TA,
            input_material_recomendation = input_material_recomendation,
            designed_trend_partial_info = designed_trend_partial_info,
            designed_trend_partial_json_sample = designed_trend_partial_json_sample,
            rag_responses_partial_Tren = str(rag_responses_partial_Tren)
            )

    designed_trend_partial_result = model.predict(
        text=designed_trend_partial_prompt)

    return designed_trend_partial_result

async def BMC2_function(input_TA, input_LO, input_BT, input_function_trend, lang: str = None):
    # 參數定義
    designed_trend_partial_info = """
    | key           | #zh-TW           | definition                                                   |
    | ------------- | ---------------- | ------------------------------------------------------------ |
    | function_trend     | 功能趨勢             |  提出三種功能趨勢，包含說明功能的限制、應用層面、適用族群，請具體的描述 |
    """
    designed_trend_partial_json_sample = """
    {
    "location": " ",
    "bike_type": " ",
    "target_audience":" ",

    "function_trend" :
    {
    "function_trend_sub" : " "
    "function_trend_idea" :
        [
        "1.功能限制：",
        "2.應用層面：",
        "3.適用族群："
        ]
    }
    }
    """

    Tren_1 = f"請檢索{input_BT}產業中最新的設計風格趨勢，包括色彩搭配、車架造型，並分析其適用於不同年齡段的族群。"
    Tren_2 = f"請檢索{input_BT}行業中常用的材質（如碳纖維、鋁合金、鋼鐵等），列出每種材質的優勢與劣勢，並說明適合的騎行者類型。"
    Tren_3 = f"請檢索各類型{input_BT}的主流功能趨勢，包括技術限制、應用層面及適用的族群。"
    Tren_4 = f"請檢索{input_BT}產業如何運用綠色設計及循環材料來實現永續目標，包括這些設計如何影響產品壽命及使用體驗。"
    Tren_5 = f"請檢索智能技術（如AI、感應器技術）在{input_BT}產業的應用情況，並分析這些技術如何提升使用者的安全性和騎行體驗。"

    queries = [Tren_1, Tren_2, Tren_3, Tren_4, Tren_5]

    q2_start = time.time()

    # 平行觸發 5 次 res()
    # responses = await asyncio.gather(*[res(q) for q in queries])
    responses = await multi_res_only(queries)

    q2_end = time.time()

    # 合併回應內容
    rag_responses_partial_Tren = "".join([r["answer"] for r in responses])
    sources = combine_sources(responses)
    all_sources = ", ".join(str(src) for src in sources)

    # designed_trend_partial full costar template
    full_designed_trend_partial_template = """
    {context}
    {objective}
    {style}
    {audience}
    {responce}
    """
    full_designed_trend_partial_prompt = PromptTemplate.from_template(full_designed_trend_partial_template)

    # designed_trend_partial "contex" template
    context_designed_trend_partial_template = """
    # Context
    你是一位專業的自行車產品設計師，不僅熟悉自行車產品開發流程，也熟悉全球自行車設計趨勢動態，在產業市場、地區經濟、設計發展、與社會文化等面向有具備相當豐富的經驗，具備協助客戶探勘設計趨勢情報、洞見技術與創新需求脈絡的能力，貼近產業立場，提供客戶全方位建言。
    """
    context_designed_trend_partial_prompt = PromptTemplate.from_template(context_designed_trend_partial_template)

    # designed_trend_partial "objective" template
    objective_designed_trend_partial_template = """
    # Objective:
    你的任務是參考下方參考文獻並結合自身看法分析適合[{LO}]地區[{TA}]的[{BT}]的設計趨勢進行設計提案，並跟據以下 <designed_trend_partial-info> 定義以及 <example> json 格式，詳細列出內容。提案請注意以下項目
    <designed_trend_partial-info>
    {designed_trend_partial_info}
    </designed_trend_partial-info>
    - 針對功能趨勢[{input_function_trend}]提出三種功能趨勢，包含說明功能的限制、應用層面、適用族群，請具體的描述

    參考文獻：
    {rag_responses_partial_Tren}
    """
    objective_designed_trend_partial_prompt = PromptTemplate.from_template(objective_designed_trend_partial_template)

    # designed_trend_partial "style" template
    style_designed_trend_partial_template = """
    # Style:
    -內容能夠輕鬆理解，用字精準無贅字，具體的描述項目。
    - 刪除任何無關緊要的文本，無關緊要的文本示例："嗯" 修正任何明顯的拼寫錯誤。
    """
    style_designed_trend_partial_prompt = PromptTemplate.from_template(style_designed_trend_partial_template)

    # designed_trend_partial "audience & tone" template
    audience_designed_trend_partial_template = """
    # Audience & Tone:
    - 目標讀者是公司內部的Product Designer、Researcher、Project Manager，他們需要蒐集市場資料分析趨勢、研究產品資料分析競品、撰寫設計提案、與團隊協作與溝通。
    - 讀者體期待瞭解市場現況，以有效將競品分析轉化成具體可行動的產品策略。因此過程中需保持清晰和有條理的語氣，提供有價值的見解和建議。"""
    audience_designed_trend_partial_prompt = PromptTemplate.from_template(audience_designed_trend_partial_template)

    # designed_trend_partial "responce" template
    responce_designed_trend_partial_template = """
    # Responce:
    - #zh-TW繁體中文進行回答。
    - 每一分析面向的條列點應「具有獨特性、敘述文字有差異性」，避免各項目彼此的內容的重複或過於相似。
    - 必須將輸出結構化為一組json格式。定義見上方 <designed_trend_partial-info>，你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！

    <example>
    {designed_trend_partial_json_sample}
    </example>

    #Start:
    根據上述資訊，針對{LO}地區的{BT}產業進行自行車近年的設計趨勢提案，使用 <example> json 格式，設計趨勢以功能趨勢為方向，分析完就停止，不要給我結論。
    如果您明白了，請開始執行
    """
    responce_designed_trend_partial_prompt = PromptTemplate.from_template(responce_designed_trend_partial_template)

    # designed_trend_partial arragement
    input_designed_trend_partial_prompts = [
        ("context", context_designed_trend_partial_prompt),
        ("objective", objective_designed_trend_partial_prompt),
        ("style", style_designed_trend_partial_prompt),
        ("audience", audience_designed_trend_partial_prompt),
        ("responce", responce_designed_trend_partial_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_designed_trend_partial_prompt, pipeline_prompts=input_designed_trend_partial_prompts
    )

    # final designed_trend_partial prompt
    designed_trend_partial_prompt = pipeline_prompt.format(
            BT = input_BT,
            LO = input_LO,
            TA = input_TA,
            input_function_trend = input_function_trend,
            designed_trend_partial_info = designed_trend_partial_info,
            designed_trend_partial_json_sample = designed_trend_partial_json_sample,
            rag_responses_partial_Tren = str(rag_responses_partial_Tren)
            )

    designed_trend_partial_result = model.predict(
        text=designed_trend_partial_prompt)

    return designed_trend_partial_result

async def BMC2_patent(input_TA, input_LO, input_BT, input_patent, lang: str = None):
    # 參數定義
    designed_trend_partial_info = """
    | key           | #zh-TW           | definition                                                   |
    | ------------- | ---------------- | ------------------------------------------------------------ |
    | patent    | 申請的專利         | 提出三種近年申請的專利，包含說明專利類型、發明內容、創新技術、設計改進及專利申請範圍，請具體的描述  |
    """
    designed_trend_partial_json_sample = """
    {
    "location": " ",
    "bike_type": " ",
    "target_audience":" ",

    "patent" :
    {
    "patent_sub" : " "
    "patent_idea" :
        "1.專利類型：",
        "2.發明內容：",
        "3.創新技術：",
        "4.設計改進：",
        "5.專利申請範圍："
        ]
    }
    }
    """

    Tren_1 = f"請檢索{input_BT}產業中最新的設計風格趨勢，包括色彩搭配、車架造型，並分析其適用於不同年齡段的族群。"
    Tren_2 = f"請檢索{input_BT}行業中常用的材質（如碳纖維、鋁合金、鋼鐵等），列出每種材質的優勢與劣勢，並說明適合的騎行者類型。"
    Tren_3 = f"請檢索各類型{input_BT}的主流功能趨勢，包括技術限制、應用層面及適用的族群。"
    Tren_4 = f"請檢索{input_BT}產業如何運用綠色設計及循環材料來實現永續目標，包括這些設計如何影響產品壽命及使用體驗。"
    Tren_5 = f"請檢索智能技術（如AI、感應器技術）在{input_BT}產業的應用情況，並分析這些技術如何提升使用者的安全性和騎行體驗。"

    queries = [Tren_1, Tren_2, Tren_3, Tren_4, Tren_5]

    q2_start = time.time()

    # 平行觸發 5 次 res()
    # responses = await asyncio.gather(*[res(q) for q in queries])
    responses = await multi_res_only(queries)

    q2_end = time.time()

    # 合併回應內容
    rag_responses_partial_Tren = "".join([r["answer"] for r in responses])
    sources = combine_sources(responses)
    all_sources = ", ".join(str(src) for src in sources)

    # designed_trend_partial full costar template
    full_designed_trend_partial_template = """
    {context}
    {objective}
    {style}
    {audience}
    {responce}
    """
    full_designed_trend_partial_prompt = PromptTemplate.from_template(full_designed_trend_partial_template)

    # designed_trend_partial "contex" template
    context_designed_trend_partial_template = """
    # Context
    你是一位專業的自行車產品設計師，不僅熟悉自行車產品開發流程，也熟悉全球自行車設計趨勢動態，在產業市場、地區經濟、設計發展、與社會文化等面向有具備相當豐富的經驗，具備協助客戶探勘設計趨勢情報、洞見技術與創新需求脈絡的能力，貼近產業立場，提供客戶全方位建言。
    """
    context_designed_trend_partial_prompt = PromptTemplate.from_template(context_designed_trend_partial_template)

    # designed_trend_partial "objective" template
    objective_designed_trend_partial_template = """
    # Objective:
    你的任務是參考下方參考文獻並結合自身看法分析適合[{LO}]地區[{TA}]的[{BT}]的設計趨勢進行設計提案，並跟據以下 <designed_trend_partial-info> 定義以及 <example> json 格式，詳細列出內容。提案請注意以下項目
    <designed_trend_partial-info>
    {designed_trend_partial_info}
    </designed_trend_partial-info>
    - 針對功能趨勢[{input_patent}]提出三種功能趨勢，包含說明功能的限制、應用層面、適用族群，請具體的描述

    參考文獻：
    {rag_responses_partial_Tren}
    """
    objective_designed_trend_partial_prompt = PromptTemplate.from_template(objective_designed_trend_partial_template)

    # designed_trend_partial "style" template
    style_designed_trend_partial_template = """
    # Style:
    -內容能夠輕鬆理解，用字精準無贅字，具體的描述項目。
    - 刪除任何無關緊要的文本，無關緊要的文本示例："嗯" 修正任何明顯的拼寫錯誤。
    """
    style_designed_trend_partial_prompt = PromptTemplate.from_template(style_designed_trend_partial_template)

    # designed_trend_partial "audience & tone" template
    audience_designed_trend_partial_template = """
    # Audience & Tone:
    - 目標讀者是公司內部的Product Designer、Researcher、Project Manager，他們需要蒐集市場資料分析趨勢、研究產品資料分析競品、撰寫設計提案、與團隊協作與溝通。
    - 讀者體期待瞭解市場現況，以有效將競品分析轉化成具體可行動的產品策略。因此過程中需保持清晰和有條理的語氣，提供有價值的見解和建議。"""
    audience_designed_trend_partial_prompt = PromptTemplate.from_template(audience_designed_trend_partial_template)

    # designed_trend_partial "responce" template
    responce_designed_trend_partial_template = """
    # Responce:
    - #zh-TW繁體中文進行回答。
    - 每一分析面向的條列點應「具有獨特性、敘述文字有差異性」，避免各項目彼此的內容的重複或過於相似。
    - 必須將輸出結構化為一組json格式。定義見上方 <designed_trend_partial-info>，你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！

    <example>
    {designed_trend_partial_json_sample}
    </example>

    #Start:
    根據上述資訊，針對{LO}地區的{BT}產業進行自行車近年的設計趨勢提案，使用 <example> json 格式，設計趨勢以功能趨勢為方向，分析完就停止，不要給我結論。
    如果您明白了，請開始執行
    """
    responce_designed_trend_partial_prompt = PromptTemplate.from_template(responce_designed_trend_partial_template)

    # designed_trend_partial arragement
    input_designed_trend_partial_prompts = [
        ("context", context_designed_trend_partial_prompt),
        ("objective", objective_designed_trend_partial_prompt),
        ("style", style_designed_trend_partial_prompt),
        ("audience", audience_designed_trend_partial_prompt),
        ("responce", responce_designed_trend_partial_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_designed_trend_partial_prompt, pipeline_prompts=input_designed_trend_partial_prompts
    )

    # final designed_trend_partial prompt
    designed_trend_partial_prompt = pipeline_prompt.format(
            BT = input_BT,
            LO = input_LO,
            TA = input_TA,
            input_patent = input_patent,
            designed_trend_partial_info = designed_trend_partial_info,
            designed_trend_partial_json_sample = designed_trend_partial_json_sample,
            rag_responses_partial_Tren = str(rag_responses_partial_Tren)
            )

    designed_trend_partial_result = model.predict(
        text=designed_trend_partial_prompt)

    return designed_trend_partial_result

async def BMC3_partial(input_TA, input_LO, input_BT, input_headline, lang: str = None):
    # 參數定義
    pds_partial_json_sample = """
    {
    "location": " ",
    "bike_type": " ",
    "target_audience": " ",

    "product_designed_trend":
        {
        "headline": " ",
        "solution": " "
        }
    }
    """

    # Define policy search queries
    pds_1 = f"請檢索與分析各品牌{input_BT}在動力系統、智能控制系統、及電池技術上的創新，包括技術實現的挑戰與優勢。"
    pds_2 = f"請檢索{input_BT}品牌如何改善騎行舒適度、智能連接性、及車身設計以滿足不同類型用戶的需求。"
    pds_3 = f"請探索國際大廠如何透過訂閱制、租賃服務及MaaS（Mobility as a Service）等新興商業模式來提升市場滲透率與盈利能力。"
    pds_4 = f"請檢索{input_BT}廠商如何使用循環設計和環保材料來降低碳排放與能源消耗，並探討其對產品與品牌價值的影響。"
    pds_5 = f"請檢索各國（特別是{input_LO}）針對電動{input_BT}的法規支持，包括稅收減免與政府補貼政策，及其對市場發展的影響。"

    queries = [pds_1, pds_2, pds_3, pds_4, pds_5]

    q3_start = time.time()

    # 平行觸發 5 次 res()
    # responses = await asyncio.gather(*[res(q) for q in queries])
    responses = await multi_res_only(queries)

    q3_end = time.time()

    # 合併回應內容
    rag_responses_pds_partial = "".join([r["answer"] for r in responses])
    sources = combine_sources(responses)
    all_sources = ", ".join(str(src) for src in sources)

    # pds_partial full costar template
    full_pds_partial_template = """
    {context}
    {objective}
    {style}
    {audience}
    {responce}
    """
    full_pds_partial_prompt = PromptTemplate.from_template(full_pds_partial_template)

    # pds_partial "contex" template
    context_pds_partial_template = """
    # Context
    你是一位專精在自行車領域的市場策略經理。掌握全球自行車產業重要發展動態，從產業市場、地區經濟、商業模式、國家政策與社會文化等面向都有具備相當豐富的經驗，協助客戶依據當前產業趨勢、開發技術與消費者需求，進行產品創新的設計建議，給出的建議貼近產業立場。
    """
    context_pds_partial_prompt = PromptTemplate.from_template(context_pds_partial_template)

    # pds_partial "objective" template
    objective_pds_partial_template = """
    # Objective:
    你的任務是基於下方分析方向以及參考文獻，統整並詳述當前產業的設計產品創新機會。
    - 先思考適合[{LO}]地區[{TA}]的[{BT}]的「五個」競品品牌
    - 基於上述的競品品牌思考適合[{LO}]地區[{TA}]的[{BT}]的設計趨勢進

    未來創新機會方案：綜合上方思考流程，分析[{input_headline}]「產品創新」面向提供並具體的切入方案。
        - 每個方案標題吸引人具前瞻性
        - 每個方案提供搭配的文案，具體描述方案的細節及市場可行性
        - 方案描述為具深刻洞見的評論，並有一定程度的合理性，指出方案創新的細節，方案描述不少於300字且為結構清晰的段落。


    參考文獻：
    {rag_responses_pds_partial}
    """
    objective_pds_partial_prompt = PromptTemplate.from_template(objective_pds_partial_template)

    # pds_partial "style" template
    style_pds_partial_template = """
    # Style:
    - 文體應專業且具描述性(descriptive)，用字精準無贅字、無拼寫錯誤、條理清晰易讀。
    - 內容必須基於真實且正確的資料，不要編撰或創造內容(Do not make information up)。
    - 以#zh-TW繁體中文進行回答，答案中提及的品牌可以是國內或國外的實際品牌，並將貨幣符號保持為簡寫。
    - 確保每一條列的字數在其分析結果中保持一致性，不應有超過30字的差異。
    - 每一分析面向的條列點應「具有獨特性」，「避免內容的重複或過於相似」。
    - 必須將輸出結構化為一組json格式。json是一種聲明性語言，可讓你對文件進行註解和確認。你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！
    範例如下：
    <example>
    {pds_partial_json_sample}
    </example>

    """
    style_pds_partial_prompt = PromptTemplate.from_template(style_pds_partial_template)

    # pds_partial "audience & tone" template
    audience_pds_partial_template = """
    # Audience & Tone:
    - 目標讀者是公司內部的Product Designer、Researcher、Project Manager，他們需要蒐集市場資料分析趨勢、研究產品資料分析競品、撰寫設計提案、與團隊協作與溝通。
    - 讀者體期待瞭解市場現況，以有效將競品分析轉化成具體可行動的產品策略。因此過程中需保持清晰和有條理的語氣，提供有價值的見解和建議。"""
    audience_pds_partial_prompt = PromptTemplate.from_template(audience_pds_partial_template)

    # pds_partial "responce" template
    responce_pds_partial_template = """
    # Responce:
    - #zh-TW繁體中文進行回答。
    - 每一分析面向的條列點應「具有獨特性、敘述文字有差異性」，避免各項目彼此的內容的重複或過於相似。
    - 必須將輸出結構化為一組json格式。格式見下方<example>，你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！

    <example>
    {pds_partial_json_sample}
    </example>

    #Start:
    根據上述資訊，針對{LO}地區的{BT}產業，參考參考文獻，統整並詳述當前產業於[{input_headline}]這個主題的設計產品創新機會，使用 <example> json 格式。
    如果您明白了，請開始執行
    """
    responce_pds_partial_prompt = PromptTemplate.from_template(responce_pds_partial_template)

    # pds_partial arragement
    input_pds_partial_prompts = [
        ("context", context_pds_partial_prompt),
        ("objective", objective_pds_partial_prompt),
        ("style", style_pds_partial_prompt),
        ("audience", audience_pds_partial_prompt),
        ("responce", responce_pds_partial_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_pds_partial_prompt, pipeline_prompts=input_pds_partial_prompts
    )

    # final pds_partial prompt
    pds_partial_prompt = pipeline_prompt.format(
            BT = input_BT,
            LO = input_LO,
            TA = input_TA,
            pds_partial_json_sample = pds_partial_json_sample,
            rag_responses_pds_partial = str(rag_responses_pds_partial),
            input_headline = input_headline
            )

    pds_partial_result = model.predict(
        text=pds_partial_prompt)

    return pds_partial_result

async def BMC4_partial(input_TA, input_LO, input_BT, lang: str = None):
    # 參數定義
    sds_partial_json_sample = """
    {
    "location": " ",
    "bike_type": " ",
    "target_audience": " ",

    "service_designed_trend":
        {
        "headline": " ",
        "solution": " "
        }
    }
    """

    # Define policy search queries
    sds_1 = f"請檢索全球{input_BT}產業中智慧化技術的應用實例，並分析這些技術如何提升使用者的使用體驗及滿足市場需求。"
    sds_2 = f"請檢索{input_BT}產業在循環經濟設計中的案例，評估如何通過技術創新實現產業的可持續性發展。"
    sds_3 = f"請檢索有關如何通過人本交通設計提升城市中{input_TA}的安全性、便捷性與舒適性的案例分析。"
    sds_4 = f"請探索{input_BT}共享租賃系統在市場中的成功應用案例，重點分析其便捷性與使用者體驗。"
    sds_5 = f"請搜索{input_BT}中創新服務的商業模式，如 B2B 一站式解決方案，並分析其如何滿足使用者需求並促進產業增長。"

    queries = [sds_1, sds_2, sds_3, sds_4, sds_5]

    q4_start = time.time()

    # 平行觸發 5 次 res()
    # responses = await asyncio.gather(*[res(q) for q in queries])
    responses = await multi_res_only(queries)

    q4_end = time.time()

    # 合併回應內容
    rag_responses_sds_partial = "".join([r["answer"] for r in responses])
    sources = combine_sources(responses)
    all_sources = ", ".join(str(src) for src in sources)

    # sds_partial full costar template
    full_sds_partial_template = """
    {context}
    {objective}
    {style}
    {audience}
    {responce}
    """
    full_sds_partial_prompt = PromptTemplate.from_template(full_sds_partial_template)

    # sds_partial "contex" template
    context_sds_partial_template = """
    # Context
    你是一位專精在自行車領域的市場策略經理。掌握全球自行車產業重要發展動態，從產業市場、地區經濟、商業模式、國家政策與社會文化等面向都有具備相當豐富的經驗，協助客戶依據當前產業趨勢、開發技術與消費者需求，進行產品創新的設計建議，給出的建議貼近產業立場。
    """
    context_sds_partial_prompt = PromptTemplate.from_template(context_sds_partial_template)

    # sds_partial "objective" template
    objective_sds_partial_template = """
    # Objective:
    你的任務是基於下方分析方向以及參考文獻，統整並詳述當前產業的設計產品創新機會。
    - 先思考適合[{LO}]地區[{TA}]的[{BT}]的「五個」競品品牌
    - 基於上述的競品品牌思考適合[{LO}]地區[{TA}]的[{BT}]的設計趨勢進

    未來創新機會方案：綜合上方思考流程，分析[{input_headline}]「服務創新」面向提供並具體的切入方案。
        - 每個方案標題吸引人具前瞻性
        - 每個方案提供搭配的文案，具體描述方案的細節及市場可行性
        - 方案描述為具深刻洞見的評論，並有一定程度的合理性，指出方案創新的細節，方案描述不少於300字且為結構清晰的段落。


    參考文獻：
    {rag_responses_sds_partial}
    """
    objective_sds_partial_prompt = PromptTemplate.from_template(objective_sds_partial_template)

    # sds_partial "style" template
    style_sds_partial_template = """
    # Style:
    - 文體應專業且具描述性(descriptive)，用字精準無贅字、無拼寫錯誤、條理清晰易讀。
    - 內容必須基於真實且正確的資料，不要編撰或創造內容(Do not make information up)。
    - 以#zh-TW繁體中文進行回答，答案中提及的品牌可以是國內或國外的實際品牌，並將貨幣符號保持為簡寫。
    - 確保每一條列的字數在其分析結果中保持一致性，不應有超過30字的差異。
    - 每一分析面向的條列點應「具有獨特性」，「避免內容的重複或過於相似」。
    - 必須將輸出結構化為一組json格式。json是一種聲明性語言，可讓你對文件進行註解和確認。你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！
    範例如下：
    <example>
    {sds_partial_json_sample}
    </example>

    """
    style_sds_partial_prompt = PromptTemplate.from_template(style_sds_partial_template)

    # sds_partial "audience & tone" template
    audience_sds_partial_template = """
    # Audience & Tone:
    - 目標讀者是公司內部的Product Designer、Researcher、Project Manager，他們需要蒐集市場資料分析趨勢、研究產品資料分析競品、撰寫設計提案、與團隊協作與溝通。
    - 讀者體期待瞭解市場現況，以有效將競品分析轉化成具體可行動的產品策略。因此過程中需保持清晰和有條理的語氣，提供有價值的見解和建議。"""
    audience_sds_partial_prompt = PromptTemplate.from_template(audience_sds_partial_template)

    # sds_partial "responce" template
    responce_sds_partial_template = """
    # Responce:
    - #zh-TW繁體中文進行回答。
    - 每一分析面向的條列點應「具有獨特性、敘述文字有差異性」，避免各項目彼此的內容的重複或過於相似。
    - 必須將輸出結構化為一組json格式。格式見下方<example>，你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！

    <example>
    {sds_partial_json_sample}
    </example>

    #Start:
    根據上述資訊，針對{LO}地區的{BT}產業，參考參考文獻，統整並詳述當前產業於[{input_headline}]這個主題的設計產品創新機會，使用 <example> json 格式。
    如果您明白了，請開始執行
    """
    responce_sds_partial_prompt = PromptTemplate.from_template(responce_sds_partial_template)

    # sds_partial arragement
    input_sds_partial_prompts = [
        ("context", context_sds_partial_prompt),
        ("objective", objective_sds_partial_prompt),
        ("style", style_sds_partial_prompt),
        ("audience", audience_sds_partial_prompt),
        ("responce", responce_sds_partial_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_sds_partial_prompt, pipeline_prompts=input_sds_partial_prompts
    )

    # final sds_partial prompt
    sds_partial_prompt = pipeline_prompt.format(
            BT = input_BT,
            LO = input_LO,
            TA = input_TA,
            sds_partial_json_sample = sds_partial_json_sample,
            rag_responses_sds_partial = str(rag_responses_sds_partial),
            input_headline = input_headline
            )

    sds_partial_result = model.predict(
        text=sds_partial_prompt)

    return sds_partial_result