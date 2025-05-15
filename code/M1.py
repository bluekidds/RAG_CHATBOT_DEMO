# from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate

from code.config import config
import os

openai_api_key = config['OPENAI_API_KEY']
azure_openai_endpoint = config['AZURE_OPENAI_ENDPOINT']
os.environ["AZURE_OPENAI_API_KEY"] = openai_api_key
os.environ["AZURE_OPENAI_ENDPOINT"] = azure_openai_endpoint
# API key
model = AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",
        api_version="2024-02-15-preview",
    )

default_location = "台灣"

class Keyword(BaseModel):
    location: str = Field(description="location mention in sentence")
    product: str = Field(description="product mention in sentence")

def get_keyword(query):
    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=Keyword)
    prompt = PromptTemplate(
        template="Extract the key word from query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser
    dic = chain.invoke({"query": query})

    try:
        print(dic['location'])
    except:
        dic = dict(dic, location=default_location)

    print(dic)

    product = dic['product']
    country = dic['location']

    return dic

def M1_1(query):
    params = get_keyword(query)
    country = params['location']
    product = params['product']
    # 參數定義
    pestel = "政治"
    pestel_whole = "政治面向（ 政治 Political ）"
    pestel_sub = ["政策支持","法規環境","國際關係"]
    objective_item_M1_1 = """
    - 未來稅率的更動
    - 可能會影響公司營運的當地法規
    - 目前正在提議修改的法規，如果修法通過，説明對公司可能的影響
    - 對公司可能有影響的外交政策
    """
    M1_1_json_sample = """
    {
    'pestel':'政治',
    'pestel_info':[
    {
        'pestel_sub':'政策支持',
        'content':'分析內文 字數約為80-90個字'
        },
    {
        'pestel_sub':'法規環境',
        'content':'分析內文 字數約為80-90個字'
        },
    {
        'pestel_sub':'國際關係',
        'content':'分析內文 字數約為80-90個字'
        }
    ]
    }
    """

    # M1-1 full costar template
    full_M1_1_template = """
    {context}
    {objective}
    {style}
    {tone}
    {audience}
    {responce}
    """
    full_M1_1_prompt = PromptTemplate.from_template(full_M1_1_template)

    # M1-1 "contex" template
    context_M1_1_template = """
    # Context
    你是一位專業的產業管理顧問。掌握全球產業重要發展動態，從產業市場、地區經濟、地方基礎建設、國家政策與社會文化等面向進行全面的分析，協助客戶探勘產業趨勢情報、洞見技術與創新需求脈絡，貼近產業立場，提供客戶全方位建言。
    """
    context_M1_1_prompt = PromptTemplate.from_template(context_M1_1_template)

    # M1-1 "objective" template
    objective_M1_1_template = """
    # Objective:
    你的任務是以PESTEL分析法，針對特定國家或區域的產業現況進行"{pestel_whole}"的分析，分析時要考量以下幾個面向與指標
    {objective_item_M1_1}
    """
    objective_M1_1_prompt = PromptTemplate.from_template(objective_M1_1_template)

    # M1-1 "style" template
    style_M1_1_template = """
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

    {M1_1_json_sample}
    """
    style_M1_1_prompt = PromptTemplate.from_template(style_M1_1_template)

    # M1-1 "tone" template
    tone_M1_1_template = """
    # Tone:
    在整個過程中保持清晰和有條理的語氣，具備說服力地說明產品與特定地區的{pestel}關係。具備清楚且有方向的分析，提供有價值的見解和建議。
    """
    tone_M1_1_prompt = PromptTemplate.from_template(tone_M1_1_template)

    # M1-1 "audience" template
    audience_M1_1_template = """
    # Audience:
    目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
    """
    audience_M1_1_prompt = PromptTemplate.from_template(audience_M1_1_template)

    # M1-1 "responce" template
    responce_M1_1_template = """
    #Responce:
    用PESTEL分析{country}地區的{product}，從'{pestel}'面向進行分析，並以'{pestel_sub}'作為 'pestel_sub'分項，字數約為80-90個字。

    #Start:
    如果您明白了，請開始進行分析
    """
    responce_M1_1_prompt = PromptTemplate.from_template(responce_M1_1_template)

    # M1-1 arragement
    input_M1_1_prompts = [
        ("context", context_M1_1_prompt),
        ("objective", objective_M1_1_prompt),
        ("style", style_M1_1_prompt),
        ("tone", tone_M1_1_prompt),
        ("audience", audience_M1_1_prompt),
        ("responce", responce_M1_1_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_M1_1_prompt, pipeline_prompts=input_M1_1_prompts
    )

    # final M1-1 prompt
    M1_1_prompt = pipeline_prompt.format(
            product= product,
            country= country,
            pestel = pestel,
            pestel_sub = pestel_sub,
            pestel_whole = pestel_whole,
            objective_item_M1_1 = objective_item_M1_1,
            M1_1_json_sample = M1_1_json_sample
            )

    print(M1_1_prompt)

    M1_1_result = model.predict(
        text=M1_1_prompt)

    print(M1_1_result)
    return M1_1_result

def M1_2(query):
    params = get_keyword(query)
    country = params['location']
    product = params['product']
    # 參數定義
    pestel = "經濟"
    pestel_whole = "經濟面向（ 經濟 Economic ）"
    pestel_sub = ["市場需求","產業競爭力","成本結構"]
    objective_item_M1_2 = """
    - 目前整體的經濟狀況與消費力道的上升/下降狀況，並說明原因
    - 當去匯率狀態及未來可能的改變
    - 失業率的上升/下降狀況及對公司人才招募的影響
    - 未來原物料價格的走勢及對公司利潤的影響
    """
    M1_2_json_sample = """
    {
    'pestel':'政治',
    'pestel_info':[
    {
        'pestel_sub':'政策支持',
        'content':'分析內文 字數約為80-90個字'
        },
    {
        'pestel_sub':'法規環境',
        'content':'分析內文 字數約為80-90個字'
        },
    {
        'pestel_sub':'國際關係',
        'content':'分析內文 字數約為80-90個字'
        }
    ]
    }
    """

    # M1-2 full costar template
    full_M1_2_template = """
    {context}
    {objective}
    {style}
    {tone}
    {audience}
    {responce}
    """
    full_M1_2_prompt = PromptTemplate.from_template(full_M1_2_template)

    # M1-2 "contex" template
    context_M1_2_template = """
    # Context
    你是一位專業的產業管理顧問。掌握全球產業重要發展動態，從產業市場、地區經濟、地方基礎建設、國家政策與社會文化等面向進行全面的分析，協助客戶探勘產業趨勢情報、洞見技術與創新需求脈絡，貼近產業立場，提供客戶全方位建言。
    """
    context_M1_2_prompt = PromptTemplate.from_template(context_M1_2_template)

    # M1-2 "objective" template
    objective_M1_2_template = """
    # Objective:
    你的任務是以PESTEL分析法，針對特定國家或區域的產業現況進行"{pestel_whole}"的分析，分析時要考量以下幾個面向與指標
    {objective_item_M1_2}
    """
    objective_M1_2_prompt = PromptTemplate.from_template(objective_M1_2_template)

    # M1-2 "style" template
    style_M1_2_template = """
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

    {M1_2_json_sample}
    """
    style_M1_2_prompt = PromptTemplate.from_template(style_M1_2_template)

    # M1-2 "tone" template
    tone_M1_2_template = """
    # Tone:
    在整個過程中保持清晰和有條理的語氣，具備說服力地說明產品與特定地區的{pestel}關係。具備清楚且有方向的分析，提供有價值的見解和建議。
    """
    tone_M1_2_prompt = PromptTemplate.from_template(tone_M1_2_template)

    # M1-2 "audience" template
    audience_M1_2_template = """
    # Audience:
    目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
    """
    audience_M1_2_prompt = PromptTemplate.from_template(audience_M1_2_template)

    # M1-2 "responce" template
    responce_M1_2_template = """
    #Responce:
    用PESTEL分析{country}地區的{product}，從'{pestel}'面向進行分析，並以'{pestel_sub}'作為 'pestel_sub'分項，字數約為80-90個字。

    #Start:
    如果您明白了，請開始進行分析
    """
    responce_M1_2_prompt = PromptTemplate.from_template(responce_M1_2_template)

    # M1-2 arragement
    input_M1_2_prompts = [
        ("context", context_M1_2_prompt),
        ("objective", objective_M1_2_prompt),
        ("style", style_M1_2_prompt),
        ("tone", tone_M1_2_prompt),
        ("audience", audience_M1_2_prompt),
        ("responce", responce_M1_2_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_M1_2_prompt, pipeline_prompts=input_M1_2_prompts
    )

    # final M1-2 prompt
    M1_2_prompt = pipeline_prompt.format(
            product= product,
            country= country,
            pestel = pestel,
            pestel_sub = pestel_sub,
            pestel_whole = pestel_whole,
            objective_item_M1_2 = objective_item_M1_2,
            M1_2_json_sample = M1_2_json_sample
            )

    print(M1_2_prompt)

    M1_2_result = model.predict(
        text=M1_2_prompt)

    print(M1_2_result)
    return M1_2_result

def M1_3(query):
    params = get_keyword(query)
    country = params['location']
    product = params['product']
    # 參數定義
    pestel = "社會"
    pestel_whole = "社會面向（ 社會 Social ）"
    pestel_sub = ["綠色意識","都市化趨勢","消費者偏好"]
    objective_item_M1_3 = """
    - 出生人口的上升/下降狀況
    - 消費者關注的文化與教育的議題
    - 消費者近期喜歡的東西，生活習慣的改變以及消費頻率的上升/下降情形
    - 可能會提高公司業績的社會的因素
    """
    M1_3_json_sample = """
    {
    'pestel':'政治',
    'pestel_info':[
    {
        'pestel_sub':'政策支持',
        'content':'分析內文 字數約為80-90個字'
        },
    {
        'pestel_sub':'法規環境',
        'content':'分析內文 字數約為80-90個字'
        },
    {
        'pestel_sub':'國際關係',
        'content':'分析內文 字數約為80-90個字'
        }
    ]
    }
    """

    # M1-3 full costar template
    full_M1_3_template = """
    {context}
    {objective}
    {style}
    {tone}
    {audience}
    {responce}
    """
    full_M1_3_prompt = PromptTemplate.from_template(full_M1_3_template)

    # M1-3 "contex" template
    context_M1_3_template = """
    # Context
    你是一位專業的產業管理顧問。掌握全球產業重要發展動態，從產業市場、地區經濟、地方基礎建設、國家政策與社會文化等面向進行全面的分析，協助客戶探勘產業趨勢情報、洞見技術與創新需求脈絡，貼近產業立場，提供客戶全方位建言。
    """
    context_M1_3_prompt = PromptTemplate.from_template(context_M1_3_template)

    # M1-3 "objective" template
    objective_M1_3_template = """
    # Objective:
    你的任務是以PESTEL分析法，針對特定國家或區域的產業現況進行"{pestel_whole}"的分析，分析時要考量以下幾個面向與指標
    {objective_item_M1_3}
    """
    objective_M1_3_prompt = PromptTemplate.from_template(objective_M1_3_template)

    # M1-3 "style" template
    style_M1_3_template = """
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

    {M1_3_json_sample}
    """
    style_M1_3_prompt = PromptTemplate.from_template(style_M1_3_template)

    # M1-3 "tone" template
    tone_M1_3_template = """
    # Tone:
    在整個過程中保持清晰和有條理的語氣，具備說服力地說明產品與特定地區的{pestel}關係。具備清楚且有方向的分析，提供有價值的見解和建議。
    """
    tone_M1_3_prompt = PromptTemplate.from_template(tone_M1_3_template)

    # M1-3 "audience" template
    audience_M1_3_template = """
    # Audience:
    目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
    """
    audience_M1_3_prompt = PromptTemplate.from_template(audience_M1_3_template)

    # M1-3 "responce" template
    responce_M1_3_template = """
    #Responce:
    用PESTEL分析{country}地區的{product}，從'{pestel}'面向進行分析，並以'{pestel_sub}'作為 'pestel_sub'分項，字數約為80-90個字。

    #Start:
    如果您明白了，請開始進行分析
    """
    responce_M1_3_prompt = PromptTemplate.from_template(responce_M1_3_template)

    # M1-3 arragement
    input_M1_3_prompts = [
        ("context", context_M1_3_prompt),
        ("objective", objective_M1_3_prompt),
        ("style", style_M1_3_prompt),
        ("tone", tone_M1_3_prompt),
        ("audience", audience_M1_3_prompt),
        ("responce", responce_M1_3_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_M1_3_prompt, pipeline_prompts=input_M1_3_prompts
    )

    # final M1-3 prompt
    M1_3_prompt = pipeline_prompt.format(
            product= product,
            country= country,
            pestel = pestel,
            pestel_sub = pestel_sub,
            pestel_whole = pestel_whole,
            objective_item_M1_3 = objective_item_M1_3,
            M1_3_json_sample = M1_3_json_sample
            )

    print(M1_3_prompt)

    M1_3_result = model.predict(
        text=M1_3_prompt)

    print(M1_3_result)
    return M1_3_result

def M1_4(query):
    params = get_keyword(query)
    country = params['location']
    product = params['product']

    # 參數定義
    pestel = "技術"
    pestel_whole = "技術面向（ 技術 Technological ）"
    pestel_sub = ["核心技術","基礎建設","週邊設備"]
    objective_item_M1_4 = """
    - 近期發表且可以拿來應用的新科技
    - 能夠推動公司成長數位技術
    - 近期政府和研究單位關注的科技議題
    """
    M1_4_json_sample = """
    {
    'pestel':'政治',
    'pestel_info':[
    {
        'pestel_sub':'政策支持',
        'content':'分析內文 字數約為80-90個字'
        },
    {
        'pestel_sub':'法規環境',
        'content':'分析內文 字數約為80-90個字'
        },
    {
        'pestel_sub':'國際關係',
        'content':'分析內文 字數約為80-90個字'
        }
    ]
    }
    """

    # M1-4 full costar template
    full_M1_4_template = """
    {context}
    {objective}
    {style}
    {tone}
    {audience}
    {responce}
    """
    full_M1_4_prompt = PromptTemplate.from_template(full_M1_4_template)

    # M1-4 "contex" template
    context_M1_4_template = """
    # Context
    你是一位專業的產業管理顧問。掌握全球產業重要發展動態，從產業市場、地區經濟、地方基礎建設、國家政策與社會文化等面向進行全面的分析，協助客戶探勘產業趨勢情報、洞見技術與創新需求脈絡，貼近產業立場，提供客戶全方位建言。
    """
    context_M1_4_prompt = PromptTemplate.from_template(context_M1_4_template)

    # M1-4 "objective" template
    objective_M1_4_template = """
    # Objective:
    你的任務是以PESTEL分析法，針對特定國家或區域的產業現況進行"{pestel_whole}"的分析，分析時要考量以下幾個面向與指標
    {objective_item_M1_4}
    """
    objective_M1_4_prompt = PromptTemplate.from_template(objective_M1_4_template)

    # M1-4 "style" template
    style_M1_4_template = """
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

    {M1_4_json_sample}
    """
    style_M1_4_prompt = PromptTemplate.from_template(style_M1_4_template)

    # M1-4 "tone" template
    tone_M1_4_template = """
    # Tone:
    在整個過程中保持清晰和有條理的語氣，具備說服力地說明產品與特定地區的{pestel}關係。具備清楚且有方向的分析，提供有價值的見解和建議。
    """
    tone_M1_4_prompt = PromptTemplate.from_template(tone_M1_4_template)

    # M1-4 "audience" template
    audience_M1_4_template = """
    # Audience:
    目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
    """
    audience_M1_4_prompt = PromptTemplate.from_template(audience_M1_4_template)

    # M1-4 "responce" template
    responce_M1_4_template = """
    #Responce:
    用PESTEL分析{country}地區的{product}，從'{pestel}'面向進行分析，並以'{pestel_sub}'作為 'pestel_sub'分項，字數約為80-90個字。

    #Start:
    如果您明白了，請開始進行分析
    """
    responce_M1_4_prompt = PromptTemplate.from_template(responce_M1_4_template)

    # M1-4 arragement
    input_M1_4_prompts = [
        ("context", context_M1_4_prompt),
        ("objective", objective_M1_4_prompt),
        ("style", style_M1_4_prompt),
        ("tone", tone_M1_4_prompt),
        ("audience", audience_M1_4_prompt),
        ("responce", responce_M1_4_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_M1_4_prompt, pipeline_prompts=input_M1_4_prompts
    )

    # final M1-4 prompt
    M1_4_prompt = pipeline_prompt.format(
            product= product,
            country= country,
            pestel = pestel,
            pestel_sub = pestel_sub,
            pestel_whole = pestel_whole,
            objective_item_M1_4 = objective_item_M1_4,
            M1_4_json_sample = M1_4_json_sample
            )

    print(M1_4_prompt)

    M1_4_result = model.predict(
        text=M1_4_prompt)

    print(M1_4_result)
    return M1_4_result

def M1_5(query):
    params = get_keyword(query)
    country = params['location']
    product = params['product']

    # 參數定義
    pestel = "環境"
    pestel_whole = "環境面向（ 環境 Eviromental ）"
    pestel_sub = ["氣候條件","環保政策","可能再生能源"]
    objective_item_M1_5 = """
    """
    M1_5_json_sample = """
    {
    'pestel':'政治',
    'pestel_info':[
    {
        'pestel_sub':'政策支持',
        'content':'分析內文 字數約為80-90個字'
        },
    {
        'pestel_sub':'法規環境',
        'content':'分析內文 字數約為80-90個字'
        },
    {
        'pestel_sub':'國際關係',
        'content':'分析內文 字數約為80-90個字'
        }
    ]
    }
    """

    # M1-5 full costar template
    full_M1_5_template = """
    {context}
    {objective}
    {style}
    {tone}
    {audience}
    {responce}
    """
    full_M1_5_prompt = PromptTemplate.from_template(full_M1_5_template)

    # M1-5 "contex" template
    context_M1_5_template = """
    # Context
    你是一位專業的產業管理顧問。掌握全球產業重要發展動態，從產業市場、地區經濟、地方基礎建設、國家政策與社會文化等面向進行全面的分析，協助客戶探勘產業趨勢情報、洞見技術與創新需求脈絡，貼近產業立場，提供客戶全方位建言。
    """
    context_M1_5_prompt = PromptTemplate.from_template(context_M1_5_template)

    # M1-5 "objective" template
    objective_M1_5_template = """
    # Objective:
    你的任務是以PESTEL分析法，針對特定國家或區域的產業現況進行"{pestel_whole}"的分析，分析時要考量以下幾個面向與指標
    {objective_item_M1_5}
    """
    objective_M1_5_prompt = PromptTemplate.from_template(objective_M1_5_template)

    # M1-5 "style" template
    style_M1_5_template = """
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

    {M1_5_json_sample}
    """
    style_M1_5_prompt = PromptTemplate.from_template(style_M1_5_template)

    # M1-5 "tone" template
    tone_M1_5_template = """
    # Tone:
    在整個過程中保持清晰和有條理的語氣，具備說服力地說明產品與特定地區的{pestel}關係。具備清楚且有方向的分析，提供有價值的見解和建議。
    """
    tone_M1_5_prompt = PromptTemplate.from_template(tone_M1_5_template)

    # M1-5 "audience" template
    audience_M1_5_template = """
    # Audience:
    目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
    """
    audience_M1_5_prompt = PromptTemplate.from_template(audience_M1_5_template)

    # M1-5 "responce" template
    responce_M1_5_template = """
    #Responce:
    用PESTEL分析{country}地區的{product}，從'{pestel}'面向進行分析，並以'{pestel_sub}'作為 'pestel_sub'分項，字數約為80-90個字。

    #Start:
    如果您明白了，請開始進行分析
    """
    responce_M1_5_prompt = PromptTemplate.from_template(responce_M1_5_template)

    # M1-5 arragement
    input_M1_5_prompts = [
        ("context", context_M1_5_prompt),
        ("objective", objective_M1_5_prompt),
        ("style", style_M1_5_prompt),
        ("tone", tone_M1_5_prompt),
        ("audience", audience_M1_5_prompt),
        ("responce", responce_M1_5_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_M1_5_prompt, pipeline_prompts=input_M1_5_prompts
    )

    # final M1-5 prompt
    M1_5_prompt = pipeline_prompt.format(
            product= product,
            country= country,
            pestel = pestel,
            pestel_sub = pestel_sub,
            pestel_whole = pestel_whole,
            objective_item_M1_5 = objective_item_M1_5,
            M1_5_json_sample = M1_5_json_sample
            )

    print(M1_5_prompt)

    M1_5_result = model.predict(
        text=M1_5_prompt)

    print(M1_5_result)
    return M1_5_result

def M1_6(query):
    params = get_keyword(query)
    country = params['location']
    product = params['product']
    # 參數定義
    pestel = "法規"
    pestel_whole = "法規（ 法規 Legal ）"
    pestel_sub = ["政治因素","經濟因素","社會文化因素"]
    objective_item_M1_6 = """
    """
    M1_6_json_sample = """
    {
    'pestel':'政治',
    'pestel_info':[
    {
        'pestel_sub':'政策支持',
        'content':'分析內文 字數約為80-90個字'
        },
    {
        'pestel_sub':'法規環境',
        'content':'分析內文 字數約為80-90個字'
        },
    {
        'pestel_sub':'國際關係',
        'content':'分析內文 字數約為80-90個字'
        }
    ]
    }
    """

    # M1-6 full costar template
    full_M1_6_template = """
    {context}
    {objective}
    {style}
    {tone}
    {audience}
    {responce}
    """
    full_M1_6_prompt = PromptTemplate.from_template(full_M1_6_template)

    # M1-6 "contex" template
    context_M1_6_template = """
    # Context
    你是一位專業的產業管理顧問。掌握全球產業重要發展動態，從產業市場、地區經濟、地方基礎建設、國家政策與社會文化等面向進行全面的分析，協助客戶探勘產業趨勢情報、洞見技術與創新需求脈絡，貼近產業立場，提供客戶全方位建言。
    """
    context_M1_6_prompt = PromptTemplate.from_template(context_M1_6_template)

    # M1-6 "objective" template
    objective_M1_6_template = """
    # Objective:
    你的任務是以PESTEL分析法，針對特定國家或區域的產業現況進行"{pestel_whole}"的分析，分析時要考量以下幾個面向與指標
    {objective_item_M1_6}
    """
    objective_M1_6_prompt = PromptTemplate.from_template(objective_M1_6_template)

    # M1-6 "style" template
    style_M1_6_template = """
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

    {M1_6_json_sample}
    """
    style_M1_6_prompt = PromptTemplate.from_template(style_M1_6_template)

    # M1-6 "tone" template
    tone_M1_6_template = """
    # Tone:
    在整個過程中保持清晰和有條理的語氣，具備說服力地說明產品與特定地區的{pestel}關係。具備清楚且有方向的分析，提供有價值的見解和建議。
    """
    tone_M1_6_prompt = PromptTemplate.from_template(tone_M1_6_template)

    # M1-6 "audience" template
    audience_M1_6_template = """
    # Audience:
    目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
    """
    audience_M1_6_prompt = PromptTemplate.from_template(audience_M1_6_template)

    # M1-6 "responce" template
    responce_M1_6_template = """
    #Responce:
    用PESTEL分析{country}地區的{product}，從'{pestel}'面向進行分析，並以'{pestel_sub}'作為 'pestel_sub'分項，字數約為80-90個字。

    #Start:
    如果您明白了，請開始進行分析
    """
    responce_M1_6_prompt = PromptTemplate.from_template(responce_M1_6_template)

    # M1-6 arragement
    input_M1_6_prompts = [
        ("context", context_M1_6_prompt),
        ("objective", objective_M1_6_prompt),
        ("style", style_M1_6_prompt),
        ("tone", tone_M1_6_prompt),
        ("audience", audience_M1_6_prompt),
        ("responce", responce_M1_6_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_M1_6_prompt, pipeline_prompts=input_M1_6_prompts
    )

    # final M1-6 prompt
    M1_6_prompt = pipeline_prompt.format(
            product= product,
            country= country,
            pestel = pestel,
            pestel_sub = pestel_sub,
            pestel_whole = pestel_whole,
            objective_item_M1_6 = objective_item_M1_6,
            M1_6_json_sample = M1_6_json_sample
            )

    print(M1_6_prompt)

    M1_6_result = model.predict(
        text=M1_6_prompt)

    print(M1_6_result)
    return M1_6_result