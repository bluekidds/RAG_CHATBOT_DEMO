from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel
from langchain_core.pydantic_v1 import Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate
from code.utils import dalle3

from code.config import config
import time
import json
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
    print('get query:')
    print(query)
    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=Keyword)
    print('is here?')
    prompt = PromptTemplate(
        template="Extract the key word from query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    print('or here?')

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

def M1M2(query, needs, subq, origin):
    if origin == 'bobai.rdlab.tw':
        print('change model to bobai')
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key="",  # if you prefer to pass api key in directly instaed of using env vars
            # base_url="...",
            # organization="...",
            # other params...
        )
    elif origin == 'dev.daivinci.punwave.com':
        print('change model to dev')
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key="",  # if you prefer to pass api key in directly instaed of using env vars
            # base_url="...",
            # organization="...",
            # other params...
        )
    params = get_keyword(query)
    country = params['location']
    product = params['product']

    M0_result = ''
    M1_1_result = '{}'
    M1_2_result = '{}'
    M1_3_result = '{}'
    M1_4_result = '{}'
    M1_5_result = '{}'
    M1_6_result = '{}'
    M2_1_result = '{}'
    M2_2_result = '{}'
    M2_3_a1_result = '{}'
    M2_3_a2_result = '{}'
    M2_3_a3_result = '{}'
    M2_3_a4_result = '{}'
    M2_3_b1_result = ''
    M2_3_b2_result = ''
    M2_3_b3_result = ''
    M2_3_b4_result = ''

    start_time = time.time()
    if ('title' in needs):
        # 參數定義
        M0_json_sample = """
        {
        'project_title':'',
        }
        """

        # M0 full template
        full_M0_template = """
        {task}
        """
        full_M0_prompt = PromptTemplate.from_template(full_M0_template)

        task_M0_template = """
        # task
        請根據下方提供的地區及產業，對其摘要並給予一個不超過十個字的命名。

        地區：{country}
        產業：{product}

        刪除任何無關緊要的文本，無關緊要的文本示例："嗯" 修正任何明顯的拼寫錯誤。

        你必須將輸出結構化為一組json格式。
        並只能以'project_title'作為json的註解,不要自行發散出其他的名稱

        json是一種聲明性語言，可讓你對文件進行註解和確認。
        你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全匹配！
        範例如下：

        {M0_json_sample}

        如果您明白了，請開始進行分析
        """
        task_M0_prompt = PromptTemplate.from_template(task_M0_template)

        # M0 arragement
        input_M0_prompts = [
            ("task", task_M0_prompt),
        ]
        pipeline_prompt = PipelinePromptTemplate(
            final_prompt=full_M0_prompt, pipeline_prompts=input_M0_prompts
        )

        # final M0 prompt
        M0_prompt = pipeline_prompt.format(
                product= product,
                country= country,
                M0_json_sample = M0_json_sample
                )

        print(M0_prompt)
        M0_result = model.predict(
        text=M0_prompt)

        print(M0_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\""))
        jsoned = json.loads(M0_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\""))
        print('json result of title')
        print(jsoned)
        M0_result = jsoned['project_title']

        print(M0_result)
    M0_time = time.time()

    if 'M1_1' in needs:
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

        M1_1_result = model.predict(
            text=M1_1_prompt)

    M1_1_time = time.time()
    if 'M1_2' in needs:
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

        M1_2_result = model.predict(
            text=M1_2_prompt)

    M1_2_time = time.time()
    if 'M1_3' in needs:
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
    M1_3_time = time.time()
    if 'M1_4' in needs:
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
    M1_4_time = time.time()
    if 'M1_5' in needs:
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
    M1_5_time = time.time()
    if 'M1_6' in needs:
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
    M1_6_time = time.time()
    if 'M1_1_R' in needs:
        # 參數定義
        pestel = "政治"
        pestel_whole = "政治面向（ 政治 Political ）"
        pestel_sub = subq
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
            'pestel_sub':'{subq}',
            'content':'分析內文 字數約為80-90個字'
            },
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
    M1_1_R_time = time.time()
    if 'M1_2_R' in needs:
        # 參數定義
        pestel = "經濟"
        pestel_whole = "經濟面向（ 經濟 Economic ）"
        pestel_sub = subq
        objective_item_M1_2 = """
        - 目前整體的經濟狀況與消費力道的上升/下降狀況，並說明原因
        - 當去匯率狀態及未來可能的改變
        - 失業率的上升/下降狀況及對公司人才招募的影響
        - 未來原物料價格的走勢及對公司利潤的影響
        """
        M1_2_json_sample = """
        {
        'pestel':'經濟',
        'pestel_info':[
            {
            'pestel_sub':'{subq}',
            'content':'分析內文 字數約為80-90個字'
            },
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
    M1_2_R_time = time.time()
    if 'M1_3_R' in needs:
        # 參數定義
        pestel = "社會"
        pestel_whole = "社會面向（ 社會 Social ）"
        pestel_sub = subq
        objective_item_M1_3 = """
        - 出生人口的上升/下降狀況
        - 消費者關注的文化與教育的議題
        - 消費者近期喜歡的東西，生活習慣的改變以及消費頻率的上升/下降情形
        - 可能會提高公司業績的社會的因素
        """
        M1_3_json_sample = """
        {
        'pestel':'社會',
        'pestel_info':[
            {
            'pestel_sub':'{subq}',
            'content':'分析內文 字數約為80-90個字'
            },
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
    M1_3_R_time = time.time()
    if 'M1_4_R' in needs:
        # 參數定義
        pestel = "技術"
        pestel_whole = "技術面向（ 技術 Technological ）"
        pestel_sub = subq
        objective_item_M1_4 = """
        - 近期發表且可以拿來應用的新科技
        - 能夠推動公司成長數位技術
        - 近期政府和研究單位關注的科技議題
        """
        M1_4_json_sample = """
        {
        'pestel':'技術',
        'pestel_info':[
            {
            'pestel_sub':'{subq}',
            'content':'分析內文 字數約為80-90個字'
            },
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
    M1_4_R_time = time.time()
    if 'M1_5_R' in needs:
        # 參數定義
        pestel = "環境"
        pestel_whole = "環境面向（ 環境 Eviromental ）"
        pestel_sub = subq
        objective_item_M1_5 = """
        """
        M1_5_json_sample = """
        {
        'pestel':'環境',
        'pestel_info':[
            {
            'pestel_sub':'{subq}',
            'content':'分析內文 字數約為80-90個字'
            },
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
    M1_5_R_time = time.time()
    if 'M1_6_R' in needs:
        # 參數定義
        pestel = "法規"
        pestel_whole = "法規（ 法規 Legal ）"
        pestel_sub = ["政治因素","經濟因素","社會文化因素"]
        objective_item_M1_6 = """
        """
        M1_6_json_sample = """
        {
        'pestel':'法規',
        'pestel_info':[
            {
            'pestel_sub':'{subq}',
            'content':'分析內文 字數約為80-90個字'
            },
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
    M1_6_R_time = time.time()
    if 'M2_1' in needs:
        # 參數定義
        M2_1_competitive_info = """
        | key           | #zh-TW           | definition                                                   |
        | ------------- | ---------------- | ------------------------------------------------------------ |
        | brand_name    | 競品品牌名稱     | 「5個」""真實存在的"" 競品品牌，國內或國外的實際品牌都可以        |
        | strengths     | 優勢             | 提供“3點條列”，每點約30字，主動語態、專業、明確的描述        |
        | weaknesses    | 劣勢             | 提供“3點條列”，每點約30字，主動語態、專業、明確的描述        |
        | features      | 產品共同特點描述 | 根據品牌特色、客群屬性，提供專業且具體的一句描述，30字左右   |
        | price_banding | 價格區間         | 根據品牌定位、市場定位以及目標客群，用[台灣]貨幣符號表示，貨幣符號請保持為簡寫     |
        | audiences     | 主要客群描述     | 根據該品牌的產品特色及定價策略，推估可能的目標使用族群 TA，30字左右精準描述 |
        | target_size   | 主要客群規模     | 根據目標客群，用1-5表示客群規模大小，1-Niche Audience、5-Mass Market |
        """
        M2_1_json_sample = """
        {
        "area": "台灣",
        "product": "膠囊咖啡機",
        "id": "2-1",

        "competitive": [
            {"price_symbol": "(NT$)"},
            {
            "brand_name": "Nespresso",
            "strengths": ["提供高品質的咖啡豆和多樣口味。", "獨特的膠囊設計，方便且衛生。", "強大的品牌知名度和廣告宣傳。"],
            "weaknesses": ["膠囊咖啡價格較高。", "限制使用特定品牌的膠囊。", "產品線較為單一，缺乏多樣性。"],
            "price_banding_min": "6000",
            "price_banding_max": "8000",
            "features": "提供便利且高品質的膠囊咖啡體驗。",
            "audiences": "追求品質、注重品牌的中高收入族群。",
            "target_size": 3
            },
            {
            "brand_name": "Breville",
            "strengths": ["多功能性，可煮不同風味的咖啡。", "提供精密的溫度和壓力控制。", "高品質的產品設計和耐用性。"],
            "weaknesses": ["價格較高，不適合預算較低的消費者。", "體積較大，不適合空間較小的家庭。", "需要較長的學習曲線來掌握操作。"],
            "price_banding_min": "10000",
            "price_banding_max": "12000",
            "features": "提供多功能性和精密控制的咖啡機。",
            "audiences": "追求咖啡品質和操作控制的咖啡愛好者。",
            "target_size": 2
            }
        ]
        }
        """

        # M2-1 full costar template
        full_M2_1_template = """
        {context}
        {objective}
        {style}
        {audience}
        {responce}
        """
        full_M2_1_prompt = PromptTemplate.from_template(full_M2_1_template)

        # M2-1 "contex" template
        context_M2_1_template = """
        # Context
        你是一位專業的產業分析顧問，掌握全球產業重要發展動態，擅長從產業市場規模、市場區隔、使用者輪廓、新產品規劃、創新策略、社會文化等面向進行全方位的產業競爭分析。
        我正在進行「競品分析」，請協助我探勘產業趨勢情報、洞見使用者需求、產業技術優勢與創新需求脈絡，貼近產業立場，提供客戶全方位建議。
        """
        context_M2_1_prompt = PromptTemplate.from_template(context_M2_1_template)

        # M2-1 "objective" template
        objective_M2_1_template = """
        # Objective:
        你的任務是分析{country}地區的{product}，給出「五個」競品品牌，針對每一個品牌，請根據以下 <competitive-info> 定義以及 <example> json 格式，詳細列出內容。
        <competitive-info>
        {M2_1_competitive_info}
        </competitive-info>
        - 回答中的品牌必須是 ""實際存在的品牌""，絕對不接受模擬品牌如'品牌A'、'品牌B'、'品牌X'、'品牌1'等。
        - 內容必須基於真實且正確的資料，不要編撰或創造內容(Do not make information up)。
        """
        objective_M2_1_prompt = PromptTemplate.from_template(objective_M2_1_template)

        # M2-1 "style" template
        style_M2_1_template = """
        # Style:
        - 用字精準無贅字、無拼寫錯誤、條理清晰易讀。
        - 寫作風格如大型研究機構 nngroup, frog design, BCG、新聞網站 Axios，撰寫 distill, descriptive, clear, straightforward 的分析摘要。


        """
        style_M2_1_prompt = PromptTemplate.from_template(style_M2_1_template)

        # M2-1 "audience & tone" template
        audience_M2_1_template = """
        # Audience & Tone:
        - 目標讀者是公司內部的Product Designer、Researcher、Project Manager，他們需要蒐集市場資料分析趨勢、研究產品資料分析競品、撰寫設計提案、與團隊協作與溝通。
        - 讀者體期待瞭解市場現況，以有效將競品分析轉化成具體可行動的產品策略。因此過程中需保持清晰和有條理的語氣，提供有價值的見解和建議。
        """
        audience_M2_1_prompt = PromptTemplate.from_template(audience_M2_1_template)

        # M2-1 "responce" template
        responce_M2_1_template = """
        #Responce:
        - #zh-TW繁體中文進行回答。
        - 每一分析面向的條列點應「具有獨特性」，避免內容的重複或過於相似。
        - 必須將輸出結構化為一組json格式。定義見上方 <competitive-info>，你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與<example>範例匹配！

        <example>
        {M2_1_json_sample}
        </example>

        #Start:
        根據上述資訊，針對{country}地區的{product}產業進行深入的「競品分析」，使用 <example> json 格式，列出5個 ""真實存在的"" {product}競品品牌，分析完就停止，不要給我結論。
        如果您明白了，請開始執行
        """
        responce_M2_1_prompt = PromptTemplate.from_template(responce_M2_1_template)

        # M2-1 arragement
        input_M2_1_prompts = [
            ("context", context_M2_1_prompt),
            ("objective", objective_M2_1_prompt),
            ("style", style_M2_1_prompt),
            ("audience", audience_M2_1_prompt),
            ("responce", responce_M2_1_prompt),
        ]
        pipeline_prompt = PipelinePromptTemplate(
            final_prompt=full_M2_1_prompt, pipeline_prompts=input_M2_1_prompts
        )

        # final M2-1 prompt
        M2_1_prompt = pipeline_prompt.format(
                product= product,
                country= country,
                M2_1_competitive_info = M2_1_competitive_info,
                M2_1_json_sample = M2_1_json_sample
                )

        print(M2_1_prompt)

        M2_1_result = model.predict(
            text=M2_1_prompt)

        print(M2_1_result)
    M2_1_time = time.time()
    if 'M2_1_R' in needs:
        print('into M2_1_r')
        print(subq)
        # 參數定義
        M2_1_2_json_sample = """
        {
        "area": "台灣",
        "product": "膠囊咖啡機",
        "competitive":
            {
                "brand_name": "Nespresso",
                "strengths":["提供高品質的咖啡豆和多樣口味。","獨特的胶囊設計，方便且衛生。","強大的品牌知名度和廣告宣傳。"],
                "weaknesses":["胶囊咖啡價格較高。","限制使用特定品牌的胶囊。","產品線較為單一，缺乏多樣性。"],
                "features": 提供便利且高品質的胶囊咖啡體驗。",
                "price_banding_min": "6000",
                "price_banding_max": "8000",
                "audiences": "追求品質、注重品牌的中高收入族群。",
                "target_size": 3
            }
        }
        """

        # M2-1-2 full costar template
        full_M2_1_2_template = """
        {context}

        {objective}

        {style}

        {tone}

        {audience}

        {responce}
        """
        full_M2_1_2_prompt = PromptTemplate.from_template(full_M2_1_2_template)

        # M2-1-2 "contex" template
        context_M2_1_2_template = """
        # Context
        你是一位專業的產業管理顧問。掌握全球產業重要發展動態，從產業市場、地區經濟、地方基礎建設、國家政策與社會文化等面向進行全面的分析，協助客戶探勘產業趨勢情報、洞見技術與創新需求脈絡，貼近產業立場，提供客戶全方位建言。
        """
        context_M2_1_2_prompt = PromptTemplate.from_template(context_M2_1_2_template)

        # M2-1-2 "objective" template
        objective_M2_1_2_template = """
        # Objective:
        你的任務是針對{country}地區的{product}的競品品牌{subq}進行分析：

        競品分析需詳細列出內容：品牌名稱、優勢、劣勢、產品特點、價格區間、主要客群、主要客群規模。

        - 優勢和劣勢各面向分別提供“3點條列”，每點約30字，並以主動語態、專業、明確的描述方式呈現。
        - 產品特點需考慮品牌特色、客群屬性，提供專業且具體的一句描述，30字左右。
        - 價格應考慮品牌定位、市場定位以及目標客群，且需用當地貨幣符號表示。
        - 依據主要客群描述，主要客群規模用1-5來表示大小。
        """
        objective_M2_1_2_prompt = PromptTemplate.from_template(objective_M2_1_2_template)

        # M2-1-2 "style" template
        style_M2_1_2_template = """
        # Style:
        - 文體應專業且具描述性(descriptive)，用字精準無贅字、無拼寫錯誤、條理清晰易讀。
        - 內容必須基於真實且正確的資料，不要編撰或創造內容(Do not make information up)。
        - 以#zh-TW繁體中文進行回答，答案中提及的品牌可以是國內或國外的實際品牌，並將貨幣符號保持為簡寫。
        - 確保每一條列的字數在其分析結果中保持一致性，不應有超過30字的差異。
        - 必須將輸出結構化為一組json格式。json是一種聲明性語言，可讓你對文件進行註解和確認。你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！
        範例如下：

        {M2_1_2_json_sample}
        """
        style_M2_1_2_prompt = PromptTemplate.from_template(style_M2_1_2_template)

        # M2-1-2 "tone" template
        tone_M2_1_2_template = """
        # Tone:
        在整個過程中保持清晰和有條理的語氣，說明產競品分析。具備清楚且有方向的分析，提供有價值的見解和建議。
        """
        tone_M2_1_2_prompt = PromptTemplate.from_template(tone_M2_1_2_template)

        # M2-1-2 "audience" template
        audience_M2_1_2_template = """
        # Audience:
        目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
        """
        audience_M2_1_2_prompt = PromptTemplate.from_template(audience_M2_1_2_template)

        # M2-1-2 "responce" template
        responce_M2_1_2_template = """
        #Responce:
        根據上述資訊，針對{country}地區的{product}產業進行深入的「競品分析」，並列出競品品牌{subq}的單一結果。分析完就停止，不須產生結論。

        #Start:
        如果您明白了，請開始進行分析
        """
        responce_M2_1_2_prompt = PromptTemplate.from_template(responce_M2_1_2_template)

        # M2-1-2 arragement
        input_M2_1_2_prompts = [
            ("context", context_M2_1_2_prompt),
            ("objective", objective_M2_1_2_prompt),
            ("style", style_M2_1_2_prompt),
            ("tone", tone_M2_1_2_prompt),
            ("audience", audience_M2_1_2_prompt),
            ("responce", responce_M2_1_2_prompt),
        ]
        pipeline_prompt = PipelinePromptTemplate(
            final_prompt=full_M2_1_2_prompt, pipeline_prompts=input_M2_1_2_prompts
        )

        # final M2-1-2 prompt
        M2_1_2_prompt = pipeline_prompt.format(
                product= product,
                country= country,
                subq= subq,
                M2_1_2_json_sample = M2_1_2_json_sample
                )

        print(M2_1_2_prompt)

        M2_1_result = model.predict(
            text=M2_1_2_prompt)

        print(M2_1_result)
    M2_1_R_time = time.time()
    if 'M2_2' in needs:
        # 參數定義
        M2_2_strategy_info = """
        | key                    | #zh-TW       | definition                                                   |
        | ---------------------- | ------------ | ------------------------------------------------------------ |
        | opportunities_headline | 設計機會標題 | 標題清晰直指重點 smarter, concrete on what matters.<br />**Be specific:** Don’t use flat, vague language when you have the space to be precise.<br />**Be brief:** Cap headlines at 10 words or 60 characters. 中文字在10個字以內。It’s highly memorable and about the maximum length someone can read and immediately repeat.<br />**Be bold**: Authority instills trust. If something is true, say it confidently.<br />**Be honest**: At times, longer is better. Smart Brevity should maximize the value of every word, while giving readers only what they need to see the big picture. Never slip into hyperbole or clickbait. Readers see through it, and it fractures long-term trust. |
        | opportunities          | 設計機會     | 總整<competitive-analysis>裡面的 strengths, features, price_banding, audiences，將其 broken down and narrated with true expertise that tells our audience why it matters right away. 從 {country} 地區的 {product} 產業的市場缺口，整理出有深刻洞見、意想不到、經過縝密分析與判斷後的機會點，幫助團隊精準決策，summarized in shareable elements. strive to sort through all of the noise to bring you substantive and meaningful content that is truly worthy of your time。內容需超過150字。 |
        | challenges_headline    | 設計挑戰標題 | 同 opportunities_headline 規則                               |
        | challenges             | 設計挑戰     | 總整<competitive-analysis>裡面的 weaknesses, features, price_banding, audiences，從 {country} 地區的 {product} 產業的市場競爭狀況、產業生態系、使用族群等限制因素，整理如果要切入該市場，團隊將會遇到的具體挑戰。內容需超過150字，其餘規則同 opportunities |
        """

        M2_2_json_sample = """
        {
        "area": "{country}",
        "product": "{product}",
        "id": "2-3",

        "opportunities-and-challenges": [
            {
            "opportunities_headline": "摺疊自行車的市場機遇",
            "opportunities": "台灣都市空間有限，人口密度高，尤其是大城市如台北、高雄等地，需求短程代步工具的市民眾多。摺疊自行車便利且易於攜帶，更能解決儲存問題，符合都市人快速、方便的生活需求。此外，便宜實惠的摺疊自行車如Dahon，對於預算有限的消費者來說，更具吸引力。",

            "challenges_headline": "摺疊自行車市場的挑戰",
            "challenges": "雖然摺疊自行車在都市市場有一定的需求，然而其設計和性能可能無法滿足專業車手的需求。在售後服務方面，由於在台灣的售後服務點較少，這可能會影響消費者的購買意願。此外，部分型號需預訂，等待時間長，也可能讓消費者猶豫不決。"
            },
            {
            "opportunities_headline": "高品質自行車的市場機遇",
            "opportunities": "對於專業車手和對自行車有高要求的消費者，品質和設計都是他們在選擇自行車時的重要考量。像TREK和Specialized等品牌的產品，以其專業和高品質的自行車，吸引了這部分消費者的青睞。",

            "challenges_headline": "高品質自行車市場的挑戰",
            "challenges": "專業和高品質的自行車價格較高，可能使預算有限的消費者望而卻步。此外，專業自行車的售後服務也是一項挑戰，特別是在台灣的售後服務點較少，可能會影響消費者的購買意願。"
            },
            {
            "opportunities_headline": "全方位自行車的市場機遇",
            "opportunities": "GIANT和MERIDA品牌提供全方位的自行車選擇，無論消費者需求如何，都能在產品線中找到合適的自行車。另外，這兩個品牌在台灣的網路與實體銷售點分佈廣泛，售後服務好，能與消費者建立深度連結，提升品牌忠誠度。",

            "challenges_headline": "全方位自行車市場的挑戰",
            "challenges": "GIANT和MERIDA品牌雖然擁有全面的產品線和廣泛的銷售網絡，但其價格相對較高，對於預算有限的消費者來說，可能會產生購買壓力。此外，部分型號的供應不足，消費者需等待，也可能影響其銷售。"
            }
        ]
        }
        """

        # M2-2 full costar template
        full_M2_2_template = """
        {context}
        {objective}
        {style}
        {audience}
        {responce}
        """
        full_M2_2_prompt = PromptTemplate.from_template(full_M2_2_template)

        # M2-2 "contex" template
        context_M2_2_template = """
        # Context
        你是一位專業的產業分析顧問，掌握全球產業重要發展動態，擅長從產業市場規模、市場區隔、使用者輪廓、新產品規劃、創新策略、社會文化等面向進行全方位的產業競爭分析。
        我正在從競品分析梳理出「設計機會及挑戰」，請協助我探勘產業趨勢情報、洞見使用者需求、產業技術優勢與創新需求脈絡，貼近產業立場，提供客戶全方位建議。
        """
        context_M2_2_prompt = PromptTemplate.from_template(context_M2_2_template)

        # M2-2 "objective" template
        objective_M2_2_template = """
        # Objective:
        你的任務是根據{country}地區的{product}分析出的 <competitive-analysis>, 統整當前產業的 3組「設計機會和挑戰」，一個機會對應一個挑戰。
        請根據以下 <strategy-info> 定義以及 <example> json 格式，讓我清楚掌握競爭環境，容易辨識自己所處的產業，或想切入的目標市場，會對應到哪些機會和潛在風險。

        <strategy-info>
        {M2_2_strategy_info}
        </strategy-info>
        以上內容必須基於真實且正確的資料，不要編撰或創造內容(Do not make information up)。

        <competitive-analysis>
        {M2_1_result}
        </competitive-analysis>
        """
        objective_M2_2_prompt = PromptTemplate.from_template(objective_M2_2_template)

        # M2-2 "style" template
        style_M2_2_template = """
        # Style:
        - 用字精準無贅字、無拼寫錯誤、條理清晰易讀。
        - 寫作風格如大型研究機構 nngroup, frog design, BCG、新聞網站 Axios，撰寫 distill, descriptive, clear, straightforward 的分析摘要。
        """
        style_M2_2_prompt = PromptTemplate.from_template(style_M2_2_template)

        # M2-2 "audience & tone" template
        audience_M2_2_template = """
        # Audience & Tone:
        - 目標讀者是公司內部的Product Designer、Researcher、Project Manager，他們需要蒐集市場資料分析趨勢、研究產品資料分析競品、撰寫設計提案、與團隊協作與溝通。
        - 讀者體期待瞭解市場現況，以有效將競品分析轉化成具體可行動的產品策略。因此過程中需保持清晰和有條理的語氣，提供有價值的見解和建議。
        """
        audience_M2_2_prompt = PromptTemplate.from_template(audience_M2_2_template)

        # M2-2 "responce" template
        responce_M2_2_template = """
        Responce:
        - #zh-TW繁體中文進行回答。
        - 每一分析面向的內容應「具有獨特性」，避免內容的重複或過於相似。
        - 必須將輸出結構化為一組json格式。定義見上方 <strategy-info>，你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與<example>範例匹配！
        <example>
        {M2_2_json_sample}
        </example>

        #Start:
        根據上述資訊，針對{country}地區的{product}產業梳理出「設計機會及挑戰」，使用 <example> json 格式，列出3組「設計機會和挑戰」，一個機會對應一個挑戰。分析完就停止，不要給我結論。
        如果您明白了，請開始執行
        """
        responce_M2_2_prompt = PromptTemplate.from_template(responce_M2_2_template)

        # M2-2 arragement
        input_M2_2_prompts = [
            ("context", context_M2_2_prompt),
            ("objective", objective_M2_2_prompt),
            ("style", style_M2_2_prompt),
            ("audience", audience_M2_2_prompt),
            ("responce", responce_M2_2_prompt),
        ]
        pipeline_prompt = PipelinePromptTemplate(
            final_prompt=full_M2_2_prompt, pipeline_prompts=input_M2_2_prompts
        )

        # final M2-2 prompt
        M2_2_prompt = pipeline_prompt.format(
                product= product,
                country= country,
                M2_2_strategy_info = M2_2_strategy_info,
                M2_1_result = M2_1_result,
                M2_2_json_sample = M2_2_json_sample
                )

        print(M2_2_prompt)

        M2_2_result = model.predict(
            text=M2_2_prompt)

        print(M2_2_result)
    M2_2_time = time.time()
    if 'M2_3_a1' in needs:
        # 參數定義
        M2_3_a1_json_sample = """
        {
        "area": "{country}",
        "product": "{product}",
        "id": "2-4",

        "product-innovation": [
            {
            "headline": "方案1標題，標題清晰直指重點",
            "star": "⭐⭐⭐⭐",
            "ideas": "方案1敘述，請說明挑選此方案的具體原因，以及如何幫助團隊制定產品設計決策、發掘創新機會"
            },
            {
            "headline": "方案2標題",
            "star": "⭐⭐⭐⭐⭐",
            "ideas": "方案2敘述"
            {
            "headline": "方案3標題",
            "star": "⭐⭐⭐",
            "ideas": "方案3敘述"
            }
        ]
        }
        """

        # M2-3-a1 full costar template
        full_M2_3_a1_template = """
        {context}
        {objective}
        {style}
        {audience}
        {responce}
        """
        full_M2_3_a1_prompt = PromptTemplate.from_template(full_M2_3_a1_template)

        # M2-3-a1 "context" template
        context_M2_3_a1_template = """
        # Context
        你是一位專業的產業分析顧問，掌握全球產業重要發展動態，擅長從產業市場規模、市場區隔、使用者輪廓、新產品規劃、創新策略、社會文化等面向進行全方位的產業競爭分析。
        我正在從競品分析、市場的機會及挑戰，尋找「創新機會」，請協助我探勘產業趨勢、洞見使用者需求、評估產業投入機會，提供客戶具體可執行的創新提案。
        """
        context_M2_3_a1_prompt = PromptTemplate.from_template(context_M2_3_a1_template)

        # M2-3-a1 "objective" template
        objective_M2_3_a1_template = """
        # Objective:
        你的任務是根據{country}地區的{product}，提出不同創新機會方案，所有方案會分成四個類型：產品創新 product innovation、服務創新 service innovation、商業模式 business model、環境建設 infrastructure development
        現在要請你產出「產品創新」，請以「產品創新 product innovation」面向提供「3個」具體的切入方案。
            - 每個方案標題吸引人具前瞻性
            - 每個方案給予一個評分（以⭐表示，範圍從0到5）
            - 方案描述為具深刻洞見的評論，並有一定程度的合理性，每一個方案描述不少於300字且為結構清晰的段落。
        """
        objective_M2_3_a1_prompt = PromptTemplate.from_template(objective_M2_3_a1_template)

        # M2-3-a1 "style" template
        style_M2_3_a1_template = """
        # Style:
        - 用字精準無贅字、無拼寫錯誤、條理清晰易讀。
        - 寫作風格如大型研究機構 nngroup, frog design, BCG、新聞網站 Axios，撰寫 distill, descriptive, clear, straightforward 的分析摘要。
        """
        style_M2_3_a1_prompt = PromptTemplate.from_template(style_M2_3_a1_template)

        # M2-3-a1 "audience & tone" template
        audience_M2_3_a1_template = """
        # Audience & Tone:
        - 目標讀者是公司內部的Product Designer、Researcher、Project Manager，他們需要蒐集市場資料分析趨勢、研究產品資料分析競品、撰寫設計提案、與團隊協作與溝通。
        - 讀者體期待瞭解市場現況，以有效將競品分析轉化成具體可行動的產品策略。因此過程中需保持清晰和有條理的語氣，提供有價值的見解和建議。
        """
        audience_M2_3_a1_prompt = PromptTemplate.from_template(audience_M2_3_a1_template)

        # M2-3-a1 "responce" template
        responce_M2_3_a1_template = """
        #Responce:
        - #zh-TW繁體中文進行回答。
        - 每一分析面向的內容應「具有獨特性」，避免內容的重複或過於相似。
        - 必須將輸出結構化為一組json格式。你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與<example>範例匹配！

        <example>
        {M2_3_a1_json_sample}
        </example>

        根據上述資訊，針對{country}地區的{product}產業，以「產品創新」為模組名稱，統整並詳述當前產業的設計「產品創新」機會。

        #Start:
        如果您明白了，請開始進行分析。分析完就停止，不要給我結論。
        """
        responce_M2_3_a1_prompt = PromptTemplate.from_template(responce_M2_3_a1_template)

        # M2-3-a1 arragement
        input_M2_3_a1_prompts = [
            ("context", context_M2_3_a1_prompt),
            ("objective", objective_M2_3_a1_prompt),
            ("style", style_M2_3_a1_prompt),
            ("audience", audience_M2_3_a1_prompt),
            ("responce", responce_M2_3_a1_prompt),
        ]
        pipeline_prompt = PipelinePromptTemplate(
            final_prompt=full_M2_3_a1_prompt, pipeline_prompts=input_M2_3_a1_prompts
        )

        # final M2-3-a1 prompt
        M2_3_a1_prompt = pipeline_prompt.format(
                product= product,
                country= country,
                M2_1_result = M2_1_result,
                M2_2_result = M2_2_result,
                M2_3_a1_json_sample = M2_3_a1_json_sample
                )

        print(M2_3_a1_prompt)

        M2_3_a1_result = model.predict(
            text=M2_3_a1_prompt)

        print(M2_3_a1_result)
    M2_3_a1_time = time.time()
    if 'M2_3_a2' in needs:
        # 參數定義
        M2_3_a2_json_sample = """
        {
        "area": "{country}",
        "product": "{product}",
        "id": "2-4",

        "service-innovation": [
            {
            "headline": "方案1標題，標題清晰直指重點",
            "star": "⭐⭐⭐⭐",
            "ideas": "方案1敘述，請說明挑選此方案的具體原因，以及如何幫助團隊制定產品設計決策、發掘創新機會"
            },
            {
            "headline": "方案2標題",
            "star": "⭐⭐⭐⭐⭐",
            "ideas": "方案2敘述"
            {
            "headline": "方案3標題",
            "star": "⭐⭐⭐",
            "ideas": "方案3敘述"
            }
        ]
        }
        """

        # M2-3-a2 full costar template
        full_M2_3_a2_template = """
        {context}

        {objective}

        {style}

        {tone}

        {audience}

        {responce}
        """
        full_M2_3_a2_prompt = PromptTemplate.from_template(full_M2_3_a2_template)

        # M2-3-a2 "context" template
        context_M2_3_a2_template = """
        # Context
        你是一位專業的產業管理顧問。掌握全球產業重要發展動態，從產業市場、地區經濟、地方基礎建設、國家政策與社會文化等面向進行全面的分析，協助客戶探勘產業趨勢情報、洞見技術與創新需求脈絡，貼近產業立場，提供客戶全方位建言。
        """
        context_M2_3_a2_prompt = PromptTemplate.from_template(context_M2_3_a2_template)

        # M2-3-a2 "objective" template
        objective_M2_3_a2_template = """
        # Objective:
        未來創新機會方案：綜合上方提供之「競品分析」及「設計機會與挑戰」，分析以下四個面向：產品創新、服務創新、商業模式、環境建設，請以「服務創新」面向提供「3個」具體的切入方案。
            - 每個方案標題吸引人具前瞻性
            - 每個方案給予一個評分（以⭐表示，範圍從0到5）
            - 方案描述為具深刻洞見的評論，並有一定程度的合理性，每一個方案描述不少於300字且為結構清晰的段落。
        """
        objective_M2_3_a2_prompt = PromptTemplate.from_template(objective_M2_3_a2_template)

        # M2-3-a2 "style" template
        style_M2_3_a2_template = """
        # Style:

        - 文體應專業且具描述性(descriptive)，用字精準無贅字、無拼寫錯誤、條理清晰易讀。
        - 內容必須基於真實且正確的資料，不要編撰或創造內容(Do not make information up)。
        - 以#zh-TW繁體中文進行回答，答案中提及的品牌可以是國內或國外的實際品牌，並將貨幣符號保持為簡寫。
        - 確保每一條列的字數在其分析結果中保持一致性，不應有超過30字的差異。
        - 每一分析面向的條列點應「具有獨特性」，「避免內容的重複或過於相似」。
        - 必須將輸出結構化為一組json格式。json是一種聲明性語言，可讓你對文件進行註解和確認。你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！
        範例如下：

        {M2_3_a2_json_sample}
        """
        style_M2_3_a2_prompt = PromptTemplate.from_template(style_M2_3_a2_template)

        # M2-3-a2 "tone" template
        tone_M2_3_a2_template = """
        # Tone:
        在整個過程中保持清晰和有條理的語氣，說明「服務創新」機會。具備清楚且有方向的分析，提供有價值的見解和建議。
        """
        tone_M2_3_a2_prompt = PromptTemplate.from_template(tone_M2_3_a2_template)

        # M2-3-a2 "audience" template
        audience_M2_3_a2_template = """
        # Audience:
        目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
        """
        audience_M2_3_a2_prompt = PromptTemplate.from_template(audience_M2_3_a2_template)

        # M2-3-a2 "responce" template
        responce_M2_3_a2_template = """
        #Responce:
        根據上述資訊，針對{country}地區的{product}產業，以「服務創新」為模組名稱，統整並詳述當前產業的設計「服務創新」機會。

        #Start:
        如果您明白了，請開始進行分析
        """
        responce_M2_3_a2_prompt = PromptTemplate.from_template(responce_M2_3_a2_template)

        # M2-3-a2 arragement
        input_M2_3_a2_prompts = [
            ("context", context_M2_3_a2_prompt),
            ("objective", objective_M2_3_a2_prompt),
            ("style", style_M2_3_a2_prompt),
            ("tone", tone_M2_3_a2_prompt),
            ("audience", audience_M2_3_a2_prompt),
            ("responce", responce_M2_3_a2_prompt),
        ]
        pipeline_prompt = PipelinePromptTemplate(
            final_prompt=full_M2_3_a2_prompt, pipeline_prompts=input_M2_3_a2_prompts
        )

        # final M2-3-a2 prompt
        M2_3_a2_prompt = pipeline_prompt.format(
                product= product,
                country= country,
                M2_1_result = M2_1_result,
                M2_2_result = M2_2_result,
                M2_3_a2_json_sample = M2_3_a2_json_sample
                )

        print(M2_3_a2_prompt)

        M2_3_a2_result = model.predict(
            text=M2_3_a2_prompt)

        print(M2_3_a2_result)
    M2_3_a2_time = time.time()
    if 'M2_3_a3' in needs:
        # 參數定義
        M2_3_a3_json_sample = """
        {
        "area": "{country}",
        "product": "{product}",
        "id": "2-4",

        "business-model": [
            {
            "headline": "方案1標題，標題清晰直指重點",
            "star": "⭐⭐⭐⭐",
            "ideas": "方案1敘述，請說明挑選此方案的具體原因，以及如何幫助團隊制定產品設計決策、發掘創新機會"
            },
            {
            "headline": "方案2標題",
            "star": "⭐⭐⭐⭐⭐",
            "ideas": "方案2敘述"
            {
            "headline": "方案3標題",
            "star": "⭐⭐⭐",
            "ideas": "方案3敘述"
            }
        ]
        }
        """

        # M2-3-a3 full costar template
        full_M2_3_a3_template = """
        {context}

        {objective}

        {style}

        {tone}

        {audience}

        {responce}
        """
        full_M2_3_a3_prompt = PromptTemplate.from_template(full_M2_3_a3_template)

        # M2-3-a3 "context" template
        context_M2_3_a3_template = """
        # Context
        你是一位專業的產業管理顧問。掌握全球產業重要發展動態，從產業市場、地區經濟、地方基礎建設、國家政策與社會文化等面向進行全面的分析，協助客戶探勘產業趨勢情報、洞見技術與創新需求脈絡，貼近產業立場，提供客戶全方位建言。
        """
        context_M2_3_a3_prompt = PromptTemplate.from_template(context_M2_3_a3_template)

        # M2-3-a3 "objective" template
        objective_M2_3_a3_template = """
        # Objective:
        未來創新機會方案：綜合上方提供之「競品分析」及「設計機會與挑戰」，分析以下四個面向：產品創新、服務創新、商業模式、環境建設，請以「商業模式」面向提供「3個」具體的切入方案。
            - 每個方案標題吸引人具前瞻性
            - 每個方案給予一個評分（以⭐表示，範圍從0到5）
            - 方案描述為具深刻洞見的評論，並有一定程度的合理性，每一個方案描述不少於300字且為結構清晰的段落。
        """
        objective_M2_3_a3_prompt = PromptTemplate.from_template(objective_M2_3_a3_template)

        # M2-3-a3 "style" template
        style_M2_3_a3_template = """
        # Style:

        - 文體應專業且具描述性(descriptive)，用字精準無贅字、無拼寫錯誤、條理清晰易讀。
        - 內容必須基於真實且正確的資料，不要編撰或創造內容(Do not make information up)。
        - 以#zh-TW繁體中文進行回答，答案中提及的品牌可以是國內或國外的實際品牌，並將貨幣符號保持為簡寫。
        - 確保每一條列的字數在其分析結果中保持一致性，不應有超過30字的差異。
        - 每一分析面向的條列點應「具有獨特性」，「避免內容的重複或過於相似」。
        - 必須將輸出結構化為一組json格式。json是一種聲明性語言，可讓你對文件進行註解和確認。你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！
        範例如下：

        {M2_3_a3_json_sample}
        """
        style_M2_3_a3_prompt = PromptTemplate.from_template(style_M2_3_a3_template)

        # M2-3-a3 "tone" template
        tone_M2_3_a3_template = """
        # Tone:
        在整個過程中保持清晰和有條理的語氣，說明「商業模式」機會。具備清楚且有方向的分析，提供有價值的見解和建議。
        """
        tone_M2_3_a3_prompt = PromptTemplate.from_template(tone_M2_3_a3_template)

        # M2-3-a3 "audience" template
        audience_M2_3_a3_template = """
        # Audience:
        目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
        """
        audience_M2_3_a3_prompt = PromptTemplate.from_template(audience_M2_3_a3_template)

        # M2-3-a3 "responce" template
        responce_M2_3_a3_template = """
        #Responce:
        根據上述資訊，針對{country}地區的{product}產業，以「商業模式」為模組名稱，統整並詳述當前產業的設計「商業模式」機會。

        #Start:
        如果您明白了，請開始進行分析
        """
        responce_M2_3_a3_prompt = PromptTemplate.from_template(responce_M2_3_a3_template)

        # M2-3-a3 arragement
        input_M2_3_a3_prompts = [
            ("context", context_M2_3_a3_prompt),
            ("objective", objective_M2_3_a3_prompt),
            ("style", style_M2_3_a3_prompt),
            ("tone", tone_M2_3_a3_prompt),
            ("audience", audience_M2_3_a3_prompt),
            ("responce", responce_M2_3_a3_prompt),
        ]
        pipeline_prompt = PipelinePromptTemplate(
            final_prompt=full_M2_3_a3_prompt, pipeline_prompts=input_M2_3_a3_prompts
        )

        # final M2-3-a3 prompt
        M2_3_a3_prompt = pipeline_prompt.format(
                product= product,
                country= country,
                M2_1_result = M2_1_result,
                M2_2_result = M2_2_result,
                M2_3_a3_json_sample = M2_3_a3_json_sample
                )

        print(M2_3_a3_prompt)

        M2_3_a3_result = model.predict(
            text=M2_3_a3_prompt)

        print(M2_3_a3_result)
    M2_3_a3_time = time.time()
    if 'M2_3_a4' in needs:
        # 參數定義
        M2_3_a4_json_sample = """
        {
        "area": "{country}",
        "product": "{product}",
        "id": "2-4",

        "infra-develop": [
            {
            "headline": "方案1標題，標題清晰直指重點",
            "star": "⭐⭐⭐⭐",
            "ideas": "方案1敘述，請說明挑選此方案的具體原因，以及如何幫助團隊制定產品設計決策、發掘創新機會"
            },
            {
            "headline": "方案2標題",
            "star": "⭐⭐⭐⭐⭐",
            "ideas": "方案2敘述"
            {
            "headline": "方案3標題",
            "star": "⭐⭐⭐",
            "ideas": "方案3敘述"
            }
        ]
        }
        """

        # M2-3-a4 full costar template
        full_M2_3_a4_template = """
        {context}

        {objective}

        {style}

        {tone}

        {audience}

        {responce}
        """
        full_M2_3_a4_prompt = PromptTemplate.from_template(full_M2_3_a4_template)

        # M2-3-a4 "context" template
        context_M2_3_a4_template = """
        # Context
        你是一位專業的產業管理顧問。掌握全球產業重要發展動態，從產業市場、地區經濟、地方基礎建設、國家政策與社會文化等面向進行全面的分析，協助客戶探勘產業趨勢情報、洞見技術與創新需求脈絡，貼近產業立場，提供客戶全方位建言。
        """
        context_M2_3_a4_prompt = PromptTemplate.from_template(context_M2_3_a4_template)

        # M2-3-a4 "objective" template
        objective_M2_3_a4_template = """
        # Objective:
        未來創新機會方案：綜合上方提供之「競品分析」及「設計機會與挑戰」，分析以下四個面向：產品創新、服務創新、商業模式、環境建設，請以「環境建設」面向提供「3個」具體的切入方案。
            - 每個方案標題吸引人具前瞻性
            - 每個方案給予一個評分（以⭐表示，範圍從0到5）
            - 方案描述為具深刻洞見的評論，並有一定程度的合理性，每一個方案描述不少於300字且為結構清晰的段落。
        """
        objective_M2_3_a4_prompt = PromptTemplate.from_template(objective_M2_3_a4_template)

        # M2-3-a4 "style" template
        style_M2_3_a4_template = """
        # Style:

        - 文體應專業且具描述性(descriptive)，用字精準無贅字、無拼寫錯誤、條理清晰易讀。
        - 內容必須基於真實且正確的資料，不要編撰或創造內容(Do not make information up)。
        - 以#zh-TW繁體中文進行回答，答案中提及的品牌可以是國內或國外的實際品牌，並將貨幣符號保持為簡寫。
        - 確保每一條列的字數在其分析結果中保持一致性，不應有超過30字的差異。
        - 每一分析面向的條列點應「具有獨特性」，「避免內容的重複或過於相似」。
        - 必須將輸出結構化為一組json格式。json是一種聲明性語言，可讓你對文件進行註解和確認。你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！
        範例如下：

        {M2_3_a4_json_sample}
        """
        style_M2_3_a4_prompt = PromptTemplate.from_template(style_M2_3_a4_template)

        # M2-3-a4 "tone" template
        tone_M2_3_a4_template = """
        # Tone:
        在整個過程中保持清晰和有條理的語氣，說明「環境建設」機會。具備清楚且有方向的分析，提供有價值的見解和建議。
        """
        tone_M2_3_a4_prompt = PromptTemplate.from_template(tone_M2_3_a4_template)

        # M2-3-a4 "audience" template
        audience_M2_3_a4_template = """
        # Audience:
        目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
        """
        audience_M2_3_a4_prompt = PromptTemplate.from_template(audience_M2_3_a4_template)

        # M2-3-a4 "responce" template
        responce_M2_3_a4_template = """
        #Responce:
        根據上述資訊，針對{country}地區的{product}產業，以「環境建設」為模組名稱，統整並詳述當前產業的設計「環境建設」機會。

        #Start:
        如果您明白了，請開始進行分析
        """
        responce_M2_3_a4_prompt = PromptTemplate.from_template(responce_M2_3_a4_template)

        # M2-3-a4 arragement
        input_M2_3_a4_prompts = [
            ("context", context_M2_3_a4_prompt),
            ("objective", objective_M2_3_a4_prompt),
            ("style", style_M2_3_a4_prompt),
            ("tone", tone_M2_3_a4_prompt),
            ("audience", audience_M2_3_a4_prompt),
            ("responce", responce_M2_3_a4_prompt),
        ]
        pipeline_prompt = PipelinePromptTemplate(
            final_prompt=full_M2_3_a4_prompt, pipeline_prompts=input_M2_3_a4_prompts
        )

        # final M2-3-a4 prompt
        M2_3_a4_prompt = pipeline_prompt.format(
                product= product,
                country= country,
                M2_1_result = M2_1_result,
                M2_2_result = M2_2_result,
                M2_3_a4_json_sample = M2_3_a4_json_sample
                )

        print(M2_3_a4_prompt)

        M2_3_a4_result = model.predict(
            text=M2_3_a4_prompt)

        print(M2_3_a4_result)
    M2_3_a4_time = time.time()
    if 'M2_3_b1' in needs:
        # 參數定義
        M2_3_b1_json_sample = """
        {
        "prompt_topic": "產品創新概念圖prompt",

        "area": " {country} ",

        "product": " {product}  ",

        prompts:[
            {
            "prompt_1." : " "
            },
            {
            "prompt_2." : " "
            },
            {
            "prompt_3." : " "
            }
        ]
        }
        """

        # M2-3-b1 full costar template
        full_M2_3_b1_template = """
        {context}

        {objective}

        {style}

        {tone}

        {audience}

        {responce}
        """
        full_M2_3_b1_prompt = PromptTemplate.from_template(full_M2_3_b1_template)

        # M2-3-b1 "context" template
        context_M2_3_b1_template = """
        # Context
        你是AI繪圖工具的專家，善用 Leonardo AI, Stable Diffusion 這類AI繪圖工具繪製各種「產品設計概念圖」提供設計師創意發想。
        """
        context_M2_3_b1_prompt = PromptTemplate.from_template(context_M2_3_b1_template)

        # M2-3-b1 "objective" template
        objective_M2_3_b1_template = """
        # Objective:
        我將提供你3種產品創新方案，你將產生3組prompt，讓我透過Leonardo AI進行圖片生成。請遵照以下規則，產生每個方案各自的"英文"prompt。

        「產品創新」3種：{M2_3_a1_result}
        """
        objective_M2_3_b1_prompt = PromptTemplate.from_template(objective_M2_3_b1_template)

        # M2-3-b1 "style" template
        style_M2_3_b1_template = """
        # Style:

        - 越往前排列的單字是越關鍵的主體、主體特徵，越往後的單字越容易被歸納成風格、裝飾、畫面細節。
        - 參考下方範例依據方案描述內容，每一項方案裡的 prompt 必須和 "{product}" 有關，建議 prompt 能包含 {country} or {product} 的 "英文關鍵字"。
        - "area"以及"product"仍以#zh-TW繁體中文生成
        - "prompt_topic"請使用範例內，內容不需更改
        - 必須將輸出結構化為一組json格式。json是一種聲明性語言，可讓你對文件進行註解和確認。你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位格式及命名方式完全與下方範例完全匹配！
        範例如下：

        {M2_3_b1_json_sample}
        """
        style_M2_3_b1_prompt = PromptTemplate.from_template(style_M2_3_b1_template)

        # M2-3-b1 "tone" template
        tone_M2_3_b1_template = """
        # Tone:
        在整個過程中保持清晰和有條理的語氣，建立產品設計概念圖提示。具備清楚且有方向的分析，提供有價值的見解和建議。
        """
        tone_M2_3_b1_prompt = PromptTemplate.from_template(tone_M2_3_b1_template)

        # M2-3-b1 "audience" template
        audience_M2_3_b1_template = """
        # Audience:
        目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。"""
        audience_M2_3_b1_prompt = PromptTemplate.from_template(audience_M2_3_b1_template)

        # M2-3-b1 "responce" template
        responce_M2_3_b1_template = """
        #Responce:
        - 請將上方提供的3種「產品創新」機會方案，依據裡面各標題的方案描述，綜合上方參考文獻以及你所熟知AI繪圖使用的keywords，產生相應的"英文" 關鍵字組合 prompt。
        - 請直接將3種方案的關鍵字 "組合成一段完整的英文prompt"！不需要寫出方案標題、不用介係詞，以逗號連結即可 convert the bulleted keywords into comma-separated。
        - 給我各個創新機會方案3組完整的"英文"prompt，每一個 prompt 不超過 50 token。

        #Start:
        如果您明白了，請開始進行分析"""
        responce_M2_3_b1_prompt = PromptTemplate.from_template(responce_M2_3_b1_template)

        # M2-3-b1 arragement
        input_M2_3_b1_prompts = [
            ("context", context_M2_3_b1_prompt),
            ("objective", objective_M2_3_b1_prompt),
            ("style", style_M2_3_b1_prompt),
            ("tone", tone_M2_3_b1_prompt),
            ("audience", audience_M2_3_b1_prompt),
            ("responce", responce_M2_3_b1_prompt),
        ]
        pipeline_prompt = PipelinePromptTemplate(
            final_prompt=full_M2_3_b1_prompt, pipeline_prompts=input_M2_3_b1_prompts
        )

        # final M2-3-b1 prompt
        M2_3_b1_prompt = pipeline_prompt.format(
                product= product,
                country= country,
                M2_3_a1_result = M2_3_a1_result,
                M2_3_b1_json_sample = M2_3_b1_json_sample
                )

        print(M2_3_b1_prompt)

        M2_3_b1_result = model.predict(
            text=M2_3_b1_prompt)

        print(M2_3_b1_result)

        json_result = json.loads(M2_3_b1_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\""))

        print(json_result)
        # pic1 = dalle3(json_result['prompts'][0][list(json_result['prompts'][0].keys())[0]])
        # pic2 = dalle3(json_result['prompts'][1][list(json_result['prompts'][1].keys())[0]])
        # pic3 = dalle3(json_result['prompts'][2][list(json_result['prompts'][2].keys())[0]])
        # pic4 = dalle3(json_result['prompts'][0][list(json_result['prompts'][0].keys())[0]])
        # pic5 = dalle3(json_result['prompts'][1][list(json_result['prompts'][1].keys())[0]])
        # pic6 = dalle3(json_result['prompts'][2][list(json_result['prompts'][2].keys())[0]])
        M2_3_b1_result = [
            json_result['prompts'][0][list(json_result['prompts'][0].keys())[0]],
            json_result['prompts'][1][list(json_result['prompts'][1].keys())[0]],
            json_result['prompts'][2][list(json_result['prompts'][2].keys())[0]]
        ]
    M2_3_b1_time = time.time()
    if 'M2_3_b2' in needs:
        # 參數定義
        M2_3_b2_json_sample = """
        {
        "prompt_topic": "服務創新概念圖prompt",

        "area": " {country} ",

        "product": " {product}  ",

        prompts:[
            {
            "prompt_1." : " "
            },
            {
            "prompt_2." : " "
            },
            {
            "prompt_3." : " "
            }
        ]
        }
        """

        # M2-3-b2 full costar template
        full_M2_3_b2_template = """
        {context}

        {objective}

        {style}

        {tone}

        {audience}

        {responce}
        """
        full_M2_3_b2_prompt = PromptTemplate.from_template(full_M2_3_b2_template)

        # M2-3-b2 "context" template
        context_M2_3_b2_template = """
        # Context
        你是AI繪圖工具的專家，善用 Leonardo AI, Stable Diffusion 這類AI繪圖工具繪製各種「產品設計概念圖」提供設計師創意發想。
        """
        context_M2_3_b2_prompt = PromptTemplate.from_template(context_M2_3_b2_template)

        # M2-3-b2 "objective" template
        objective_M2_3_b2_template = """
        # Objective:
        我將提供你3種服務創新方案，你將產生3組prompt，讓我透過Leonardo AI進行圖片生成。請遵照以下規則，產生每個方案各自的"英文"prompt。

        「服務創新」3種：{M2_3_a2_result}
        """
        objective_M2_3_b2_prompt = PromptTemplate.from_template(objective_M2_3_b2_template)

        # M2-3-b2 "style" template
        style_M2_3_b2_template = """
        # Style:

        - 越往前排列的單字是越關鍵的主體、主體特徵，越往後的單字越容易被歸納成風格、裝飾、畫面細節。
        - 參考下方範例依據方案描述內容，每一項方案裡的 prompt 必須和 "{product}" 有關，建議 prompt 能包含 {country} or {product} 的 "英文關鍵字"。
        - "area"以及"product"仍以#zh-TW繁體中文生成
        - "prompt_topic"請使用範例內，內容不需更改
        - 必須將輸出結構化為一組json格式。json是一種聲明性語言，可讓你對文件進行註解和確認。你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位格式及命名方式完全與下方範例完全匹配！
        範例如下：

        {M2_3_b2_json_sample}
        """
        style_M2_3_b2_prompt = PromptTemplate.from_template(style_M2_3_b2_template)

        # M2-3-b2 "tone" template
        tone_M2_3_b2_template = """
        # Tone:
        在整個過程中保持清晰和有條理的語氣，建立產品設計概念圖提示。具備清楚且有方向的分析，提供有價值的見解和建議。
        """
        tone_M2_3_b2_prompt = PromptTemplate.from_template(tone_M2_3_b2_template)

        # M2-3-b2 "audience" template
        audience_M2_3_b2_template = """
        # Audience:
        目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。"""
        audience_M2_3_b2_prompt = PromptTemplate.from_template(audience_M2_3_b2_template)

        # M2-3-b2 "responce" template
        responce_M2_3_b2_template = """
        #Responce:
        - 請將上方提供的3種「服務創新」機會方案，依據裡面各標題的方案描述，綜合上方參考文獻以及你所熟知AI繪圖使用的keywords，產生相應的"英文" 關鍵字組合 prompt。
        - 請直接將3種方案的關鍵字 "組合成一段完整的英文prompt"！不需要寫出方案標題、不用介係詞，以逗號連結即可 convert the bulleted keywords into comma-separated。
        - 給我各個創新機會方案3組完整的"英文"prompt，每一個 prompt 不超過 50 token。

        #Start:
        如果您明白了，請開始進行分析"""
        responce_M2_3_b2_prompt = PromptTemplate.from_template(responce_M2_3_b2_template)

        # M2-3-b2 arragement
        input_M2_3_b2_prompts = [
            ("context", context_M2_3_b2_prompt),
            ("objective", objective_M2_3_b2_prompt),
            ("style", style_M2_3_b2_prompt),
            ("tone", tone_M2_3_b2_prompt),
            ("audience", audience_M2_3_b2_prompt),
            ("responce", responce_M2_3_b2_prompt),
        ]
        pipeline_prompt = PipelinePromptTemplate(
            final_prompt=full_M2_3_b2_prompt, pipeline_prompts=input_M2_3_b2_prompts
        )

        # final M2-3-b2 prompt
        M2_3_b2_prompt = pipeline_prompt.format(
                product= product,
                country= country,
                M2_3_a2_result = M2_3_a2_result,
                M2_3_b2_json_sample = M2_3_b2_json_sample
                )

        print(M2_3_b2_prompt)

        M2_3_b2_result = model.predict(
            text=M2_3_b2_prompt)

        print(M2_3_b2_result)

        json_result = json.loads(M2_3_b2_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\""))

        print(json_result)
        # pic1 = dalle3(json_result['prompts'][0][list(json_result['prompts'][0].keys())[0]])
        # pic2 = dalle3(json_result['prompts'][1][list(json_result['prompts'][1].keys())[0]])
        # pic3 = dalle3(json_result['prompts'][2][list(json_result['prompts'][2].keys())[0]])
        # pic4 = dalle3(json_result['prompts'][0][list(json_result['prompts'][0].keys())[0]])
        # pic5 = dalle3(json_result['prompts'][1][list(json_result['prompts'][1].keys())[0]])
        # pic6 = dalle3(json_result['prompts'][2][list(json_result['prompts'][2].keys())[0]])
        M2_3_b2_result = [
            json_result['prompts'][0][list(json_result['prompts'][0].keys())[0]],
            json_result['prompts'][1][list(json_result['prompts'][1].keys())[0]],
            json_result['prompts'][2][list(json_result['prompts'][2].keys())[0]]
        ]
    M2_3_b2_time = time.time()
    if 'M2_3_b3' in needs:
        # 參數定義
        M2_3_b3_json_sample = """
        {
        "prompt_topic": "商業模式概念圖prompt",

        "area": " {country} ",

        "product": " {product}  ",

        prompts:[
            {
            "prompt_1." : " "
            },
            {
            "prompt_2." : " "
            },
            {
            "prompt_3." : " "
            }
        ]
        }
        """

        # M2-3-b3 full costar template
        full_M2_3_b3_template = """
        {context}

        {objective}

        {style}

        {tone}

        {audience}

        {responce}
        """
        full_M2_3_b3_prompt = PromptTemplate.from_template(full_M2_3_b3_template)

        # M2-3-b3 "context" template
        context_M2_3_b3_template = """
        # Context
        你是AI繪圖工具的專家，善用 Leonardo AI, Stable Diffusion 這類AI繪圖工具繪製各種「產品設計概念圖」提供設計師創意發想。
        """
        context_M2_3_b3_prompt = PromptTemplate.from_template(context_M2_3_b3_template)

        # M2-3-b3 "objective" template
        objective_M2_3_b3_template = """
        # Objective:
        我將提供你3種商業模式方案，你將產生3組prompt，讓我透過Leonardo AI進行圖片生成。請遵照以下規則，產生每個方案各自的"英文"prompt。

        「商業模式」3種：{M2_3_a3_result}
        """
        objective_M2_3_b3_prompt = PromptTemplate.from_template(objective_M2_3_b3_template)

        # M2-3-b3 "style" template
        style_M2_3_b3_template = """
        # Style:

        - 越往前排列的單字是越關鍵的主體、主體特徵，越往後的單字越容易被歸納成風格、裝飾、畫面細節。
        - 參考下方範例依據方案描述內容，每一項方案裡的 prompt 必須和 "{product}" 有關，建議 prompt 能包含 {country} or {product} 的 "英文關鍵字"。
        - "area"以及"product"仍以#zh-TW繁體中文生成
        - "prompt_topic"請使用範例內，內容不需更改
        - 必須將輸出結構化為一組json格式。json是一種聲明性語言，可讓你對文件進行註解和確認。你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位格式及命名方式完全與下方範例完全匹配！
        範例如下：

        {M2_3_b3_json_sample}
        """
        style_M2_3_b3_prompt = PromptTemplate.from_template(style_M2_3_b3_template)

        # M2-3-b3 "tone" template
        tone_M2_3_b3_template = """
        # Tone:
        在整個過程中保持清晰和有條理的語氣，建立產品設計概念圖提示。具備清楚且有方向的分析，提供有價值的見解和建議。
        """
        tone_M2_3_b3_prompt = PromptTemplate.from_template(tone_M2_3_b3_template)

        # M2-3-b3 "audience" template
        audience_M2_3_b3_template = """
        # Audience:
        目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。"""
        audience_M2_3_b3_prompt = PromptTemplate.from_template(audience_M2_3_b3_template)

        # M2-3-b3 "responce" template
        responce_M2_3_b3_template = """
        #Responce:
        - 請將上方提供的3種「商業模式」機會方案，依據裡面各標題的方案描述，綜合上方參考文獻以及你所熟知AI繪圖使用的keywords，產生相應的"英文" 關鍵字組合 prompt。
        - 請直接將3種方案的關鍵字 "組合成一段完整的英文prompt"！不需要寫出方案標題、不用介係詞，以逗號連結即可 convert the bulleted keywords into comma-separated。
        - 給我各個創新機會方案3組完整的"英文"prompt，每一個 prompt 不超過 50 token。

        #Start:
        如果您明白了，請開始進行分析"""
        responce_M2_3_b3_prompt = PromptTemplate.from_template(responce_M2_3_b3_template)

        # M2-3-b3 arragement
        input_M2_3_b3_prompts = [
            ("context", context_M2_3_b3_prompt),
            ("objective", objective_M2_3_b3_prompt),
            ("style", style_M2_3_b3_prompt),
            ("tone", tone_M2_3_b3_prompt),
            ("audience", audience_M2_3_b3_prompt),
            ("responce", responce_M2_3_b3_prompt),
        ]
        pipeline_prompt = PipelinePromptTemplate(
            final_prompt=full_M2_3_b3_prompt, pipeline_prompts=input_M2_3_b3_prompts
        )

        # final M2-3-b3 prompt
        M2_3_b3_prompt = pipeline_prompt.format(
                product= product,
                country= country,
                M2_3_a3_result = M2_3_a3_result,
                M2_3_b3_json_sample = M2_3_b3_json_sample
                )

        print(M2_3_b3_prompt)

        M2_3_b3_result = model.predict(
            text=M2_3_b3_prompt)

        print(M2_3_b3_result)

        json_result = json.loads(M2_3_b3_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\""))

        print(json_result)
        # pic1 = dalle3(json_result['prompts'][0][list(json_result['prompts'][0].keys())[0]])
        # pic2 = dalle3(json_result['prompts'][1][list(json_result['prompts'][1].keys())[0]])
        # pic3 = dalle3(json_result['prompts'][2][list(json_result['prompts'][2].keys())[0]])
        # pic4 = dalle3(json_result['prompts'][0][list(json_result['prompts'][0].keys())[0]])
        # pic5 = dalle3(json_result['prompts'][1][list(json_result['prompts'][1].keys())[0]])
        # pic6 = dalle3(json_result['prompts'][2][list(json_result['prompts'][2].keys())[0]])
        M2_3_b3_result = [
            json_result['prompts'][0][list(json_result['prompts'][0].keys())[0]],
            json_result['prompts'][1][list(json_result['prompts'][1].keys())[0]],
            json_result['prompts'][2][list(json_result['prompts'][2].keys())[0]]
        ]
    M2_3_b3_time = time.time()
    if 'M2_3_b4' in needs:
        # 參數定義
        M2_3_b4_json_sample = """
        {
        "prompt_topic": "環境建設概念圖prompt",

        "area": " {country} ",

        "product": " {product}  ",

        prompts:[
            {
            "prompt_1." : " "
            },
            {
            "prompt_2." : " "
            },
            {
            "prompt_3." : " "
            }
        ]
        }
        """

        # M2-3-b4 full costar template
        full_M2_3_b4_template = """
        {context}

        {objective}

        {style}

        {tone}

        {audience}

        {responce}
        """
        full_M2_3_b4_prompt = PromptTemplate.from_template(full_M2_3_b4_template)

        # M2-3-b4 "context" template
        context_M2_3_b4_template = """
        # Context
        你是AI繪圖工具的專家，善用 Leonardo AI, Stable Diffusion 這類AI繪圖工具繪製各種「產品設計概念圖」提供設計師創意發想。
        """
        context_M2_3_b4_prompt = PromptTemplate.from_template(context_M2_3_b4_template)

        # M2-3-b4 "objective" template
        objective_M2_3_b4_template = """
        # Objective:
        我將提供你3種環境建設方案，你將產生3組prompt，讓我透過Leonardo AI進行圖片生成。請遵照以下規則，產生每個方案各自的"英文"prompt。

        「環境建設」3種：{M2_3_a4_result}
        """
        objective_M2_3_b4_prompt = PromptTemplate.from_template(objective_M2_3_b4_template)

        # M2-3-b4 "style" template
        style_M2_3_b4_template = """
        # Style:

        - 越往前排列的單字是越關鍵的主體、主體特徵，越往後的單字越容易被歸納成風格、裝飾、畫面細節。
        - 參考下方範例依據方案描述內容，每一項方案裡的 prompt 必須和 "{product}" 有關，建議 prompt 能包含 {country} or {product} 的 "英文關鍵字"。
        - "area"以及"product"仍以#zh-TW繁體中文生成
        - "prompt_topic"請使用範例內，內容不需更改
        - 必須將輸出結構化為一組json格式。json是一種聲明性語言，可讓你對文件進行註解和確認。你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位格式及命名方式完全與下方範例完全匹配！
        範例如下：

        {M2_3_b4_json_sample}
        """
        style_M2_3_b4_prompt = PromptTemplate.from_template(style_M2_3_b4_template)

        # M2-3-b4 "tone" template
        tone_M2_3_b4_template = """
        # Tone:
        在整個過程中保持清晰和有條理的語氣，建立產品設計概念圖提示。具備清楚且有方向的分析，提供有價值的見解和建議。
        """
        tone_M2_3_b4_prompt = PromptTemplate.from_template(tone_M2_3_b4_template)

        # M2-3-b4 "audience" template
        audience_M2_3_b4_template = """
        # Audience:
        目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。"""
        audience_M2_3_b4_prompt = PromptTemplate.from_template(audience_M2_3_b4_template)

        # M2-3-b4 "responce" template
        responce_M2_3_b4_template = """
        #Responce:
        - 請將上方提供的3種「環境建設」機會方案，依據裡面各標題的方案描述，綜合上方參考文獻以及你所熟知AI繪圖使用的keywords，產生相應的"英文" 關鍵字組合 prompt。
        - 請直接將3種方案的關鍵字 "組合成一段完整的英文prompt"！不需要寫出方案標題、不用介係詞，以逗號連結即可 convert the bulleted keywords into comma-separated。
        - 給我各個創新機會方案3組完整的"英文"prompt，每一個 prompt 不超過 50 token。

        #Start:
        如果您明白了，請開始進行分析"""
        responce_M2_3_b4_prompt = PromptTemplate.from_template(responce_M2_3_b4_template)

        # M2-3-b4 arragement
        input_M2_3_b4_prompts = [
            ("context", context_M2_3_b4_prompt),
            ("objective", objective_M2_3_b4_prompt),
            ("style", style_M2_3_b4_prompt),
            ("tone", tone_M2_3_b4_prompt),
            ("audience", audience_M2_3_b4_prompt),
            ("responce", responce_M2_3_b4_prompt),
        ]
        pipeline_prompt = PipelinePromptTemplate(
            final_prompt=full_M2_3_b4_prompt, pipeline_prompts=input_M2_3_b4_prompts
        )

        # final M2-3-b4 prompt
        M2_3_b4_prompt = pipeline_prompt.format(
                product= product,
                country= country,
                M2_3_a4_result = M2_3_a4_result,
                M2_3_b4_json_sample = M2_3_b4_json_sample
                )

        print(M2_3_b4_prompt)

        M2_3_b4_result = model.predict(
            text=M2_3_b4_prompt)

        print(M2_3_b4_result)

        json_result = json.loads(M2_3_b4_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\""))

        print(json_result)
        # pic1 = dalle3(json_result['prompts'][0][list(json_result['prompts'][0].keys())[0]])
        # pic2 = dalle3(json_result['prompts'][1][list(json_result['prompts'][1].keys())[0]])
        # pic3 = dalle3(json_result['prompts'][2][list(json_result['prompts'][2].keys())[0]])
        # pic4 = dalle3(json_result['prompts'][0][list(json_result['prompts'][0].keys())[0]])
        # pic5 = dalle3(json_result['prompts'][1][list(json_result['prompts'][1].keys())[0]])
        # pic6 = dalle3(json_result['prompts'][2][list(json_result['prompts'][2].keys())[0]])
        M2_3_b4_result = [
            json_result['prompts'][0][list(json_result['prompts'][0].keys())[0]],
            json_result['prompts'][1][list(json_result['prompts'][1].keys())[0]],
            json_result['prompts'][2][list(json_result['prompts'][2].keys())[0]]
        ]
    M2_3_b4_time = time.time()

    print("======M1M2 TIME======")
    print("======M0 TIME======")
    print(M0_time - start_time)
    print("======M1_1 TIME======")
    print(M1_1_time - M0_time)
    print("======M1_2 TIME======")
    print(M1_2_time - M1_1_time)
    print("======M1_3 TIME======")
    print(M1_3_time - M1_2_time)
    print("======M1_4 TIME======")
    print(M1_4_time - M1_3_time)
    print("======M1_5 TIME======")
    print(M1_5_time - M1_4_time)
    print("======M1_6 TIME======")
    print(M1_6_time - M1_5_time)
    print("======M1_1_R TIME======")
    print(M1_1_R_time - M1_6_time)
    print("======M1_2_R TIME======")
    print(M1_2_R_time - M1_1_R_time)
    print("======M1_3_R TIME======")
    print(M1_3_R_time - M1_2_R_time)
    print("======M1_4_R TIME======")
    print(M1_4_R_time - M1_3_R_time)
    print("======M1_5_R TIME======")
    print(M1_5_R_time - M1_4_R_time)
    print("======M1_6_R TIME======")
    print(M1_6_R_time - M1_5_R_time)
    print("======M2_1 TIME======")
    print(M2_1_time - M1_6_R_time)
    print("======M2_1_R TIME======")
    print(M2_1_R_time - M2_1_time)
    print("======M2_2 TIME======")
    print(M2_2_time - M2_1_R_time)
    print("======M2_3_a1 TIME======")
    print(M2_3_a1_time - M2_2_time)
    print("======M2_3_a2 TIME======")
    print(M2_3_a2_time - M2_3_a1_time)
    print("======M2_3_a3 TIME======")
    print(M2_3_a3_time - M2_3_a2_time)
    print("======M2_3_a4 TIME======")
    print(M2_3_a4_time - M2_3_a3_time)
    print("======M2_3_b1 TIME======")
    print(M2_3_b1_time - M2_3_a4_time)
    print("======M2_3_b2 TIME======")
    print(M2_3_b2_time - M2_3_b1_time)
    print("======M2_3_b3 TIME======")
    print(M2_3_b3_time - M2_3_b2_time)
    print("======M2_3_b4 TIME======")
    print(M2_3_b4_time - M2_3_b3_time)

    print("======Total TIME======")
    print(M2_3_b4_time - start_time)

    return {
        "title": M0_result,
        "pestel": {
            "politic": json.loads(M1_1_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
            "economic": json.loads(M1_2_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
            "social": json.loads(M1_3_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
            "technology": json.loads(M1_4_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
            "environment": json.loads(M1_5_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
            "law": json.loads(M1_6_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
        },
        "competitive_product_analysis": {
            "competitive": json.loads(M2_1_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
            "opportunities-and-challenges": json.loads(M2_2_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
            "product_innovation": json.loads(M2_3_a1_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
            "service_innovation": json.loads(M2_3_a2_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
            "business_model": json.loads(M2_3_a3_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
            "infra_develop": json.loads(M2_3_a4_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
            "product-innovation_images": M2_3_b1_result,
            "service-innovation_images": M2_3_b2_result,
            "business-model_images": M2_3_b3_result,
            "infra-develop_images": M2_3_b4_result,
        }
    }