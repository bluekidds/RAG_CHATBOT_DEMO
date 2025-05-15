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

# Define data structure for keyword module.
class Keyword(BaseModel):
    behavior: str = Field(description="user's behavioral traits in sentence")
    profession: str = Field(description="person profession in sentence")
    task: str = Field(description="user goal mention in sentence")

# Set up a parser + inject instructions into the prompt template.
def get_keyword(query):
    parser = JsonOutputParser(pydantic_object=Keyword)
    prompt = PromptTemplate(
        template="Extract the key word from query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser
    dic = chain.invoke({"query": query})

    behavior = dic['behavior']
    profession = dic['profession']
    task = dic['task']

    print(dic)
    return dic

def M3_1(query):
    params = get_keyword(query)
    behavior = params['behavior']
    profession = params['profession']
    task = params['task']

    # 參數定義
    U3_1_json_sample = """
    {
    "user_scenario": "身為一名熱衷於推動健康生活方式的物理治療師，因為深信規律運動與預防性護理能改善心血管健康，我需要將自行車作為患者康復計劃的一部分，並積極參與社區健康推廣活動，以達到讓自行車成為安全、可行的全民運動選擇的目標。我希望通過推薦具有醫療認證的自行車產品，確保每位患者都能安心使用，最終推動整個社區走向更健康的生活方式。",
    "name": "李大衛",
    "age": "42",
    "male_or_female": "男性",
    "occupation": "物理治療師兼自行車運動愛好者",
    "task": "推廣自行車運動，讓更多人享受到騎乘的樂趣，並認識到其對健康的好處。"
    }
    """

    # U3-1 full costar template
    full_U3_1_template = """
    {context}
    {objective}
    {style}
    {tone}
    {audience}
    {responce}
    """
    full_U3_1_prompt = PromptTemplate.from_template(full_U3_1_template)

    # U3-1 "contex" template
    context_U3_1_template = """
    # Context
    I want you to act as a UX researcher. I will provide some details about 'How to create a persona/user scenarios', and it will be your job to gain a comprehensive understanding of the customer experience. My first request is "I need a best practices for creating user personas to guide design decisions on different user and task i.e.
    [{profession},{task}].
    """
    context_U3_1_prompt = PromptTemplate.from_template(context_U3_1_template)

    # U3-1 "objective" template
    objective_U3_1_template = """
    # Objective:
    人物誌的方法論如下
        #1 Indi Young的書《Mental Models: Aligning Design Strategy with Human Behavior》參考書中 Define Task-Based Audience Segment 章節講解的方法，架構所謂「Mental Model」，先定義使用者，不同於以往的人口研究(demographic)或是心理側寫(psychographic)等方法，Indi Young的Mental Model中使用的是task-based：找出會執行某樣作業的人，不管年齡、職業、性別。

        #2 藉由想像自己站在使用者的立場，試著了解他/她最真實的感受，然後思考自己會怎麼做並描繪使用者同理心地圖 (Empathy Map)，預設這次目標族群 [{profession}, {task}]

        #3 參考Preece, Rogers and Sharp(2002)列舉下列使用者經驗目標：1.滿意的(Satisfying) 2.樂趣的(Enjoyable) 3.快樂的(Fun) 4.娛樂的(Entertaining) 5.有益的(Helpful) 6.刺激的(Motivating) 7.美的愉悅感(Aesthetically Pleasing) 8.創造力的支援(Supportive of Creativity) 9.實現個人抱負的情感(Emotionally Fulfilling)，進一步描述 [{profession}, {task}] 的角色原型：包含他每天可能的生活場景、他每天下班回到家可能要執行的任務與痛點及道具需求

        #4 人物誌（Persona）的功能是為我們的使用者創造「具體形象」，它是一個半虛擬的人物，用來描述使用者的詳細資訊。完整的人物誌（Persona）需考慮以下內容：
        - 個人資料（姓名、性別、年齡、住址、婚姻狀況、座右銘）
        - 個性（價值觀）
        - 任務目標與需求，Goals and concerns when they perform relevant 任務(tasks): speed, accuracy, thoroughness, or any other needs that may factor into their usage
        - 進行任務過程的痛點及挑戰（可能的工作風險、可能的情緒變化、可能取得的專業機構協助）
        - Create a believable and alive character. Avoid adding extraneous details that do not have any implications for design.
    """
    objective_U3_1_prompt = PromptTemplate.from_template(objective_U3_1_template)

    # U3-1 "style" template
    style_U3_1_template = """
    # Style:

    - 文體具敘事性(narrative)，用字精準無贅字、無拼寫錯誤、清晰易讀。#zh-TW繁體中文為主，專有名詞、品牌等關鍵字可使用英文。
    - 必須將輸出結構化為一組json格式。json是一種聲明性語言，可讓你對文件進行註解和確認。你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！
    範例如下：

    {U3_1_json_sample}
    """
    style_U3_1_prompt = PromptTemplate.from_template(style_U3_1_template)

    # U3-1 "tone" template
    tone_U3_1_template = """
    # Tone:
    在整個過程中保持清晰和有條理的語氣，說明人物誌描述。具備清楚且有方向的分析，提供有價值的見解和建議。
    """
    tone_U3_1_prompt = PromptTemplate.from_template(tone_U3_1_template)

    # U3-1 "audience" template
    audience_U3_1_template = """
    # Audience:
    目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
    """
    audience_U3_1_prompt = PromptTemplate.from_template(audience_U3_1_template)

    # U3-1 "responce" template
    responce_U3_1_template = """
    #Responce:
    請依據以上方#Objective中 #1, #2, #3, #4 的方法論，以 [{profession}, {task}] 為題目，用"一整篇不分段的文章"來描述該 Persona，
    角色需包含：姓名、年齡、性別。Persona 總字數不少於250字。不需解釋也不要結語，""請直接給我Persona描述""。

    #Start:
    如果您明白了，請開始進行分析
    """
    responce_U3_1_prompt = PromptTemplate.from_template(responce_U3_1_template)

    # U3-1 arragement
    input_U3_1_prompts = [
        ("context", context_U3_1_prompt),
        ("objective", objective_U3_1_prompt),
        ("style", style_U3_1_prompt),
        ("tone", tone_U3_1_prompt),
        ("audience", audience_U3_1_prompt),
        ("responce", responce_U3_1_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_U3_1_prompt, pipeline_prompts=input_U3_1_prompts
    )

    # final U3-1 prompt
    U3_1_prompt = pipeline_prompt.format(
            behavior = behavior,
            profession = profession,
            task = task,
            U3_1_json_sample = U3_1_json_sample
            )

    print(U3_1_prompt)

    U3_1_result = model.predict(
        text=U3_1_prompt)

    print(U3_1_result)
    return U3_1_result

def M3_2(query):
    params = get_keyword(query)
    behavior = params['behavior']
    profession = params['profession']
    task = params['task']

    # 參數定義
    U3_2_json_sample = """
    {
    "events": [
        {
        "touchpoint": "個人興趣探索",
        "action": ["參加自行車同好會活動", "學習與自行車相關的知識"],
        "channels_and_stakeholders": "#自行車同好會 #社交媒體群組 #線上自行車論壇 #同好成員"
        },
        {
        "touchpoint": "病患診療",
        "action": ["了解病患生活習慣並提出建議", "鼓勵患者透過騎自行車來改善心血管健康"],
        "channels_and_stakeholders": "#診間 #平板 #心臟掃描影片 #亞健康患者"
        },
        {
        "touchpoint": "社區健康推廣活動",
        "action": ["組織並參加社區健康講座", "推廣自行車運動的益處"],
        "channels_and_stakeholders": "#社區中心 #公共廣播 #社區成員 #健康講座"
        },
        {
        "touchpoint": "合作夥伴關係建立",
        "action": ["與當地自行車店和公共健康組織合作", "提供自行車安全和健康騎行的課程"],
        "channels_and_stakeholders": "#自行車店 #公共健康組織 #課程推廣 #潛在騎行者"
        },
        {
        "touchpoint": "反饋與持續改進",
        "action": ["收集患者和社區成員的反饋", "改進推廣策略"],
        "channels_and_stakeholders": "#患者反饋表 #社交媒體調查 #健康數據分析 #推廣團隊"
        }
    ]
    }
    """

    # U3-2 full costar template
    full_U3_2_template = """
    {context}
    {objective}
    {style}
    {tone}
    {audience}
    {responce}
    """
    full_U3_2_prompt = PromptTemplate.from_template(full_U3_2_template)

    # U3-2 "contex" template
    context_U3_2_template = """
    # Context
    I want you to act as a UX researcher. I will provide some details about 'How to create a customer journey map',
    and it will be your job to use customer journey mapping to gain a comprehensive understanding of the customer experience.
    My first request is 'I need a best practices for creating user journeys based on user personas.
    """
    context_U3_2_prompt = PromptTemplate.from_template(context_U3_2_template)

    # U3-2 "objective" template
    objective_U3_2_template = """
    # Objective:
    你需要參考下方的顧客旅程地圖參考文獻，並針對提供的Persona，以 [{profession}, {task}] 為主題條列出「五個」「接觸點 (touchpoints)」。每個條列內容需包含Persona角色之姓名、任務以及「使用者的道具需求與可能接觸的角色」。各別條列不少於200字。

    顧客旅程地圖參考文獻如下：

    How to create a customer journey map?

        1.Define the behavioral stages.
        Depending on {profession}, customers may go through different stages while navigating your site. A B2C ecommerce company may have just a few clearly defined phases; a B2B SaaS company selling to the Fortune 100 may have more.
        {profession} personas should give you a pretty good idea of the process that customers go through from their first landing to an eventual purchase and subsequent interactions. The next step identifies which interactions fit into which stages.

        2.Align customer goals with each stage.
        What do customers want to achieve as they move through each behavioral stage? You can mine a number of data sources to get that information:
        • Survey answers;
        • User testing;
        • Interview transcripts;
        • Customer service emails or support transcripts.
        Then, you’ll be able to see if your website supports each of those goals.

        3.Plot the touchpoints.
        Think of touchpoints as places where customers engage with your site and where you support the completion of their goals. These touchpoints will be grouped under the relevant stage in your customer’s journey. For retailers, a common touchpoint might be a product description page; for a service business, it may anything from a pricing page to a contact form. You can identify touchpoints along the user journey in two reports in Google Analytics:
        • Behavior Flow report;
        • Goal flow report.
        You’ll be able to determine if users—or a subset of them—are unexpectedly leaving in the middle of their journey on the path to the goal, or if there’s a place where your traffic loops back.

        4.Determine if customers achieve their goals.
        This is where you take the data you’ve collected and measure it against how easily your customers can get done what they need to do. Ask yourself the following types of questions:
        • Where are there roadblocks?
        • Do tons of people abandon their carts on the checkout page?
        • Do users go to your opt-in download page but not fill out the form?
        The Google Analytics reports you’ve mined for insights will show you where issues crop up. The existing qualitative research you have—the same research you used to build your personas—should help you understand the why behind the problems.
        Analyze the actions (or lack thereof) of your customers. How well are their needs met at each touchpoint and during each phase?

        5.Effective customer journey mapping follows five key high-level steps:
        • Aspiration and allies: Building a core cross disciplinary team and defining the scope of the mapping initiative
        • Internal investigation: Gathering existing customer data and research that exists throughout the organization
        • Assumption formulation: Formulating a hypothesis of the current state of the journey and planning additional customer research
        • External research: Collecting new user data to validate (or invalidate) the hypothesis journey map
        • Narrative visualization: Combining existing insights and new research to create a visual narrative that depicts the customer journey in a sound way

    Persona 如下：
    {U3_1_result}
    """
    objective_U3_2_prompt = PromptTemplate.from_template(objective_U3_2_template)

    # U3-2 "style" template
    style_U3_2_template = """
    # Style:
    - 文體具敘事性(narrative)，用字精準無贅字、無拼寫錯誤、清晰易讀。"#zh-TW繁體中文"為主，專有名詞、品牌等關鍵字可使用英文。
    - 必須將輸出結構化為一組json格式。json是一種聲明性語言，可讓你對文件進行註解和確認。你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！
    範例如下：

    {U3_2_json_sample}
    """
    style_U3_2_prompt = PromptTemplate.from_template(style_U3_2_template)

    # U3-2 "tone" template
    tone_U3_2_template = """
    # Tone:
    在整個過程中保持清晰和有條理的語氣，說明接觸點。具備清楚且有方向的分析，提供有價值的見解和建議。
    """
    tone_U3_2_prompt = PromptTemplate.from_template(tone_U3_2_template)

    # U3-2 "audience" template
    audience_U3_2_template = """
    # Audience:
    目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
    """
    audience_U3_2_prompt = PromptTemplate.from_template(audience_U3_2_template)

    # U3-2 "responce" template
    responce_U3_2_template = """
    #Responce:
    請依據上方#Objective中的參考文獻，以及提供的Persona，以 [{profession}, {task}] 為主題，條列出「五個」「接觸點 (touchpoints)」。每個條列內容需包含Persona角色之姓名、任務。各別條列不少於200字。

    #Start:
    如果您明白了，請開始進行分析
    """
    responce_U3_2_prompt = PromptTemplate.from_template(responce_U3_2_template)

    # U3-2 arragement
    input_U3_2_prompts = [
        ("context", context_U3_2_prompt),
        ("objective", objective_U3_2_prompt),
        ("style", style_U3_2_prompt),
        ("tone", tone_U3_2_prompt),
        ("audience", audience_U3_2_prompt),
        ("responce", responce_U3_2_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_U3_2_prompt, pipeline_prompts=input_U3_2_prompts
    )

    # final U3-2 prompt
    U3_2_prompt = pipeline_prompt.format(
            behavior = behavior,
            profession = profession,
            task = task,
            U3_1_result = U3_1_result,
            U3_2_json_sample = U3_2_json_sample
            )

    print(U3_2_prompt)

    U3_2_result = model.predict(
        text=U3_2_prompt)

    print(U3_2_result)
    return U3_2_result

def M3_3(query):
    params = get_keyword(query)
    behavior = params['behavior']
    profession = params['profession']
    task = params['task']

    # 參數定義
    U3_3_json_sample = """
    {
    "emotions": [
        {
        "touchpoint": "個人興趣探索",
        "score": 4
        },
        {
        "touchpoint": "病患診療",
        "score": 3
        },
        {
        "touchpoint": "社區健康推廣活動",
        "score": 5
        },
        {
        "touchpoint": "合作夥伴關係建立",
        "score": 4
        },
        {
        "touchpoint": "反饋與持續改進",
        "score": 3
        }
    ]
    }
    """

    # U3-3 full costar template
    full_U3_3_template = """
    {context}
    {objective}
    {style}
    {tone}
    {audience}
    {responce}
    """
    full_U3_3_prompt = PromptTemplate.from_template(full_U3_3_template)

    # U3-3 "contex" template
    context_U3_3_template = """
    # Context
    I want you to act as a UX researcher with a specialization in customer journey mapping. My primary role is to analyze user perosna and touchpoints.
    These touchpoints focus on the emotional journey of the [{profession}, {task}] encounter with a [{profession}, {task}] task, ranging from -2 (negative emotion) to 2 (positive emotion).
    """
    context_U3_3_prompt = PromptTemplate.from_template(context_U3_3_template)

    # U3-3 "objective" template
    objective_U3_3_template = """
    # Objective:
    你需要參考下方的情緒分數參考文獻，參考提供的 Persona，並基於提供的 Touchpoints 五個階段，依據各階段的情緒體驗，分別給一個「分數 (emotion score)」：positive emotion 最高給2分，negative emotion 最低給-2分。無須任何描述「只須給我分數即可」。。

    情緒分數考文獻如下：
    Thoughts and feelings: what the customer thinks and feels at each touchpoint.
    Emotions are plotted as a single line across the journey phases, literally signaling the emotional “ups” and “downs” of the experience. Think of this line as a contextual layer of emotion that tells us where the user is delighted versus frustrated.

    Persona 如下：
    {U3_1_result}

    Touchpoints 內容如下
    {U3_2_result}
    """
    objective_U3_3_prompt = PromptTemplate.from_template(objective_U3_3_template)

    # U3-3 "style" template
    style_U3_3_template = """
    # Style:
    - The journey includes at least 5 stages, with some stages intentionally negative to reflect realistic user experiences.
    - 必須將輸出結構化為一組json格式。json是一種聲明性語言，可讓你對文件進行註解和確認。你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！
    範例如下：

    {U3_3_json_sample}
    """
    style_U3_3_prompt = PromptTemplate.from_template(style_U3_3_template)

    # U3-3 "tone" template
    tone_U3_3_template = """
    # Tone:
    在整個過程中保持清楚且有方向的分析，提供情緒分數。
    """
    tone_U3_3_prompt = PromptTemplate.from_template(tone_U3_3_template)

    # U3-3 "audience" template
    audience_U3_3_template = """
    # Audience:
    目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
    """
    audience_U3_3_prompt = PromptTemplate.from_template(audience_U3_3_template)

    # U3-3 "responce" template
    responce_U3_3_template = """
    #Responce:
    請依據上方#Objective的參考文獻、並參考上方提供的 Persona，並基於上方提供的 Touchpoints 五個階段，依據各階段的情緒體驗，分別給一個「分數 (emotion score)」：positive emotion 最高給2分，negative emotion 最低給-2分。無須任何描述「只須給我分數即可」。

    #Start:
    如果您明白了，請開始進行分析
    """
    responce_U3_3_prompt = PromptTemplate.from_template(responce_U3_3_template)

    # U3-3 arragement
    input_U3_3_prompts = [
        ("context", context_U3_3_prompt),
        ("objective", objective_U3_3_prompt),
        ("style", style_U3_3_prompt),
        ("tone", tone_U3_3_prompt),
        ("audience", audience_U3_3_prompt),
        ("responce", responce_U3_3_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_U3_3_prompt, pipeline_prompts=input_U3_3_prompts
    )

    # final U3-3 prompt
    U3_3_prompt = pipeline_prompt.format(
            behavior = behavior,
            profession = profession,
            task = task,
            U3_1_result = U3_1_result,
            U3_2_result = U3_2_result,
            U3_3_json_sample = U3_3_json_sample
            )

    print(U3_3_prompt)

    U3_3_result = model.predict(
        text=U3_3_prompt)

    print(U3_3_result)
    return U3_3_result

def M3_4(query):
    params = get_keyword(query)
    behavior = params['behavior']
    profession = params['profession']
    task = params['task']

    # 參數定義
    U3_4_json_sample = """
    {
    "opportunities_and_painpoints": [
        {
        "touchpoint": "個人興趣探索",
        "painpoint": "使用者在探索自行車運動時，可能因資訊來源繁多而感到困惑，尤其是初學者，難以選擇最適合自己的自行車類型或品牌，這增加了入門門檻。",
        "opportunity": "開發一個整合式的資訊平台或推薦系統，根據使用者的健康狀況、騎行目標和個人需求，提供具體的自行車選擇建議和教育資源，幫助使用者輕鬆入門。"
        },
        {
        "touchpoint": "病患診療",
        "painpoint": "醫療專業人員可能對推薦自行車運動存在顧慮，擔心產品是否能安全有效地支持病患，特別是有特定健康問題的患者，如心血管疾病。",
        "opportunity": "設計專門針對健康問題的自行車產品，並提供臨床認證和醫療數據支持，確保醫療人員能夠放心推薦，提升病患的康復效果。"
        },
        {
        "touchpoint": "社區健康推廣活動",
        "painpoint": "社區活動的參與度可能受到地理範圍和宣傳渠道的限制，難以吸引更廣泛的受眾，影響推廣效果。",
        "opportunity": "與地方政府和企業合作，擴大推廣渠道，如利用社群媒體和當地媒體進行廣告宣傳，並舉辦大型活動如全市騎行比賽，來吸引更多潛在使用者參與。"
        },
        {
        "touchpoint": "合作夥伴關係建立",
        "painpoint": "在與不同類型的合作夥伴合作時，可能面臨目標和利益不一致的挑戰，例如自行車店更關注銷售，而健康組織重視健康教育，這會導致資源分配不均和推廣效率低下。",
        "opportunity": "制定清晰的合作框架和共同目標，確保各方在推動產品和健康理念時能夠協同工作，並通過定期溝通機制來解決潛在衝突。"
        },
        {
        "touchpoint": "反饋與持續改進",
        "painpoint": "收集的使用者回饋可能偏向表層意見，例如外觀設計或基本功能，而忽略了對深層次使用體驗的探討，如長期使用的舒適度和耐用性，這限制了產品的深度改進。",
        "opportunity": "建立一個全面的回饋收集和分析系統，定期進行深度訪談和使用者調查，聚焦於實際使用中的關鍵問題，以此為基礎進行精準的產品優化和創新。"
        }
    ]
    }
    """

    # U3-4 full costar template
    full_U3_4_template = """
    {context}
    {objective}
    {style}
    {tone}
    {audience}
    {responce}
    """
    full_U3_4_prompt = PromptTemplate.from_template(full_U3_4_template)

    # U3-4 "contex" template
    context_U3_4_template = """
    # Context
    I want you to act as a UX researcher. I will provide some details about 'How to create a customer journey map',
    and it will be your job to use customer journey mapping to gain a comprehensive understanding of the customer experience. My first request is
    'I need a best practices for creating user journeys based on user personas and touchpoints.'
    """
    context_U3_4_prompt = PromptTemplate.from_template(context_U3_4_template)

    # U3-4 "objective" template
    objective_U3_4_template = """
    # Objective:
    你需要參考下方的機會點參考文獻，並基於提供的Persona、Touchpoints以及Interaction，以 [{profession}, {task}] 為主題條列出「五個」「人物可能的痛點與機會點」。每個條列內容需包含Persona角色之姓名、任務。各別條列不少於150字。

    機會點參考文獻如下：
    Opportunities (along with additional context such as ownership and metrics) are insights gained from mapping; they speak to how the user experience can be optimized. Insights and opportunities help the team draw knowledge from the map:
    - What needs to be done with this knowledge?
    - Who owns what change?
    - Where are the biggest opportunities?
    - How are we going to measure improvements we implement?

    Persona 如下：
    {U3_1_result}

    Touchpoints 內容如下
    {U3_2_result}
    """
    objective_U3_4_prompt = PromptTemplate.from_template(objective_U3_4_template)

    # U3-4 "style" template
    style_U3_4_template = """
    # Style:
    - 文體具敘事性(narrative)，用字精準無贅字、無拼寫錯誤、清晰易讀。#zh-TW繁體中文為主，專有名詞、品牌等關鍵字可使用英文。
    - 必須將輸出結構化為一組json格式。json是一種聲明性語言，可讓你對文件進行註解和確認。你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！
    範例如下：

    {U3_4_json_sample}
    """
    style_U3_4_prompt = PromptTemplate.from_template(style_U3_4_template)

    # U3-4 "tone" template
    tone_U3_4_template = """
    # Tone:
    在整個過程中保持清晰和有條理的語氣，說明人物可能的痛點與機會點。具備清楚且有方向的分析，提供有價值的見解和建議。
    """
    tone_U3_4_prompt = PromptTemplate.from_template(tone_U3_4_template)

    # U3-4 "audience" template
    audience_U3_4_template = """
    # Audience:
    目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
    """
    audience_U3_4_prompt = PromptTemplate.from_template(audience_U3_4_template)

    # U3-4 "responce" template
    responce_U3_4_template = """
    #Responce:
    請依據上方#Objective中的參考文獻，並參考上方提供的Persona, 基於各階段Touchpoints以及Interactions，以 [{profession}, {task}] 為主題，條列出「五個」「人物可能的痛點與機會點」。每個條列內容需包含Persona角色之姓名、任務。各別條列不少於150字。

    #Start:
    如果您明白了，請開始進行分析"""
    responce_U3_4_prompt = PromptTemplate.from_template(responce_U3_4_template)

    # U3-4 arragement
    input_U3_4_prompts = [
        ("context", context_U3_4_prompt),
        ("objective", objective_U3_4_prompt),
        ("style", style_U3_4_prompt),
        ("tone", tone_U3_4_prompt),
        ("audience", audience_U3_4_prompt),
        ("responce", responce_U3_4_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_U3_4_prompt, pipeline_prompts=input_U3_4_prompts
    )

    # final U3-4 prompt
    U3_4_prompt = pipeline_prompt.format(
            behavior = behavior,
            profession = profession,
            task = task,
            U3_1_result = U3_1_result,
            U3_2_result = U3_2_result,
            U3_4_json_sample = U3_4_json_sample
            )

    print(U3_4_prompt)

    U3_4_result = model.predict(
        text=U3_4_prompt)

    print(U3_4_result)
    return U3_4_result

def M3_5(query):
    params = get_keyword(query)
    behavior = params['behavior']
    profession = params['profession']
    task = params['task']

    # 參數定義
    U3_5_sample = """
    Possible details of the scene of [{profession},{task}] are presented from a third-person perspective, including [{profession}]’s
    partners in performing tasks and the tools and operational details needed to perform [{task}], as well as the scene’s picture
    and style design. Set to 8K UHD, real world, dramatic lighting, 90s, far field of view. elegant, super-detailed, very verbose,
    Style Med = –s 100.
    """

    # U3-5 full costar template
    full_U3_5_template = """
    {Prompt}
    """
    full_U3_5_prompt = PromptTemplate.from_template(full_U3_5_template)

    # U3-5 "prompt" template
    prompt_U3_5_template = """

    # Context
    你是AI繪圖工具的專家，善用 Leonardo AI, Stable Diffusion 這類AI繪圖工具繪製「顧客旅程概念圖」提供設計師創意發想。

    # Objective:
    我將提供你一段人物描述，你將產生一組prompt，可以讓我透過Leonardo AI進行圖片生成。請遵照以下規則，產生"英文"prompt。

    人物描述描述如下:
    {U3_1_result}

    # Style:
    - 越往前排列的單字是越關鍵的主體、主體特徵，越往後的單字越容易被歸納成風格、裝飾、畫面細節。
    - 參考下方範例並依據方案描述內容，prompt 必須和[{profession},{task}]有關。

    參考範例:
    {U3_5_sample}

    # Tone:
    在整個過程中保持清晰和有條理，建立產品設計概念圖提示。具備清楚且有方向的分析，提供有價值的見解和建議。

    # Audience:
    目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。

    #Responce:
    - 請將上方提供的參考文獻，以[{profession},{task}]為主題，生成一組"英文"prompt。
    - 注意：生成一組prompt即可，不需生成多個。

    #Start:
    如果您明白了，請開始進行生成
    """
    prompt_U3_5_prompt = PromptTemplate.from_template(prompt_U3_5_template)

    # U3-5 arragement
    input_U3_5_prompts = [
        ("Prompt", prompt_U3_5_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_U3_5_prompt, pipeline_prompts=input_U3_5_prompts
    )

    # final U3-5 prompt
    U3_5_prompt = pipeline_prompt.format(
            profession = profession,
            task = task,
            U3_5_sample = U3_5_sample,
            U3_1_result = U3_1_result
            )

    print(U3_5_prompt)

    U3_5_result = model.predict(
        text=U3_5_prompt)

    print(U3_5_result)
    return U3_5_result
