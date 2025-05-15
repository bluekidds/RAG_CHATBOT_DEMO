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

def M3M4(query, needs, subq, origin):
    start_time = time.time()
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
    behavior = params['behavior']
    profession = params['profession']
    task = params['task']

    U0_result = ''
    U3_1_result = '{}'
    U3_2_result = '{}'
    U3_3_result = '{}'
    U3_4_result = '{}'
    U3_5_result = '{}'
    U4_1_result = '{}'
    U4_2_result = '{}'
    U4_3_result = '{}'
    U4_4_result = '{}'

    if ('title' in needs):
        # 參數定義
        U0_json_sample = """
        {
        'project_title':'',
        }
        """

        # U0 full template
        full_U0_template = """
        {task}
        """
        full_U0_prompt = PromptTemplate.from_template(full_U0_template)

        task_U0_template = """
        # task
        請根據下方提供的行為及職業，對其摘要並給予一個不超過十個字的命名。

        行為：{behavior}
        職業：{profession}
        任務：{task}

        刪除任何無關緊要的文本，無關緊要的文本示例："嗯" 修正任何明顯的拼寫錯誤。

        你必須將輸出結構化為一組json格式。
        並只能以'project_title'作為json的註解,不要自行發散出其他的名稱

        json是一種聲明性語言，可讓你對文件進行註解和確認。
        你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全匹配！
        範例如下：

        {U0_json_sample}

        如果您明白了，請開始進行分析
        """
        task_U0_prompt = PromptTemplate.from_template(task_U0_template)

        # U0 arragement
        input_U0_prompts = [
            ("task", task_U0_prompt),
        ]
        pipeline_prompt = PipelinePromptTemplate(
            final_prompt=full_U0_prompt, pipeline_prompts=input_U0_prompts
        )

        # final U0 prompt
        U0_prompt = pipeline_prompt.format(
                behavior= behavior,
                profession= profession,
                task = task,
                U0_json_sample = U0_json_sample
                )

        print(U0_prompt)

        U0_result = model.predict(
            text=U0_prompt)

        jsoned = json.loads(U0_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\""))
        print('json result of title')
        print(jsoned)
        U0_result = jsoned['project_title']

        print(U0_result)
    U0_time = time.time()
    if 'M3_1' in needs:
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
    U3_1_time = time.time()
    if 'M3_2' in needs:
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
    U3_2_time = time.time()
    if 'M3_3' in needs:
        # 參數定義
        U3_3_json_sample = """
        {
        "emotions": [
            {
            "touchpoint": "個人興趣探索",
            "score": 2
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
        {audience&tone}
        {responce}
        """
        full_U3_3_prompt = PromptTemplate.from_template(full_U3_3_template)

        # U3-3 "contex" template
        context_U3_3_template = """
        # Context
        I want you to act as a UX researcher and Service Designer. It will be your job to gain a comprehensive understanding of the customer experience. I need a best practices for creating user journey map to guide design decisions.
        """
        context_U3_3_prompt = PromptTemplate.from_template(context_U3_3_template)

        # U3-3 "objective" template
        objective_U3_3_template = """
        # Objective:
        你的任務是分析想要完成[{task}]且有[{behavior}]行為特徵的[{profession}]，此角色<persona>在體驗產品/服務過程中所經歷的<touchpoints>，分別有哪些情緒分數 (emotion score)？請根據<emotion-info>定義，生成角色的情緒分數。

        <persona>
        {U3_1_result}
        </persona>

        <touchpoints>
        {U3_2_result}
        </touchpoints>

        <emotion-info>
        | key        | #zh-TW     | definition                                                   |
        | ---------- | ---------- | ------------------------------------------------------------ |
        | touchpoint | 對照事件點 | 事件點名稱                                                   |
        | score      | 情緒分數   | 不同接觸點會有不同情緒分數，請根據每個 touchpoint 底下的 action 敘述，給予 ranging from 1 (negative emotion) to 5 (positive emotion)。分數要合理，需要表現出情緒起伏，不能每個階段分數都很高看不出差異。<br />Emotions are plotted as a single line across the journey phases, literally signaling the emotional “ups” and “downs” of the experience. Think of this line as a contextual layer of emotion that tells us where the user is delighted versus frustrated.<br />The best way to measure emotion is through a combination of subjective self-report measures, physiological activation measures, analysis of nonverbal expressive behavior, and behavioral tasks. These four components provide a comprehensive assessment of emotions, including subjective feelings, physiological responses, facial expressions, and motivation. Self-report questionnaires allow individuals to report their own emotional experiences, while physiological measures capture changes associated with emotions. Analysis of nonverbal behavior focuses on facial expressions and other cues, while behavioral tasks assess how emotions influence behavior.<br />Emotion analysis, like sentiment analysis, aims to detect the affective aspects of text. However, emotion analysis goes beyond identifying positive, negative, or neutral sentiments, focusing instead on specific psychological reactions such as happiness, sadness, or fear. While emotion detection and prediction are considered shallow analyses, deeper insights are gained by exploring the causes and consequences of emotions. Traditional psychology views the cause of emotion as an event or object that triggers a corresponding emotional response. Recent research suggests that a clause-level analysis is more effective for identifying the reasons behind emotions, leading to more precise emotion-cause pair extraction. |
        </emotion-info>
        """
        objective_U3_3_prompt = PromptTemplate.from_template(objective_U3_3_template)

        # U3-3 "style" template
        style_U3_3_template = """
        # Style:
        - 用字精準無贅字、無拼寫錯誤、條理清晰易讀。
        - 寫作風格如大型研究機構 nngroup, frog design, BCG，撰寫 distill, narrative, insightful 的使用者分析研究洞察。
        """
        style_U3_3_prompt = PromptTemplate.from_template(style_U3_3_template)

        # U3-3 "audience&tone" template
        audience_tone_U3_3_template = """
        # Audience&Tone:
        - 目標讀者是公司內部的Product Designer、Researcher、Project Manager，他們需要蒐集使用產品的回饋、研究不同用戶旅程的切入機會、撰寫設計提案。
        - 讀者體期待瞭解使用者在5個不同階段的接觸點體驗到的情緒變化，以有效轉化產品開發方向的實用建議，找出未被滿足的痛點及需求，提供有價值的見解和建議。
        """
        audience_tone_U3_3_prompt = PromptTemplate.from_template(audience_tone_U3_3_template)

        # U3-3 "responce" template
        responce_U3_3_template = """
        #Responce:
        - #zh-TW繁體中文進行回答，數字維持整數。
        - 必須根據下方 <example> 將輸出結構化為json格式。定義見上方 <emotion-info>，你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！
        <example>
        {U3_3_json_sample}
        </example>

        #Start:
        根據上述要求，給我該角色情緒分數，使用 <example> json 格式。分析完就停止，不要給我結論。
        如果您明白了，請開始執行
        """
        responce_U3_3_prompt = PromptTemplate.from_template(responce_U3_3_template)

        # U3-3 arragement
        input_U3_3_prompts = [
            ("context", context_U3_3_prompt),
            ("objective", objective_U3_3_prompt),
            ("style", style_U3_3_prompt),
            ("audience&tone", audience_tone_U3_3_prompt),
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
    U3_3_time = time.time()
    if 'M3_4' in needs:
        # 參數定義
        U3_4_json_sample = """
        {
        "opportunities_and_painpoints": [
            {
            "touchpoint": "個人興趣探索",
            "painpoint_topic":"入門門檻高"
            "painpoint": "使用者在探索自行車運動時，可能因資訊來源繁多而感到困惑，尤其是初學者，難以選擇最適合自己的自行車類型或品牌，這增加了入門門檻。",
            "opportunity_topic":"入門平台建立"
            "opportunity": "開發一個整合式的資訊平台或推薦系統，根據使用者的健康狀況、騎行目標和個人需求，提供具體的自行車選擇建議和教育資源，幫助使用者輕鬆入門。"
            },
            {
            "touchpoint": "病患診療",
            "painpoint_topic":"特定疾病限制"
            "painpoint": "醫療專業人員可能對推薦自行車運動存在顧慮，擔心產品是否能安全有效地支持病患，特別是有特定健康問題的患者，如心血管疾病。",
            "opportunity_topic":"專門產品設計"
            "opportunity": "設計專門針對健康問題的自行車產品，並提供臨床認證和醫療數據支持，確保醫療人員能夠放心推薦，提升病患的康復效果。"
            },
            {
            "touchpoint": "社區健康推廣活動",
            "painpoint_topic":"社區限制"
            "painpoint": "社區活動的參與度可能受到地理範圍和宣傳渠道的限制，難以吸引更廣泛的受眾，影響推廣效果。",
            "opportunity_topic":"在地'產''官'合作"
            "opportunity": "與地方政府和企業合作，擴大推廣渠道，如利用社群媒體和當地媒體進行廣告宣傳，並舉辦大型活動如全市騎行比賽，來吸引更多潛在使用者參與。"
            },
            {
            "touchpoint": "合作夥伴關係建立",
            "painpoint_topic":"跨領域問題"
            "painpoint": "在與不同類型的合作夥伴合作時，可能面臨目標和利益不一致的挑戰，例如自行車店更關注銷售，而健康組織重視健康教育，這會導致資源分配不均和推廣效率低下。",
            "opportunity_topic":"協同機制建立"
            "opportunity": "制定清晰的合作框架和共同目標，確保各方在推動產品和健康理念時能夠協同工作，並通過定期溝通機制來解決潛在衝突。"
            },
            {
            "touchpoint": "反饋與持續改進",
            "painpoint_topic":"使用者回饋收集"
            "painpoint": "收集的使用者回饋可能偏向表層意見，例如外觀設計或基本功能，而忽略了對深層次使用體驗的探討，如長期使用的舒適度和耐用性，這限制了產品的深度改進。",
            "opportunity_topic":"系統建立"
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
        {audience&tone}
        {responce}
        """
        full_U3_4_prompt = PromptTemplate.from_template(full_U3_4_template)

        # U3-4 "contex" template
        context_U3_4_template = """
        # Context
        I want you to act as a UX researcher and Service Designer. It will be your job to gain a comprehensive understanding of the customer experience. I need a best practices for creating user journey map to guide design decisions.
        """
        context_U3_4_prompt = PromptTemplate.from_template(context_U3_4_template)

        # U3-4 "objective" template
        objective_U3_4_template = """
        # Objective:
        你的任務是分析想要完成[{task}]且有[{behavior}]行為特徵的[{profession}]，此角色<persona>在體驗產品/服務過程中所經歷的<touchpoints>，分別有哪些痛點與機會點 (pain-points and opportunities)？請根據<oppo-info>定義，生成角色的痛點與機會點。

        <persona>
        {U3_1_result}
        </persona>

        <touchpoints>
        {U3_2_result}
        </touchpoints>

        <oppo-info>
        | key         | #zh-TW     | definition                                                   |
        | ----------- | ---------- | ------------------------------------------------------------ |
        | touchpoint  | 對照事件點 | 事件點名稱                                                   |
        | painpoint_topic  | 基於痛點之摘要標題 ｜
        | painpoint   | 痛點       | Pain points are the challenges, frustrations, or obstacles users encounter during their journey. These could include difficulty finding information, complex navigation, slow load times, or unclear instructions. Identifying pain points helps in pinpointing areas that need improvement.<br />Identify the low points or points of friction. See where the journey reaches its lowest point and compare it to other low points in the journey. |
        | opportunity_topic  | 基於機會點之摘要標題 ｜
        | opportunity | 機會點     | Opportunities are insights gained from mapping; they speak to how the user experience can be optimized. 尋找機會點可透過以下方法：<br />1. Look for points in the journey where expectations are not met. To identify these instances, first reflect on who the persona is（習慣[{behavior}]的[{profession}]）. Ask yourself; what is important to this persona（他的任務是[{task}]）, where did she come from before this journey, what has she seen and what does she know already?<br />2. Identify any unnecessary touchpoints or interactions. Look for logical ways to optimize the process to reduce total interaction cost. That may mean removing an existing step that is no longer needed or adding something to the experience that bring efficiency to the overall journey.<br />3. Pinpoint high-friction channel transitions. Many journeys take place across devices or channels. A lot of times the journey breaks down and friction appears when users change channels. Think outside of the box: rather than forcing users to work hard, build a bridge for them to get to the other side easily.<br />4. Identify high points or points where expectations are met or exceeded. Look at the high points in the journey — the interactions that users are happy with. Where do they express positive thoughts and emotions? These insights are also valuable. You may be able to amplify them or recreate similar experiences elsewhere in the journey. |
        </oppo-info>
        """
        objective_U3_4_prompt = PromptTemplate.from_template(objective_U3_4_template)

        # U3-4 "style" template
        style_U3_4_template = """
        # Style:
        - 用字精準無贅字、無拼寫錯誤、條理清晰易讀。
        - 寫作風格如大型研究機構 nngroup, frog design, BCG，撰寫 distill, narrative, insightful 的使用者分析研究洞察。
        """
        style_U3_4_prompt = PromptTemplate.from_template(style_U3_4_template)


        # U3-4 "audience&tone" template
        audience_tone_U3_4_template = """
        # Audience&Tone:
        - 目標讀者是公司內部的Product Designer、Researcher、Project Manager，他們需要蒐集使用產品的回饋、研究不同用戶旅程的切入機會、撰寫設計提案。
        - 讀者體期待瞭解使用者在5個不同階段的接觸點有哪些行為及需求，以有效找出未被滿足的痛點及產品/服務優化的機會，提供有價值的見解和建議。
        """
        audience_tone_U3_4_prompt = PromptTemplate.from_template(audience_tone_U3_4_template)

        # U3-4 "responce" template
        responce_U3_4_template = """
        #Responce:
        - #zh-TW繁體中文進行回答，專有名詞、品牌等關鍵字可使用英文。
        - 必須根據下方 <example> 將輸出結構化為json格式。定義見上方 <oppo-info>，你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！
        <example>
        {U3_4_json_sample}
        </example>

        #Start:
        根據上述要求，給我該角色痛點與機會點，使用 <example> json 格式。分析完就停止，不要給我結論。
        如果您明白了，請開始執行
        """
        responce_U3_4_prompt = PromptTemplate.from_template(responce_U3_4_template)

        # U3-4 arragement
        input_U3_4_prompts = [
            ("context", context_U3_4_prompt),
            ("objective", objective_U3_4_prompt),
            ("style", style_U3_4_prompt),
            ("audience&tone", audience_tone_U3_4_prompt),
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
    U3_4_time = time.time()
    if 'M3_5' in needs:
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
        你是AI繪圖工具的專家，善用 Dalle 3, Leonardo AI, Stable Diffusion 這類AI繪圖工具繪製「顧客旅程概念圖」提供設計師創意發想。

        # Objective:
        我將提供你一段人物描述以及接觸點描述，你將產生一組prompt，可以讓我透過Dalle 3進行與下列描述完全相符的圖片生成。

        人物描述如下:
        {U3_1_result}

        接觸點描述:
        {U3_2_result}

        # Style:
        - 越往前排列的單字是越關鍵的主體、主體特徵，越往後的單字越容易被歸納成風格、裝飾、畫面細節。
        - 請遵照規則，產生"英文"prompt。
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
                U3_1_result = U3_1_result,
                U3_2_result = U3_2_result
                )

        print(U3_5_prompt)

        U3_5_result = model.predict(
            text=U3_5_prompt)

        print(U3_5_result)

        # pic = dalle3(U3_5_result)

        # U3_5_result = pic

    U3_5_time = time.time()
    if 'M4_1' in needs:
        # 參數定義
        U4_1_json_sample = """
        {
        "name": "李大衛",
        "male_or_female": "男性",
        "age": "42",
        "occupation": "物理治療師兼自行車運動愛好者",
        "family_status": "已婚，有兩個孩子",
        "frequently_used_items": ["自行車", "用於病患紀錄的平板", "心率監測器", "騎行應用程式", "物理治療設備"],
        "motivations_and_behaviors":"李大衛致力於通過騎自行車來促進更健康的生活方式，這源於他對預防性護理和規律運動的治療效果的深刻信念。他的動機來自於親眼目睹心血管健康如何通過持續的運動得到改善，因此，他熱衷於讓自行車成為患者和更廣泛社區的更安全、更可行的選擇。大衛每天都在將他的職業專長與他對自行車運動的熱愛結合起來，積極地向他人宣傳自行車運動的健康益處，同時解決安全問題。他的目標是將騎自行車納入主流的預防性護理，確保它不僅被視為一項運動，更是一種有助於整體健康的日常活動。"
        }
        """

        # U4-1 full costar template
        full_U4_1_template = """
        {context}
        {objective}
        {style}
        {audience&tone}
        {responce}
        """
        full_U4_1_prompt = PromptTemplate.from_template(full_U4_1_template)

        # U4-1 "contex" template
        context_U4_1_template = """
        # Context
        I want you to act as a UX researcher and Service Designer. It will be your job to gain a comprehensive understanding of the customer experience. I need a best practices for creating user persona to guide design decisions.
        """
        context_U4_1_prompt = PromptTemplate.from_template(context_U4_1_template)

        # U4-1 "objective" template
        objective_U4_1_template = """
        # Objective:
        你的任務是分析想要完成[{task}]且有[behavior]行為特徵的[profession]，想像自己站在使用者的立場，了解他/她最真實的感受、可能的行為及反應。然後從「產品/服務設計師」的角度，描繪對你而言具有洞察的 persona 會有怎樣的人物基本介紹？請根據<persona-info>定義，生成角色的人物描述。

        A persona is depicted as a specific person but is not a real individual; rather, it is synthesized from observations of many people. Each persona represents a significant portion of people in the real world and enables the designer to focus on a manageable and memorable cast of characters. 人物誌的目的是幫助團隊描述目標受眾的 goals & behavior patterns，從而在產品/服務設計和行銷策略中做出更明智的決策，提昇使用者滿意度。專業的人物誌需要達到：
        - Build empathy. 站在使用者的角度察覺目標及需求
        - Develop focus. 鎖定目標族群
        - Communicate and form consensus. 凝聚產品開發的共識
        - Make and defend decisions. 人物誌作為工具幫助收斂產品設計決策
        - Measure effectiveness. 模擬不同角色使用產品的狀況是否能有效滿足他的需求

        Persona creation step by step:
        1. Market segmentation: It's all about grouping users into segments that share similar characteristics. The segmentation can be geographical, demographical, behavioural, etc.
        2. Persona creation: Alan Cooper introduced a few types of personas in his book《The Inmates Are Running the Asylum》:
        >- Primary personas: The main focus of the design. They represent the primary target audience. Your product should meet the needs of the primary persona.
        >- Secondary personas: Users who have additional needs not covered by the primary personas. They are important, but their needs should not be addressed at the expense of the primary persona.
        >- Customer personas: Represent those who make the purchasing decision. Important in contexts where the user and the buyer are different.
        >- Supplemental personas: Users who might interact with the product in a more limited way. They help to understand less important use cases.
        3. Jobs to Be Done (JTBD): JTBD is a powerful tool that focuses on understanding the underlying motivations and needs driving user behavior. It shifts the perspective from WHO the users are to WHAT they are trying to accomplish. Define the fundamental tasks that users aim to achieve with a product and focus on the context and motivations behind why users use a product or service.
        >- Understand context: Analyze the situations and contexts in which users perform these jobs. Consider factors like time, location, and circumstances that influence their behavior.
        >- Define success criteria: Identify what users consider a successful outcome for each job. Understand the criteria they use to judge the effectiveness of the solution.

        Well-crafted personas include details about user goals that are similar to those in jobs-to-be-done descriptions, but are enriched with attitudinal, contextual, behavioral, and personal data that can provide a well-rounded set of considerations to guide UX designers and product teams in decision making.

        <persona-info>
        | key                       | #zh-TW     | definition                                                   |
        | ------------------------- | ---------- | ------------------------------------------------------------ |
        | name                      | 角色名稱   | 角色名稱                                                     |
        | age                       | 年齡       | 年齡                                                         |
        | occupation                | 職業       | 職業                                                         |
        | family_status             | 家庭狀況   | 推測角色的家庭狀況，單身、有伴侶、結婚、有小孩、獨居…等      |
        | frequently_used_items     | 常用物件   | 從使用者的[behavior]行為、職業以及[{task}]，推測他每天經常使用的物件，不侷限於實體產品、數位產品、軟體服務 |
        | motivations_and_behaviors | 動機與行為 | Motivation: the reason why the user wants to solve the above problem, comes from a deeper, more personal level. The core motivations, the reasons, usually require a higher level of knowledge about the user and real insights into the user’s mind.<br />參考 Indi Young 的《Mental Models: Aligning Design Strategy with Human Behavior》書中 Creating Mental Model Diagrams、Define Task-Based Audience Segment 章節講解的 Task-Based Mental Models 方法：聚焦於分析用戶在完成特定任務時的動機、行為、思維方式。<br />請以第三人稱的方式，中文200字，描寫角色的動機與行為，包含他平日或假日可能的生活場景、與家庭或不同利害關係人的互動、這個互動行為會用到的物件、資訊來源以及背後的動機和想法。 |
        </persona-info>
        """
        objective_U4_1_prompt = PromptTemplate.from_template(objective_U4_1_template)

        # U4-1 "style" template
        style_U4_1_template = """
        # Style:
        - 用字精準無贅字、無拼寫錯誤、條理清晰易讀。
        - 寫作風格如大型研究機構 nngroup, frog design, BCG，撰寫 distill, narrative, insightful 的使用者分析研究洞察。
        """
        style_U4_1_prompt = PromptTemplate.from_template(style_U4_1_template)

        # U4-1 "audience&tone" template
        audience_tone_U4_1_template = """
        # Audience&Tone:
        - 目標讀者是公司內部的Product Designer、Researcher、Project Manager，他們需要蒐集使用產品的回饋、研究不同用戶旅程的切入機會、撰寫設計提案。
        - 讀者體期待瞭解使用者行為的動機與需求，以有效轉化產品開發方向的實用建議與可行動的步驟，提供有價值的見解和建議。
        """
        audience_tone_U4_1_prompt = PromptTemplate.from_template(audience_tone_U4_1_template)

        # U4-1 "responce" template
        responce_U4_1_template = """
        #Responce:
        - #zh-TW繁體中文進行回答，專有名詞、品牌等關鍵字可使用英文。
        - 必須根據下方 <example> 將輸出結構化為json格式。定義見上方 <persona-info>，你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！
        <example>
        {U4_1_json_sample}
        </example>

        #Start:
        根據上述要求，以「習慣[{behavior}]的[{profession}]，任務是[{task}]」為題目給我該角色的人物描述，使用 <example> json 格式。分析完就停止，不要給我結論。
        如果您明白了，請開始執行
        """
        responce_U4_1_prompt = PromptTemplate.from_template(responce_U4_1_template)

        # U4-1 arragement
        input_U4_1_prompts = [
            ("context", context_U4_1_prompt),
            ("objective", objective_U4_1_prompt),
            ("style", style_U4_1_prompt),
            ("audience&tone", audience_tone_U4_1_prompt),
            ("responce", responce_U4_1_prompt),
        ]
        pipeline_prompt = PipelinePromptTemplate(
            final_prompt=full_U4_1_prompt, pipeline_prompts=input_U4_1_prompts
        )

        # final U4-1 prompt
        U4_1_prompt = pipeline_prompt.format(
                task = task,
                behavior = behavior,
                profession = profession,
                U4_1_json_sample = U4_1_json_sample
                )

        print(U4_1_prompt)

        U4_1_result = model.predict(
            text=U4_1_prompt)

        print(U4_1_result)
    U4_1_time = time.time()
    if 'M4_2' in needs:
        # 參數定義
        U4_2_json_sample = """
        {
        "quote": ["當我推薦自行車給患者時，我需要確信它能提升健康而非帶來風險。這款車必須成為他們康復之路上的可靠夥伴。", "自行車對我來說不僅是運動工具，而是推動全民健康的關鍵。我希望每個人都能輕鬆將騎行融入日常生活。"],
        }
        """

        # U4-2 full costar template
        full_U4_2_template = """
        {context}
        {objective}
        {style}
        {tone}
        {audience}
        {responce}
        """
        full_U4_2_prompt = PromptTemplate.from_template(full_U4_2_template)

        # U4-2 "contex" template
        context_U4_2_template = """
        # Context
        I want you to act as a journalist. You will report on breaking news, write behavior stories and opinion pieces,
        develop research techniques for verifying information and uncovering sources, adhere to journalistic ethics, and deliver accurate reporting using your own distinct style.
        """
        context_U4_2_prompt = PromptTemplate.from_template(context_U4_2_template)

        # U4-2 "objective" template
        objective_U4_2_template = """
        # Objective:
        請依據以下方的參考文獻以及人物誌描述，參照給定的人物誌基本資料內容，以[{profession},{behavior}]為主題，
        用第一人稱視角生成一份 [{profession},{behavior}]的人物誌(Persona)引言(quote)。不需解釋也不要結語，""請直接給我quote描述""。

        My first suggestion request is "I need help writing an article about
        [{profession},{behavior}]".


        參考文獻如下：
            #1
            Indi Young的書《Mental Models: Aligning Design Strategy with Human Behavior》參考書中 Define Task-Based Audience Segment 章節講解的方法，架構所謂「Mental Model」，
            先定義使用者，不同於以往的人口研究(demographic)或是心理側寫(psychographic)等方法，Indi Young的Mental Model中使用的是task-based：找出會執行某樣作業的人，不管年齡、職業、性別。

            #2
            使用同理心地圖 (Empathy Map)：一個藉由想像自己站在使用者的立場，試著了解他/她最真實的感受，然後思考自己會怎麼做的工具。我們首先要預設這次討論的目標族群資訊，並想像他們可能的行為、反應。

            #3
            參考Preece , Rogers and Sharp(2002)列舉下列使用者經驗目標：

            1.滿意的(Satisfying)
            2.樂趣的(Enjoyable)
            3.快樂的(Fun)
            4.娛樂的(Entertaining)
            5.有益的(Helpful)
            6.刺激的(Motivating)
            7.美的愉悅感(Aesthetically Pleasing)
            8.創造力的支援(Supportive of Creativity)
            9.實現個人抱負的情感(Emotionally Fulfilling)，

            進一步描述上面建構作業裡面提到的角色原型：包含他每天可能的生活場景、他每天下班回到家可能要執行的任務與痛點及道具需求

            #4
            人物誌（Persona）的功能是為我們的使用者創造「具體形象」。它是一個半虛擬的人物，用來描述使用者的詳細資訊。
            Well-crafted personas include details about user goals that are similar to those in jobs-to-be-done descriptions, but are enriched with attitudinal,
            contextual, behavioral, and personal data that can provide a well-rounded set of considerations to guide UX designers and product teams in decision making.

            人物誌描述如下：
            {U4_1_result}
        """
        objective_U4_2_prompt = PromptTemplate.from_template(objective_U4_2_template)

        # U4-2 "style" template
        style_U4_2_template = """
        # Style:
        Persona描述的書櫥格式規範如下：
        - 用字精準無贅字、無拼寫錯誤、條理清晰易讀。
        - 重要!!! 除了專有名詞、品牌等關鍵字之外，生成內容皆需以"#zh-TW繁體中文"生成
        - 個生成內容字數不得少於50字。
        - 必須將輸出結構化為一組json格式。json是一種聲明性語言，可讓你對文件進行註解和確認。你的輸出將根據提供的提示進行解析，請參照下方範例，確保輸出中的所有欄位完全與範例匹配, 不得生成範例中沒有的項目！
        範例如下：

        {U4_2_json_sample}
        """
        style_U4_2_prompt = PromptTemplate.from_template(style_U4_2_template)

        # U4-2 "tone" template
        tone_U4_2_template = """
        # Tone:
        在整個過程中保持清晰和有條理的語氣，說明人物誌介紹。具備清楚且有方向的分析，提供有價值的見解和建議。
        """
        tone_U4_2_prompt = PromptTemplate.from_template(tone_U4_2_template)

        # U4-2 "audience" template
        audience_U4_2_template = """
        # Audience:
        目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
        """
        audience_U4_2_prompt = PromptTemplate.from_template(audience_U4_2_template)

        # U4-2 "responce" template
        responce_U4_2_template = """
        #Responce:
        請依據以上方的參考文獻，以[{profession},{behavior}]為主題，
        用第一人稱視角生成一份{behavior}的人物誌(Persona)引言(quote)。不需解釋也不要結語，""請直接給我quote描述""。

        #Start:
        如果您明白了，請開始進行分析
        """
        responce_U4_2_prompt = PromptTemplate.from_template(responce_U4_2_template)

        # U4-2 arragement
        input_U4_2_prompts = [
            ("context", context_U4_2_prompt),
            ("objective", objective_U4_2_prompt),
            ("style", style_U4_2_prompt),
            ("tone", tone_U4_2_prompt),
            ("audience", audience_U4_2_prompt),
            ("responce", responce_U4_2_prompt),
        ]
        pipeline_prompt = PipelinePromptTemplate(
            final_prompt=full_U4_2_prompt, pipeline_prompts=input_U4_2_prompts
        )

        # final U4-2 prompt
        U4_2_prompt = pipeline_prompt.format(
                behavior = behavior,
                profession = profession,
                U4_1_result = U4_1_result,
                U4_2_json_sample = U4_2_json_sample
                )

        print(U4_2_prompt)

        U4_2_result = model.predict(
            text=U4_2_prompt)

        print(U4_2_result)
    U4_2_time = time.time()
    if 'M4_3' in needs:
        # 參數定義
        U4_3_json_sample = """
        {
        "characteristics_and_focus": ["熱情", "專注", "身材壯碩", "鴨舌帽", "預防性護理", "健康倡導", "自行車安全"],
        "lifestyle_preferences": ["規律運動：喜歡每天進行規律的自行車騎行或其他形式的體能鍛鍊，保持良好的身體狀態。", "數據驅動決策：偏好使用健康追蹤設備和應用程式，以數據作為改善健康狀況和患者治療計劃的依據。", "社區參與：積極參與社區健康活動，並且喜歡與志同道合的人分享健康生活方式的理念。"],
        "user_end_goal": ["推廣自行車作為健康生活的核心部分：使自行車不僅成為個人的運動選擇，更是廣泛應用於預防性醫療的一部分。", "提升患者的康復效果：通過推薦安全、有效的自行車運動，幫助患者在康復過程中獲得更好的健康結果。", "建立健康社區：與合作夥伴一起推動全民參與的騎行文化，打造一個健康、活力的社區環境。"],
        "painpoints": ["患者安全顧慮：擔心推薦的自行車產品無法滿足特定健康需求，可能帶來潛在的風險。", "資源有限：在推廣自行車運動時，常常受到資金、人力等資源的限制，難以擴大影響力。", "多方協作的挑戰：在與不同的合作夥伴協作時，面臨溝通不暢、目標不一致的問題，影響推廣效率。"],
        "needs": ["醫療認證的自行車產品：需要具備專門設計和醫療認證的自行車產品，以便能安全有效地推薦給不同健康狀況的患者。", "社區推廣資源：需要更多的資源和支持來擴大自行車運動在社區的影響力，增加社會參與度。", "協作工具和機制：需要一個有效的協作平台和清晰的合作機制，以確保與合作夥伴之間的溝通順暢，共同推動健康項目。"]
        }
        """

        # U4-3 full costar template
        full_U4_3_template = """
        {context}
        {objective}
        {style}
        {tone}
        {audience}
        {responce}
        """
        full_U4_3_prompt = PromptTemplate.from_template(full_U4_3_template)

        # U4-3 "contex" template
        context_U4_3_template = """
        # Context
        I want you to act as a journalist. You will report on breaking news, write behavior stories and opinion pieces,
        develop research techniques for verifying information and uncovering sources, adhere to journalistic ethics, and deliver accurate reporting using your own distinct style.
        """
        context_U4_3_prompt = PromptTemplate.from_template(context_U4_3_template)

        # U4-3 "objective" template
        objective_U4_3_template = """
        # Objective:
        請依據以下方的參考文獻以及人物誌描述，參照給定的人物誌基本資料內容，以[{profession},{behavior}]為主題，
        用第一人稱視角生成一份 [{profession},{behavior}]的人物誌(Persona)人物分析。""請直接給我人物分析""。

        My first suggestion request is "I need help writing an article about
        [{profession},{behavior}]".


        參考文獻如下：
            #1
            Indi Young的書《Mental Models: Aligning Design Strategy with Human Behavior》參考書中 Define Task-Based Audience Segment 章節講解的方法，架構所謂「Mental Model」，
            先定義使用者，不同於以往的人口研究(demographic)或是心理側寫(psychographic)等方法，Indi Young的Mental Model中使用的是task-based：找出會執行某樣作業的人，不管年齡、職業、性別。

            #2
            使用同理心地圖 (Empathy Map)：一個藉由想像自己站在使用者的立場，試著了解他/她最真實的感受，然後思考自己會怎麼做的工具。我們首先要預設這次討論的目標族群資訊，並想像他們可能的行為、反應。

            #3
            參考Preece , Rogers and Sharp(2002)列舉下列使用者經驗目標：

            1.滿意的(Satisfying)
            2.樂趣的(Enjoyable)
            3.快樂的(Fun)
            4.娛樂的(Entertaining)
            5.有益的(Helpful)
            6.刺激的(Motivating)
            7.美的愉悅感(Aesthetically Pleasing)
            8.創造力的支援(Supportive of Creativity)
            9.實現個人抱負的情感(Emotionally Fulfilling)，

            進一步描述上面建構作業裡面提到的角色原型：包含他每天可能的生活場景、他每天下班回到家可能要執行的任務與痛點及道具需求

            #4
            人物誌（Persona）的功能是為我們的使用者創造「具體形象」。它是一個半虛擬的人物，用來描述使用者的詳細資訊。
            Well-crafted personas include details about user goals that are similar to those in jobs-to-be-done descriptions, but are enriched with attitudinal,
            contextual, behavioral, and personal data that can provide a well-rounded set of considerations to guide UX designers and product teams in decision making.

            人物誌描述如下：
            {U4_1_result}
        """
        objective_U4_3_prompt = PromptTemplate.from_template(objective_U4_3_template)

        # U4-3 "style" template
        style_U4_3_template = """
        # Style:
        Persona描述的書櫥格式規範如下：
        - 用字精準無贅字、無拼寫錯誤、條理清晰易讀。
        - 重要!!! 除了專有名詞、品牌等關鍵字之外，生成內容皆需以"#zh-TW繁體中文"生成
        - "characteristics_and_focus"為關鍵字，而非一句話，請參考下方範例
        - "lifestyle_preferences"、"user_end_goal"、"painpoints"、"needs"各需生成三項，描述方式請參考下方範例
        - 必須將輸出結構化為一組json格式。json是一種聲明性語言，可讓你對文件進行註解和確認。你的輸出將根據提供的提示進行解析，請參照下方範例，確保輸出中的所有欄位完全與範例匹配, 不得生成範例中沒有的項目！
        範例如下：

        {U4_3_json_sample}
        """
        style_U4_3_prompt = PromptTemplate.from_template(style_U4_3_template)

        # U4-3 "tone" template
        tone_U4_3_template = """
        # Tone:
        在整個過程中保持清晰和有條理的語氣，說明人物誌介紹。具備清楚且有方向的分析，提供有價值的見解和建議。
        """
        tone_U4_3_prompt = PromptTemplate.from_template(tone_U4_3_template)

        # U4-3 "audience" template
        audience_U4_3_template = """
        # Audience:
        目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
        """
        audience_U4_3_prompt = PromptTemplate.from_template(audience_U4_3_template)

        # U4-3 "responce" template
        responce_U4_3_template = """
        #Responce:
        請依據以上方的參考文獻，以[{profession},{behavior}]為主題，
        用第一人稱視角生成一份{behavior}的人物誌(Persona)人物分析。""請直接給我人物分析""。

        #Start:
        如果您明白了，請開始進行分析
        """
        responce_U4_3_prompt = PromptTemplate.from_template(responce_U4_3_template)

        # U4-3 arragement
        input_U4_3_prompts = [
            ("context", context_U4_3_prompt),
            ("objective", objective_U4_3_prompt),
            ("style", style_U4_3_prompt),
            ("tone", tone_U4_3_prompt),
            ("audience", audience_U4_3_prompt),
            ("responce", responce_U4_3_prompt),
        ]
        pipeline_prompt = PipelinePromptTemplate(
            final_prompt=full_U4_3_prompt, pipeline_prompts=input_U4_3_prompts
        )

        # final U4-3 prompt
        U4_3_prompt = pipeline_prompt.format(
                behavior = behavior,
                profession = profession,
                U4_1_result = U4_1_result,
                U4_3_json_sample = U4_3_json_sample
                )

        print(U4_3_prompt)

        U4_3_result = model.predict(
            text=U4_3_prompt)

        print(U4_3_result)
    U4_3_time = time.time()
    if 'M4_4' in needs:
        # 參數定義
        U4_4_sample = """
        [{profession},{behavior}] Showing flexible body movements and scene details of [{profession}] and attitude at work, and enhancing the expression of
        facial emotions, 8K UHD, real world, dramatic lighting, 90s, far field of view.
        """

        # U4-4 full costar template
        full_U4_4_template = """
        {Prompt}
        """
        full_U4_4_prompt = PromptTemplate.from_template(full_U4_4_template)

        # U4-4 "prompt" template
        prompt_U4_4_template = """

        # Context
        你是AI繪圖工具的專家，善用 Dalle 3, Leonardo AI, Stable Diffusion 這類AI繪圖工具繪製「人物誌概念圖」提供設計師創意發想。

        # Objective:
        我將提供你一段人物描述以及個性、偏好、目標、痛點、需求等五點描述，你將產生一組prompt，可以讓我透過Dalle 3進行與下列描述完全相符的圖片生成

        人物描述如下:
        {U4_1_result}

        五點描述:
        {U4_3_result}

        # Style:
        - 越往前排列的單字是越關鍵的主體、主體特徵，越往後的單字越容易被歸納成風格、裝飾、畫面細節。
        - 請遵照以下規則，產生一組"英文"prompt。
        - 參考下方範例並依據方案描述內容，prompt 必須和[{profession},{behavior}] 有關。

        參考範例:
        {U4_4_sample}

        # Tone:
        在整個過程中保持清晰和有條理，建立產品設計概念圖提示。具備清楚且有方向的分析，提供有價值的見解和建議。

        # Audience:
        目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。

        #Responce:
        - 請將上方提供的參考文獻，以[{profession},{behavior}]為主題，生成一組"英文"prompt。
        - 注意：生成一組prompt即可，不需生成多個。

        #Start:
        如果您明白了，請開始進行生成
        """
        prompt_U4_4_prompt = PromptTemplate.from_template(prompt_U4_4_template)

        # U4-4 arragement
        input_U4_4_prompts = [
            ("Prompt", prompt_U4_4_prompt),
        ]
        pipeline_prompt = PipelinePromptTemplate(
            final_prompt=full_U4_4_prompt, pipeline_prompts=input_U4_4_prompts
        )

        # final U4-4 prompt
        U4_4_prompt = pipeline_prompt.format(
                profession = profession,
                behavior = behavior,
                U4_4_sample = U4_4_sample,
                U4_1_result = U4_1_result,
                U4_3_result = U4_3_result
                )

        print(U4_4_prompt)

        U4_4_result = model.predict(
            text=U4_4_prompt)

        print(U4_4_result)

        # pic1 = dalle3(U4_4_result)
        # pic2 = dalle3(U4_4_result)
        # pic3 = dalle3(U4_4_result)
        # pic4 = dalle3(U4_4_result)

        U4_4_result = [U4_4_result, U4_4_result, U4_4_result, U4_4_result]
    U4_4_time = time.time()

    print("======M3M4 TIME======")
    print("======U0 TIME======")
    print(U0_time - start_time)
    print("======U3_1 TIME======")
    print(U3_1_time - U0_time)
    print("======U3_2 TIME======")
    print(U3_2_time - U3_1_time)
    print("======U3_3 TIME======")
    print(U3_3_time - U3_2_time)
    print("======U3_4 TIME======")
    print(U3_4_time - U3_3_time)
    print("======U3_5 TIME======")
    print(U3_5_time - U3_4_time)
    print("======U4_1 TIME======")
    print(U4_1_time - U3_5_time)
    print("======U4_2 TIME======")
    print(U4_2_time - U4_1_time)
    print("======U4_3 TIME======")
    print(U4_3_time - U4_2_time)
    print("======U4_4 TIME======")
    print(U4_4_time - U4_3_time)

    print("======total TIME======")
    print(U4_4_time - start_time)

    return {
        "title": U0_result,
        "customer_journey": {
            "user_info": json.loads(U3_1_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
            "journeys": json.loads(U3_2_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
            "emotions": json.loads(U3_3_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
            "opportunities_and_painpoints": json.loads(U3_4_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
            "U3_5_result": U3_5_result,
        },
        "persona": {
            "user_info": json.loads(U4_1_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
            "quote": json.loads(U4_2_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
            "health_advocate_profile": json.loads(U4_3_result.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")),
            "U4_4_result": U4_4_result,
        }
    }
