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

def M4_1(query):
    params = get_keyword(query)
    behavior = params['behavior']
    profession = params['profession']
    task = params['task']

    # 參數定義
    U4_1_json_sample = """
    {
    "name": "李大衛",
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
    {tone}
    {audience}
    {responce}
    """
    full_U4_1_prompt = PromptTemplate.from_template(full_U4_1_template)

    # U4-1 "contex" template
    context_U4_1_template = """
    # Context
    I want you to act as a journalist. You will report on breaking news, write behavior stories and opinion pieces,
    develop research techniques for verifying information and uncovering sources, adhere to journalistic ethics, and deliver accurate reporting using your own distinct style.
    """
    context_U4_1_prompt = PromptTemplate.from_template(context_U4_1_template)

    # U4-1 "objective" template
    objective_U4_1_template = """
    # Objective:
    請依據以下方的參考文獻，以[{profession},{behavior}]為主題，
    用第一人稱視角生成一份 [{profession},{behavior}]的人物誌(Persona)基本輪廓介紹(short biography)。不需解釋也不要結語，""請直接給我Persona描述""。

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
    """
    objective_U4_1_prompt = PromptTemplate.from_template(objective_U4_1_template)

    # U4-1 "style" template
    style_U4_1_template = """
    # Style:
    Persona描述的書櫥格式規範如下：
    - 用字精準無贅字、無拼寫錯誤、條理清晰易讀。
    - 重要!!! 除了專有名詞、品牌等關鍵字之外，生成內容皆需以"#zh-TW繁體中文"生成
    - "regularly_used_goods" 為3-5個元素,物品名稱即可，不需任何形容詞。
    - 必須將輸出結構化為一組json格式。json是一種聲明性語言，可讓你對文件進行註解和確認。你的輸出將根據提供的提示進行解析，因此請確保輸出中的所有欄位完全與範例匹配！
    範例如下：

    {U4_1_json_sample}
    """
    style_U4_1_prompt = PromptTemplate.from_template(style_U4_1_template)

    # U4-1 "tone" template
    tone_U4_1_template = """
    # Tone:
    在整個過程中保持清晰和有條理的語氣，說明人物誌介紹。具備清楚且有方向的分析，提供有價值的見解和建議。
    """
    tone_U4_1_prompt = PromptTemplate.from_template(tone_U4_1_template)

    # U4-1 "audience" template
    audience_U4_1_template = """
    # Audience:
    目標讀者是公司內部的決策層或高階主管。假設讀者群體期待瞭解市場現況，以有效轉化產品開發方向的實用建議與可行動的步驟。
    """
    audience_U4_1_prompt = PromptTemplate.from_template(audience_U4_1_template)

    # U4-1 "responce" template
    responce_U4_1_template = """
    #Responce:
    請依據以上方的參考文獻，以[{profession},{behavior}]為主題，
    用第一人稱視角生成一份{behavior}的人物誌(Persona)基本輪廓介紹(short biography)。不需解釋也不要結語，""請直接給我Persona描述""。

    #Start:
    如果您明白了，請開始進行分析
    """
    responce_U4_1_prompt = PromptTemplate.from_template(responce_U4_1_template)

    # U4-1 arragement
    input_U4_1_prompts = [
        ("context", context_U4_1_prompt),
        ("objective", objective_U4_1_prompt),
        ("style", style_U4_1_prompt),
        ("tone", tone_U4_1_prompt),
        ("audience", audience_U4_1_prompt),
        ("responce", responce_U4_1_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_U4_1_prompt, pipeline_prompts=input_U4_1_prompts
    )

    # final U4-1 prompt
    U4_1_prompt = pipeline_prompt.format(
            behavior = behavior,
            profession = profession,
            U4_1_json_sample = U4_1_json_sample
            )

    print(U4_1_prompt)

    U4_1_result = model.predict(
        text=U4_1_prompt)

    print(U4_1_result)
    return U4_1_result

def M4_2(query):
    params = get_keyword(query)
    behavior = params['behavior']
    profession = params['profession']
    task = params['task']

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
    return U4_2_result

def M4_3(query):
    params = get_keyword(query)
    behavior = params['behavior']
    profession = params['profession']
    task = params['task']

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
    return U4_3_result

def M4_4(query):
    params = get_keyword(query)
    behavior = params['behavior']
    profession = params['profession']
    task = params['task']

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
    你是AI繪圖工具的專家，善用 Leonardo AI, Stable Diffusion 這類AI繪圖工具繪製「人物誌概念圖」提供設計師創意發想。

    # Objective:
    我將提供你一段人物描述，你將產生一組prompt，可以讓我透過Leonardo AI進行圖片生成。請遵照以下規則，產生一組"英文"prompt。

    人物至描述如下:
    1.
    {U4_1_result}

    2.
    {U4_3_result}

    # Style:
    - 越往前排列的單字是越關鍵的主體、主體特徵，越往後的單字越容易被歸納成風格、裝飾、畫面細節。
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
    return U4_4_result