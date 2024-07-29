config = {
    "OPENAI_API_KEY": "your-key",
    "ANYSCALE_API_KEY": "your-key",
    "MAX_CONTEXT_LENGTHS": {
        "gpt-3.5-turbo-1106": 16385,
        "gpt-4": 8192,
        "gpt-4-0125-preview": 128000,
        "gpt-4-turbo" : 128000
    },
    "CHUNKS" :{
        500: {
            "json_file" : "chunks/ChunkSize500.json",
            "lexial_index_file" : "chunks/ChunkSize500_LexicalIndex",
            "chromadb_collection" : "ChunkSize500Collection"
        }
    },
    "LLM": "gpt-4-turbo",
    "MAX_TOKENS": 4096,
    "SYSTEM_CONTENT": "Answer the query using the context provided. Be succinct. Do not produce any content that does not appear in the context.",
    "ASSISTANT_CONTENT": "",
    "DIRECTORY_PATH": "files",
    "TEMPERATURE": 0.0,
    "EMBEDDING_MODEL_NAME": "text-embedding-3-small",
    "CHUNK_SIZE": 500,
    "CHUNK_OVERLAP": 50,
    "COLLECTION_NAME": "0626_Test_Collection",
    "LEXICAL_SEARCH_K": 3,
    "NUM_CHUNKS": 5,
    "RETRIEVAL_REFERENCE": 'file_name',
    "EVALUATION_LLM": "gpt-4",
    "EVALUATION_SYSTEM_CONTENT": """
    Your job is to rate the quality of our generated answer {generated_answer}
    given a query {query} and a reference answer {reference_answer}.
    Your score has to be between 1 and 5.
    You must return your response in a line with only the score.
    Do not return answers in any other format.
    On a separate line provide your reasoning for the score as well.
    Note : your reasoning should be traditional Chinese, rather than English.
    """,
    "FILE_ID" : {
        "電動輔助自行車及微型電動二輪車型式安全審驗管理辦法_統整.pdf" : "RoadTrafficSafetyRules",
        "產業設計創新趨勢與指引_全文.pdf" : "IndustrialDesignInnovationTrendsAndGuidelines",
        "107民眾日常使用運具狀況.pdf" : "Year107PeopleDailyUseofTransportationVehicles",
        "109年民眾日常使用運具狀況調查.pdf" : "Year109PeopleDailyUseofTransportationVehicles",
        "2022_2023產業技術白皮書.pdf" : "Year2022_2023IndustrialTechnologyWhitePaper",
        "2022年重要產業技術-智慧感測與控制-全文下載-+.pdf" : "Year2022ImportantIndustrialTechnologiesInSmartSensingAndControl",
        "2022年重要產業技術-運輸能源智慧化-全文下載-+.pdf" : "Year2022ImportantIndustrialTechnologiesInIntelligentTransportationEnergy",
        "2022年重要產業技術-擴增實境_混合實境-全文下載-+.pdf" : "Year2022ImportantIndustrialTechnologiesInAugmentedReality_MixedReality",
        "2023_2024產業技術白皮書.pdf" : "Year2023_2024IndustrialTechnologyWhitePaper",
        "公共運輸發展成果檢討及推動策略研析.pdf" : "ReviewOfPublicTransportDevelopmentResultsAndAnalysisOfPromotionStrategies",
        "日本虛實整合科技發展之推動政策與產業應用.pdf" : "PromotingPoliciesAndIndustrialApplicationsOfVirtualAndRealIntegratedTechnologyDevelopmentInJapan",
        "以人為本的綠色交通－自行車導向的阿姆斯特丹城市規劃（上）.pdf" : "People-centeredGreenTransportation-Bicycle-OrientedUrbanPlanningOfAmsterdam(Part1)",
        "以人為本的綠色交通－自行車導向的阿姆斯特丹城市規劃（下）.pdf" : "People-centeredGreenTransportation-Bicycle-OrientedUrbanPlanningOfAmsterdam(Part2)",
        "未來車站報告_ARUP.pdf" : "FutureStationReport_ARUP",
        "全球自行車市場熱門議題暨代表廠商與技術.pdf" : "HotTopicsInTheGlobalBicycleMarketAndRepresentativeManufacturersAndTechnologies.",
        "全球自行車與電動自行車市場趨勢.pdf" : "GlobalBicycleAndElectricBicycleMarketTrends",
        "我國民眾外出旅次運具轉乘分析.pdf" : "AnalysisOfTransportVehicleTransfersforTaiwanesePeopleOutboundTravel",
        "科學與文明- 運輸科技的演進.pdf" : "ScienceAndCivilization-TheEvolutionOfTransportationTechnology",
        "旅運者運具選擇行為異質性之研析：混合羅吉特模式之應用.pdf" : "AnalysisOfTheHeterogeneityOfTravelersVehicleChoiceBehavior:ApplicationOfTheMixedLogitModel",
        "高齡者旅運行為之初探.pdf" : "PreliminaryStudyOnTheTravelBehaviorOfTheElderly",
        "從2023台北國際自行車展看指標廠商關鍵技術布局.pdf" : "LookingAtTheKeyTechnologyLayoutOfIndicatorManufacturersFromThe2023TaipeiInternationalBicycleExhibition",
        "溫室氣體排放量盤查作業指引(2022_05)-final.pdf" : "May2022GreenhouseGasEmissionsInventoryOperationGuidelines",
        "歐盟CBAM指引.pdf" : "EUCBAMGuidelines",
        "Arup Tomorrows Public Transport System.pdf" : "ArupTomorrowsPublicTransportSystem",
        "Arup Urban Transport Group Equitable Mobility.pdf" : "ArupUrbanTransportGroupEquitableMobility",
        "Arup-Future-Mobility-Hubs.pdf" : "Arup-Future-Mobility-Hubs",
        "Cities Alive_Towards a walking world_lowres.pdf" : "CitiesAlive_TowardsAWalkingWorld_lowres",
        "Driverless_Futures_Publication.pdf" : "Driverless_Futures_Publication",
        "Future mobility for rural communities.pdf" : "FutureMobilityForRuralCommunities",
        "Future transport asia pacific main report.pdf" : "FutureTransportAsiaPacificMainReport",
        "Future_of_Tourism_after_Corona_-_Copenhagen_Institute_for_Futures_Studies_for_ANTOR_-_April_2020_.pdf" : "April2020FutureOfTourismAfterCoronaCopenhagenInstituteForFuturesStudiesForANTOR",
        "McKinsey-Tech-Trends-Outlook-2022-Mobility.pdf" : "McKinsey-Tech-Trends-Outlook-2022-Mobility",
        "Moving beyond financial value.pdf" : "MovingBeyondFinancialValue" ,
        "Rethinking Urban Mobility Report 2018.pdf" : "Year2018RethinkingUrbanMobilityReport"
    }
}
