這個repo是清理掉過去commit進git的大型檔案後上傳的，架構與原先的一致
其中包含自行車工具廠商需要使用的程式

基本邏輯是使用run code/Data_Processing.py
處理資料

chunks
這個資料夾儲存由Data_Processing.py產生的chunk檔案

streamlit_chatbot
這個資料夾是卡神爺

code
- config.py key和utils.py用到的設定等需要放在這裡
- 23ai.json 建構卡神爺用的資料
- Query.py RAG的主程式

- RAG_CHATBOT.py
- pages 這兩個部分是streamlit介面，主要for自行車工具使用

- multi_vector
- potent.py
- nonpotent.py
- OCIGEN.py 這四個是使用不同拆解方式/不同文件/不同資料庫等做的嘗試

- fa.py fastapi的主程式，用來把query等功能搭建成api

- Data_Processing.py
- utils.py 主要function都放在這邊，內部function還沒有做文件
- agentic_chunker.py Data_Processing過程用以將chunk送到openai做分類和簡短化的檔案


自行車工具用（其中prompt的部分可以參考，但應該不需用到）



------------------------------------------------------------------------------------------

如何使用此程式？

- 環境設置：
    - Python版本：3.8.8
    - SQLite > 3.35 （這是向量資料庫chromadb的要求）
    - 其他的套件，直接 pip install -r requirements.txt 即可
    - 請至code底下的config.json檔，將您的openai key填進去


- 程式說明：
    - demo_Query.py：單純問答，所有使用到的function/class，皆放在QueryAndEvaluation.py
    - demo_QueryAndEvaluation.py : 對不同參數設定下的回答進行評估（evaluation），並將結果儲存成csv和excel檔，以便進行比較和評估，進而選擇最佳的參數。所有使用到的function/class，皆放在QueryAndEvaluation.py
    - Data_Processing.py : 將文字組塊儲存在向量資料庫、儲存成json檔、儲存成lexical _index的相關檔案。目前只有建立文字組塊長度為500的向量資料庫以及相關檔案，若您要測試其他長度，請使用此程式建立相對應的向量資料庫以及檔案



