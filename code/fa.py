from typing import Union, Annotated, Optional

import httpx
from fastapi import FastAPI, Header, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from code.Query import res
from code.M1M2 import M1M2
from code.M3M4 import M3M4
from code.M1 import M1_1,M1_2,M1_3,M1_4,M1_5,M1_6
from code.M2 import M2_1,M2_2,M2_3_a1,M2_3_a2,M2_3_a3,M2_3_a4,M2_3_b1,M2_3_b2,M2_3_b3,M2_3_b4
from code.M3 import M3_1,M3_2,M3_3,M3_4,M3_5
from code.M4 import M4_1,M4_2,M4_3,M4_4
from code.utils import dalle3
from code.BicycleM1 import BM1,BM2,BM3,BM4,BM5,BM6
from code.BicycleM2 import BMC,BMC1,BMC2,BMC3,BMC4,BMC5,BMC2_theme,BMC2_material,BMC2_function,BMC2_patent,BMC3_partial,BMC4_partial

# timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=5.0)
timeout = httpx.Timeout(connect=None,read=None,write=None,pool=None)

class Item(BaseModel):
    q: str
    TA: str
    LO: str
    BT: str
    lang: Optional[str] = None

class Item2(BaseModel):
    TA: str
    LO: str
    BT: str
    lang: Optional[str] = None

class Item3(BaseModel):
    pds_result: str
    sds_result: str
    lang: Optional[str] = None

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prompts = {
    "M1M2": M1M2,
    "M3M4": M3M4,
    "M1_1": M1_1,
    "M1_2": M1_2,
    "M1_3": M1_3,
    "M1_4": M1_4,
    "M1_5": M1_5,
    "M1_6": M1_6,
    "M2_1": M2_1,
    "M2_2": M2_2,
    "M2_3_a1": M2_3_a1,
    "M2_3_a2": M2_3_a2,
    "M2_3_a3": M2_3_a3,
    "M2_3_a4": M2_3_a4,
    "M2_3_b1": M2_3_b1,
    "M2_3_b2": M2_3_b2,
    "M2_3_b3": M2_3_b3,
    "M2_3_b4": M2_3_b4,
    "M3_1": M3_1,
    "M3_2": M3_2,
    "M3_3": M3_3,
    "M3_4": M3_4,
    "M3_5": M3_5,
    "M4_1": M4_1,
    "M4_2": M4_2,
    "M4_3": M4_3,
    "M4_4": M4_4
}

paths = {
    "pestel": {
        # "economic_partial_generation": {},
        "politic": ['M1_1'],
        "economic": ['M1_2'],
        "social": ['M1_3'],
        "technology": ['M1_4'],
        "environment": ['M1_5'],
        "law": ['M1_6'],
        "politic_partial_generation": ['M1_1_R'],
        "economic_partial_generation": ['M1_2_R'],
        "social_partial_generation": ['M1_3_R'],
        "technology_partial_generation": ['M1_4_R'],
        "environment_partial_generation": ['M1_5_R'],
        "law_partial_generation": ['M1_6_R'],
        # "politic_partial_generation": {},
    },
    "persona": {
        "health_advocate_profile": ['M4_3'],
        "quote": ['M4_2'],
        "user_info": ['M4_1', 'M4_4'],
        "all": ['M4_1','M4_2','M4_3','M4_4']
    },
    "customer_journey": {
        "emotions": ['M3_1', 'M3_3'],
        "journeys": ['M3_1', 'M3_2'],
        "opportunities_and_painpoints": ['M3_1', 'M3_2', 'M3_3'],
        "user_info": ['M3_1', 'M3_5'],
        "all": ['M3_1','M3_2','M3_3','M3_4','M3_5']
    },
    "competitive_product_analysis": {
        "competitive_partial_generation": ['M2_1_R'],
        "competitive": ['M2_1'],
        "opportunities-and-challenges": ['M2_1','M2_2'],
        "opportunities-and-challenges_partial_generation": ['M2_1','M2_2'],
        "business_model": ['M2_3_a3','M2_3_b3'],
        "infra_develop": ['M2_3_a4','M2_3_b4'],
        "product_innovation": ['M2_3_a1','M2_3_b1'],
        "service_innovation": ['M2_3_a2','M2_3_b2'],
        "business_model_partial_generation": ['M2_3_a3','M2_3_b3'],
        "infra_develop_partial_generation": ['M2_3_a4','M2_3_b4'],
        "product_innovation_partial_generation": ['M2_3_a1','M2_3_b1'],
        "service_innovation_partial_generation": ['M2_3_a2','M2_3_b2'],
        "all": ['M2_1','M2_2','M2_3_a1','M2_3_b1','M2_3_a2','M2_3_b2','M2_3_a3','M2_3_b3','M2_3_a4','M2_3_b4'],
    }
}

Bicycles = {
    "politic": BM1,
    "economic": BM2,
    "social": BM3,
    "technology": BM4,
    "environment": BM5,
    "law": BM6,
    "competitive": BMC1,
    "designed": BMC2,
    "product_designed": BMC3,
    "service_designed": BMC4,
    "image_prompt": BMC5,
    "theme_style": BMC2_theme,
    "material_recomendation": BMC2_material,
    "function_trend_recomendation": BMC2_function,
    "patent": BMC2_patent,
    "product_designed_trend": BMC3_partial,
    "service_designed_trend": BMC4_partial,
}


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/blob")
def query(name: Union[str, None] = None, size: Union[str, None] = None):
    print('receive item from blob trigger')
    print(name)
    print(size)
    return {"response": ''}

@app.get("/query/{q}")
async def query(q: Union[str, None] = None, token: Annotated[str | None, Header()] = None):
    print("checking token:")
    print(token)
    if (token != "Basic YWRtaW45OTk6YWRtaW4tMTIzNDU2"):
        return {"error": "error"}
    else:
        response = await res(q)    
        return {"response": response, "q": q}

@app.get("/bicycle/{q}")
async def query(q: Union[str, None] = None, token: Annotated[str | None, Header()] = None, TA: Union[str, None] = None, LO: Union[str, None] = None, BT: Union[str, None] = None, lang: Union[str, None] = None):
    print("here")
    print(TA)
    print(LO)
    print(BT)
    print("lang=")
    print(lang)
    if (token != "Basic YWRtaW45OTk6YWRtaW4tMTIzNDU2"):
        return {"error": "error"}
    if (Bicycles[q]):
        response = await Bicycles[q](TA, LO, BT, lang)
        return {"response": response}
    else:
        return {"error": "error"}
    # print("checking token:")
    # print(token)
    # if (token != "Basic YWRtaW45OTk6YWRtaW4tMTIzNDU2"):
    #     return {"error": "error"}
    # else:
    #     response = res(q)    
    #     return {"response": response, "q": q}

@app.post("/bicycle")
async def query(item: Item, token: Annotated[str | None, Header()] = None):
    print(item)
    print(item.TA)
    print(item.LO)
    print(item.BT)
    print(item.lang)
    print(token)
    print(item.q)
    if (token != "Basic YWRtaW45OTk6YWRtaW4tMTIzNDU2"):
        return {"error": "error"}
    if (item.q):
        if '/' in item.q:
            parts = item.q.split('/')
            path, subpath, subq = parts[0], parts[1], parts[2]
            response = await Bicycles[subpath](item.TA, item.LO, item.BT, subq, item.lang)
            print("===sending response===")
            print(response)
            return {"response": response}
        else:
            response = await Bicycles[item.q](item.TA, item.LO, item.BT, item.lang)
            return {"response": response}
    else:
        return {"error": "error"}

@app.post("/bicycle/image_prompt")
async def query(item: Item3, token: Annotated[str | None, Header()] = None):
    if (token != "Basic YWRtaW45OTk6YWRtaW4tMTIzNDU2"):
        return {"error": "error"}
    response = await Bicycles["image_prompt"](item.pds_result, item.sds_result)
    return {"response": response}

# @app.post("/bicycle/M2/{q}")
# async def query(item: Item2, token: Annotated[str | None, Header()] = None, q: Union[str, None] = None):
#     print(item)
#     print(item.TA)
#     print(item.LO)
#     print(item.BT)
#     print(token)
#     if (token != "Basic YWRtaW45OTk6YWRtaW4tMTIzNDU2"):
#         return {"error": "error"}
#     if (Bicycles[q]):
#         response = await Bicycles[q](item.TA, item.LO, item.BT)
#         return {"response": response}
#     else:
#         return {"error": "error"}

@app.post("/bicycle/M2")
async def query(item: Item2, token: Annotated[str | None, Header()] = None):
    print(item)
    print(item.TA)
    print(item.LO)
    print(item.BT)
    print(item.lang)
    print(token)
    if token != "Basic YWRtaW45OTk6YWRtaW4tMTIzNDU2":
        raise HTTPException(status_code=401, detail="Unauthorized")
    # if (token != "Basic YWRtaW45OTk6YWRtaW4tMTIzNDU2"):
    #     return {"error": "error"}
    payload = {
        "TA": item.TA,
        "LO": item.LO,
        "BT": item.BT,
        "lang": item.lang
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(
                "https://v2.ds.punwave.com/draft/bicycle/M2",
                headers={
                    "token": "Bearer ZG9ub3RicmVha3Rocm91Z2hpdHNub3RhbGxvd2Vk",
                    "Content-Type": "application/json"
                },
                json=payload,
            )
            print('===new version call===')
            resp.raise_for_status()
        except httpx.RequestError as e:
            print(type(e), repr(e))
            if e.__cause__:
                print('底層錯誤：', repr(e.__cause__))
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {e}")
        except httpx.HTTPStatusError as e:
            # 上游回傳非 2xx
            raise HTTPException(status_code=e.response.status_code,
                                detail=f"Upstream error: {e.response.text}")
    # response = await BMC(item.TA, item.LO, item.BT)
    # return {"response": response}
    return {"response": resp.json()}

@app.get("/dalle3/{q}")
async def query(q: Union[str, None] = None, origin: Union[str, None] = None):
    response = await dalle3(q, origin)
    return {"response": response, "q": q}

@app.get("/n1/{q}")
def query(q: Union[str, None] = None, origin: Union[str, None] = None):
    if q is None:
        print('q is not set')
        return { "error": "q is not set" }
    response = M1M2(q, ['title'], '', origin)
    return {"response": response['title'], "q": q}

@app.get("/n2/{q}")
def query(q: Union[str, None] = None, origin: Union[str, None] = None):
    if q is None:
        print('q is not set')
        return { "error": "q is not set" }
    response = M3M4(q, ['title'], '', origin)
    return {"response": response['title'], "q": q}

@app.get("/{path}/{subpath}/{q}")
def query(request: Request, path: Union[str, None] = None, subpath: Union[str, None] = None, q: Union[str, None] = None, subq: Union[str, None] = None, origin: Union[str, None] = None):
    client_host = request.client.host
    print(client_host)
    print(origin)
    if q is None:
        print('q is not set')
        return { "error": "q is not set" }
    if path == "pestel" or path == "competitive_product_analysis":
        response = M1M2(q, paths[path][subpath], subq, origin)
    else:
        response = M3M4(q, paths[path][subpath], subq, origin)
    # trimmed = response.replace("```json", "").replace("```", "").replace(" ", "").replace("\n", "").replace("'", "\"")
    # print(trimmed)
    # parsed = json.loads(trimmed)
    # print(parsed)
    return {"response": response, "q": q}
