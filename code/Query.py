# -*- coding: utf-8 -*-
import time

from code.utils import generated_answer_result,generated_multi_answer_result

async def res(query):
    start_time = time.time()

    result = await generated_answer_result(query=query, stream=False)

    end_time = time.time()


    # print(f"question :{query}\n\n" )

    # print(f"answer : {result["answer"]}\n\n")

    # print(f"source : {result["sources"]}\n\n")

    # print(f"context :{result["context"]}\n\n" )
    # f"context : {list(set(result["context"]))}\n\n"

    # generated_answer_result_df = pd.DataFrame(generated_answer_result)
    print('time spent:', end_time - start_time )
    return f"question :{query}\n\n" f"answer : {result["answer"]}\n\n" f"source : {list(set(result["sources"]))}\n\n" f"time spent: {end_time - start_time}"

async def res_only(query):
    start_time = time.time()

    result = await generated_answer_result(query=query, stream=False)

    end_time = time.time()


    # print(f"question :{query}\n\n" )

    # print(f"answer : {result["answer"]}\n\n")

    # print(f"source : {result["sources"]}\n\n")

    # print(f"context :{result["context"]}\n\n" )
    # f"context : {list(set(result["context"]))}\n\n"

    # generated_answer_result_df = pd.DataFrame(generated_answer_result)
    print('time spent:', end_time - start_time )
    # return result["answer"]
    return {"answer": result["answer"], "source": result["sources"]}

async def multi_res_only(querys):
    start_time = time.time()

    results = await generated_multi_answer_result(querys=querys, stream=False)

    end_time = time.time()

    print(f"questions :{querys}\n\n" )

    # print(results)

    # print(f"answer : {result["answer"]}\n\n")

    # print(f"source : {result["sources"]}\n\n")

    # print(f"context :{result["context"]}\n\n" )
    # # f"context : {list(set(result["context"]))}\n\n"

    # # generated_answer_result_df = pd.DataFrame(generated_answer_result)
    # print('time spent:', end_time - start_time )
    # # return result["answer"]
    return results

#generated_answer_result_df.to_csv('result.csv')
