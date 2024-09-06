# -*- coding: utf-8 -*-
import time

from utils import generated_answer_result

def res(query):
    start_time = time.time()

    result = generated_answer_result(query=query, stream=False)

    end_time = time.time()


    print(f"question :{query}\n\n" )

    print(f"answer : {result["answer"]}\n\n")

    print(f"source : {result["sources"]}\n\n")

    print(f"context :{result["context"]}\n\n" )
    # f"context : {list(set(result["context"]))}\n\n"

    # generated_answer_result_df = pd.DataFrame(generated_answer_result)
    print('time spent:', end_time - start_time )
    return f"question :{query}\n\n" f"answer : {result["answer"]}\n\n" f"source : {list(set(result["sources"]))}\n\n" f"time spent: {end_time - start_time}"

#generated_answer_result_df.to_csv('result.csv')
