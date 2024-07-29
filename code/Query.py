# -*- coding: utf-8 -*-
import time

from utils import generated_answer_result

def res(query):
    start_time = time.time()

    result = generated_answer_result(query=query, stream=False)

    end_time = time.time()


    print(f"question :{query}" )

    print(f"answer : {result["answer"]}")

    # generated_answer_result_df = pd.DataFrame(generated_answer_result)
    print('time spent:', end_time - start_time )
    return f"question :{query} answer : {result["answer"]} time spent: {end_time - start_time}"

#generated_answer_result_df.to_csv('result.csv')
