
import time

from utils import get_evaluation_result

def evaluation():
    chunk_size_list = [500]
    num_chunks_list = [5]
    lexical_search_k_list = [0,1,3]
    llm_answer = "gpt-4-turbo"
    llm_evaluate = "gpt-3.5-turbo"

    total_start_time = time.time()
    result = get_evaluation_result(
        chunk_size_list=chunk_size_list,
        num_chunks_list=num_chunks_list,
        lexical_search_k_list=lexical_search_k_list,
        llm_answer=llm_answer,
        llm_evaluate=llm_evaluate
    )
    total_end_time = time.time()
    print(result)
    total_elapsed_time = total_end_time - total_start_time

if __name__ == "__main__":
  evaluation()