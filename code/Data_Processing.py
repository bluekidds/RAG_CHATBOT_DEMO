from utils import load_documents, split_text, save_to_chroma, save_to_bm25

def generate_data_store():
  """
  Function to generate vector database in chroma from documents.
  """
  print("---loading documents---")
  documents = load_documents() # Load documents from a source

  print("---spliting chunks---")
  chunks = split_text(documents) # Split documents into manageable chunks

  save_to_chroma(chunks) # Save the processed data to a data store

  save_to_bm25(chunks) # Save the raw documents to bm25 store
  return

if __name__ == "__main__":
  generate_data_store()


        
    





    











        

