from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from typing import List
from langchain_core.documents import Document


#Extract data from the pdf fle 

def load_pdf_files(data_path):
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()

extracted_data = load_pdf_files(r"D:\Desktop\medical chatbot\data")



def filter_to_minimal_docs(docs:List[Document])->List[Document]:
    """Given a list of Document objects, return a new list of documents objects
    containing only 'source' in metadata and the original page_content"""

    minimal_docs:List[Document]=[]
    
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src}
            )
        )
    return minimal_docs


#split the documents into smaller chunks
def text_split(minimal_docs):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,  #	Chunk → बड़े टेक्स्ट को manageable बनाता है।
        chunk_overlap=20, # Chunk Overlap → context loss रोकता है, ताकि model को समझने में मदद मिले।
        
    )

    texts_chunk= text_splitter.split_documents(minimal_docs)
    return texts_chunk


from langchain_community.embeddings import HuggingFaceEmbeddings
#download the embeddings from hugging face
def download_embeddings():

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        
    )

    return embeddings
