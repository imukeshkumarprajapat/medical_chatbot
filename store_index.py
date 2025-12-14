from dotenv import load_dotenv
from pinecone import Pinecone
import os
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.helper import load_pdf_files, filter_to_minimal_docs,text_split, download_embeddings

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")


os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["GROQ_API_KEY"]=GROQ_API_KEY


extracted_data = load_pdf_files(r"D:\Desktop\medical chatbot\data")
filter_data=filter_to_minimal_docs(extracted_data)
texts_chunk=text_split(filter_data)

embeddings=download_embeddings()



pc=Pinecone(api_key=PINECONE_API_KEY)



index_name = "medical-chatbot"

# Initialize Pinecone client
#pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # dimension of your embedding model
        metric="cosine",  # cosine similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Access the index (correct method)
index = pc.Index(index_name)


docsearch=PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embeddings,
    index_name=index_name
)
#यह code आपके  को embeddings में बदलकर Pinecone index में store कर रहा है, ताकि बाद में आप fast similarity search कर सकें।
