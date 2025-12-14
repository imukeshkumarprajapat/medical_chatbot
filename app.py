from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
from operator import itemgetter

from src.helper import download_embeddings
from src.prompt import system_prompt
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq

app = Flask(__name__)
load_dotenv()

# API keys
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Embeddings + Pinecone retriever
embeddings = download_embeddings()
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Groq LLM
chatModel = ChatGroq(model="llama-3.1-8b-instant")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Format docs function
def format_docs(docs):
    return "\n\n".join([getattr(d, "page_content", str(d)) for d in docs])

# Context branch
context_branch = itemgetter("input") | retriever | RunnableLambda(format_docs)

# RAG chain
rag_chain = (
    {"context": context_branch, "input": itemgetter("input")}
    | prompt
    | chatModel
)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    #print("User:", msg)

    try:
        response = rag_chain.invoke({"input": msg})
        #print("Raw response:", response)

        # AIMessage object → .content
        if hasattr(response, "content"):
            answer = response.content
        else:
            answer = str(response)

        #print("Final Answer:", answer)
        return str(answer)
    except Exception as e:
        print("Error in rag_chain:", e)
        return "Error: Could not get response"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)





# import streamlit as st
# from dotenv import load_dotenv
# import os
# from operator import itemgetter

# from src.helper import download_embeddings
# from src.prompt import system_prompt
# from langchain_pinecone import PineconeVectorStore
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableLambda
# from langchain_groq import ChatGroq

# # ---------------- Setup ----------------
# load_dotenv()
# os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# # ✅ Meta tensor fix: cache pipeline so it doesn't reload every rerun
# @st.cache_resource
# def load_pipeline():
#     # Embeddings + Pinecone retriever
#     embeddings = download_embeddings()
#     index_name = "medical-chatbot"
#     docsearch = PineconeVectorStore.from_existing_index(
#         index_name=index_name,
#         embedding=embeddings
#     )
#     retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

#     # Groq LLM
#     chatModel = ChatGroq(model="llama-3.1-8b-instant")

#     # ✅ Meta tensor safeguard: if model accidentally loads on meta device
#     try:
#         if hasattr(chatModel, "to") and hasattr(chatModel, "device") and str(chatModel.device) == "meta":
#             # move empty structure to CPU
#             chatModel = chatModel.to_empty(device="cpu")
#     except Exception as e:
#         st.warning(f"Meta tensor fix applied: {e}")

#     # Prompt template
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ])

#     # Format docs function
#     def format_docs(docs):
#         return "\n\n".join([getattr(d, "page_content", str(d)) for d in docs])

#     # Context branch
#     context_branch = itemgetter("input") | retriever | RunnableLambda(format_docs)

#     # RAG chain
#     rag_chain = (
#         {"context": context_branch, "input": itemgetter("input")}
#         | prompt
#         | chatModel
#     )
#     return rag_chain

# rag_chain = load_pipeline()

# # ---------------- Streamlit UI ----------------
# st.set_page_config(page_title="Medical Chatbot", page_icon="🩺", layout="centered")
# st.title("🩺 Medical Chatbot (Groq + Pinecone)")

# # Chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat history
# for msg in st.session_state.messages:
#     if msg["role"] == "user":
#         st.markdown(f"<div style='text-align:right; background:#007bff; color:white; padding:8px; border-radius:8px; margin:4px;'>You: {msg['content']}</div>", unsafe_allow_html=True)
#     else:
#         st.markdown(f"<div style='text-align:left; background:#f1f1f1; padding:8px; border-radius:8px; margin:4px;'>Bot: {msg['content']}</div>", unsafe_allow_html=True)

# # User input
# user_input = st.text_input("Type your message:", key="input")

# if st.button("Send") and user_input:
#     # Add user message
#     st.session_state.messages.append({"role": "user", "content": user_input})

#     # Get bot response
#     with st.spinner("Thinking..."):
#         response = rag_chain.invoke({"input": user_input})
#         answer = response.content if hasattr(response, "content") else str(response)

#     # Add bot message
#     st.session_state.messages.append({"role": "bot", "content": answer})

#     # Rerun to show updated chat
#     st.rerun()





# import streamlit as st
# import os
# import time
# from operator import itemgetter

# # Langchain/RAG components
# from dotenv import load_dotenv
# from langchain_pinecone import PineconeVectorStore
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableLambda
# from langchain_groq import ChatGroq

# # ⚠️ सुनिश्चित करें कि src/helper.py और src/prompt.py उपलब्ध हैं
# try:
#     from src.helper import download_embeddings
#     from src.prompt import system_prompt
# except ImportError:
#     st.warning("RAG files (src/helper.py, src/prompt.py) not found. Using Dummy Chat Response.")
#     system_prompt = "You are a helpful medical assistant."
    
#     def initialize_rag_chain():
#         return lambda x: f"Dummy RAG Response: Please set up your RAG chain correctly. You asked: {x['input']}"
    
#     initialize_rag_chain_full = initialize_rag_chain # Dummy assignment

# load_dotenv()
# PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
# GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# # ⚙️ RAG Chain Initialization (Unchanged - uses st.cache_resource)
# @st.cache_resource
# def initialize_rag_chain_full():
#     # ... (Your RAG setup logic here) ...
#     embeddings = download_embeddings()
#     index_name = "medical-chatbot"
#     try:
#         docsearch = PineconeVectorStore.from_existing_index(
#             index_name=index_name,
#             embedding=embeddings
#         )
#         retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
#     except Exception as e:
#         st.error(f"Failed to connect to Pinecone index '{index_name}': {e}")
#         st.stop()
#     chatModel = ChatGroq(model="llama-3.1-8b-instant")
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ])
#     def format_docs(docs):
#         return "\n\n".join([getattr(d, "page_content", str(d)) for d in docs])
#     context_branch = itemgetter("input") | retriever | RunnableLambda(format_docs)
#     rag_chain = (
#         {"context": context_branch, "input": itemgetter("input")}
#         | prompt
#         | chatModel
#     )
#     return rag_chain

# if PINECONE_API_KEY and GROQ_API_KEY and 'initialize_rag_chain_full' in locals():
#     rag_chain = initialize_rag_chain_full()
# else:
#     rag_chain = initialize_rag_chain()


# # 🎨 3. Streamlit UI and Styling (FIXED HEADER & DARK STYLE)
# st.set_page_config(page_title="Medical Chatbot (Fixed Header)", layout="centered")

# # 🖼️ CUSTOM CSS STYLING (Fixed Header Logic Added)
# st.markdown(
#     """
#     <style>
#     /* 1. Global Background (Dark Gradient) */
#     .stApp { 
#         background: linear-gradient(to right, #26333d, #323741, #21214e); 
#         color: white;
#     }

#     /* 2. Main Container (The Chatbox Border/Box) */
#     .main .block-container { 
#         max-width: 600px;
#         height: 85vh; /* Fixed height for the entire card */
        
#         background-color: rgba(0,0,0,0.4) !important; 
#         border-radius: 15px !important; 
#         box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5); 
        
#         padding: 0 !important; /* Remove all internal padding */
#         margin-top: 30px; 
#         margin-bottom: 30px;
#         position: relative; /* Base for fixed/absolute children */
#     }
    
#     /* 3. Chat Header (ABSOLUTELY POSITIONED within the fixed container) */
#     .chat-header {
#         background-color: #007bff; 
#         color: white; 
#         padding: 15px; 
#         display: flex; 
#         align-items: center;
        
#         /* Make it stick to the top edge of the block-container */
#         position: absolute; 
#         top: 0; 
#         left: 0;
#         right: 0;
#         width: 100%;
#         z-index: 11;
#         border-radius: 15px 15px 0 0 !important;
#     }
    
#     /* 4. Chat Input Bar (ABSOLUTELY POSITIONED at the bottom) */
#     .stChatInputContainer {
#         position: absolute; /* Make it stick to the bottom edge */
#         bottom: 0; 
#         left: 0;
#         right: 0;
#         width: 100%;
#         background-color: rgba(0,0,0,0.4) !important;
#         padding: 10px 15px;
#         z-index: 10; 
#         border-radius: 0 0 15px 15px !important;
#     }
    
#     /* 5. Scrollable Chat Content Area (This holds all the messages) */
#     .main div[data-testid="stVerticalBlock"] {
#         /* Height adjustment: Total height (85vh) - Header Height (~60px) - Footer Height (~80px) */
#         height: calc(85vh - 140px); 
#         overflow-y: auto; /* Only messages scroll */
#         padding-top: 75px; /* Add padding equal to the header height */
#         padding-bottom: 75px; /* Add padding equal to the footer height */
#         padding-left: 10px;
#         padding-right: 10px;
#         color: white;
#         width: 100%; /* Ensure it spans the full width */
#     }

#     /* --- Message Bubble Styles (Unchanged from Dark Theme) --- */
    
#     /* USER Message (Right Aligned, Green Bubble - #58cc71) */
#     .stChatMessage[data-testid="stChatMessage"]:has(> div > [data-testid="stImage"]):first-child .stMarkdown {
#         background-color: #58cc71; 
#         color: white;
#         border-radius: 25px !important; 
#         padding: 10px; 
#         margin-left: 30%; 
#         text-align: right;
#         margin-right: 5px; /* Adjusted margin */
#         margin-top: 10px;
#     }
#     .stChatMessage[data-testid="stChatMessage"]:has(> div > [data-testid="stImage"]):first-child .stImage {
#         display: none;
#     }

#     /* ASSISTANT Message (Left Aligned, Blue Bubble - #52acff) */
#     .stChatMessage[data-testid="stChatMessage"]:has(> div > [data-testid="stImage"]):last-child .stMarkdown {
#         background-color: #52acff; 
#         color: white;
#         border-radius: 25px !important; 
#         padding: 10px; 
#         margin-right: 30%; 
#         text-align: left;
#         margin-left: 5px; /* Adjusted margin */
#         margin-top: 10px;
#     }
#     .stChatMessage[data-testid="stChatMessage"]:has(> div > [data-testid="stImage"]):last-child .stImage {
#          height: 40px; 
#          width: 40px; 
#          border: 1.5px solid #f5f6fa;
#          border-radius: 50%;
#          margin-right: 10px;
#     }

#     /* Input Field and Button Styles (Unchanged from Dark Theme) */
#     .stChatInput > div > div > input {
#         background-color: rgba(0,0,0,0.3) !important;
#         border: 0 !important;
#         color: white !important;
#         height: 60px !important;
#         border-radius: 15px 0 0 15px !important;
#     }
#     .stChatInput > div > div:last-child {
#         background-color: rgba(0,0,0,0.3) !important;
#         border-radius: 0 15px 15px 0 !important;
#         color: white !important;
#         width: 60px;
#         display: flex;
#         align-items: center;
#         justify-content: center;
#     }

#     /* Timestamp (Unchanged) */
#     .timestamp { 
#         font-size: 10px; 
#         color: rgba(255,255,255,0.6);
#         margin-top: 5px; 
#         display: block; 
#         line-height: 1;
#         text-align: right;
#     }
#     .stChatMessage:has(> div > [data-testid="stImage"]):last-child .timestamp {
#         text-align: left;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # 🕒 टाइमस्टैम्प फंक्शन
# def get_current_time():
#     return time.strftime("%I:%M %p")

# # Chat Header (HTML) - Fixed Header (Added a surrounding container to ensure absolute positioning works)
# header_placeholder = st.empty()
# with header_placeholder.container():
#     st.markdown(
#         """
#         <div class='chat-header'>
#             <img src="https://cdn-icons-png.flaticon.com/512/387/387569.png" class="rounded-circle mr-2" width="40" height="40" style="background-color: #f5f6fa; border: 1.5px solid #f5f6fa;">
#             <div>
#                 <strong>Medical Chatbot</strong><br>
#                 <small>Ask me anything!</small>
#             </div>
#         </div>
#         """, 
#         unsafe_allow_html=True
#     )

# # 📜 4. चैट इतिहास (Session State)
# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {"role": "assistant", "content": "Hello. How can I assist you today, particularly in relation to medical topics?", "time": get_current_time()}
#     ]

# # 💬 पुराने संदेशों को प्रदर्शित करें
# for message in st.session_state.messages:
#     avatar = "https://cdn-icons-png.flaticon.com/512/387/387569.png" if message["role"] == "assistant" else None
    
#     with st.chat_message(message["role"], avatar=avatar):
#         st.markdown(message["content"])
#         st.markdown(f'<span class="timestamp">{message["time"]}</span>', unsafe_allow_html=True) 


# # ⌨️ 5. नया उपयोगकर्ता इनपुट हैंडल करें
# if prompt := st.chat_input("Type your message...", key="chat_input_fixed"):
#     current_time = get_current_time()
    
#     # 1. User Message
#     user_message = {"role": "user", "content": prompt, "time": current_time}
#     st.session_state.messages.append(user_message)
    
#     with st.chat_message("user"):
#         st.markdown(prompt)
#         st.markdown(f'<span class="timestamp">{current_time}</span>', unsafe_allow_html=True)

#     # 2. Assistant Response
#     with st.chat_message("assistant", avatar="https://cdn-icons-png.flaticon.com/512/387/387569.png"):
#         with st.spinner("Processing your query..."):
#             try:
#                 response = rag_chain.invoke({"input": prompt})
#                 if hasattr(response, "content"):
#                     answer = response.content
#                 else:
#                     answer = str(response)

#             except Exception as e:
#                 answer = f"Error: Could not get response from RAG chain. Details: {e}"
#                 st.error(answer)

#         st.markdown(answer)
#         assistant_time = get_current_time()
#         st.markdown(f'<span class="timestamp">{assistant_time}</span>', unsafe_allow_html=True)

#     # 3. Add to History
#     st.session_state.messages.append({"role": "assistant", "content": answer, "time": assistant_time})
    
#     st.rerun()