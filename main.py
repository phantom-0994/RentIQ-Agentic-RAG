from importlib.resources import path
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain.tools import tool   
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st   

if "vector_db" not in st.session_state:    
    st.session_state.vector_db = None 

if "agent" not in st.session_state:    
    st.session_state.agent = None 

if "is_uploaded" not in st.session_state:    
    st.session_state.is_uploaded = False                 

if "messages" not in st.session_state:
    st.session_state.messages = []

def process_documents(path:str):
    loader = PyPDFLoader(path)
    docs = loader.load()    

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    vector_db = InMemoryVectorStore.from_documents(documents=docs,embedding=embeddings)
    st.session_state.vector_db = vector_db  
    return vector_db
if st.session_state.is_uploaded:
    vector_db = process_documents("./document.pdf")
else:
    vector_db = process_documents("./rentIQ_Report_5_2026 (1).pdf")
@tool
def retriever_tool(query:str):
    """retrieve document relevent to a query from the knowledge base"""
    documents = vector_db.similarity_search(query=query, k = 2)
    context = ""
    for doc in documents:
        page = doc.metadata.get("page", "Unknown")
        context += f"[Page {page}]: {doc.page_content}\n\n"
    
    return context



system_prompt=f"""You are a helpful assistant that answers questions using retrieved document context.
My knowledge base consist of the details from the uploaded document.
ALWAYS use the 'retriever_tool' tool for questions requiring external knowledge.
When answering, ALWAYS cite the source page number provided in the context (e.g., 'According to Page 3...')."""

def get_agent():
    memory = InMemorySaver()  
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)  
    agent = create_agent(
        model = llm,
        system_prompt = system_prompt,
        checkpointer = memory,
        tools = [retriever_tool]    
    )
    return agent

agent = get_agent()

if not st.session_state.agent:
    st.session_state.agent = get_agent()    


def call_agent(query:str):
    response = agent.invoke({"messages": [{"role": "user", "content": query}]},{"configurable": {"thread_id": "rentiq"}})
    print("AI:", response["messages"][-1].content)
    return response["messages"][-1].content

print("Agentic RAG System is running. Type 'quit' to exit.")

st.subheader("RentIQ Agentic RAG System")

with st.sidebar:
    st.header("Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.write("### Active Document")
    if st.session_state.is_uploaded:
        st.write("📄 `document.pdf` (User Uploaded)")
    else:
        st.write("📄 `rentIQ_Report_5_2026 (1).pdf` (Default)")
if not st.session_state.is_uploaded:
    uploaded_file = st.file_uploader(label="Upload your document here", type=["pdf"])
    if uploaded_file:
        with open("document.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success("File uploaded successfully")
        st.session_state.is_uploaded = True
        st.rerun()
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask a question about your document:")
    if query:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.spinner("Agent is searching and thinking..."):
            answer = call_agent(query)
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role": "ai", "content": answer})
