# RentIQ Agentic RAG System

An advanced, interactive Agentic Retrieval-Augmented Generation (RAG) system built with **Streamlit**, **LangGraph**, and **Groq**. 

This application allows users to upload PDF documents and dynamically query them. The system uses an intelligent agent (powered by `llama-3.3-70b-versatile`) equipped with a custom retriever tool to search through the embedded document chunks and provide precise, cited answers.

## 🚀 Features

- **Dynamic Document Processing**: Upload any PDF directly through the UI. The document is automatically chunked and embedded using Google Generative AI Embeddings.
- **Agentic Reasoning**: Uses a LangGraph checkpointer and a tool-calling Groq model to intelligently decide when to use the retriever tool and how to synthesize the information.
- **Source Citations**: The agent is explicitly instructed to cite the page numbers of the documents it references.
- **Interactive UI**: A seamless Streamlit chat interface that preserves conversation history and provides visual feedback (spinners) during agent execution.
- **Sidebar Controls**: Easily clear chat history or monitor the currently active document.

## 🛠️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Agent Framework**: [LangGraph](https://python.langchain.com/v0.1/docs/langgraph/) & [LangChain](https://python.langchain.com/)
- **LLM**: [Groq](https://groq.com/) (`llama-3.3-70b-versatile`)
- **Embeddings**: Google Generative AI (`models/gemini-embedding-2`)
- **Vector Store**: InMemoryVectorStore

## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/phantom-0994/RentIQ-Agentic-RAG.git
   cd RentIQ-Agentic-RAG
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Environment Variables:**
   Rename `.env.example` to `.env` and add your API keys:
   ```env
   GOOGLE_API_KEY="your_google_api_key_here"
   GROQ_API_KEY="your_groq_api_key_here"
   ```

5. **Run the Application:**
   ```bash
   streamlit run main.py
   ```

## 🧠 How It Works

1. **Ingestion**: The user uploads a PDF. `PyPDFLoader` reads it, and `RecursiveCharacterTextSplitter` breaks it into manageable chunks.
2. **Embedding**: The chunks are embedded using Google's Gemini embeddings and stored in an in-memory vector database.
3. **Retrieval**: When the user asks a question, the LangGraph agent intercepts the prompt. If it requires external knowledge, the agent autonomously invokes the `@tool` decorated `retriever_tool`, passing in the query.
4. **Synthesis**: The tool performs a similarity search, formats the context (with page citations), and returns it to the agent, which then formulates the final conversational response.
