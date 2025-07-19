import os
from typing import List
import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings

# Initialize the Ollama LLM and embeddings
llm = OllamaLLM(model="qwen2.5:latest")
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

# Streamlit UI setup
st.set_page_config(page_title="Multi-Hop RAG Chatbot", page_icon="🤖")
st.title("Multi-Hop RAG Chatbot")
st.caption("Upload PDF documents and chat with AI using multi-hop retrieval")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your document assistant. Please upload PDF files to get started."}
    ]

# Initialize session state for vector store and processed files
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Sidebar for document upload
with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True
    )
    process_button = st.button("Process Documents")
    clear_button = st.button("Clear Documents")

    if st.session_state.processed_files:
        st.write("### Processed Files:")
        for file in st.session_state.processed_files:
            st.write(f"- {file}")

# Clear documents if button is pressed
if clear_button:
    # Remove all PDF files from current directory
    for filename in st.session_state.processed_files:
        if os.path.exists(filename):
            os.remove(filename)
    st.session_state.vector_store = None
    st.session_state.processed_files = []
    st.session_state.messages = [{"role": "assistant", "content": "All documents have been cleared. You can upload new files now."}]
    st.rerun()

# Process uploaded documents
if process_button and uploaded_files:
    with st.spinner("Processing documents..."):
        # Save uploaded files to current directory
        pdf_paths = []
        for uploaded_file in uploaded_files:
            # Check if file already exists
            if uploaded_file.name in st.session_state.processed_files:
                st.warning(f"File {uploaded_file.name} was already processed. Skipping...")
                continue
            
            file_path = uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf_paths.append(file_path)
            st.session_state.processed_files.append(file_path)
        
        # Load and split documents
        documents = []
        for pdf_path in pdf_paths:
            try:
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
            except Exception as e:
                st.error(f"Error loading {pdf_path}: {str(e)}")
                continue
        
        if documents:
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            st.session_state.vector_store = Chroma.from_documents(
                documents=splits, embedding=embeddings
            )
            
            st.session_state.messages.append(
                {"role": "assistant", "content": f"✅ Processed {len(pdf_paths)} documents with {len(splits)} chunks. Ask me anything about them!"}
            )
            st.rerun()
        else:
            st.error("No valid documents were processed.")

# RAG functions
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def get_retriever(top_k: int = 3):
    if st.session_state.vector_store is None:
        return None
    return st.session_state.vector_store.as_retriever(search_kwargs={"k": top_k})

def generate_sub_questions(question: str) -> List[str]:
    """Generate sub-questions for multi-hop retrieval"""
    prompt = ChatPromptTemplate.from_template(
        """Break down the following question into 2-3 sub-questions that would help answer it.
        Original question: {question}
        Return only the sub-questions as a bulleted list, nothing else."""
    )
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question})
    
    # Parse the bulleted list
    sub_questions = [q.strip()[2:] for q in result.split("\n") if q.strip().startswith("-")]
    return sub_questions or [question]

def get_rag_chain():
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:
        {context}
        
        Question: {question}
        
        Provide a detailed answer with reasoning. If you don't know, say you don't know.
        Use a conversational tone and format your response nicely."""
    )
    
    return (
        {"context": RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"])),
         "question": RunnablePassthrough()}
        | prompt
        | llm
    )

# Chat interface
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if st.session_state.vector_store is None:
            st.error("Please upload and process documents first.")
            st.stop()
        
        retriever = get_retriever()
        
        # Generate sub-questions
        with st.spinner("Analyzing your question..."):
            sub_questions = generate_sub_questions(prompt)
        
        # Retrieve documents for each sub-question
        with st.spinner("Gathering relevant information..."):
            retrieved_docs = []
            for sub_q in sub_questions:
                docs = retriever.invoke(sub_q)
                retrieved_docs.extend(docs)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_docs = []
            for doc in retrieved_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_docs.append(doc)
        
        # Prepare the RAG chain
        rag_chain = get_rag_chain()
        
        # Stream the response
        response = st.write_stream(
            rag_chain.stream({"question": prompt, "docs": unique_docs})
        )
        
        # Add retrieval details to the message
        retrieval_details = {
            "sub_questions": sub_questions,
            "retrieved_docs": [{"content": doc.page_content[:200] + "...", 
                              "source": doc.metadata.get("source", "unknown")} 
                             for doc in unique_docs]
        }
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "metadata": retrieval_details
    })
    
    # Show retrieval details in expander
    with st.expander("Retrieval Details"):
        st.write("### Sub-questions generated:")
        for i, q in enumerate(retrieval_details["sub_questions"], 1):
            st.write(f"{i}. {q}")
        
        st.write("### Retrieved documents for each sub-question:")
        for doc in retrieval_details["retrieved_docs"]:
            st.write(f"- {doc['content']}")
            st.write(f"  *Source: {doc['source']}*")
            st.write("---")