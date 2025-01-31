import gradio as gr
import os
from typing import List, Optional, Tuple, Dict, Any
from gradio import FileData

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceEndpoint

# Configuration
api_token = os.getenv("HF_TOKEN")
if not api_token:
    raise ValueError("HF_TOKEN environment variable not set")

# Available LLM models
list_llm = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2"
]
list_llm_simple = [os.path.basename(llm) for llm in list_llm]

def validate_files(file_list: List[Optional[FileData]]) -> Tuple[bool, str]:
    """
    Validates uploaded files to ensure they are PDFs and accessible
    
    Args:
        file_list: List of uploaded file objects
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not file_list:
        return False, "No files were uploaded. Please upload at least one PDF file."
        
    for file in file_list:
        if file is None:
            continue
            
        if not file.name.lower().endswith('.pdf'):
            return False, f"File '{file.name}' is not a PDF. Please upload only PDF files."
            
        if not os.path.exists(file.name):
            return False, f"Unable to access file '{file.name}'. Please try uploading again."
    
    return True, ""

def load_doc(list_file_path: List[str]) -> List[Any]:
    """
    Loads and splits PDF documents into chunks
    
    Args:
        list_file_path: List of PDF file paths
    
    Returns:
        List of document chunks
    """
    try:
        loaders = [PyPDFLoader(x) for x in list_file_path]
        pages = []
        for loader in loaders:
            pages.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=64
        )
        doc_splits = text_splitter.split_documents(pages)
        return doc_splits
    except Exception as e:
        raise Exception(f"Error loading documents: {str(e)}")

def create_db(splits: List[Any]) -> FAISS:
    """
    Creates a FAISS vector database from document splits
    
    Args:
        splits: List of document chunks
    
    Returns:
        FAISS vector store
    """
    try:
        embeddings = HuggingFaceEmbeddings()
        vectordb = FAISS.from_documents(splits, embeddings)
        return vectordb
    except Exception as e:
        raise Exception(f"Error creating vector database: {str(e)}")

def initialize_llmchain(
    llm_model: str,
    temperature: float,
    max_tokens: int,
    top_k: int,
    vector_db: FAISS,
    progress: gr.Progress = gr.Progress()
) -> ConversationalRetrievalChain:
    """
    Initializes the LLM chain for question answering
    
    Args:
        llm_model: Name of the LLM model to use
        temperature: Temperature parameter for text generation
        max_tokens: Maximum number of tokens to generate
        top_k: Top-k parameter for token selection
        vector_db: FAISS vector database
        progress: Gradio progress indicator
    
    Returns:
        Conversational retrieval chain
    """
    try:
        llm = HuggingFaceEndpoint(
            repo_id=llm_model,
            huggingfacehub_api_token=api_token,
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_k=top_k,
            task="text-generation"
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key='answer',
            return_messages=True
        )
        
        retriever = vector_db.as_retriever()
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            chain_type="stuff",
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        
        return qa_chain
    except Exception as e:
        raise Exception(f"Error initializing LLM chain: {str(e)}")

def initialize_database(
    list_file_obj: List[Optional[FileData]],
    progress: gr.Progress = gr.Progress()
) -> Tuple[Optional[FAISS], str]:
    """
    Initializes the vector database with proper error handling
    
    Args:
        list_file_obj: List of uploaded file objects
        progress: Gradio progress indicator
    
    Returns:
        tuple: (vector_db, status_message)
    """
    # Validate files first
    is_valid, error_msg = validate_files(list_file_obj)
    if not is_valid:
        return None, error_msg
        
    try:
        list_file_path = [x.name for x in list_file_obj if x is not None]
        doc_splits = load_doc(list_file_path)
        vector_db = create_db(doc_splits)
        return vector_db, "Database created successfully!"
        
    except Exception as e:
        return None, f"Error creating database: {str(e)}"

def initialize_LLM(
    llm_option: int,
    llm_temperature: float,
    max_tokens: int,
    top_k: int,
    vector_db: FAISS,
    progress: gr.Progress = gr.Progress()
) -> Tuple[Optional[ConversationalRetrievalChain], str]:
    """
    Initializes the LLM with selected parameters
    
    Args:
        llm_option: Index of selected LLM model
        llm_temperature: Temperature parameter
        max_tokens: Maximum tokens to generate
        top_k: Top-k parameter
        vector_db: FAISS vector database
        progress: Gradio progress indicator
    
    Returns:
        tuple: (qa_chain, status_message)
    """
    try:
        if vector_db is None:
            return None, "Please create a vector database first"
            
        llm_name = list_llm[llm_option]
        qa_chain = initialize_llmchain(
            llm_name,
            llm_temperature,
            max_tokens,
            top_k,
            vector_db,
            progress
        )
        return qa_chain, "QA chain initialized. Chatbot is ready!"
        
    except Exception as e:
        return None, f"Error initializing LLM: {str(e)}"

def format_chat_history(message: str, chat_history: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """
    Formats the chat history for the LLM
    
    Args:
        message: Current user message
        chat_history: List of (user_message, bot_message) tuples
    
    Returns:
        Formatted chat history as list of message dictionaries
    """
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append({"role": "user", "content": user_message})
        formatted_chat_history.append({"role": "assistant", "content": bot_message})
    return formatted_chat_history

def conversation(
    qa_chain: ConversationalRetrievalChain,
    message: str,
    history: List[Tuple[str, str]]
) -> Tuple[ConversationalRetrievalChain, gr.update, List[Tuple[str, str]], str, int, str, int, str, int]:
    """
    Handles the conversation with the chatbot
    
    Args:
        qa_chain: Conversational retrieval chain
        message: User message
        history: Chat history
    
    Returns:
        tuple: Updated chain, message, history, and source information
    """
    try:
        formatted_chat_history = format_chat_history(message, history)
        response = qa_chain.invoke({
            "question": message,
            "chat_history": formatted_chat_history
        })
        
        response_answer = response["answer"]
        if response_answer.find("Helpful Answer:") != -1:
            response_answer = response_answer.split("Helpful Answer:")[-1]
            
        response_sources = response["source_documents"]
        response_source1 = response_sources[0].page_content.strip()
        response_source2 = response_sources[1].page_content.strip()
        response_source3 = response_sources[2].page_content.strip()
        
        response_source1_page = response_sources[0].metadata["page"] + 1
        response_source2_page = response_sources[1].metadata["page"] + 1
        response_source3_page = response_sources[2].metadata["page"] + 1
        
        new_history = history + [(message, response_answer)]
        
        return (
            qa_chain,
            gr.update(value=""),
            new_history,
            response_source1,
            response_source1_page,
            response_source2,
            response_source2_page,
            response_source3,
            response_source3_page
        )
        
    except Exception as e:
        return (
            qa_chain,
            gr.update(value=""),
            history + [(message, f"Error: {str(e)}")],
            "", 0, "", 0, "", 0
        )

def demo():
    """Creates and launches the Gradio interface with proper port binding for Render"""
    with gr.Blocks(theme=gr.themes.Default(
        primary_hue="red",
        secondary_hue="pink",
        neutral_hue="sky"
    )) as demo:
        # State variables
        vector_db = gr.State()
        qa_chain = gr.State()
        
        # Header
        gr.HTML("<center><h1>RAG PDF chatbot</h1></center>")
        gr.Markdown("""
        <b>Query your PDF documents!</b> This AI agent is designed to perform 
        retrieval augmented generation (RAG) on PDF documents. The app is hosted 
        on Hugging Face Hub for demonstration purposes.
        <b>Please do not upload confidential documents.</b>
        """)
        
        with gr.Row():
            with gr.Column(scale=86):
                # Document upload section
                gr.Markdown("<b>Step 1 - Upload PDF documents and Initialize RAG pipeline</b>")
                with gr.Row():
                    document = gr.Files(
                        height=300,
                        file_count="multiple",
                        file_types=["pdf"],
                        interactive=True,
                        label="Upload PDF documents"
                    )
                with gr.Row():
                    db_btn = gr.Button("Create vector database")
                with gr.Row():
                    db_progress = gr.Textbox(
                        value="Not initialized",
                        show_label=False
                    )
                
                # LLM selection section
                gr.Markdown("""
                <style>body { font-size: 16px; }</style>
                <b>Select Large Language Model (LLM) and input parameters</b>
                """)
                with gr.Row():
                    llm_btn = gr.Radio(
                        list_llm_simple,
                        label="Available LLMs",
                        value=list_llm_simple[0],
                        type="index"
                    )
                
                # LLM parameters
                with gr.Row():
                    with gr.Accordion("LLM input parameters", open=False):
                        with gr.Row():
                            slider_temperature = gr.Slider(
                                minimum=0.01,
                                maximum=1.0,
                                value=0.5,
                                step=0.1,
                                label="Temperature",
                                info="Controls randomness in token generation",
                                interactive=True
                            )
                        with gr.Row():
                            slider_maxtokens = gr.Slider(
                                minimum=128,
                                maximum=9192,
                                value=4096,
                                step=128,
                                label="Max New Tokens",
                                info="Maximum number of tokens to be generated",
                                interactive=True
                            )
                        with gr.Row():
                            slider_topk = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=3,
                                step=1,
                                label="top-k",
                                info="Number of tokens to select the next token from",
                                interactive=True
                            )
                
                with gr.Row():
                    qachain_btn = gr.Button("Initialize Question Answering Chatbot")
                with gr.Row():
                    llm_progress = gr.Textbox(
                        value="Not initialized",
                        show_label=False
                    )
            
            # Chat interface
            with gr.Column(scale=200):
                gr.Markdown("<b>Step 2 - Chat with your Document</b>")
                chatbot = gr.Chatbot(height=505, type="messages")
                
                # Source documents section
                with gr.Accordion("Relevant context from the source document", open=False):
                    with gr.Row():
                        doc_source1 = gr.Textbox(
                            label="Reference 1",
                            lines=2,
                            container=True,
                            scale=20
                        )
                        source1_page = gr.Number(label="Page", scale=1)
                    with gr.Row():
                        doc_source2 = gr.Textbox(
                            label="Reference 2",
                            lines=2,
                            container=True,
                            scale=20
                        )
                        source2_page = gr.Number(label="Page", scale=1)
                    with gr.Row():
                        doc_source3 = gr.Textbox(
                            label="Reference 3",
                            lines=2,
                            container=True,
                            scale=20
                        )
                        source3_page = gr.Number(label="Page", scale=1)
                
                # Chat input
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask a question",
                        container=True
                    )
                with gr.Row():
                    submit_btn = gr.Button("Submit")
                    clear_btn = gr.ClearButton(
                        [msg, chatbot],
                        value="Clear"
                    )
        
        # Event handlers
        db_btn.click(
            initialize_database,
            inputs=[document],
            outputs=[vector_db, db_progress]
        )
