import os
import shutil
import logging
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import LlamaCpp
import gradio as gr
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

DATA_PATH = os.getcwd() + "/data/"
DB_FAISS_PATH = os.getcwd() + "/vectorstores/"
MODEL_PATH = "models/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/snapshots/3a6fbf4a41a1d52e415a4958cde6856d34b2db93/mistral-7b-instruct-v0.2.Q4_0.gguf"

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(DB_FAISS_PATH, exist_ok=True)

# LLM
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.1,
    n_threads=8,
    max_tokens=1024,   
    n_ctx=2048, 
    verbose=False
)

def log_event(event):
    with open("access_log.txt", "a") as f:
        f.write(f"{datetime.now()} - {event}\n")

# Embeddings Model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(DB_FAISS_PATH + "/index.faiss"):
    vector_store = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    vector_store = None

retriever_chain = None

def create_retriever_chain(vector_store):
    retriever = vector_store.as_retriever()

    query_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant extracting answers from a lab report PDF based on the user's questions."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Generate a focused search query to retrieve relevant information from the report for the latest question.")
    ])
    history_aware = create_history_aware_retriever(llm, retriever, query_prompt)

    # answer_prompt = ChatPromptTemplate.from_messages([
    #     ("system", "You are a helpful assistant answering questions based on the provided lab report context. Answer concisely, using the report details when relevant. Be aware of the prior chat history for references and pronouns, but only respond to the current user message. Do not continue the conversation or ask follow-up questions."),
    #     MessagesPlaceholder(variable_name="chat_history"),
    #     ("user", "Context from the report:\n{context}\n\nQuestion:\n{input}\n\nAnswer:")
    # ])
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant answering questions from a lab report. Use the provided context to answer the user's question. Only provide the direct answer. Do not repeat the context or the question. Be aware of prior conversation history for references."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Context:\n{context}\n\nQuestion:\n{input}\n\nAnswer concisely:")
    ])

    doc_chain = create_stuff_documents_chain(llm, answer_prompt)

    return create_retrieval_chain(history_aware, doc_chain)


def create_vector_db_from_pdf(pdf_file):
    global vector_store, retriever_chain

    pdf_filename = os.path.basename(pdf_file)
    pdf_path = os.path.join(DATA_PATH, pdf_filename)

    if not os.path.exists(pdf_path):
        shutil.copy(pdf_file, pdf_path)

    if os.path.exists(DB_FAISS_PATH + "/index.faiss"):
        shutil.rmtree(DB_FAISS_PATH)
        os.makedirs(DB_FAISS_PATH, exist_ok=True)

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    vector_store = FAISS.from_documents(texts, embeddings)
    vector_store.save_local(DB_FAISS_PATH)

    retriever_chain = create_retriever_chain(vector_store)

    log_event(f"Ingested PDF: {pdf_filename}")
    return "PDF successfully ingested."

# Gradio App
with gr.Blocks() as demo:
    gr.Markdown("Medical Chatbot")

    with gr.Row():
        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")

    auth_message = gr.Textbox(label="Auth Status", interactive=False)
    auth_flag = gr.State(False)

    def authenticate(user, pwd):
        return ("Login successful.", True) if user == "user" and pwd == "12345" else ("Invalid login.", False)

    login_btn = gr.Button("Login")
    login_btn.click(authenticate, inputs=[username, password], outputs=[auth_message, auth_flag])

    with gr.Row():
        pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
        ingest_btn = gr.Button("Ingest PDF")

    chatbot = gr.Chatbot(label="Chat")
    user_input = gr.Textbox(label="Ask something")
    send_btn = gr.Button("Send")

    chat_history_state = gr.State([HumanMessage(content="Hello"), AIMessage(content="Hi! I'm your assistant.")])

    # Ingest PDF Handler
    def handle_ingest(pdf_file, is_auth):
        if not is_auth:
            return "Unauthorized."
        if pdf_file is None:
            return "No file provided."
        try:
            result = create_vector_db_from_pdf(pdf_file)
            return result
        except Exception as e:
            logger.error(f"Ingestion error: {e}")
            return "Error during ingestion."

    ingest_btn.click(
        handle_ingest,
        inputs=[pdf_upload, auth_flag],
        outputs=[auth_message]
    )

    # Chat Handler
    def handle_chat(message, chat_history, is_auth):
        if not is_auth:
            return [["System", "You must be logged in to chat."]], chat_history

        if vector_store is None or retriever_chain is None:
            return [["System", "Please ingest a PDF first."]], chat_history

        chat_history = (chat_history + [HumanMessage(content=message)])[-8:]

        try:
            result = retriever_chain.invoke({
                "chat_history": chat_history,
                "input": message
            })
            response = result["answer"]
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return [["System", "Error occurred during chat."]], chat_history

        chat_history = (chat_history + [AIMessage(content=response)])[-8:]
        display = [[chat_history[i].content, chat_history[i + 1].content]
                   for i in range(0, len(chat_history) - 1, 2)]
        return display, chat_history

    send_btn.click(
        handle_chat,
        inputs=[user_input, chat_history_state, auth_flag],
        outputs=[chatbot, chat_history_state]
    )

demo.launch(server_name="0.0.0.0", server_port=7861, debug=True)