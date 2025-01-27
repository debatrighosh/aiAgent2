import pinecone as pinecone
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from pinecone import Pinecone
from pymongo import MongoClient
import os


from pymongo.mongo_client import MongoClient

uri = "mongodb+srv://subai646:Joydisha123$@clusterrag.bdwrc.mongodb.net/?retryWrites=true&w=majority&appName=ClusterRAG"

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


# MongoDB setup
client = MongoClient("mongodb+srv://subai646:Joydisha123$@clusterrag.bdwrc.mongodb.net/?retryWrites=true&w=majority&appName=ClusterRAG",tls=True,
                     tlsAllowInvalidCertificates=False,
                     tlsInsecure=False)
db = client["rag_database"]
collection = db["pdf_data"]
load_dotenv()

# Pinecone setup
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "joydisha"
dimension = 768  # Assuming you're using OpenAI Embeddings which are 768-dimensional vectors

# Create Pinecone index if not exists
if index_name not in pc.list_indexes().names():
    pinecone.create_index(index_name=index_name, dimension=dimension)

host = "joydisha-wy1rl5k.svc.aped-4627-b74a.pinecone.io"
index = pinecone.Index(index_name, host=host)

def store_pdf_in_mongodb(pdf_file):
    pdf_binary = pdf_file.read()
    collection.update_one(
        {"pdf_name": pdf_file.name},
        {"$set": {"pdf_data": pdf_binary}},
        upsert=True
    )

def store_data_in_mongodb(text_chunks,  pdf_docs):
    # Insert text chunks, images, and tables into MongoDB with unique identifiers
    for i, chunk in enumerate(text_chunks):
        collection.update_one({"chunk_id": f"{pdf_docs[0].name}_chunk_{i}"}, {"$set": {"chunk_text": chunk}}, upsert=True)


def generate_and_store_embeddings(text_chunks):
    embeddings = OpenAIEmbeddings()

    # Text embeddings
    for i, chunk in enumerate(text_chunks):
        embedding = embeddings.embed_text(chunk)
        index.upsert([(f"chunk_{i}", embedding)])

def get_pdf_text(aaa):
    text = ""
    for pdf in aaa:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
                text = text+page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore


def create_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.write(response)

def main():
    print("hii")
    load_dotenv()
    st.set_page_config(page_title="chat with people", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("chat with people :books:")
    user_question = st.text_input("Ask me:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("your doc")
        pdf_docs = st.file_uploader("upload your pdf here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Prcessing"):
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text)

                #get text chunks
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)

                #create vector store
                vectorstore = get_vectorstore(text_chunks)

                #create conversation chain
                st.session_state.conversation = create_conversation_chain(vectorstore)
                for pdf in pdf_docs:
                    store_pdf_in_mongodb(pdf)
                store_data_in_mongodb(text_chunks,  pdf_docs)

                # Step 5: Generate embeddings for text, images, and tables and store in Pinecone
                generate_and_store_embeddings(text_chunks)

if __name__ == '__main__':
    main()