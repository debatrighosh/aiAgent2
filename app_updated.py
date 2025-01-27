import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import tabula
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
import pinecone
from pymongo import MongoClient
import openai
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from PIL import Image
import io
import os
from pinecone import Pinecone, ServerlessSpec

# MongoDB setup
client = MongoClient("mongodb+srv://subai646:Joydisha123$@clusterrag.bdwrc.mongodb.net/?retryWrites=true&w=majority&appName=ClusterRAG")
db = client["rag_database"]
collection = db["pdf_data"]
load_dotenv()
# Pinecone setup
#pinecone.init(api_key="pcsk_3qN8G8_KfwNv5u2mfHBiTuMPEN8xWG2tj8AjgkjwppDHxnSYjggPRcNFSmBactnGiS8gWq")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
print(f"Pinecone API Key: {pinecone_api_key}")

# Initialize Pinecone by creating an instance
pc = Pinecone(api_key=pinecone_api_key)

index_name = "joydisha"
dimension = 768  # Assuming you're using OpenAI Embeddings which are 768-dimensional vectors

#index = pc.index(index_name)
# Create Pinecone index if not exists
if index_name not in pc.list_indexes().names():
#if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name=index_name, dimension=dimension)
host="https://joydisha-wy1rl5k.svc.aped-4627-b74a.pinecone.io"
index = pinecone.Index(index_name,host=host)

def get_pdf_text(aaa):
    text = ""
    for pdf in aaa:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text = text + page.extract_text()
    return text

def get_images_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    return images

def extract_tables_from_pdf(pdf_path):
    tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
    return tables

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to store PDF as binary data in MongoDB
def store_pdf_in_mongodb(pdf_file):
    pdf_binary = pdf_file.read()
    collection.update_one(
        {"pdf_name": pdf_file.name},
        {"$set": {"pdf_data": pdf_binary}},
        upsert=True
    )

def store_data_in_mongodb(text_chunks, images, tables, pdf_docs):
    # Insert text chunks, images, and tables into MongoDB with unique identifiers
    for i, chunk in enumerate(text_chunks):
        collection.update_one({"chunk_id": f"{pdf_docs[0].name}_chunk_{i}"}, {"$set": {"chunk_text": chunk}}, upsert=True)

    for i, img in enumerate(images):
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        collection.update_one({"image_id": f"{pdf_docs[0].name}_img_{i}"}, {"$set": {"image_data": img_byte_arr}}, upsert=True)

    for i, table in enumerate(tables):
        collection.update_one({"table_id": f"{pdf_docs[0].name}_table_{i}"}, {"$set": {"table_data": table}}, upsert=True)

# Function to check if the data is stored in MongoDB
def check_data_in_mongodb(pdf_name):
    # Check for stored PDF data
    pdf_data = collection.find_one({"pdf_name": pdf_name})
    if not pdf_data:
        return "PDF data not found in MongoDB."

    # Check if text chunks are stored
    text_chunks = collection.find({"chunk_id": {"$regex": f"^{pdf_name}_chunk"}})
    if not text_chunks:
        return "Text chunks not found in MongoDB."

    # Check if images are stored
    images = collection.find({"image_id": {"$regex": f"^{pdf_name}_img"}})
    if not images:
        return "Image data not found in MongoDB."

    # Check if tables are stored
    tables = collection.find({"table_id": {"$regex": f"^{pdf_name}_table"}})
    if not tables:
        return "Table data not found in MongoDB."

    return "All data stored successfully in MongoDB."

def generate_and_store_embeddings(text_chunks, images, tables):
    embeddings = OpenAIEmbeddings()

    # Text embeddings
    for i, chunk in enumerate(text_chunks):
        embedding = embeddings.embed_text(chunk)
        index.upsert([(f"chunk_{i}", embedding)])

    # Image embeddings - Assuming an image embedding model like CLIP or similar is used
    for i, img in enumerate(images):
        img_embedding = embeddings.embed_image(img)  # Using OpenAI's CLIP or another model to generate image embeddings
        index.upsert([(f"img_{i}", img_embedding)])

    # Table embeddings - You can use a specialized table-to-text model or encode tables as text
    for i, table in enumerate(tables):
        table_text = str(table)  # Convert the table to a string format
        table_embedding = embeddings.embed_text(table_text)
        index.upsert([(f"table_{i}", table_embedding)])

def create_conversation_chain():
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=index.as_retriever(), memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.write(response)

def main():
    load_dotenv()
    st.set_page_config(page_title="chat with people", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("chat with people :books:")
    user_question = st.text_input("Ask me:")
    print(pinecone_api_key)
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("your doc")
        pdf_docs = st.file_uploader("upload your pdf here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Step 1: Get the PDF text
                raw_text = get_pdf_text(pdf_docs)

                # Step 2: Get text chunks
                text_chunks = get_text_chunks(raw_text)

                # Step 3: Extract images and tables
                images = []
                tables = []
                for pdf in pdf_docs:
                    images += get_images_from_pdf(pdf)
                    tables += extract_tables_from_pdf(pdf)

                # Step 4: Store the PDF, chunks, images, and tables in MongoDB
                for pdf in pdf_docs:
                    store_pdf_in_mongodb(pdf)
                store_data_in_mongodb(text_chunks, images, tables, pdf_docs)

                # Step 5: Generate embeddings for text, images, and tables and store in Pinecone
                generate_and_store_embeddings(text_chunks, images, tables)

                # Step 6: Check if data is stored in MongoDB
                result = check_data_in_mongodb(pdf_docs[0].name)
                st.write(result)

                # Step 7: Create a conversation chain with Pinecone as the retriever
                st.session_state.conversation = create_conversation_chain()

if __name__ == '__main__':
    main()
