
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

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

if __name__ == '__main__':
    main()