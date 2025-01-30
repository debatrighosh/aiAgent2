import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    # response = st.session_state.conversation({'question': user_question})
    # st.session_state.chat_history = response['chat_history']
    #
    # for i, message in enumerate(st.session_state.chat_history):
    #     if i % 2 == 0:
    #         st.write(user_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)
    #     else:
    #         st.write(bot_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)





        # Ensure chat history is initialized
        if "chat_history" not in st.session_state or st.session_state.chat_history is None:
            st.session_state.chat_history = []

        # Define system-level prompts for different query types
        resume_summary_prompt = """
    You are an AI assistant specialized in analyzing resumes. Your task is to extract structured insights from the provided resume text.

    Provide a summary using the following structure:
    1. **Executive Summary** (3-4 lines summarizing the candidate’s overall experience and key strengths).
    2. **Work Experience** (List each job separately with Company, Role, Duration, and Key Achievements).
    3. **Education** (Degree, Institution, Year, Major Achievements).
    4. **Key Skills** (List of primary technical and soft skills).

    Only return the extracted resume insights. Do not include any system instructions or formatting explanations.
    """

        # Identify user intent and select the correct system prompt
        if any(keyword in user_question.lower() for keyword in ["summary", "cv", "resume", "overview"]):
            system_prompt = resume_summary_prompt
        else:
            system_prompt = "You are an AI assistant that helps answer document-related queries."

        # Combine system prompt and user question as a single input
        formatted_question = f"{system_prompt}\n\nUser Question: {user_question}"

        # Call the conversation chain while keeping history
        response = st.session_state.conversation({'question': formatted_question})

        # Append the user's question and bot's response to the chat history
        st.session_state.chat_history.append(("user", user_question))
        bot_reply = response['chat_history'][-1].content
        st.session_state.chat_history.append(("bot", bot_reply))

        # Ensure no system prompt is leaked in the response
        if "**Follow this format for the response:**" in bot_reply:
            bot_reply = bot_reply.split("**Follow this format for the response:**")[-1].strip()

        # Display conversation history
        for role, message in st.session_state.chat_history:
            if role == "user":
                st.write(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask Away!!",
                       page_icon=":speech_balloon:")
    #st.write(css, unsafe_allow_html=True)
    st.markdown(css, unsafe_allow_html=True)


    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Ask Away!! :speech_balloon:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    st.write(user_template.replace("{{MSG}}", "Ask your Question"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Answer - "), unsafe_allow_html=True)


    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()