import streamlit as st
import os
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models import ChatOpenAI
from langchain import hub

def save_file(file):
    folder = 'tmp'
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = f'./{folder}/{file.name}'
    with open(file_path, 'wb') as f:
        f.write(file.getvalue())
    return file_path


def get_pdf_docs(pdf_docs):
    docs = []
    for file in pdf_docs:
        file_path = save_file(file)
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())

    return docs


def get_docs_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    return splits


def add_vector_store(splits):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    SupabaseVectorStore.from_documents(splits, embeddings, client=supabase, table_name="documents",
                                       query_name="match_documents")


def get_vector_store():
    return SupabaseVectorStore(
        embedding=OpenAIEmbeddings(model='text-embedding-3-small'),
        client=supabase,
        table_name="documents",
        query_name="match_documents",
    )


def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={'k': 6}
    )

    prompt = hub.pull("rlm/rag-prompt")

    qa_system_prompt = """As a security manager for Spoon Consulting Ltd. A consulting company in digital transformation, 
        your primary role is to locate and reference documents within the company's Information Security Management System. 
        Your task involves identifying relevant documents and providing concise explanations about how specific 
        sections of these documents relate to the query at hand. You should focus on delivering precise, 
        relevant information while ensuring that your responses remain within the scope of the company's 
        information security guidelines.{context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    retriever_with_history = create_history_aware_retriever(
        llm, retriever, qa_prompt
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever_with_history, question_answer_chain)

    return rag_chain


def handle_userinput(user_question):
    vector_db = get_vector_store()
    rag_chain = get_conversation_chain(vector_db)
    chat_history = []

    ai_message = rag_chain.invoke({"input": user_question, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=user_question), ai_message["answer"]])
    st.session_state.chat_history.append(chat_history)

    for item in st.session_state.chat_history:
        question, ai_response = item
        with st.chat_message("user"):
            st.markdown(question.content)
        with st.chat_message("ai"):
            st.markdown(ai_response)


def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with Spoon Consulting ISMS")
    user_question = st.chat_input("Ask a question about Spoon Consulting iSMS : ")
    if user_question:
        handle_userinput(user_question)

    # Sidebar
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get pdf text
                docs = get_pdf_docs(pdf_docs)

                # Get text chunks
                splits = get_docs_chunks(docs)

                # Add vector store into database vector
                add_vector_store(splits)


if __name__ == '__main__':
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    name, authentication_status, username = authenticator.login('main')

    if authentication_status:
        authenticator.logout('Logout', 'sidebar')
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_SERVICE_KEY"]
        supabase: Client = create_client(supabase_url, supabase_key)
        main()
    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')
