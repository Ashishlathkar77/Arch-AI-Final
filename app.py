import os
import streamlit as st
import pickle
import json
import nltk
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import warnings

# NLTK setup
nltk.download('punkt')

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables
load_dotenv()

# Streamlit session state setup
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'summaries' not in st.session_state:
    st.session_state.summaries = {}
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Streamlit app title and sidebar
st.title("Arch AI")

st.sidebar.info(
    "Enter up to three URLs to process the content. Click 'Process URLs' to prepare the content for the chatbot. "
    "Use 'Summarize Articles' to generate brief summaries of the content."
)
url1 = st.sidebar.text_input("Enter URL 1")
url2 = st.sidebar.text_input("Enter URL 2")
url3 = st.sidebar.text_input("Enter URL 3")
process_urls_clicked = st.sidebar.button("Process URLs")
summarize_articles_clicked = st.sidebar.button("Summarize Articles")

# Constants
index_file_path = "faiss_index"
active_learning_file = "interactions_dataset.jsonl"

# Initialize the Language Model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.2,
    max_tokens=1000
)

# Helper Functions
def clean_text(text):
    soup = BeautifulSoup(text, 'html.parser')
    return ' '.join(soup.get_text().split())

def process_urls(urls):
    try:
        all_docs = []
        if os.path.exists(index_file_path):
            os.remove(index_file_path)

        for idx, url in enumerate(urls, start=1):
            if not url.strip():
                continue
            loader = UnstructuredURLLoader(urls=[url])
            data = loader.load()
            if not data:
                st.warning(f"No content found at URL {url}.")
                continue
            for doc in data:
                doc.metadata = {"source": url}
            all_docs.extend(data)

        if not all_docs:
            st.error("No valid URLs provided for processing.")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=200
        )
        docs = text_splitter.split_documents(all_docs)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        st.session_state.qa_chain = qa_chain
        st.session_state.processing_complete = True
        st.success("URLs processed successfully!")
    except Exception as e:
        st.error(f"Error processing URLs: {str(e)}")

@st.cache_resource
def load_summarizer():
    return load_summarize_chain(llm, chain_type="refine")

def summarize_articles(urls):
    try:
        st.info("Generating summaries...")
        summaries = {}
        summarize_chain = load_summarizer()

        for idx, url in enumerate(urls, start=1):
            if not url.strip():
                continue
            loader = UnstructuredURLLoader(urls=[url])
            data = loader.load()
            if not data:
                st.warning(f"No content found at URL {url}.")
                continue
            documents = [Document(page_content=clean_text(doc.page_content), metadata={"source": url}) for doc in data]
            summary = summarize_chain.run(documents)
            summaries[url] = summary

        st.session_state.summaries = summaries
        st.success("Summaries generated successfully!")
    except Exception as e:
        st.error(f"Error generating summaries: {str(e)}")

# Button Actions
if process_urls_clicked:
    urls = [url1, url2, url3]
    process_urls(urls)

if summarize_articles_clicked:
    urls = [url1, url2, url3]
    summarize_articles(urls)

# Display Summaries
if st.session_state.summaries:
    st.subheader("Summaries")
    for idx, (url, summary) in enumerate(st.session_state.summaries.items(), start=1):
        st.write(f"**Summary for URL {idx}: {url}**")
        st.write(summary)
        st.markdown("---")

# Chatbot Interface
if st.session_state.processing_complete:
    user_input = st.chat_input("Ask your question...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        try:
            result = st.session_state.qa_chain({"query": user_input})
            answer = result["result"]
            source_documents = result.get("source_documents", [])
            sources_text = "\n".join([doc.metadata.get("source", "Unknown") for doc in source_documents])
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources_text
            })
            with st.chat_message("assistant"):
                st.write(answer)
                if sources_text:
                    st.markdown(f"**Sources:** {sources_text}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
