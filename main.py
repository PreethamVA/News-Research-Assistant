import os
import streamlit as st
import pickle
import time
from newspaper import Article
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Get up to 3 URLs from user
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()

# LLM init
llm = OpenAI(temperature=0.9, max_tokens=500)

# Article downloader using newspaper3k
def fetch_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return ""

# Process on button click
if process_url_clicked and urls:
    main_placeholder.text("Fetching and parsing articles...")
    texts = [fetch_article_text(url) for url in urls if url.strip()]
    texts = [t for t in texts if t.strip()]  # remove empty strings

    if not texts:
        st.error("❌ No data could be fetched. Try different news URLs.")
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", ","]
        )
        docs = text_splitter.create_documents(texts)

        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(docs, embeddings)

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)

        main_placeholder.success("✅ FAISS index built and saved successfully!")

# Ask questions
query = main_placeholder.text_input("Ask a question based on the articles:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )
        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
    else:
        st.error("Vector store not found. Process URLs first.")
