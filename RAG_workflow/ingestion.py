from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()



hf_embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},        
    encode_kwargs={"normalize_embeddings": False},
)

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

docs = [WebBaseLoader(url).load() for url in urls]

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250,chunk_overlap=0)

doc_splits = text_splitter.split_documents(docs_list)

# vectorstore = Chroma.from_documents(
#     documents=docs_list,
#     collection_name="rag-chroma",
#     embedding=hf_embeddings,
#     persist_directory="./chroma"
# )

retriver = Chroma(
    collection_name="rag-chroma",
    persist_directory="./chroma",
    embedding_function=hf_embeddings,
).as_retriever()