from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_astradb.vectorstores import AstraDBVectorStore
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()





# --- Load ENV variables ---
astra_db_config = {
    "ASTRA_DB_ID": os.getenv("ASTRA_DB_ID"),
    "ASTRA_DB_REGION": os.getenv("ASTRA_DB_REGION"),
    "ASTRA_DB_APPLICATION_TOKEN": os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    "ASTRA_DB_KEYSPACE": os.getenv("ASTRA_DB_KEYSPACE"),
}
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Load and process document ---
loader = TextLoader("FAQ.txt")
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

# --- Create embeddings ---
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# --- Store documents in AstraDB vectorstore ---
vectorstore = AstraDBVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    namespace="default_keyspace",
    collection_name="faq_qa",  # ✅ Required
    astra_db_id=os.getenv("ASTRA_DB_ID"),
    astra_db_region=os.getenv("ASTRA_DB_REGION"),
    astra_db_application_token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    astra_db_keyspace=os.getenv("ASTRA_DB_KEYSPACE")
)
retriever = vectorstore.as_retriever()

# --- Create prompt template ---
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
{context}

---

Given the context above, answer the question as best as possible.

Question: {question}

Answer:
"""
)

# --- Initialize OpenAI LLM ---
llm = ChatOpenAI(
    temperature=0.7,
    model_name = "gpt-4.1-nano", # Use a smaller model for faster responses
    openai_api_key=openai_api_key
)

# --- Build RetrievalQA chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,  # ✅ This makes .run() work again
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
)

# --- FastAPI app ---
app = FastAPI()

class Query(BaseModel):
    message: str
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(query: Query):
    try:
        # Use invoke() and extract the correct field
        response = qa_chain.invoke({"query": query.message})
        return {"response": response["result"]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": "Internal Server Error", "detail": str(e)}

@app.post("/chat")
async def chat(query: Query):
    try:
        result_dict = qa_chain.invoke({"query": query.message})
        return {"response": result_dict["result"]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": "Internal Server Error", "detail": str(e)}
    
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return Path("index.html").read_text()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ For development only! Use specific origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
