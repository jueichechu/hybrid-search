# Install dependencies:
# pip install psycopg2-binary python-dotenv langchain langchain-openai rank_bm25 faiss-cpu
# only library missing from notebook is `pip install psycopg2-binary`

# Again, make sure to have .env file with OPENAI_API_KEY set up

import os
from dotenv import load_dotenv

# DB driver
import psycopg2
from psycopg2.extras import RealDictCursor

# LangChain imports
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Step 1: Load credentials
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env")

# Step 2: Connect to PostgreSQL and fetch rows
# Note: Please refer to the guide on setting up the database
conn = psycopg2.connect(
    dbname="retail_store",
    cursor_factory=RealDictCursor
)
cur = conn.cursor()
cur.execute("SELECT * FROM products;")
rows = cur.fetchall()
cur.close()
conn.close()

# Step 3: Build LangChain Documents
documents = []
for r in rows:
    text = (
        f"{r['sku']} — {r['name']}\n"
        f"Category: {r['category']}\n"
        f"Location: aisle {r['aisle']}, shelf {r['shelf']}\n"
        f"Stock: {r['stock_quantity']} (reorder @ {r['reorder_threshold']})\n"
        f"Price: ${r['price']}\n"
    )
    documents.append(Document(page_content=text, metadata=r))

# Step 4: Create BM25 retriever
bm25 = BM25Retriever.from_documents(documents, k=2)

# Step 5: Create FAISS vector retriever
embeddings = OpenAIEmbeddings(openai_api_key=API_KEY, model="text-embedding-3-small")
vector_store = FAISS.from_documents(documents, embeddings)
vector = vector_store.as_retriever(search_kwargs={"k": 2})

# Step 6: Combine into hybrid retriever (40% BM25, 60% FAISS)
hybrid = EnsembleRetriever(retrievers=[bm25, vector], weights=[0.4, 0.6])

# Step 7: Sample query
query = "Where can I find the Logitech mouse, and how much does it cost?"
results = hybrid.invoke(query)

for idx, doc in enumerate(results, 1):
    print(f"Result #{idx}\n{doc.page_content}\n---\n")
