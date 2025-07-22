#!/usr/bin/env python
# coding: utf-8

# # Hybrid Search: BM25 + OpenAI Embeddings
# 
# Build a hybrid retriever combining:
# - BM25 (traditional keyword matching)
# - OpenAI embeddings (semantic search)
# 
# ### Why Hybrid Search Matters
# Traditional search methods like BM25 excel at exact keyword matching, making them perfect for finding documents containing specific phrases. However, they struggle with capturing semantic meaning — if a document discusses the same topic using different words, it won’t match. Neural embedding-based search, on the other hand, understands context and semantics but sometimes misses exact matches.
# 
# By combining both approaches, we get the best of both worlds: precise keyword matching and semantic understanding. This creates a more robust search system that can handle diverse queries effectively.

# ### Core Concepts
# #### 1. BM25 (Best Match 25) - Keyword Search
# - Statistical ranking algorithm for text retrieval
# - Full-text search excels at keyword matching and relevance scoring
# - Particularly strong for exact phrase matching
# 
# #### 2. Vector Search (FAISS)
# - Converts text into dense vectors in high-dimensional space
# - Captures semantic meaning beyond exact keywords
# - Enables similarity-based searching
# - Excels in identifying information that is close in meaning to the search query, even when there are no direct keyword matches
# 
# #### 3. Hybrid Approach
# - Combines results and strengths from both search methods
# - Uses weighted ensemble to balance precision and recall
# - Allows flexible tuning of search behavior

# ### System Architecture
# Let’s visualize how our hybrid search system works:
# 
# ![System Architecture](attachment:image.png)
# **Figure:** [System Architecture](https://photokheecher.medium.com/hybrid-search-made-easy-bm25-openai-embeddings-34e16a08cc17)
# 

# The diagram above illustrates our hybrid search system’s architecture. Both search paths operate **simultaneously** on the same query:
# - The top path uses BM25 to find exact matches in the text documents
# - The below path converts text to vectors and performs semantic similarity search
# - Results from both paths are combined using weighted averaging, where BM25 contributes 40% and FAISS contributes 60% to the final score
# 

# #### Create virtual environment
# In your command line/terminal, go to project directory, run:
# 
# ``` python -m venv hybrid```
# 
# This will create a directory with necessary files to run an isolated Python environment. Then run:
# 
# `source hybrid/bin/activate`
# 
# Once activated, use pip to install packages inside the virtual environment:
# 
# ```pip install <package-name>```

# #### Requirements
# Install these libraries if you haven't already:
# 
# - ```pip install openai```
# - ```pip install langchain```
# - `pip install -U langchain-community`
# - `pip install -U langchain-openai`
# - `pip install rank_bm25`
# - `pip install faiss-cpu` (depending on Python version)

# In[29]:


import faiss
print(faiss.__version__)   # should show a version, no ImportError


# #### Steps to use `.env` for `OPEN_API_KEY`
# 1. Install `python-dotenv`
# - `pip install python-dotenv`
# 
# 2. Create a `.env` file in your project folder (same directory as your script), and within the file type:
# `OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
# 
# Replace `sk-xxx...` with your actual OpenAI API key.
# 
# 3. Update your Python script to load the .env file:

# In[30]:


import os
from dotenv import load_dotenv

# Load environment variables from .env file (e.g. you OpenAI API key)
load_dotenv()

# Get the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("Missing OPENAI_API_KEY in environment")


# In[31]:


# Import retrievers and document schema
from langchain.retrievers import BM25Retriever, EnsembleRetriever
# BM25Retriever: (without Elasticsearch) pure keyword-based (lexical) retriever based on the Okapi BM25 algorithm  
#     A ranking function used in information retrieval systems to estimate the relevance of documents to a given search query
#     BM25 is a probabilistic model that considers term frequency, document length, and avg document length. (builds upon TF-IDF)
# EnsembleRetriever: combines multiple retrievers with a weighted voting scheme  

from langchain.schema import Document # Class for storing page content (string) and associated metadata (optional)

# Import OpenAI embedding model and FAISS vector store
from langchain_openai import OpenAIEmbeddings # OpenAIEmbeddings: LangChain wrapper over OpenAI’s embeddings API  
from langchain.vectorstores import FAISS # FAISS: in-memory vector index from Facebook AI Research for fast similarity search  

# Define a function to create a hybrid retriever
def hybrid_retriever(texts: list, bm25_top_k: int = 2, vector_top_k: int = 2, weights: list = [0.4, 0.6]):
    """
    Create a hybrid retriever combining BM25 and OpenAI Embeddings.
    Args:
        texts (list): List of text documents.
        bm25_top_k (int): Number of top results to fetch from BM25.
        vector_top_k (int): Number of top results fetch from vector (FAISS) similarity search.
        weights (list): Weights for ensemble retriever [BM25 weight, vector weight].
    Returns:
        EnsembleRetriever: A hybrid retriever combining BM25 and vector search.
    """

    # Step 1: Create LangChain Document objects from raw texts 
    documents = [Document(page_content=text) for text in texts] # list of Document objects, each containing page content (text) 

    # Step 2: Create BM25 retriever from a list of documents (created in step 1)
    bm25_retriever = BM25Retriever.from_documents(
        documents, # List of documents to use for retrieval
        k = bm25_top_k, # Set number of top results to retrieve (Number of documents to return)
        # preprocess_func = word_tokenize # custom tokenizer function (e.g. NLTK, spaCy)
    )

    # Step 3: Create Vector (FAISS) retriever/vector_store using OpenAI embeddings
    # set the embeddings function to OpenAI's text embedding. Other Langchain embedding models: https://js.langchain.com/docs/integrations/text_embedding/
    # An embedding is a vector (list) of floating point numbers. The distance between two vectors measures their relatedness. 
    # Small distances suggest high relatedness and large distances suggest low relatedness.
    embed_model = OpenAIEmbeddings(
        openai_api_key=api_key,
        model="text-embedding-3-small" # or "text-similarity-ada-002"
        # Optional: specify number of dimensions the resulting output embeddings should have
        # By default, the length of the embedding vector is 1536 for text-embedding-3-small or 3072 for text-embedding-3-large
        # dimensions=256 # To reduce the embedding's dimensions without losing its concept-representing properties, pass in the dimensions parameter
    ) 

    # Facebook AI Similarity Search (FAISS) is a library for efficient similarity search and clustering of dense vectors
    # Return VectorStore initialized from documents and embeddings
    vector_store = FAISS.from_documents(documents, embed_model) # Store vectors in FAISS, documents = list of Documents to add to the vectorstore
    # Wrap FAISS vector store as a LangChain retriever
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": vector_top_k}) # Basic top-k similarity search

    # Step 4: Combine BM25 and vector retrievers using EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], # a list of retrievers to ensemble
        weights=weights # a list of weights corresponding to the retrievers. Defaults to equal weighting for all retrievers.
    )

    return ensemble_retriever # returns the hybrid retriever


# In[ ]:


# Sample product list with pricing details
texts = [
    "The Apple iPhone 14 features a 6.1-inch display and advanced dual-camera system. Price: $799.",
    "The Samsung Galaxy S23 Ultra comes with a 200MP camera and a 5000mAh battery. Price: $1,199.",
    "The Sony WH-1000XM5 headphones offer industry-leading noise cancellation. Price: $399.",
    "Our organic avocados are sourced from sustainable farms in California. Price: $2.49 each.",
    "The Dyson V15 Detect is a cordless vacuum cleaner with laser dust detection. Price: $749.",
    "Lavazza Super Crema is a medium roast coffee with notes of honey and almonds. Price: $23.99 for a 2.2 lb bag.",
    "The Instant Pot Duo 7-in-1 is a multifunctional pressure cooker for fast meals. Price: $99.95.",
    "The Logitech MX Master 3S is an ergonomic wireless mouse for productivity. Price: $99.99.",
    "Philips Hue smart bulbs allow color customization and voice control via Alexa. Price: $49.99 for a 2-pack.",
    "The Lenovo ThinkPad X1 Carbon Gen 11 is a lightweight business laptop. Price: $1,649.",
    "Our bamboo cutting boards are durable, eco-friendly, and knife-safe. Price: $29.95 for a 3-piece set.",
    "Tide Pods 3-in-1 offer detergent, stain remover, and color protector in one capsule. Price: $21.99 for 81 pods."
]

# Create the hybrid retriever with equal weights
# BM25 returns top `bm25_top_k` documents by keyword match and FAISS (vector) returns top `vector_top_k` results by semantic similarity
# So, we can get up to `bm25_top_k` + `vector_top_k` results in total, but if there are overlaps/duplicate results
# then LangChain combines these using weighted score fusion and deduplicates based on document ID or content
# Result Merging: Implement deduplication and sorting mechanisms using score fusion algorithms
# Performance Optimization: Enhance efficiency through parallel retrieval and reduce redundant calculations with caching mechanisms.
retriever = hybrid_retriever(
    texts=texts,
    bm25_top_k=2,
    vector_top_k=2,
    weights=[0.5, 0.5]  # Equal weight to both BM25 and vector search
)

# Sample query to search the hybrid retriever
query = "What's the price of the Samsung phone?"

# Fetch relevant documents
results = retriever.invoke(query) # Invoke the retriever to get relevant documents, returns a List of relevant documents


# In[38]:


results # results is a list of Document objects containing the retrieved results


# In[ ]:


for idx, doc in enumerate(results, start=1): # idx starts at 1
    print(f"Result {idx}: {doc.page_content}")


# ### Before commiting to GitHub
# 
# #### Step 1: Generate requirements.txt for jupyter notebook
# `pip install pipreqs`
# 
# `pip install nbconvert`
# 
# Then, convert the hybrid_search.ipynb to the same name but python file ending in .py, and store in new folder named reqs:
# 
# `jupyter nbconvert --output-dir="./reqs" --to script hybrid_search.ipynb`    
# 
# `cd reqs`
# 
# `pipreqs`
# 
# So what we've done here is converted our notebook into a .py file in a new directory called reqs, then run pipreqs in the new directory. The reason for this is that pipreqs only works on .py files and I can't seem to get it to work when there are other files in the folder. The requirements.txt will be generated in the same folder.
# 
# #### Step 2: Generate .gitignore
# 
# Method 1: Visit https://github.com/github/gitignore, find the Python/gitignore, copy-paste it into your own .gitignore file.
# 
# Method 2: Visit https://www.toptal.com/developers/gitignore, type: python, macos, jupyter, visualstudiocode, virtualenv. 
# Then click “Create” and save the result as .gitignore.
