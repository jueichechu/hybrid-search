# Hybrid Search

**A hybrid information retrieval tool leveraging BM25 and FAISS-powered embeddings**

## Getting Started

### Steps

1.  Clone this repository:

    ```bash
    git clone https://github.com/jueichechu/hybrid-search.git
    cd hybrid-search
    ```

2.  Create and activate a virtual environment:

    ```bash
    python -m venv .venv
    source .venv/bin/activate   # macOS/Linux
    ```

3.  Install Python dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4.  Create a `.env` file in the project root:

    ```text
    OPENAI_API_KEY=your_openai_api_key_here
    ```

5.  Database & Index Setup:

    Follow the detailed instructions in `db_setup/db_setup.md` to initialize your database and indices.

### Run

    python hybrid_search.py
