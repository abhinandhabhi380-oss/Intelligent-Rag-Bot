# Intelligent RAG Bot

An HR-focused Retrieval-Augmented Generation (RAG) assistant that answers employee questions using company policy documents stored in a local corpus. The project loads PDF files, splits them into chunks, embeds them using a sentence-transformer model, stores them in a FAISS vector database, and uses a Groq-hosted LLM to answer HR-related questions.

## Project Overview

This project includes:

- PDF document loading from the `zyro-dynamics-hr-corpus` folder
- Chunking with `RecursiveCharacterTextSplitter`
- Embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Vector search using FAISS
- Response generation using Groq + LangChain
- A guardrail check to reject out-of-scope questions
- A CSV-based evaluation workflow for generating a submission file

## Tech Stack

- Python
- LangChain
- LangChain Community
- LangChain Groq
- Hugging Face Embeddings
- FAISS
- PyPDF
- Sentence Transformers
- Groq LLM API
- Pandas

## Folder Structure

```text
Intelligent Rag Bot/
├── code.ipynb
├── README.md
├── submission.csv
├── test (1).csv
├── zyro-dynamics-hr-corpus/
│   └── ...PDF documents...
└── .env
```

## Prerequisites

Before running the project, make sure you have:

- Python 3.9 or later
- A Groq API key
- Access to the local HR policy PDF files in `zyro-dynamics-hr-corpus`

## Environment Setup

1. Create a `.env` file in the project root.
2. Add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Installation

Install the required dependencies:

```bash
pip install langchain langchain-community langchain-text-splitters langchain-huggingface langchain-groq langchain-google-genai langchain-openai langchain-core faiss-cpu pypdf sentence-transformers transformers torch huggingface_hub groq langsmith python-dotenv tiktoken pandas ipykernel
```

## How it Works

The notebook follows this flow:

1. Load all PDFs from the corpus directory.
2. Split the documents into smaller, overlapping chunks.
3. Generate vector embeddings for each chunk.
4. Store the embeddings in a FAISS vector store.
5. Retrieve the most relevant chunks for a user question.
6. Pass those chunks and the question to the Groq LLM.
7. Apply a guardrail prompt to ensure only HR-related queries are answered.

## Usage

Open the notebook `code.ipynb` and run the cells in order.

The main workflow is:

```python
result = ask_bot("How many casual leaves do I get per year?")
print(result["answer"])
```

The notebook also evaluates questions from `test (1).csv` and writes predictions to `submission.csv`.

## Guardrail Logic

The assistant checks whether a question is in scope before using the RAG pipeline. It rejects non-HR or unrelated queries such as:

- general knowledge questions
- coding help
- unrelated small talk

Example refusal message:

> I'm an HR assistant and can only help with questions about company HR policies (leave, reimbursement, code of conduct, etc.). I don't have information to answer that question.

## Important Notes

- The model is configured with:
  - `LLM_Provider = 'groq'`
  - `LLM_Model = 'openai/gpt-oss-20b'`
- The search retriever is configured to return the top 3 relevant document chunks.
- The project is designed for internal HR knowledge retrieval and is not a general-purpose chatbot.

## License

This project is for learning and internal use unless stated otherwise by the project owner.

## Future Improvements

Possible enhancements include:

- user-friendly web UI
- better source attribution
- multilingual support
- improved evaluation metrics
- persistent vector database storage
- advanced prompt tuning and retrieval optimization
