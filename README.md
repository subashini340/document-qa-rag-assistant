# 📚 Document Q&A Assistant using RAG

A PDF-based Question Answering Assistant built using Retrieval-Augmented Generation (RAG), Streamlit, LangChain, ChromaDB, local Hugging Face embeddings, and Google Gemini AI.

## 🚀 Features

- Upload PDF documents
- Extract text from PDF pages
- Split text into smaller chunks
- Generate embeddings locally using Hugging Face
- Store and retrieve relevant content using ChromaDB
- Ask questions about uploaded documents
- Generate answers using Google Gemini AI
- Display retrieved source page numbers

## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain
- ChromaDB
- Hugging Face Sentence Transformers
- Google Gemini API
- PyPDF

## ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_FOLDER
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## 🔑 API Configuration

Create a `.env` file in the project folder and add your Gemini API key:

```text
GEMINI_API_KEY=your_actual_api_key
```

**Important:** Never upload your `.env` file or API key to GitHub.

## ▶️ Run the Application

```bash
python -m streamlit run main.py
```

Open the local URL displayed in the terminal.

## 💡 How It Works

1. Upload a PDF document.
2. Extract and split its text into chunks.
3. Generate local embeddings using Hugging Face.
4. Store the embeddings in ChromaDB.
5. Retrieve relevant chunks for the user's question.
6. Generate an answer using Google Gemini.
7. Display the answer and retrieved source pages.

## 🎯 Project Objective

To build a document-based question-answering system that uses Retrieval-Augmented Generation to retrieve relevant information from PDF documents and generate context-based answers.

## 👩‍💻 Author

Subashini E