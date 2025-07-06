# TextbookGenie
This project is an AI chatbot that answers questions about the NCERT Science Class 10 textbook by leveraging Retrieval-Augmented Generation (RAG) with a local Llama 2 model via Ollama and text embeddings.

## Features

* **Intelligent Q&A**: Get precise answers to your questions based on the content of the NCERT Science Class 10 textbook.
* **Contextual Retrieval**: Utilizes advanced embedding models and vector search to find the most relevant sections of the textbook.
* **Generative AI**: Employs a powerful Large Language Model (LLM) to synthesize coherent and accurate answers from the retrieved context.
* **User-Friendly Interface**: An intuitive web interface built with Gradio for seamless interaction.
* **Local LLM Support**: Runs an LLM locally via Ollama, offering privacy and potentially faster response times without relying on cloud APIs.

## Technologies Used

* **PDF Processing**: `PyPDF2` for robust text extraction from PDF documents. (Conceptual `pytesseract` for scanned text fallback).
* **Text Processing**: `langchain` (RecursiveCharacterTextSplitter) for efficient text chunking.
* **Embeddings**: `sentence-transformers` (`msmarco-bert-base-dot-v5` - a MobileBERT-based model) for generating semantic representations of text.
* **Vector Database**: `FAISS` (Facebook AI Similarity Search) for high-performance similarity search on vector embeddings.
* **Large Language Model (LLM)**: `Ollama` running the `llama2` model for local, powerful text generation.
* **Database**: `sqlite3` for an in-memory database to store text chunks.
* **User Interface**: `Gradio` for creating an interactive web-based chatbot.
* **Image Processing **: `Pillow (PIL)` and `OpenCV (cv2)` are imported for potential future OCR enhancements.

## Setup and Installation

Follow these steps to get the project up and running on your local machine.

### 1. Prerequisites

* **Python 3.8+**: Ensure you have a compatible Python version installed.
* **Ollama**: Download and install Ollama from [ollama.com](https://ollama.com/).

### 2. Download the Textbook

* Place your `Ncert_science_10th.pdf` file in the same directory as your Python script.
    * *(Note: The provided code processes pages 0-70. You can adjust `TARGET_PDF_PAGE_START` and `TARGET_PDF_PAGE_END` in the script to include more pages if needed).*
### 3. Install Python Dependencies

Open your terminal or command prompt (or a new cell in your Jupyter Notebook) and run:

pip install PyPDF2 gradio langchain-community sentence-transformers faiss-cpu "langchain>=0.2.0" "huggingface_hub>=0.23.0"
# Note: faiss-gpu can be used for GPU acceleration if available.
# The specific langchain and huggingface_hub versions are recommended for compatibility. 

### 4. Pull the Llama2 Model with Ollama
In a separate terminal window, start Ollama and pull the llama2 model. This will download the model weights and make them available to your application. Keep this terminal window open as long as you want your RAG chatbot to run.
Run the command : ollama run llama2 in terminal. You should see >>> Send a message when the model is ready.

### 5. Run the Application
Now, navigate to the directory containing your script (script_name.py) and run it.
If you are in a Jupyter Notebook, simply run the cell containing the entire code.

The script will:
Process the PDF and chunk the text.
Initialize the embedding model and vector store.
Attempt to connect to your running Ollama server.
Launch the Gradio web interface.
You will see a URL printed in your terminal. Open this URL in your web browser.

Usage
Once the Gradio interface loads in your browser:
Ask a Question: Type your question related to the NCERT Science Class 10 textbook into the "Your Question" textbox.
Submit: Press Enter or click the "Submit Question" button.
Get Answer: The chatbot will display the answer in the "Chat History" area, referencing the textbook content.
Clear History: Use the "Clear" button to clear the conversation.
(Troubleshooting Tip: If the text box is non-interactive or you see setup errors, check your terminal for critical messages about Ollama or missing Python packages. Also check if there is any error while running your script)
