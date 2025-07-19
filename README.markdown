# Multi-Hop RAG Chatbot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

A powerful Streamlit-Based web application that allows users to upload PDF documents and interact with an AI chatbot powered by a multi-hop retrieval-augmented generation (RAG) technique. Designed for researchers, students, and knowledge enthusiasts, this chatbot processes complex queries by breaking them into sub-questions, retrieving relevant document chunks, and generating detailed, conversational answers.

## Features

- **PDF Document Upload**: Upload multiple PDF files for processing and indexing.
- **Multi-Hop RAG**: Breaks down complex queries into sub-questions for precise information retrieval.
- **Conversational AI**: Delivers detailed, conversational answers based on document content.
- **Interactive Interface**: Built with Streamlit for a seamless and intuitive user experience.
- **Document Management**: Easily clear uploaded documents to start fresh.
- **Retrieval Details**: Displays sub-questions and retrieved document snippets for transparency.

## Technologies Used

| Technology            | Description                                                  |
|-----------------------|--------------------------------------------------------------|
| **Python 3.8+**       | Core programming language for the application.                |
| **Streamlit**         | Creates the interactive web-based user interface.             |
| **LangChain**         | Framework for building applications with language models.     |
| **LangChain-Ollama**  | Integrates Ollama for text generation and embeddings.         |
| **Chroma**            | Vector database for storing and retrieving document chunks.   |
| **PyPDFLoader**       | Loads and parses PDF documents.                               |
| **RecursiveCharacterTextSplitter** | Splits documents into manageable chunks for indexing. |

## Prerequisites

- **Python 3.8 or later**: Download from [python.org](https://www.python.org/downloads/).
- **Ollama**: Local AI server for running language models. Install from [ollama.com](https://ollama.com).
- **Internet Access**: Required for downloading Ollama models and accessing the app.

## Installation

1. **Install Python**:
   - Ensure Python 3.8 or later is installed. Verify with:
     ```bash
     python --version
     ```

2. **Install Ollama**:
   - Follow instructions at [ollama.com](https://ollama.com).
   - Pull the required models:
     ```bash
     ollama pull qwen2.5:latest
     ollama pull nomic-embed-text:latest
     ```
   - Start the Ollama server:
     ```bash
     ollama serve
     ```

3. **Clone the Repository**:
   ```bash
   git clone https://github.com/armanjscript/Multi-Hop-RAG-Chatbot.git
   cd Multi-Hop-RAG-Chatbot
   ```

4. **Install Python Libraries**:
   ```bash
   pip install streamlit langchain langchain-ollama langchain-chroma langchain-community
   ```

## Usage

1. **Run the Application**:
   - Ensure the Ollama server is running:
     ```bash
     ollama serve
     ```
   - Start the Streamlit app:
     ```bash
     streamlit run multi_hop_rag.py
     ```
   - Open your browser and navigate to `http://localhost:8501`.

2. **Upload Documents**:
   - In the sidebar, use the file uploader to select PDF files.
   - Click "Process Documents" to load and index the documents into the vector store.

3. **Chat with the AI**:
   - Enter a question in the chat input box (e.g., "What is the capital of France and its population?").
   - The chatbot will break the question into sub-questions, retrieve relevant document chunks, and generate a detailed answer.
   - View retrieval details (sub-questions and document snippets) in the expandable section.

4. **Clear Documents**:
   - Click "Clear Documents" in the sidebar to remove all uploaded files and reset the vector store.

## Multi-Hop RAG Technique

The multi-hop RAG technique enhances the chatbot’s ability to handle complex queries by breaking them into smaller, manageable sub-questions. Here’s how it works:

1. **Question Decomposition**:
   - The user’s query is analyzed by the `qwen2.5:latest` model to generate 2-3 sub-questions.
   - Example: For "What is the capital of France and its population?", sub-questions might be:
     - "What is the capital of France?"
     - "What is the population of Paris?"

2. **Document Retrieval**:
   - For each sub-question, the Chroma vector store retrieves the top 3 relevant document chunks using the `nomic-embed-text:latest` embeddings.
   - Duplicates are removed to ensure efficiency.

3. **Answer Generation**:
   - The retrieved documents are combined and passed to the `qwen2.5:latest` model along with the original question.
   - The model generates a detailed, conversational answer based on the context.

### Diagram of the Multi-Hop RAG Technique

```
[User Question] --> [Generate Sub-Questions] --> [Retrieve Docs for Each Sub-Question] --> [Combine Unique Docs] --> [Generate Answer]
```

**Example Workflow**:
- **Input**: "What is the capital of France and its population?"
- **Sub-Questions**: 
  - "What is the capital of France?"
  - "What is the population of Paris?"
- **Retrieval**: Fetch relevant document chunks for each sub-question.
- **Combination**: Merge unique document chunks.
- **Output**: Generate a response like: "The capital of France is Paris, with a population of approximately 2.2 million."

## Limitations

- **Document Quality**: Answer accuracy depends on the quality and relevance of uploaded PDFs.
- **Model Performance**: The effectiveness of sub-question generation and answer quality relies on the `qwen2.5:latest` model.
- **File Handling**: Uploaded PDFs are saved locally and deleted when cleared, which may affect other files with the same name.
- **Processing Time**: Large PDFs or complex queries may take longer to process.

## Contributing

Contributions are welcome! To contribute:
- Fork the repository.
- Make changes, ensuring they align with the project’s coding style.
- Submit a pull request with a clear description of your changes.
- Include tests to maintain quality.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Disclaimer

This tool is for informational purposes only. Answers depend on the quality of uploaded documents and model performance. Always verify critical information with reliable sources.

## Contact

For questions or feedback, contact [Arman Daneshdoost] at [armannew73@gmail.com].