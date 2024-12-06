# Arch-AI Final - Intelligent Document Research Tool

Welcome to the Arch-AI Final project! This tool provides a robust and scalable solution for intelligent document research, leveraging cutting-edge technologies in Natural Language Processing (NLP), vector embeddings, and information retrieval. The tool processes scraped data from various sources and offers a user-friendly interface to query and interact with the documents efficiently.

## Live Demo

You can access the live version of the project at the following link:  
[Arch-AI Final Streamlit App](https://arch-ai-final.streamlit.app/)

## Project Overview

The Arch-AI Final project aims to assist users in conducting intelligent research on documents. It uses a combination of web scraping, text summarization, and vector-based search techniques. Here's a breakdown of the project's main components:

- **Document Scraping**: Extracts data from various online sources, including text files and other document formats.
- **Text Preprocessing**: The raw data is cleaned and preprocessed to prepare it for NLP tasks.
- **Text Summarization**: Summarizes large blocks of text into concise summaries to help users quickly understand the content.
- **Vector Search**: Utilizes FAISS (Facebook AI Similarity Search) to create a searchable index from the document text, enabling fast and accurate retrieval of relevant information based on user queries.
- **Streamlit Interface**: A web-based interface built with Streamlit that allows users to interact with the tool and retrieve summarized results and insights from documents.

## Features

- **Web Scraping**: Automatically collects data from pre-configured URLs and stores it in a structured format.
- **Document Summarization**: Automatically summarizes lengthy documents into shorter versions for easier consumption.
- **Searchable Index**: Creates an index of documents using vector embeddings for efficient searching.
- **Intelligent Querying**: Users can type queries to retrieve relevant summaries or information from the documents in the index.
- **Interactive User Interface**: The user-friendly interface built with Streamlit enables a seamless experience to interact with the tool.

## Technologies Used

- **Python**: The core programming language used to implement the project.
- **Streamlit**: The framework used for building the web-based user interface.
- **FAISS**: Used for creating and querying vector indexes.
- **OpenAI**: For NLP tasks such as document summarization and text analysis.
- **Pandas & NumPy**: Data manipulation libraries used for data processing and handling.
- **JSONL**: Data format used for storing structured data (interactions and document information).

## Requirements

To run this project locally, you need to have Python installed on your system. It is recommended to use a virtual environment to manage dependencies.

### Prerequisites:

- Python 3.x
- pip (Python package installer)

### Dependencies:

- **streamlit**: For building and running the web interface.
- **openai**: For utilizing OpenAI APIs to perform NLP tasks.
- **faiss-cpu**: For vector search operations.
- **pandas**: For data manipulation.
- **numpy**: For numerical operations.
- **jsonlines**: For reading and writing JSONL files.

You can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

### Running the Project Locally
To run the project on your local machine, follow these steps:

1. Clone the repository: If you haven't already cloned the repository, run the following command:

```bash
git clone https://github.com/Ashishlathkar77/arch-ai-final.git
```

2. Navigate to the project directory:

```bash
cd arch-ai-final
```

3. Install the dependencies: Make sure you have all the required libraries installed by running:

```bash
pip install -r requirements.txt
```

4. Run the application: To start the Streamlit app, use the following command:

```bash
streamlit run app.py
```

5. Open the application in your browser: After running the command, Streamlit will automatically open a local server, usually accessible at http://localhost:8501/. You can interact with the web interface to perform document research and queries.

### How the System Works
- Data Scraping: The project collects raw data from a variety of pre-configured sources (e.g., URLs, documents).

- Text Processing: The scraped data is preprocessed, including cleaning, tokenization, and other necessary transformations.

- Text Summarization: The preprocessed data is then summarized using OpenAI’s language model APIs to condense lengthy documents into more manageable summaries.

- Vectorization: After summarizing the documents, they are transformed into vector representations (embeddings) using advanced NLP techniques, and stored in FAISS for fast searching.

- Search and Query: Users can enter a query through the Streamlit interface, and the system will return the most relevant documents or summaries by performing a similarity search using the FAISS index.

### Contributing
We welcome contributions to this project! If you'd like to contribute, please follow these steps:

- Fork the repository
- Create a new branch for your changes
- Commit your changes
- Push your changes to your fork
- Create a pull request
