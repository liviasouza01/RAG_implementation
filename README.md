# RAG Implementation

This is a simple implementation of a Retrieval-Augmented Generation (RAG) system using Langchain, OpenAI, and Pinecone. The system retrieves relevant documents from a vector store and uses a language model to generate responses based on the retrieved information.

## Features

- **OpenAI Embeddings**: Utilizes OpenAI's embeddings to encode text data.
- **Pinecone Vector Store**: Stores and retrieves document embeddings for efficient similarity search.
- **Langchain**: Integrates with Langchain's language models and prompt templates.
- **Retrieval-Augmented Generation**: Combines document retrieval with language model generation to provide contextually relevant answers.

## Prerequisites

- Python 3.7+
- An OpenAI API key
- A Pinecone API key and index
- A Langchain API key

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/liviasouza01/RAG_implementation.git
   cd RAG_implementation
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables. Create a `.env` file in the root directory or use the provided `.env-example` file:

   ```bash
   cp .env-example .env
   ```

   Fill in your API keys and the Pinecone index name in the `.env` file.

## Usage

Run the main script to start the RAG system:

The script will retrieve relevant documents from the Pinecone vector store and generate a response using the OpenAI language model.

## Configuration

- **Query**: Modify the `query` variable in `main.py` to change the input question.
- **Environment Variables**: Ensure your `.env` file contains the correct API keys and index name.


