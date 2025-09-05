from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="granite-embedding:30m",
)
