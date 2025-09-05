from langchain_milvus import Milvus

URI = "./milvus_example.db"
embeddings = None

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
)