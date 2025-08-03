# 创建文档检索索引, 用标准方法就行, 不用自己组织node
# nohup python llamaindex/construct_index.py > log/construct_index616.log 2>&1 &

import os
import json
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader,StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.schema import TextNode,BaseNode
from llama_index.core import StorageContext,  get_response_synthesizer,load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
Settings.embed_model = HuggingFaceEmbedding(model_name = "models/GTE/bge-base-en-v1.5",trust_remote_code=True)




def initialize_from_files(file_path,index_save_path,chunk_size,collection_name):
    
    client = qdrant_client.QdrantClient(path=index_save_path)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
 
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f.readlines()]
    nodes = []
    for line in lines:

        sentence_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=100)
        chunks = sentence_splitter.split_text(line['plain_text'])
        for i, chunk in enumerate(chunks):
            node = TextNode(
                text=chunk,
                metadata={'id': line['id'], 'chunk_id': i,'article_title': line['article_title'],'name':line['name'],'source':line['source']},
            )
            nodes.append(node)
        nodes.append(node)
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    storage_context = StorageContext.from_defaults(docstore=docstore,vector_store=vector_store)
    

    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=Settings.embed_model,
        storage_context=storage_context,
        show_progress=True
        )
    return index

    
def load_index_from_file(path:str,collection_name):
    client = qdrant_client.QdrantClient(path=path)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store,embed_model=Settings.embed_model)
    
    return index



if __name__ == '__main__':
    index = initialize_from_files(file_path="",chunk_size=512,index_save_path='llamaindex/qdrant',collection_name="RAG-index0616-bge")
    
    query_engine = index.as_query_engine(
        llm = HuggingFaceLLM(
            model_name="models/Qwen2.5-0.5B-Instruct",
            tokenizer_name="models/Qwen2.5-0.5B-Instruct",
            context_window=30000,
            max_new_tokens=2000,
            generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
            device_map="auto",))
    response = query_engine.query("")
    print(response.response)
    
