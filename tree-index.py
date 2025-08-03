
import os
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
import sys
from utils.tree_utils import convert_trees,add_parent
from utils.clean_answer import clean_data
from utils.tree_utils import summarize_tree,find_tree_path_through_similarity
import json
from tqdm import tqdm
from utils.similarity import ST

Settings.embed_model = HuggingFaceEmbedding(model_name = "models/GTE/bge-base-en-v1.5")

def initialize_nodes_from_file(file_path):
    '''
    read jsonl file and convert to nodes,return nodes list
    '''
    tree_list = []
    with open(file_path, 'r') as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line)
            answer = line.get('answer')
            summary_list = line.get('summary')
            # clean and convert 
            cleaned_answer = clean_data(answer)
            trees = convert_trees(cleaned_answer)
            if trees != []:
                assert len(trees) == len(summary_list)
                source_id = line.get('id')
                source_name = ''
                if line.get('name') is not None:
                    source_name = line['name']
                if line.get('article_title') is not None:
                    article_title = line['article_title']
        
                for i, tree in enumerate(trees):
                    tree = str(tree)
                    summary = summary_list[i]
                    result = {
                        'source_id': source_id,
                        'article_title': article_title,
                        'source_name': source_name,
                        'tree_order': i,
                        'answer': tree,
                        'summary': summary,
                    }
                    tree_list.append(result)
    nodes = []
    
    for i in tree_list:
        source_id = i["source_id"]
        source_name = i["source_name"]
        tree_order = i["tree_order"]
        tree = i["answer"]
        summary = i["summary"]
        article_title = i["article_title"]
        node = TextNode(text=summary,metadata={"tree":tree,'source_id':source_id,'article_title':article_title,'source_name':source_name,'tree_order':tree_order})
        # 不使用metadata嵌入
        node.excluded_embed_metadata_keys = ['tree','source_id','article_title','source_name','tree_order']
        node.excluded_llm_metadata_keys = ['tree','source_id','article_title','source_name','tree_order']
        
        nodes.append(node)
    return nodes

def save_nodes_to_file(nodes,path,collection_name):
    # 从零开始创建并保存
    client = qdrant_client.QdrantClient(
    # you can use :memory: mode for fast and light-weight experiments,
    # it does not require to have Qdrant deployed anywhere
    # but requires qdrant-client >= 1.1.1
    # location=":memory:"
    # otherwise set Qdrant instance address with:
    # url="http://:"
    # otherwise set Qdrant instance with host and port:
    path=path
    # set API KEY for Qdrant Cloud
    # api_key="",
    )
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    
    
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    storage_context = StorageContext.from_defaults(docstore=docstore,vector_store=vector_store)
    
    vector_index = VectorStoreIndex(nodes=nodes,embed_model=Settings.embed_model,storage_context=storage_context,show_progress=True,)

    #vector_index.storage_context.persist(path)
    return vector_index

    
def load_index_from_file(path,collection_name):
    client = qdrant_client.QdrantClient(
        path=path
    )
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=Settings.embed_model)
    return index


    
with open('utils/prompt/total_tree_query.txt', 'r', encoding='utf-8') as f:
    total_tree_query_prompt = f.read()


total_tree_prompt = PromptTemplate(
    template=total_tree_query_prompt,
)

def gen_question_for_dataset_total_tree(retriever, dataset_path_list):

    for dataset_path in dataset_path_list:
        dir_path = os.path.dirname(dataset_path)
        base_name = os.path.basename(dataset_path)

        save_dataset_path = os.path.join(dir_path, f'{base_name.split(".")[0]}_tree_rag_question_total_tree_0719_ablation_wo_tree_sum.jsonl')
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data_list = [json.loads(line) for line in f]

        new_data_list = []
        for data in tqdm(data_list, desc=f"Processing {base_name}"):

            rag_text = data['rag_text']
            nodes = retriever.retrieve(rag_text)

            retrieved_tree_list = []

            metadata_list = []
            for n in nodes: 
                retrieved_tree_list.append(
                    eval(n.metadata['tree'])
                    )
                metadata_list.append({
                        'source_id': n.metadata['source_id'],
                        'article_title': n.metadata['article_title'],
                        'source_name': n.metadata['source_name'],
                        'tree_order': n.metadata['tree_order'],
                        'tree': n.metadata['tree'],
                        'text': n.text,
                     })

            tree_num=25
            tree_in_text_num = 6
            tree_path_result = [str(i) for i in retrieved_tree_list]
            tree_path_result = tree_path_result[:tree_num]
            metadata_list = metadata_list[:tree_num]
            
            new_data_list.append([tree_path_result,metadata_list])

        dataset_path_temp = save_dataset_path
        data_list_temp  = new_data_list
        print(dataset_path_temp)

        assert len(data_list_temp) == len(data_list)
        
        with open(dataset_path_temp, 'w', encoding='utf-8') as f:
            for i, singel_data_temp_metadata in tqdm(enumerate(data_list_temp),total=len(data_list_temp), desc=f"writing in dataset path  {dataset_path_temp}"):
                
                metadata = singel_data_temp_metadata[1]
                singel_data_temp = singel_data_temp_metadata[0]
                
                rag_tree_text = ''
                singel_data_temp = singel_data_temp[:tree_in_text_num] # 只取前3个树, 减少长度
                for j , tree in enumerate(singel_data_temp):
                    rag_tree_text += f'Medical tree{j+1}:\n{tree}\n'
                
                question = data_list[i]['question']
                gen_question= total_tree_prompt.format(context_str=rag_tree_text, query_str=question)
                f.write(json.dumps({
                        **data_list[i],
                        'gen_question': gen_question, 
                        'metadata': metadata,
                        },ensure_ascii=False) + '\n')
        


        
if __name__ == '__main__':
    # save
    nodes = initialize_nodes_from_file('datasets/qwen_gen/gen/0621/qwen1m_ans_0621_all-select-question_WITH_r1_prompt_with_id_summary_title.jsonl')

    
    vector_index = save_nodes_to_file(nodes=nodes,path='llamaindex/qdrant',collection_name='tree-index-0719-ablation')
    #read
    #vector_index = load_index_from_file(path='llamaindex/qdrant',collection_name='tree-index-0510')
    
    retriever = vector_index.as_retriever(similarity_top_k=25)

    dataset_list = [
    'datasets/mimic_2k_0508_medicationsadd_rag_text.jsonl',
    'datasets/mimic_2k_0508_treatmentadd_rag_text.jsonl',
    ]
    gen_question_for_dataset_total_tree(retriever, dataset_list)
    #gen_question_for_dataset(retriever, dataset_list)