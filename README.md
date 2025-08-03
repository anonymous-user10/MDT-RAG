Code and dataset for the paper "Medical Decision Tree-Enhanced LLMs for Interpretable Reasoning"  

### Code  

`finetuning.py`: Code for fine-tuning the LLM to perform MDT extraction tasks.  

`text-index.py`: Code for building the text-based RAG index.  

`tree-index.py`: Code for building the MDT-RAG index.  

### Data  

`data/mdt.jsonl`: All decision tree data. The first 2,400 entries were extracted by DeepSeek R1, while the rest were extracted by the fine-tuned model.  

`data/plain-text.jsonl`: All text data. These texts are guidelines, serving both as the retrieval source for text-based RAG and as the source for MDT-RAG decision tree extraction.  

`data/mimic`: This folder is used to construct the evaluation dataset. After obtaining the MIMIC-IV dataset, execute the scripts inside this folder to build the evaluation dataset.
Due to the large size of the dataset, it has been placed in the release.
