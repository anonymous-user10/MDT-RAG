from sentence_transformers import SentenceTransformer

class ST():
    def __init__(self, 
                 model_name="models/GTE/gte-Qwen2-1.5B-instruct",
                 device="cuda:0"
                 ):
        self.device = device
        self.model = SentenceTransformer(
                                        model_name, 
                                        trust_remote_code=True,
                                        device=self.device
                                         )


    def similarity(self, text_list1, text_list2,prompt_name="triple"):
        if text_list1 == []:
            if text_list2 == []:
                return []
            else:
                return [[0 for _ in range(len(text_list2))]]
        elif text_list2 == []:
            return [[0 for _ in range(len(text_list1))]]
        query_embeddings = self.model.encode(text_list1, prompt_name=prompt_name)
        document_embeddings = self.model.encode(text_list2)
        scores = (query_embeddings @ document_embeddings.T) * 100
        return scores.tolist()

if __name__ == '__main__':
    '''
    ,
    "path":"Given a medical reasoning path composed of a series of conditions and a final conclusion, both conditions and conclusions are represented using triplets. For negation of triplets, use the prefix \"doesn't satisfy\" to indicate negation. You need to find a medical reasoning path with similar relationships.\n The given medical reasoning path is:"
    
    '''
    
    queries = [
        "how much protein should a female eat",
        "summit define",
        "lalalabalabala"
    ]
    documents = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    ]
    ST = ST(device='cuda:3')
    scores = ST.similarity(queries, documents)
    print(len(scores))
    print(len(scores[0]))
    print(scores)
    for i in range(len(scores)):
        if any(x > 65 for x in scores[i]):
            print(i)
    
    
    
    