import os 
import re
import json
import sys
import gc
from tqdm import tqdm
from utils.similarity import ST
from utils.clean_answer import clean_data
from utils.bertscore import BERTScore
from utils.tree_utils import (
    tree_traversal,
    path2str,
    empty_list_metric,
    multi_text_tree_list_format_metric,
    convert_trees)
import copy
import torch
import time

import logging
from datetime import datetime
now = datetime.now()
formatted_data = now.strftime("%m%d")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(f'log/eval_metric/{__name__}_{formatted_data}.log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s %(funcName)s [line:%(lineno)d] - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)



import signal
from contextlib import contextmanager

class TimeoutError(Exception):
    """自定义超时异常"""
    pass

@contextmanager
def time_limit(seconds):
  
    def signal_handler(signum, frame):
        raise TimeoutError(f"{seconds} ")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)



    
class MultiWayTreeMetric:
    def __init__(self,tokenizer,
                 sentence_transformer_model='models/GTE/gte-Qwen2-1.5B-instruct',
                 bert_model='models/GTE/biobert-v1.1',
                 device='cuda:0'
                 ):
        self.device = device
        self.tokenizer = tokenizer
        self.sentence_transformer_model = sentence_transformer_model
        self.bert_model = bert_model
        #self.BERTScore = None
        #self.similarity = None
        self.predicts = []
        self.references = []
        self.predicts_cleaned = [] 
        self.references_cleaned = []
        self.predicts_cleaned_pos = []
        self.references_cleaned_pos = []
        self.predicts_converted = []
        self.references_converted = []
        self.triple_thereshold = [45,50,55,60,65,70,75,80,85,90]
        self.path_threshold = [45,50,55,60,65,70,75,80,85,90]
        
    

    def triple_metric(self,ref_triple_list,pred_triple_list,threshold=[55,65]):

        if type(ref_triple_list[0]) == dict:
            ref_triple_list = [list(x.values()) for x in ref_triple_list]
            pred_triple_list = [list(x.values()) for x in pred_triple_list]
        p_dict,r_dict,f1_dict = {},{},{}
        for single_threshold in threshold:
            p_dict[f'p_{single_threshold}'] = []
            r_dict[f'r_{single_threshold}'] = []
            f1_dict[f'f1_{single_threshold}'] = []
        
        
        for ref,pred in zip(ref_triple_list,pred_triple_list):
            ref = [', '.join(x) for x in ref]
            pred = [', '.join(x) for x in pred]
            sm_matrix = self.similarity.similarity(pred,ref,prompt_name='triple')
            for single_threshold in threshold:
                tp_p = 0
                tp_r = 0
                for i in range(len(sm_matrix)):
                    if any(x>single_threshold for x in sm_matrix[i]):
                        tp_p += 1
                transposed_matrix = list(zip(*sm_matrix)) if sm_matrix else []
                for i in range(len(transposed_matrix)):
                    if any(x>single_threshold for x in transposed_matrix[i]):
                        tp_r += 1
                p = tp_p/(len(pred)+1e-8)
                r = tp_r/(len(ref)+1e-8)
                f1 = 2*p*r/(p+r+1e-8)
                p_dict[f'p_{single_threshold}'].append(p)
                r_dict[f'r_{single_threshold}'].append(r)
                f1_dict[f'f1_{single_threshold}'].append(f1)
        result = {}
        for single_threshold in threshold:
            p = round(sum(p_dict[f'p_{single_threshold}'])/len(p_dict[f'p_{single_threshold}']) * 100,2)
            r = round(sum(r_dict[f'r_{single_threshold}'])/len(r_dict[f'r_{single_threshold}']) * 100,2)
            f1 = round(sum(f1_dict[f'f1_{single_threshold}'])/len(f1_dict[f'f1_{single_threshold}']) * 100,2)
            result[f'triple_macro_p_{single_threshold}'] = p
            result[f'triple_macro_r_{single_threshold}'] = r
            result[f'triple_macro_f1_{single_threshold}'] = f1
            
        return result
    
    
    def path_metric(self,ref_triple_dict_list,pred_triple_dict_list,ref_path_list,pred_path_list,threshold=[55]):
        
        p_dict,r_dict,f1_dict = {},{},{}
        for single_threshold in threshold:  
            p_dict[f'p_{single_threshold}'] = []
            r_dict[f'r_{single_threshold}'] = []
            f1_dict[f'f1_{single_threshold}'] = []
        for ref_triple_dict,pred_triple_dict,ref_path,pred_path in zip(ref_triple_dict_list,pred_triple_dict_list,ref_path_list,pred_path_list):
            ref_single_tree_path = path2str(ref_triple_dict,ref_path)
            pred_single_tree_path = path2str(pred_triple_dict,pred_path)
            sm_matrix = self.similarity.similarity(pred_single_tree_path,ref_single_tree_path,prompt_name='path')
            for single_threshold in threshold:
                tp_p = 0
                tp_r = 0
                for i in range(len(sm_matrix)):
                    if any(x>single_threshold for x in sm_matrix[i]):
                        tp_p += 1
                transposed_matrix = list(zip(*sm_matrix)) if sm_matrix else []
                for i in range(len(transposed_matrix)):
                    if any(x>single_threshold for x in transposed_matrix[i]):
                        tp_r += 1
                        
                        
                p = tp_p/(len(pred_single_tree_path)+1e-8)
                r = tp_r/(len(ref_single_tree_path)+1e-8)
                f1 = 2*p*r/(p+r+1e-8)
                p_dict[f'p_{single_threshold}'].append(p)
                r_dict[f'r_{single_threshold}'].append(r)
                f1_dict[f'f1_{single_threshold}'].append(f1)
                
        result = {}
        for single_threshold in threshold:
            p = round(sum(p_dict[f'p_{single_threshold}'])/len(p_dict[f'p_{single_threshold}'])*100,2)
            r = round(sum(r_dict[f'r_{single_threshold}'])/len(r_dict[f'r_{single_threshold}'])*100,2)
            f1 = round(sum(f1_dict[f'f1_{single_threshold}'])/len(f1_dict[f'f1_{single_threshold}'])*100,2)
            result[f'path_macro_p_{single_threshold}'] = p
            result[f'path_macro_r_{single_threshold}'] = r
            result[f'path_macro_f1_{single_threshold}'] = f1
        
        return result
            
    def triple_path_metric(self):

        ref_triple_dict_list=[]
        pred_triple_dict_list=[]
        ref_path_list=[]
        pred_path_list=[]
        for ref_tree,pred_tree in zip(self.references_converted,self.predicts_converted):
            try:
                with time_limit(5):  
                    ref_result = tree_traversal(ref_tree)
                    pred_result = tree_traversal(pred_tree)
            except TimeoutError as e:
                print("遇到错误, 打印并推出")
                print(ref_tree)
                print('\n\n')
                print(pred_tree)
                exit()
            ref_triple_dict_list.append(ref_result['triples_dict'])
            pred_triple_dict_list.append(pred_result['triples_dict'])
            ref_path_list.append(ref_result['paths'])
            pred_path_list.append(pred_result['paths'])
        print('triple metric start')
        triple_result = self.triple_metric(ref_triple_dict_list,pred_triple_dict_list,self.triple_thereshold)
        print('path metric start')
        path_result = self.path_metric(ref_triple_dict_list,pred_triple_dict_list,ref_path_list,pred_path_list,self.path_threshold)
        triple_path_result = {}
        triple_path_result.update(triple_result)
        triple_path_result.update(path_result)
        return triple_path_result
        
    def bert_score_metric(self,predicts,references):
        f1_list = []
        for predict,reference in zip(predicts,references):
            f1 = self.BERTScore.sentence_bert_score(str(predict),str(reference))
            f1_list.append(f1)
            
        return {'bert_score':round(sum(f1_list)/len(f1_list)*100,2)}
    def length_weighted_bert_score_metric(self,predicts,references):
        total_length = 0
        f1_list = []
        for predict,reference in zip(predicts,references):
            f1 = self.BERTScore.sentence_bert_score(str(predict),str(reference))
            f1 = f1 * (len(predict)+len(reference))/2
            total_length += (len(predict)+len(reference))/2
            f1_list.append(f1)
            
        return {
            'length_weighted_bert_score':round(sum(f1_list)/total_length*100,3)
        }
        
        
    def add_batch(self,predicts=None,references=None):

        torch.cuda.empty_cache()
        gc.collect()
        
        self.BERTScore = BERTScore(model_path=self.bert_model,num_layers=8,device=self.device)
        self.similarity = ST(model_name=self.sentence_transformer_model,device=self.device)
        
        
        for predict,reference in zip(predicts,references):
            # decode
            if reference[0].isdecimal():# 有可能已经是tokenized了
                predict_decoded = self.tokenizer.decode(predict,skip_special_tokens=True)
                reference_decoded = self.tokenizer.decode(reference,skip_special_tokens=True)
            # 其实label已经是cleaned的了, 但是为了保险起见, 还是再洗一下
            else:
                predict_decoded = predict
                reference_decoded = reference
                
            predict_cleaned = clean_data(predict_decoded,-1)
            if isinstance(predict_cleaned,list):
                predict_cleaned = predict_cleaned[1]
                self.predicts_cleaned_pos.append(False)
            else:
                self.predicts_cleaned_pos.append(True)
            reference_cleaned = clean_data(reference_decoded,-1)
            if isinstance(reference_cleaned,list):
                reference_cleaned = reference_cleaned[1]
                self.references_cleaned_pos.append(False)
            else:
                self.references_cleaned_pos.append(True)
            
            predict_convert_trees = convert_trees(copy.deepcopy(predict_cleaned))
            reference_convert_trees = convert_trees(copy.deepcopy(reference_cleaned))
            
            self.predicts.append(predict_decoded)
            self.references.append(reference_decoded)
            self.predicts_cleaned.append(predict_cleaned)
            self.references_cleaned.append(reference_cleaned)
            self.predicts_converted.append(predict_convert_trees)
            self.references_converted.append(reference_convert_trees)
    
    def compute(self):
        print("compute start")
        result = {}
        print("multi_text_tree_list_format_metric start")
        format_result = multi_text_tree_list_format_metric(self.predicts_cleaned)
        print('triple_path_metric start')
        tuiple_path_result = self.triple_path_metric()
        empty_result = empty_list_metric(predict_origin=self.predicts,reference_origin=self.references,
            predict_cleaned=self.predicts_cleaned,reference_cleaned=self.references_cleaned,
            predicts_converted=self.predicts_converted,references_converted=self.references_converted,
            predicts_cleaned_pos=self.predicts_cleaned_pos,references_cleaned_pos=self.references_cleaned_pos)
        print("bertscore start")
        bertscore_result = self.bert_score_metric(self.predicts,self.references)
        print("length_weighted_bertscore start")
        legth_weighted_bertscore_result = self.length_weighted_bert_score_metric(self.predicts,self.references)
        
        result.update(format_result)
        result.update(bertscore_result)
        result.update(tuiple_path_result)
        result.update(empty_result)
        result.update(legth_weighted_bertscore_result)
        self.predicts = [] # decoded
        self.references = []    
        self.predicts_cleaned = []
        self.references_cleaned = []
        self.predicts_converted = []
        self.references_converted = []
        self.predicts_cleaned_pos = []
        self.references_cleaned_pos = []
        del self.BERTScore 
        del self.similarity 
        
        torch.cuda.empty_cache()
        gc.collect()
        print("compute finished")
        return result
    
if __name__ == '__main__':
    pass
    
        




    