import re 
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
import sys
from nltk.metrics import edit_distance
from utils.dnfcnf import Parser
import json
import warnings
import copy
from collections import deque

warnings.filterwarnings("ignore", category=SyntaxWarning)

triples_str_dict = {}



with open('utils/prompt/triple2nl.json','r',encoding='utf-8') as f:
    triples_str_dict = json.load(f)

with open('utils/prompt/prompt_path.txt','r',encoding='utf-8') as f:
    prompt_path = f.read()

def add_parent(node,parent):
    node['parent'] = parent
    for child in node['children']:
        add_parent(child,node)
    return node

def edit_distance_metric(a,b,t=0.85):
    s = edit_distance(a,b)
    max_len = max(len(a),len(b))
    if max_len == 0:
        k = 0
    else:
        k = 1 - s/max_len
    if k >= t:
        return True
    else:
        return False

def triple2str(triple,negative=False):
    if len(triple) != 3:
        logger.error(f"triples2str error, in triple2str, triple length not equal to 3 : {triple}")
        return ''
    head = triple[0]
    relation = triple[1]
    tail = triple[2]
    if head in triples_str_dict.keys():
        
        template_dict = triples_str_dict[head]
    else:
        # 模糊匹配
        for k in triples_str_dict.keys():
            if edit_distance_metric(head,k):
                template_dict = triples_str_dict[k]
                break
        else:
            template_dict = triples_str_dict['default']
    if negative:
        template = template_dict[1]
    else :
        template = template_dict[0]
    
    strs = template.format(head=head,relation=relation,tail=tail)
    
    return strs
    
    
def extract_triples(node):
    triples = []
    try:
        for k,v in node['triples'].items():
            triples.append(tuple(v))
        for child in node['children']:
            triples += extract_triples(child)
        return triples
    except:
        logger.debug(f"Error in extract_triples: {node}")
        return []

def extract_cnf_node(node,triples_dict):
    if node['parent'] == 'ROOT':
        return None
    select_condition = node['select_condition']
    num_set = set()
    i = 0
    while i < len(select_condition):
        if select_condition[i].isdecimal():
            start = i
            while i < len(select_condition) and select_condition[i].isdecimal():
                i += 1
            num_set.add(int(select_condition[start:i]))
        else:
            i += 1
    parent_triples = node['parent']['triples']
    old_num2new_num = {}
    for num in num_set:

        triple = parent_triples[str(num)]
        for k,v in triples_dict.items():
            if v==tuple(triple):
                old_num2new_num[str(num)] = k
                
    new_select_condition = select_condition
    ptn = r'(and)|(or)|(not)|(\()|(\))'
    result_re = re.split(ptn,new_select_condition)
    for i in range(len(result_re)):
        if result_re[i] == "" or result_re[i] is None:
            continue
        else:
            if result_re[i] in old_num2new_num.keys():
                result_re[i] = str(old_num2new_num[result_re[i]])
    result_re = [x for x in result_re if x != "" and x is not None]
    
    new_select_condition = ''.join(result_re)
    
    parse = Parser(new_select_condition)
    ast = parse.parse()
    cnf = parse.to_cnf(ast)
    return {
        'new_select_condition':new_select_condition,
        'cnf':cnf
        }
        
def extract_cnf_tree(tree,triples_dict):
    paths = []
    try:
        def dfs(node,path):
            path=path.copy()
            try:
                node_result = extract_cnf_node(node,triples_dict)
            except:
                logger.debug(f"Error in extract_cnf_tree: {node}")
                return
            if node_result is not None:
                path.append(node_result)
            if node['children'] == []:
                new_select_condition = 'and'.join(node['triples'].keys())
                try:
                    parse = Parser(new_select_condition)
                    ast = parse.parse()
                    cnf = parse.to_cnf(ast)
                except:
                    logger.debug(f"Error in extract_cnf_tree: {new_select_condition}")
                    return
                
                leaf_result ={
                    'new_select_condition':new_select_condition,
                    'cnf':cnf
                    }
                path.append(leaf_result)
                
                paths.append(path)
                return
            for child in node['children']:
                dfs(child,path)
        dfs(tree,[])
        return paths
    except:
        logger.debug(f"Error in extract_cnf_tree: {tree}")
        return []
def tree_traversal(trees):
    new_trees = []
    triples = []
    for tree in trees:
        tree = add_parent(tree,"ROOT")
        triples.extend(extract_triples(tree))
        new_trees.append(tree)
    
    triples = list(set(triples))
    triples_dict = {}
    for i , triple in enumerate(triples):
        triples_dict[i] = triple
    paths = []
    for tree in new_trees:
        paths += extract_cnf_tree(tree,triples_dict)
    return {
        "triples_dict":triples_dict,
        "paths":paths
        }
    


def path2str(triple_dict,path_list):
    path_str_list = []
    for path in path_list:
        text = ''
        for node_order, node in enumerate(path):
            ptn = r'(and)|(or)|(not)|(\()|(\))'
            result_re = re.split(ptn,node['new_select_condition'])
            result_re = [x for x in result_re if x != "" and x is not None]
            for i in range(len(result_re)):
                if result_re[i].isdecimal():
                    triple_temp = triple_dict.get(int(result_re[i]))
                    if triple_temp is None:
                        logger.error(f"Error in path2str:result_re[i]: {result_re[i]} triple_dict: {triple_dict},path_list: {path_list},path ;{path}")
                        triple_temp = ["DEBUG","DEBUG","DEBUG"]
                    triple_temp = ', '.join(triple_temp)
                    result_re[i] = '(' + triple_temp + ')'
                elif result_re[i] in ['and','or','not']:
                    result_re[i] = result_re[i].upper()
            if len(path)==1:
                text += ' '.join(result_re) + '\n'
            
            elif node_order != len(path)-1:
                text += ' '.join(result_re)+'\nAND\n'
            else:
                text = text[-4:] + 'THEN\n' + ' '.join(result_re)
        path_str_list.append(text)
    return path_str_list
def path2nl(triple_dict,path_list):
    prompt_text = prompt_path
    for path_order, path in enumerate(path_list):
        path_text = f"Rule {path_order+1}:\n"
        group_order = 1
        for node_order, node in enumerate(path):
            if node_order == len(path)-1:
                conclusion_text = f"Conclusions:\n"
                cnf = node['cnf']
                for dnf_order, dnf in enumerate(cnf):
                    for triple_order in dnf:
                        negative = True if '-' in triple_order else False
                        triple_temp = triple_dict.get(abs(int(triple_order)))
                        if triple_temp is None:
                            logger.error(f"Error in path2nl:triple_order: {triple_order} triple_dict: {triple_dict},path_list: {path_list},path ;{path}")
                            triple_temp = ["DEBUG","DEBUG","DEBUG"]
                        triple_str = triple2str(triple_temp,negative=negative)
                        conclusion_text += triple_str + '\n'
                    if dnf_order != len(cnf)-1:
                        conclusion_text += 'OR\n'
                path_text += conclusion_text
                
            else:
                cnf = node['cnf']
                for dnf in cnf:
                    gropu_text = f"Group {group_order}:\n"
                    for triple_order in dnf:
                        try:
                            negative = True if '-' in triple_order else False
                            triple_temp = triple_dict.get(abs(int(triple_order)))
                            if triple_temp is None:
                                logger.error(f"Error in path2nl:triple_order: {triple_order} triple_dict: {triple_dict},path_list: {path_list},path ;{path}")
                                triple_temp = ["DEBUG","DEBUG","DEBUG"]
                            triple_str = triple2str(triple_temp,negative=negative)
                        except:
                            triple_str = ''
                        gropu_text += triple_str + '\n'
                    path_text += gropu_text
                    group_order += 1

        prompt_text += path_text
        prompt_text += '-----------------------\n'
    return prompt_text
        
    




def find_tree_path_through_similarity(query_triples,tree_list,sentence_transformer_class,thresholds=[75,80],path_num=None):

    tree_traversal_result = tree_traversal(tree_list)
    tree_triple_dict = tree_traversal_result['triples_dict']
    tree_triple_list = list(tree_triple_dict.values())
    tree_triple_list = [','.join(triple) for triple in tree_triple_list]
    tree_path_list = tree_traversal_result['paths'] 
    unique_tree_path_list = []
    for path in tree_path_list:
        if path not in unique_tree_path_list:
            unique_tree_path_list.append(path)
    if len(unique_tree_path_list) != len(tree_path_list):
        pass
    tree_path_list = unique_tree_path_list
    
    last = -1
    for k in tree_triple_dict.keys():
        if int(k)>last:
            last = int(k)
        else:
            print(f"tree_triple_dict key error: {str(tree_triple_dict)}")
            return None
    
    thresholds_result = {}
    path_num_pos = False 
    sm_matrix = sentence_transformer_class.similarity(tree_triple_list,query_triples)
    for t in thresholds:
        path_num_temp = path_num
        target_keys = []
        for i in range(len(sm_matrix)):
            if any(x > t for x in sm_matrix[i]):
                target_keys.append(i)
        if target_keys == []:
            
            pass
        path_count = {}
        ptn = r'(and)|(or)|(not)|(\()|(\))'
            
        for i, path in enumerate(tree_path_list):
            for node in path:
                result_re = re.split(ptn,node['new_select_condition'])
                result_re = [x for x in result_re if x is not None and  x.isdecimal()]# only num
                for target_key in target_keys:
                    if str(target_key) in result_re:
                        if i not in path_count.keys():
                            path_count[i] = 1
                        else:
                            path_count[i] += 1

        new_path_list = []
        if path_num_temp is None:
            path_num_temp = len(path_count)
        # descent sort
        path_count = dict(sorted(path_count.items(), key=lambda item: item[1], reverse=True))
        for k,v in path_count.items():
            new_path_list.append(tree_path_list[k])
            path_num_temp -= 1
            if path_num_temp == 0:
                break
                    
        if len(new_path_list) == path_num:
            path_num_pos = True
        
        paths_str = path2nl(tree_triple_dict,new_path_list)
        thresholds_result[t] = {"paths_str":paths_str,"path_num":len(new_path_list)}
    if not path_num_pos:
        pass
    return thresholds_result


    
def empty_list_metric(predict_origin=None,reference_origin=None,
                      predict_cleaned=None,reference_cleaned=None,
                      predicts_converted=None,references_converted=None,
                      predicts_cleaned_pos=None,references_cleaned_pos=None):

    p_empty_origin = 0
    r_empty_origin = 0
    all_empty_origin = 0
    p_empty_cleaned = 0
    p_empty_caused_by_clean_fail = 0
    r_empty_cleaned = 0
    r_empty_caused_by_clean_fail = 0
    all_empty_cleaned = 0
    all_empty_caused_by_clean_fail = 0
    p_empty_converted = 0
    r_empty_converted = 0
    all_empty_converted = 0
    result = {}
    if predict_origin is not None:
        if reference_origin is not None:
            for pred, ref in zip(predict_origin,reference_origin):
                if len(pred) < 5:
                    p_empty_origin += 1
                if len(ref) < 5:
                    r_empty_origin += 1
                if len(pred) < 5 and len(ref) < 10:
                    all_empty_origin += 1
            result['predict_empty_origin'] = round(p_empty_origin/len(predict_origin)*100,3)
            result['reference_empty_origin'] = round(r_empty_origin/len(reference_origin)*100,3)
            result['together_empty_origin'] = round(all_empty_origin/len(predict_origin)*100,3)
        else:
            for pred in predict_origin:
                if len(pred) < 5:
                    p_empty_origin += 1
            result['predict_empty_origin'] = round(p_empty_origin/len(predict_origin)*100,3)
    if predict_cleaned is not None and predicts_cleaned_pos is not None:
        if reference_cleaned is not None and references_cleaned_pos is not None:
            for i, (pred, ref) in enumerate(zip(predict_cleaned,reference_cleaned)):
                if len(pred) == 0 :
                    p_empty_cleaned += 1
                    if predicts_cleaned_pos[i] == False:
                        p_empty_caused_by_clean_fail += 1
                if len(ref) == 0 :
                    r_empty_cleaned += 1
                    if references_cleaned_pos[i] == False:
                        r_empty_caused_by_clean_fail += 1
                if len(pred) == 0 and len(ref) == 0 :
                    all_empty_cleaned += 1
                    if predicts_cleaned_pos[i] == False and references_cleaned_pos[i] == False:
                        all_empty_caused_by_clean_fail += 1
            result['predict_empty_cleaned'] = round(p_empty_cleaned/len(predict_cleaned)*100,3)
            result['reference_empty_cleaned'] = round(r_empty_cleaned/len(reference_cleaned)*100,3)
            result['together_empty_cleaned'] = round(all_empty_cleaned/len(predict_cleaned)*100,3)
            result['predict_empty_caused_by_clean_fail'] = round(p_empty_caused_by_clean_fail/len(predicts_cleaned_pos)*100,3)
            result['reference_empty_caused_by_clean_fail'] = round(r_empty_caused_by_clean_fail/len(references_cleaned_pos)*100,3)
            result['together_empty_caused_by_clean_fail'] = round(all_empty_caused_by_clean_fail/len(predicts_cleaned_pos)*100,3)
        else:
            for i,  pred in enumerate(predict_cleaned):
                if len(pred) == 0 :
                    p_empty_cleaned += 1
                    if predicts_cleaned_pos[i] == False:
                        p_empty_caused_by_clean_fail += 1
            result['predict_empty_cleaned'] = round(p_empty_cleaned/len(predict_cleaned)*100,3) 
            result['predict_empty_caused_by_clean_fail'] = round(p_empty_caused_by_clean_fail/len(predicts_cleaned_pos)*100,3)
    
    if predicts_converted is not None:
        if references_converted is not None:
            for pred, ref in zip(predicts_converted,references_converted):
                # list
                if len(pred) == 0:
                    p_empty_converted += 1
                if len(ref) == 0:
                    r_empty_converted += 1
                if len(pred) == 0 and len(ref) == 0:
                    all_empty_converted += 1
            result['predict_empty_converted'] = round(p_empty_converted/len(predicts_converted)*100,3)
            result['reference_empty_converted'] = round(r_empty_converted/len(references_converted)*100,3)
            result['together_empty_converted'] = round(all_empty_converted/len(predicts_converted)*100,3)
        else:
            for pred in predicts_converted:
                if len(pred) == 0:
                    p_empty_converted += 1
            result['predict_empty_converted'] = round(p_empty_converted/len(predicts_converted)*100,3)

    return result       



def trees_only_check(tree_list,in_detail=False):
    tree_list = copy.deepcopy(tree_list)
    
    if isinstance(tree_list,dict):
        if tree_list == {}:
            tree_list = []
        else:
            if tree_list.get('tree1') is not None:
                new_tree_list = []
                for k ,v in tree_list.items():
                    if isinstance(v,dict):
                        new_tree_list.append(v)
                    else:
                        return False,ERROR_TYPE_CHILDREN
                tree_list = new_tree_list
    
    ERROR_TYPE_IS_LEAF = "is_leaf"
    ERROR_TYPE_TRIPLE_FORMAT = "triples"
    ERROR_TYPE_SELECT_CONDITION = "select_condition"
    ERROR_TYPE_CHILDREN = "children"
    ERROR_TYPE_NONE = "correct"

    def tree_check_func(tree):
        temp_is_leaf = tree.get('is_leaf')
        temp_select_condition = tree.get('select_condition')
        temp_triples = tree.get('triples')
        temp_children = tree.get('children')
        
        # check is_leaf
        if temp_is_leaf is None or temp_is_leaf not in ['0','1']:
            return False,ERROR_TYPE_IS_LEAF
        # check select_condition
        if temp_select_condition is None:
            return False,ERROR_TYPE_SELECT_CONDITION
        if temp_select_condition == "" and tree.get('parent') != 'ROOT':
            return False,ERROR_TYPE_SELECT_CONDITION
        if not check_logic_expression(temp_select_condition):
            return False,ERROR_TYPE_SELECT_CONDITION        
        temp_parent = tree.get('parent')
        if temp_parent != 'ROOT':
            parent_triples = temp_parent.get('triples')
            parent_triples_keys = parent_triples.keys()
            ptn = r'(and)|(or)|(not)|(\()|(\))'
            result_re = re.split(ptn,temp_select_condition)
            logic_expression_list = [i for i in result_re  if i!="" and i is not None]
            result_num = [i for i in logic_expression_list if i.isdecimal()]
            for i in result_num:
                if i not in parent_triples_keys:
                    return False,ERROR_TYPE_SELECT_CONDITION
        # check triples
        if isinstance(temp_triples,dict):
            for k,v in temp_triples.items():
                if not check_triples_list(v) or not isinstance(k,str) or not k.isdecimal():
                    return False,ERROR_TYPE_TRIPLE_FORMAT
        else: 
            return False,ERROR_TYPE_TRIPLE_FORMAT
        # check children
        if not isinstance(temp_children,list):
            return False,ERROR_TYPE_CHILDREN
        for child in temp_children:
            temp_result = tree_check_func(child)
            if not temp_result[0]:
                return temp_result
        return True,ERROR_TYPE_NONE
    
    for tree in tree_list:
        try:
            tree = add_parent(tree,"ROOT")
        except:
            return False,ERROR_TYPE_CHILDREN
        temp_result = tree_check_func(tree)
        if not temp_result[0]:
            if in_detail:
                return temp_result
            else:
                return False
    return True,ERROR_TYPE_NONE

def check_logic_expression(logic_expression):
    ''' only check bracket and item and operator
    '''
    logic_expression = re.sub(r'\s+', '', logic_expression)
    bracket = 0
    i = 0
    while i < len(logic_expression):
        if logic_expression[i] == '(':
            bracket += 1
            i += 1
        elif logic_expression[i] == ')':
            bracket -= 1
            i += 1
        elif logic_expression[i]=='a' and i+2<len(logic_expression) and logic_expression[i:i+3] == 'and':
            i += 3
        elif logic_expression[i]=='o' and i+1<len(logic_expression) and logic_expression[i:i+2] == 'or':
            i += 2
        elif logic_expression[i]=='n' and i+2<len(logic_expression) and logic_expression[i:i+3] == 'not':
            i += 3
        elif logic_expression[i].isdecimal():
            i += 1
            while i<len(logic_expression) and logic_expression[i].isdecimal():
                i += 1
        else:
            return False
    if bracket != 0:
        return False
    return True


def remove_logic_expression_item(logic_expression_list,key):
    while key in logic_expression_list:
        index = logic_expression_list.index(key)
        # check not, not is unary operator
        if index-1>=0 and logic_expression_list[index-1] == 'not':
            # there may be more than one not
            logic_expression_list.pop(index-1)
            continue
        if index-1>=0 and (logic_expression_list[index-1] == 'and' or logic_expression_list[index-1] == 'or'):
            logic_expression_list = logic_expression_list[:index-1] + logic_expression_list[index+1:]
        elif index+1<len(logic_expression_list) and (logic_expression_list[index+1] == 'and' or logic_expression_list[index+1] == 'or'):
            logic_expression_list = logic_expression_list[:index] + logic_expression_list[index+2:]
        elif len(logic_expression_list) == 1:
            logic_expression_list = []
        elif index-1>=0 and index+1<len(logic_expression_list) and logic_expression_list[index-1] == '(' and logic_expression_list[index+1] == ')':
            #
            logic_expression_list.pop(index+1)
            logic_expression_list.pop(index-1)

        else:
            logger.debug(f"remove_logic_expression_item error: {str(logic_expression_list)}")
            break
    # simplify by remove bracket at the end and beginning
    while len(logic_expression_list)>1 and logic_expression_list[0] == '(' and logic_expression_list[-1] == ')':
        logic_expression_list.pop(0)
        logic_expression_list.pop(-1)


    return logic_expression_list

def check_triples_list(single_triplets):
    # 检查三元组格式
    if isinstance(single_triplets,list) and len(single_triplets) == 3:
        for j in single_triplets:
            if not isinstance(j,str):
                return False
        return True
    else: 
        return False
      
def convert_tree(tree):
    # is_leaf
    if tree == {}:
        return None
    try:
        temp_is_leaf = tree.get('is_leaf')
        temp_select_condition = tree.get('select_condition')
        temp_triples = tree.get('triples')
        temp_children = tree.get('children')
        if  temp_is_leaf is None or temp_select_condition is None or temp_triples is None or temp_children is None:
            return None
    except:
        return None
    
    #select_condition
    
    if tree.get('select_condition') is None or tree.get('parent') == 'ROOT':
        tree['select_condition'] = ""
        
    select_condition = tree.get('select_condition')
    if  select_condition != '':
        # convert remove space
        select_condition = re.sub(r'\s+', '', select_condition)
        # check and modify format error    , 自顶向下处理, 可以保证parent节点的正确性
        parent_triples = tree.get('parent').get('triples')
        k_list_parent = []
        if parent_triples is not None:
            for k,v in parent_triples.items():
                k_list_parent.append(str(k))
        pos = check_logic_expression(select_condition)
        if not pos:
            # check "else"
            if select_condition == "else" and k_list_parent != []:
                # get parent triples
                select_condition = 'not' +"("+ 'or'.join(k_list_parent) + ")"
            else:
                # 
                logger.debug(f"select_condition format error:{str(select_condition)}")
                
                select_condition = ""
        
    # check if the select_condition is  out of parent's triples
    if select_condition != "":
        
        ptn = r'(and)|(or)|(not)|(\()|(\))'
        result_re = re.split(ptn,select_condition)
        select_condition_list = [i for i in result_re  if i!="" and i is not None]
        result_num = [i for i in select_condition_list if i.isdecimal()]
        bracket = 0
        if len(result_num) > 10:
            for i in range(len(select_condition_list)):
                if select_condition_list[i] == '(':
                    bracket += 1
                elif select_condition_list[i] == ')':
                    bracket -= 1
                if bracket == 0 and i >= 10:
                    if select_condition_list[i] ==')' or select_condition_list[i].isdecimal():
                        select_condition_list = select_condition_list[:i+1]
                    else:
                         # logic 
                        select_condition_list = select_condition_list[:i]
                    break
        result_num = [i for i in select_condition_list if i.isdecimal()]


        
        for i in result_num:
            if i not in k_list_parent :
                logger.debug(f"select_condition not in parent triples:{str(select_condition)},parent triples:{str(tree.get('parent').get('triples'))}")
                select_condition_list = remove_logic_expression_item(select_condition_list,i)
        select_condition = "".join(select_condition_list)
    tree['select_condition'] = select_condition
    
    
    if tree.get('select_condition') == '' and tree.get('parent') != "ROOT":
        #tree['parent']['children'].remove(tree)
        tree['parent'] = None
        tree['children'] = []
        return None
        
    
    # check and modify triples format    
    triples = tree.get('triples')
    if triples is None or triples == {} or triples == []:
        tree['triples'] = {}
    else:
        if isinstance(triples,list):
            temp = {}
            for i in range(len(triples)):
                if check_triples_list(triples[i]):
                    temp[str(i)] = triples[i]
                else:
                    logger.debug("triples format error, not dict or list of length 3: " + str(triples))
                    break
            tree['triples'] = temp
        elif isinstance(triples,dict):
            temp = {}
            for k,v in triples.items():
                if check_triples_list(v):
                    temp[str(k)] = v
                else:
                    logger.debug(f"triples format error, not dict or list of length 3:{str(triples)}")
                    break
            tree['triples'] = temp
        else:
            logger.debug(f"triples format error, neither dict nor list : {str(triples)}")
            tree['triples'] = {}
    
    # children
    new_children = []
    
    children = tree.get('children')
    if children == [] or children == {} : 
        tree['children'] = []
    
    
    elif type(children) == list:
        i = 0
        while i < len(children):
            temp = convert_tree(children[i])
            if temp is not None:
                new_children.append(temp)
                i += 1
            else:
                logger.debug(f"children format error, not dict or list of length 3 , children: {str(children[i])}, parent_triples: {str(tree['triples'])}")

                children.pop(i)
                
                pass
        tree['children'] = new_children
        
        
    elif type(children) == dict :
        # single node
        if len(children)==1 and children.get['is_leaf'] is not None:
            child = convert_tree(children['children']) 
            if child is not None:
                tree['children'] = [child]
            else:
                tree['children'] = []
        # multi node    
        else:
            for k in children.keys():
                v = children[k]
                if v.get('is_leaf') is not None:
                    temp = convert_tree(v)
                    if temp is not None:
                        new_children.append(temp)
            tree['children'] = new_children
    else:
        logger.debug(f"children format error: {str(children)}")
        tree['children'] = []
    
    
    #convert is_leaf
    tree['is_leaf'] = str(tree['is_leaf'])
    if tree.get('children') == []:
        tree['is_leaf'] = '1'
    else:
        tree['is_leaf'] = '0'

    return tree      
    
    
    

def convert_trees(single_article_tree_list):

    if single_article_tree_list == [] or single_article_tree_list == {}:
        return []
    new_single_article_tree_list = []
    if isinstance(single_article_tree_list,dict):
        try: 
            for k,v in single_article_tree_list.items():
                new_single_article_tree_list.append(v)
        except:
            pass
    single_article_tree_list = copy.deepcopy(new_single_article_tree_list)
    new_tree_list = []
    for single_tree in single_article_tree_list:
        try:
            new_tree = add_parent(single_tree,"ROOT")

            new_tree = convert_tree(new_tree)
            new_tree = delete_parent_field(new_tree)
            if new_tree is None:
                logger.debug(f"convert tree error, return None : {str(single_tree)}")
                continue
            new_tree_list.append(new_tree)
        except:
            logger.debug(f"convert tree error, exception : {str(single_tree)}")
            continue
    return new_tree_list


        
def multi_text_tree_list_format_metric(tree_list):

    if isinstance(tree_list[0],dict):
        new_tree_list = []
        for singel_article_trees in tree_list:
            temp_list = []
            for k,v in singel_article_trees.items():
                temp_list.append(v)
            new_tree_list.append(temp_list)
        tree_list = copy.deepcopy(new_tree_list)

    format_correct = 0
    format_result = {}
    for predict in tree_list:
        result = trees_only_check(predict,in_detail=True)
        if result[0]:
            format_correct += 1
            continue
        else:
            if format_result.get(result[1]) is None:
                format_result[result[1]] = 1
            else:
                format_result[result[1]] += 1
    for k in format_result.keys():
        format_result[k] = round(format_result[k]/len(tree_list)*100,2)
    new_format_result = {}  
    for k,v in format_result.items():
        key = f'format_error_{k}'
        new_format_result[key] = v
    new_format_result['format_correct'] = round(format_correct/len(tree_list)*100,2)
    return new_format_result     
def delete_parent_field(tree):
    tree = copy.deepcopy(tree)
    def delect_parent_func(tree):
        if tree.get('parent') is None:
            return
        else:
            tree.pop('parent')
            for child in tree['children']:
                delect_parent_func(child)
    delect_parent_func(tree)
    return tree

def remove_space_from_select_condition(tree):

    tree = copy.deepcopy(tree)
    def remove_space_func(tree):
        if tree.get('select_condition') is not None:
            tree['select_condition'] = re.sub(r'\s+', '', tree['select_condition'])
        for child in tree['children']:
            remove_space_func(child)
    remove_space_func(tree)
    return tree
            
def check_select_condition_in_parent(select_condition,parent_triples_keys):
    ptn = r'(and)|(or)|(not)|(\()|(\))'
    result_re = re.split(ptn,select_condition)
    logic_expression_list = [i for i in result_re  if i!="" and i is not None]
    result_num = [i for i in logic_expression_list if i.isdecimal()]
    for i in result_num:
        if i not in parent_triples_keys:
            return False
    return True
def revise_single_tree_triples(tree):    
    tree = copy.deepcopy(tree)
    tree = add_parent(tree,"ROOT")
    def revise_single_tree_triples_func(tree):
        triples = tree.get('triples')
        
        for k in triples.keys():
            for i in range(len(triples[k])):
                triples[k][i] == str(triples[k][i])
            
            if not check_triples_list(triples[k]):
                if len(triples[k]) <= 1:
                    triples.pop(k)
                elif len(triples[k]) == 2:
                    triples[k] = ['patient'] + triples[k]
                elif len(triples[k]) > 3:
                    triples[k] = triples[k][:3]
                else:
                    pass
                
        tree['triples'] = triples
        
        for child in tree.get('children'):
            revise_single_tree_triples_func(child)
    revise_single_tree_triples_func(tree)
    return tree
                
def revise_single_tree_select_condition(tree):
    tree = copy.deepcopy(tree)
    tree = add_parent(tree,"ROOT")
    def revise_select_condition_func(tree):
        parent = tree.get('parent')
        if parent == 'ROOT':
            pass
        else:
            select_condition = tree.get('select_condition')
            parent_triples = parent.get('triples')
            parent_triples_keys = parent_triples.keys()
            if select_condition == '':
                temp_select_condition = 'and'.join(parent_triples_keys)
                tree['select_condition'] = temp_select_condition
            
            if not check_select_condition_in_parent(select_condition,parent_triples_keys):
                if len(parent.get('children')) == 1:
                    temp_select_condition = 'and'.join(parent_triples_keys)
                    tree['select_condition'] = temp_select_condition
                    
                elif len(parent.get('children')) == 2:
                    for child in parent.get('children'):
                        if child == tree:
                            continue
                        other_select_condition = child.get('select_condition')  
                        if check_select_condition_in_parent(other_select_condition,parent_triples_keys):
                            tree['select_condition'] = 'not' + '(' + other_select_condition + ')'
                            
                        else:
                            temp_select_condition = 'and'.join(parent_triples_keys)
                            tree['select_condition'] = temp_select_condition
                            child['select_condition'] = temp_select_condition
                else:
                    for child in parent.get('children'):
                        child['select_condition'] = 'and'.join(parent_triples_keys)
                    
                    
                    
        for child in tree.get('children'):
            revise_select_condition_func(child)
    revise_select_condition_func(tree)
    return tree
    

def revise_singele_article_trees(tree_dict,single_tree_pos=False):
    tree_dict = copy.deepcopy(tree_dict)
    if not single_tree_pos:
        for tree_order in tree_dict.keys():
            tree = tree_dict[tree_order]
            tree = remove_space_from_select_condition(tree)
            tree = revise_single_tree_triples(tree)
            tree = revise_single_tree_select_condition(tree)
            
            tree = delete_parent_field(tree)
            tree_dict[tree_order] = tree
    else:
        tree = tree_dict
        tree = remove_space_from_select_condition(tree)
        tree = revise_single_tree_triples(tree)
        tree = revise_single_tree_select_condition(tree)
        tree = delete_parent_field(tree)
        tree_dict = tree
    return tree_dict

def total_tree_to_str(tree_dict):
    tree_dict=add_parent(tree_dict,'ROOT')
    triples = []
    triples.extend(extract_triples(tree_dict))
    triples = list(set(triples))
    triples_dict = {}
    for i , triple in enumerate(triples):
        triples_dict[i] = triple
    
    queue = deque()
    result_line = []
    for child in tree_dict.get('children'):
        queue.append((child,0))
    while queue:
        current_node,parent_indent = queue.popleft()
        node_result = extract_cnf_node(current_node,triples_dict)
        cnf = node_result['cnf']
        condition_text_list = []
        for dnf_order, dnf in enumerate(cnf):
            second_condition_text_list = []
            for triple_order in dnf:
                negative = True if '-' in triple_order else False
                triple_order = triple_order[0] if isinstance(triple_order,list) and len(triple_order) == 1 else triple_order
                triple_temp = triples_dict.get(abs(int(triple_order)))
                if triple_temp is None:
                    logger.error(f"Error in total_tree_to_str:triple_order: {triple_order} triples_dict: {triples_dict},current_node :{str(current_node)}")
                    triple_temp = ["","",""]
                triple_str = triple2str(triple_temp,negative=negative)
                second_condition_text_list.append(triple_str)
            second_condition_text = ' OR '.join(second_condition_text_list)
            second_condition_text = '( ' + second_condition_text + ' )'
            condition_text_list.append(second_condition_text)
        condition_text = ' AND '.join(condition_text_list)
        condition_text = 'IF: ' + condition_text + ' : '
        condition_text_line = ' ' * parent_indent + condition_text
        result_line.append(condition_text_line)
        if current_node.get('is_leaf') == '0':
            for child in current_node['children']:
                queue.append((child,parent_indent + 4))
        else:
            decision_indent = parent_indent + 4
            for k, v in current_node['triples'].items():
                triple_str = triple2str(v)
                result_line.append(' ' * decision_indent  + triple_str)
    tree_dict = delete_parent_field(tree_dict)
    result_text = '\n'.join(result_line)
    return result_text
            
            
            
            
            
        

        
    
                
                
            
            
        
            
            
            
    
    