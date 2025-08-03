import json
import os
from tqdm import tqdm
path1 = "datasets/mimic/origin/1.json"
path2 = "datasets/mimic/origin/2.json"
path3 = "datasets/mimic/origin/3.json"
path4 = "datasets/mimic/origin/4.json"
path5 = "datasets/mimic/origin/5.json"
path_temp = "datasets/mimic/origin/temp.jsonl"
def is_float(s):
    try:
        float(s)
        return True
    except:
        return False

def single(path):
    # 筛选单病人单次就诊
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sub_id_hadm_id= {}
    for i in data:
        if i['subject_id'] not in sub_id_hadm_id:
            sub_id_hadm_id[i['subject_id']] = set()
            sub_id_hadm_id[i['subject_id']].add(i['hadm_id'])
        else:
            sub_id_hadm_id[i['subject_id']].add(i['hadm_id'])
    # 统计就诊次数占比
    num_dic = {}
    for k,v in sub_id_hadm_id.items():
        if len(v) not in num_dic:
            num_dic[len(v)] = 1
        else:
            num_dic[len(v)] += 1
    percentage = {}
    for k,v in num_dic.items():
        percentage[k] = v/len(sub_id_hadm_id)
    print(percentage)
    singel_sub_id = []
    for k,v in sub_id_hadm_id.items():
        if len(v) == 1:
            singel_sub_id.append(k)
    print("total subject_id:", len(sub_id_hadm_id))
    print("single subject_id:", len(singel_sub_id))
    return set(singel_sub_id)
        
def merge(path1, path2, path3, path4, path5,  singel_sub_id):
    # 合并数据集
    with open(path1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
        data1_subject_dict = {}
        for i in tqdm(data1):
            if i['subject_id'] not in singel_sub_id:
                continue
            else:
                if data1_subject_dict.get(i['subject_id']) is None:
                    data1_subject_dict[i['subject_id']] = {
                                                        "subject_id": i['subject_id'],
                                                        "hadm_id": i['hadm_id'], 
                                                        "gender": i["gender"],
                                                        "anchor_age": i["anchor_age"],
                                                        "diagnosis": set([i["long_title"]])
                                                            }
                else:
                    data1_subject_dict[i['subject_id']]["diagnosis"].add(i["long_title"])
                    assert data1_subject_dict[i['subject_id']]["hadm_id"] == i['hadm_id']
        # convert diagnosis to list
        for k in list(data1_subject_dict.keys()):
            data1_subject_dict[k]["diagnosis"] = list(data1_subject_dict[k]["diagnosis"])
            
   
    with open(path2, 'r', encoding='utf-8') as f:
        # labevents
        data2 = json.load(f)
        for i in tqdm(data2):
            if i['subject_id'] not  in singel_sub_id:
                continue
            if data1_subject_dict[i['subject_id']].get("labevents") is None:
                data1_subject_dict[i['subject_id']]["labevents"] = []
            labevent_value = i.get("labevent_value")
            valuenum = i.get("valuenum")
            labevent_comments = i.get("labevent_comments")
            valueuom = "" if  i.get("valueuom") is  None else i.get("valueuom")
            labevent_label = i.get("labevent_label")
            value = ""
            if labevent_value is not None:
                if is_float(labevent_value):
                    value = str(valuenum)+valueuom
                else:
                    if is_float(valuenum):
                        value = labevent_value + ' ' + str(valuenum)+valueuom
                    else:
                        if labevent_comments is not None:
                            value = labevent_value + ' ' +  labevent_comments
                        else:
                            value = labevent_value
            elif labevent_comments is not None:
                value = labevent_comments
            else:
                continue
            singel_labevent = {
                "itemid": i["itemid"],
                "name":labevent_label,
                "value": value,
            }
            pos = True
            for j in data1_subject_dict[i['subject_id']]["labevents"]:
                if j['name'] == labevent_label:
                    pos = False
            if pos:
                data1_subject_dict[i['subject_id']]["labevents"].append(singel_labevent)
            
    with open(path3, 'r', encoding='utf-8') as f:
        #microbiologyevents
        data3 = json.load(f)
        for i in tqdm(data3):
            if i['subject_id'] not  in singel_sub_id:
                continue
            if data1_subject_dict[i['subject_id']].get("microbiologyevents") is None:
                data1_subject_dict[i['subject_id']]["microbiologyevents"] = []
            
            spec_type_desc = "" if  i.get("microevent_spec_type_desc") is None else i.get("microevent_spec_type_desc")
            test_name = "" if  i.get("microevent_test_name") is None else i.get("microevent_test_name")
            org_name = "" if  i.get("microevent_org_name") is None else i.get("microevent_org_name")
            comments = "" if  i.get("microevent_comments") is None else i.get("microevent_comments")
            value = spec_type_desc+' '+test_name+' '+org_name+' '+comments
            singel_microbiologyevent = {
                "microevent_id": i["microevent_id"],
                "value": value,
            }
            pos = True
            for j in data1_subject_dict[i['subject_id']]["microbiologyevents"]:
                if j['value'] == value:
                    pos = False
            if pos:
                data1_subject_dict[i['subject_id']]["microbiologyevents"].append(singel_microbiologyevent)
        
    with open(path4, 'r', encoding='utf-8') as f:
        # durgs
        data4 = json.load(f)
        for i in tqdm(data4):
            if i['subject_id'] not  in singel_sub_id:
                continue
            if data1_subject_dict[i['subject_id']].get("durgs") is None:
                data1_subject_dict[i['subject_id']]["durgs"] = []
            drug_name = "" if  i.get("prescriptions_drug") is None else i.get("prescriptions_drug")
            value = drug_name
            singel_drug = {
                "value": value,
            }
            pos = True
            for j in data1_subject_dict[i['subject_id']]["durgs"]:
                if j['value'] == value:
                    pos = False
            if pos:
                data1_subject_dict[i['subject_id']]["durgs"].append(singel_drug)

    with open(path5, 'r', encoding='utf-8') as f:
        # procedures
        data5 = json.load(f)
        for i in tqdm(data5):
            if i['subject_id'] not  in singel_sub_id:
                continue
            if data1_subject_dict[i['subject_id']].get("procedures") is None:
                data1_subject_dict[i['subject_id']]["procedures"] = []
            value = i.get("d_icd_procedures_long_title")
            singel_procedure = {
                "value": value,
            }
            pos = True
            for j in data1_subject_dict[i['subject_id']]["procedures"]:
                if j['value'] == value:
                    pos = False
            if pos:
                data1_subject_dict[i['subject_id']]["procedures"].append(singel_procedure)
        return data1_subject_dict
        
        
        
        
if __name__ == '__main__':
    singel_sub_id = single(path1)
    data_subject_dict = merge(path1, path2, path3, path4, path5, singel_sub_id)
    pass
    with open(path_temp, 'w', encoding='utf-8') as f:
        for k,v in data_subject_dict.items():
            f.write(json.dumps(v, ensure_ascii=False)+'\n')
            



















