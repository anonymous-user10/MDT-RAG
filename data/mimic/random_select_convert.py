
import os
import json
import random
random.seed(42)

with open('datasets/mimic/origin/temp.jsonl', 'r') as f:
    lines = [json.loads(line) for line in f.readlines()]

random_index_list = list(range(len(lines))) 
random.shuffle(random_index_list)
count = 0

save_list_medications = []
save_list_treatment = []
save_list_items = []
save_path_medications = 'datasets/mimic/mimic_2k_medications.jsonl'
save_path_treatment = 'datasets/mimic/mimic_2k_treatment.jsonl'
save_path_items = 'datasets/mimic/mimic_2k_items.jsonl'


for index in random_index_list:
    line = lines[index]

    diagnosis_temp = line.get("diagnosis")
    if diagnosis_temp is None or len(diagnosis_temp) == 0 or len(diagnosis_temp) > 7:
        continue
    labevents = line.get("labevents")
    if labevents is None or len(labevents) == 0 or len(labevents) > 10:
        continue
    microbiologyevents_temp = line.get("microbiologyevents")
    if microbiologyevents_temp is None or len(microbiologyevents_temp) == 0 or len(microbiologyevents_temp) > 8:
        continue
    durgs_temp = line.get("durgs")
    if durgs_temp is None or len(durgs_temp) == 0 or len(durgs_temp) > 14:
        continue
    procedures_temp = line.get("procedures")
    if procedures_temp is None or len(procedures_temp) == 0 or len(procedures_temp) > 4:
        continue

    #prompt_medications = "A {age}-year-old {gender} patient, with a doctor's diagnosis of {diagnosis}, laboratory test results of {labevents}, and microbiological test results of {microbiologyevents}. Please analyze the patient's condition based on the diagnostic information and medical test results. Based on this analysis, recommend the appropriate medications and provide the reasoning process."
    prompt_medications = "A {age}-year-old {gender} patient, with a doctor's diagnosis of {diagnosis}, laboratory test results of {labevents}, and microbiological test results of {microbiologyevents}. Please recommend appropriate medication for the patient based on their information and explain the reasons."
    
    #prompt_treatment = "A {age}-year-old {gender} patient, with a doctor's diagnosis of {diagnosis}, laboratory test results of {labevents}, and microbiological test results of {microbiologyevents}. Please analyze the patient's condition using the diagnostic information and medical test results. Then, recommend suitable treatment methods and explain the reasoning process behind your choice."
    prompt_treatment = "A {age}-year-old {gender} patient, with a doctor's diagnosis of {diagnosis}, laboratory test results of {labevents}, and microbiological test results of {microbiologyevents}. Please recommend a suitable treatment plan for the patient based on the patient's information and explain the reasons."
    gender = 'male' if line['gender']=='M' else 'famale'
    age = line['anchor_age']
    diagnosis =  ', '.join(line.get("diagnosis"))
    labevents = ""
    for i in line["labevents"]:
        labevents += i['name'] + ": " + i['value'] + ", "
    labevents = labevents[:-2]
    microbiologyevents = ""
    for i in line["microbiologyevents"]:
        microbiologyevents +=  i['value'] + ", "
    microbiologyevents = microbiologyevents[:-2]
    medications = set()
    for i in line["durgs"]:
        medications.add(i['value'])
    medications = list(medications)
    procedures = set()
    for i in line["procedures"]:
        procedures.add(i['value'])
    procedures = list(procedures)
    question_medications = prompt_medications.format(age=age,
                            gender=gender,
                            diagnosis=diagnosis,
                            labevents=labevents,
                            microbiologyevents=microbiologyevents)
    question_treatment = prompt_treatment.format(age=age,
                            gender=gender,
                            diagnosis=diagnosis,
                            labevents=labevents,
                            microbiologyevents=microbiologyevents)
    
    
    std_ans_for_medications = "The recommended medications are: " + ', '.join(medications) + "."
    std_ans_for_treatment = "The recommended treatment methods are: " + ', '.join(procedures) + "."
    
    save_list_medications.append({
        "question": question_medications,
        "answer": std_ans_for_medications,
        'diagnosis': diagnosis,
        "subject_id": line["subject_id"],
        "hadm_id": line["hadm_id"]
    })

    save_list_treatment.append({
        "question": question_treatment,
        "answer": std_ans_for_treatment,
        'diagnosis': diagnosis,
        "subject_id": line["subject_id"],
        "hadm_id": line["hadm_id"]
    })
    save_list_items.append({
        "subject_id": line["subject_id"],
        "hadm_id": line["hadm_id"],
        "diagnosis": line.get("diagnosis"),
        "labevents": line.get("labevents"),
        "microbiologyevents": line.get("microbiologyevents"),
        "medications": line.get("durgs"),
        "procedures": line.get("procedures"),
    })
    
# save to jsonl
with open(save_path_items, 'w') as f:
    for item in save_list_items:
        f.write(json.dumps(item) + '\n')


with open(save_path_medications, 'w') as f:
    for item in save_list_medications:
        f.write(json.dumps(item) + '\n')
with open(save_path_treatment, 'w') as f:
    for item in save_list_treatment:
        f.write(json.dumps(item) + '\n')


        
                                 
                                 
           
    
        
        
        
        
        
        
        
        
    







