import os
import json
import pandas as pd

def gen_status():
    split_path = 'data/splits/tcga_luad/splits_0.csv'
    clinical_path = 'data/LUAD/clinical_luad.json'
    output_path = 'data/LUAD/status_luad.json'
    
    split_df = pd.read_csv(split_path)
    train_list= split_df.loc[:,'train'].dropna().to_list()
    val_list = split_df.loc[:,'val'].dropna().to_list()
    case_list = train_list+val_list
    clinical_dict = json.load(open(clinical_path,'r'))
    
    status_dict = {}
    for item_dict in clinical_dict:
        if 'exposures' not in item_dict.keys():
            case_id = item_dict['submitter_id']
            print(item_dict)
        else:
            case_id = item_dict['exposures'][0]['submitter_id'].split('_')[0]
        
        if case_id not in case_list:
            continue
        
        status_dict[case_id] = {'demographic':{}, 
                                'diagnosis':{},
                                'treatments':{}}
        
        race = item_dict['demographic']['race']
        gender = item_dict['demographic']['gender']
        ethnicity = item_dict['demographic']['ethnicity']
        if 'age_at_index' not in item_dict['demographic'].keys():
            age = -1
        else:
            age = item_dict['demographic']['age_at_index']
        status_dict[case_id]['demographic'] = {
            'race' : race,
            'gender': gender,
            'ethnicity': ethnicity,
            'age': age,
        }
        
        diagnose = item_dict['diagnoses'][0]
        primary_diagnosis = diagnose['primary_diagnosis']
        if 'ajcc_pathologic_stage' in diagnose.keys():
            ajcc_pathologic_stage = diagnose['ajcc_pathologic_stage']
        else:
            ajcc_pathologic_stage = 'not reported'
        if 'ajcc_pathologic_t' in diagnose.keys():
            ajcc_pathologic_t = diagnose['ajcc_pathologic_t']
        elif 'ajcc_clinical_t' in diagnose.keys():
            ajcc_pathologic_t = diagnose['ajcc_clinical_t']
        else:
            ajcc_pathologic_t = 'not reported'
        if 'ajcc_pathologic_n' in diagnose.keys():
            ajcc_pathologic_n = diagnose['ajcc_pathologic_n']
        else:
            ajcc_pathologic_n = 'not reported'
        if 'ajcc_pathologic_m' in diagnose.keys():
            ajcc_pathologic_m = diagnose['ajcc_pathologic_m']
        else:
            ajcc_pathologic_m = 'not reported'

        status_dict[case_id]['diagnosis'] = {
            'primary_diagnosis': primary_diagnosis,
            "ajcc_pathologic_stage": ajcc_pathologic_stage,
            "ajcc_pathologic_t": ajcc_pathologic_t,
            "ajcc_pathologic_n": ajcc_pathologic_n,
            "ajcc_pathologic_m": ajcc_pathologic_m,
        }
        
        treatments = diagnose['treatments']
        for treatment in treatments:
            treatment_type = treatment['treatment_type'].split(',')[0]
            if_applied = treatment["treatment_or_therapy"]
            status_dict[case_id]['treatments'][treatment_type] = if_applied
            
    all_dict = {}
    for case_id in case_list:
        if case_id in status_dict.keys():
            all_dict[case_id] = status_dict[case_id]
    
    json.dump(all_dict, open(output_path,'w'), indent=2)

if __name__=="__main__":
    gen_status()
