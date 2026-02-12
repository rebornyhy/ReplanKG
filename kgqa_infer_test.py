import pickle
import random
import time
import jsonlines
from FlagEmbedding import FlagAutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer
#sfrom peft import AutoPeftModelForCausalLM
import json
from tqdm import tqdm
from pathlib import Path
from kg_process import *
from get_sim_rels_list import *
from check_modify_plan import *
from execute_plan import *

def load_model_and_tokenizer(model_dir):
    
    model_dir = Path(model_dir).expanduser().resolve()
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, torch_dtype="auto", device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype="auto", device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    return model, tokenizer

def infer(prompt, model, tokenizer):
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    start_time = time.time()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    end_time = time.time()
    infer_time = end_time - start_time
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response, infer_time

def get_infer_main(model, tokenizer, iter_n, data_path, save_path, width=3, retry=1):
    data = list(jsonlines.open(data_path))
    if 'simpleques' in data_path:
        data = random.sample(data, 1000)
    data_cate = []
    data_stat_info = {
        'infer_num':0,
        'w/o_stage3_hit1_num': 0,
        'stage3_hit1_num': 0,
        'stage3_iter_hit1_num':0,
        'w/o_stage3_f1_sum':0,
        'stage3_f1_sum':0,
        'stage3_iter_f1_sum':0,
    }
    for d_idx, dd in tqdm(enumerate(data), total=len(data)):
        #if d_idx > 158:
        #    continue   
        # get ready
        question = dd['question']
        exe_num = 0
        exe_time = 0
        if 'parses' in dd:
            d = dd['parses']['0']
        else:
            d = dd
        data_source = d['source']
        data_class = data_source.split('-')[0]
        #if data_class != 'webqsp':
        #    continue 
        if data_class not in data_cate:
            data_cate.append(data_class)
            print(data_class, 'begin infer!!!')
        raw_idx = d['raw_idx']
        data_kg = data_class_2_data_kg[data_class]

        # stage 1
        prompt = d["prompt"]
        retry_n = retry
        while retry_n > 0:
            try:
                raw_output, infer_time = infer(prompt, model, tokenizer)
                check, whole_plan = check_extract_modify_get_plan(raw_output)
                if check == False:
                    retry_n -= 1
                    #print(raw_output)
                    continue   
                retr_hop = data_class_2_hop[data_class]
                sim_rels_dict = get_sim_rels_list(data_kg, d, whole_plan.split('\n'), data_class, raw_idx, mid2name, emb_model, retr_hop, 5) 
                break
            except Exception as e:
                print(e, 'something wrong!')
                #raise ValueError('wrong!')
                retry_n -= 1        
                continue
        if retry_n == 0:
            print(d_idx, 'infer failed! give up!')
            continue
        d['stage1_input'] = prompt
        d['stage1_raw_output'] = raw_output
        d['stage1_plan'] = whole_plan
        d['stage1_infer_time'] = infer_time
        d['sim_n_hop_rels'] = sim_rels_dict 
           
        if 'parses' in dd:
            dd['parses']['0'] = d
        else:
            dd = d
        if 'cwq' in data_class:
            prompt = get_ai_input_stage2(PROMPT_2_shot, dd, width = width, few_shot = True)
        else:
            prompt = get_ai_input_stage2(PROMPT_2, dd, width = width)

        # stage 2
        retry_n = retry
        while retry_n > 0:
            try:
                raw_output, infer_time = infer(prompt, model, tokenizer)
                check, whole_plan = check_extract_modify_get_plan(raw_output)
                if check == False:
                    retry_n -= 1
                    #print(raw_output)
                    continue 
                start_time = time.time()
                raw_plan_list, ret, return_ent, wrong_dict = execute_whole_plan(data_kg, dd, data_class, whole_plan, mid2name, emb_model)  
                end_time = time.time()
                exe_time2 = end_time - start_time
                report = get_retr_result_report(raw_plan_list, ret, return_ent, wrong_dict)
                f1_score = cal_pre_f1_score(dd, ret, return_ent, mid2name, data_class)
                break
            except Exception as e:
                #print(raw_output)
                print(e, 'something wrong!')
                retry_n -= 1        
                continue
        if retry_n == 0:
            print(d_idx, 'infer failed! give up!')
            continue
        d['stage2_input'] = prompt
        d['stage2_output'] = raw_output
        d['stage2_plan'] = whole_plan
        d['stage2_f1_score'] = f1_score
        d['stage2_infer_time'] = infer_time
        d['stage2_wrong_dict'] = wrong_dict
        d['stage2_exe_report'] = report 
        exe_num += 1
        exe_time += exe_time2   
        if 'parses' in dd:
            dd['parses']['0'] = d
        else:
            dd = d
        if 'cwq' in data_class:
            prompt = get_ai_input_stage3(PROMPT_3_shot, dd, few_shot = True)
        else:
            prompt = get_ai_input_stage3(PROMPT_3, dd)
        
        # stage 3
        d['reflection_list'] = []
        for _ in range(iter_n):
            cur_ref_dict = dict()
            retry_n = retry
            while retry_n > 0:
                try:
                    raw_output, infer_time = infer(prompt, model, tokenizer)
                    #check, whole_plan = check_extract_modify_get_plan(raw_output)
                    if '<answer>' in raw_output and '</answer>' in raw_output:
                        if is_valid_ai_output_stage3(raw_output) == False:
                            retry_n -= 1
                            #print(raw_output)
                            continue 
                        if d['reflection_list'] == []:
                            cur_ref_dict['stage3_input'] = prompt
                            cur_ref_dict['stage3_output'] = raw_output
                            cur_ref_dict['stage3_f1_score'] = d['stage2_f1_score']
                            cur_ref_dict['stage3_infer_time'] = infer_time
                        else:
                            cur_ref_dict['stage3_input'] = prompt
                            cur_ref_dict['stage3_output'] = raw_output
                            cur_ref_dict['stage3_f1_score'] = d['reflection_list'][-1]['stage3_f1_score']
                            cur_ref_dict['stage3_infer_time'] = infer_time
                    else:
                        check, whole_plan = check_extract_modify_get_plan(raw_output)
                        if check == False:
                            retry_n -= 1
                            #print(raw_output)
                            continue 
                        start_time = time.time()
                        raw_plan_list, ret, return_ent, wrong_dict = execute_whole_plan(data_kg, dd, data_class, whole_plan, mid2name, emb_model)  
                        end_time = time.time()
                        exe_time3 = end_time - start_time
                        exe_num += 1
                        exe_time += exe_time3
                        report = get_retr_result_report(raw_plan_list, ret, return_ent, wrong_dict)
                        f1_score = cal_pre_f1_score(dd, ret, return_ent, mid2name, data_class)
                        cur_ref_dict['stage3_input'] = prompt
                        cur_ref_dict['stage3_output'] = raw_output
                        cur_ref_dict['plan'] = whole_plan
                        cur_ref_dict['stage3_wrong_dict'] = wrong_dict
                        cur_ref_dict['stage3_exe_report'] = report
                        cur_ref_dict['stage3_f1_score'] = f1_score
                        cur_ref_dict['stage3_infer_time'] = infer_time
                    break
                except Exception as e:
                    #print(raw_output)
                    print(e, 'something wrong!')
                    retry_n -= 1        
                    continue
            if retry_n == 0:
                print(d_idx, 'infer failed! give up!')
                continue
            if 'plan' not in cur_ref_dict:
                d['reflection_list'].append(cur_ref_dict)
                d['avg_exe_time'] = exe_time / exe_num
                break
            else:
                d['reflection_list'].append(cur_ref_dict)
                d['avg_exe_time'] = exe_time / exe_num
                if 'parses' in dd:
                    dd['parses']['0'] = d
                else:
                    dd = d
                if 'cwq' in data_class:
                    prompt = get_ai_input_stage3_ref(PROMPT_3_shot, dd, few_shot = True)
                else:
                    prompt = get_ai_input_stage3_ref(PROMPT_3, dd)
        if d['reflection_list'] == []:
            continue
        data_stat_info['infer_num'] += 1
        if d['stage2_f1_score'] > 0:
            data_stat_info['w/o_stage3_hit1_num'] += 1
        data_stat_info['w/o_stage3_f1_sum'] += d['stage2_f1_score']
        if d['reflection_list'][0]['stage3_f1_score'] > 0:
            data_stat_info['stage3_hit1_num'] += 1
        data_stat_info['stage3_f1_sum'] += d['reflection_list'][0]['stage3_f1_score']
        if d['reflection_list'][-1]['stage3_f1_score'] > 0:
            data_stat_info['stage3_iter_hit1_num'] += 1
        data_stat_info['stage3_iter_f1_sum'] += d['reflection_list'][-1]['stage3_f1_score']
        if 'stage3_exe_report' in d['reflection_list'][-1]:
            if 'Head node of the retrieval path' in d['reflection_list'][-1]['stage3_exe_report'] and 'is not in KG!' in d['reflection_list'][-1]['stage3_exe_report']:
                d['reflection_list'][-1]['stage3_f1_score'] = 1
        else:
            if 'Head node of the retrieval path' in d['stage2_exe_report'] and 'is not in KG!' in d['stage2_exe_report']:
                d['reflection_list'][-1]['stage3_f1_score'] = 1
        if 'parses' in dd:
            dd['parses']['0'] = d
        else:
            dd = d
        cur_save_path = save_path.format(data_class, exp_name)
        save = open(cur_save_path, 'a',encoding='utf-8')
        save.write(json.dumps(dd, ensure_ascii=False) + '\n')
        save.close()
    data_stat_info['w/o_stage3_hit1_avg'] = data_stat_info['w/o_stage3_hit1_num'] / data_stat_info['infer_num']
    data_stat_info['stage3_hit1_avg'] = data_stat_info['stage3_hit1_num'] / data_stat_info['infer_num']
    data_stat_info['stage3_iter_hit1_avg'] = data_stat_info['stage3_iter_hit1_num'] / data_stat_info['infer_num']
    data_stat_info['stage3_f1_avg'] = data_stat_info['stage3_f1_sum'] / data_stat_info['infer_num']
    data_stat_info['w/o_stage3_f1_avg'] = data_stat_info['w/o_stage3_f1_sum'] / data_stat_info['infer_num']
    data_stat_info['stage3_iter_f1_avg'] = data_stat_info['stage3_iter_f1_sum'] / data_stat_info['infer_num']
    return 

if __name__ == '__main__':
    print('#'*10, 'import mid2name')
    with open('/apdcephfs_cq7/share_1447896/hongyiyang/data/mid2name.pkl', 'rb') as file:
        mid2name = pickle.load(file)
        
    print('#'*10, 'import embedding model')
    emb_model = FlagAutoModel.from_finetuned('/apdcephfs_gy2/share_303215196/hongyiyang/model/bge-large-en-v1.5',
                                        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                                        use_fp16=True)

    print('#'*10, 'import data_kg')
    print('#'*10, 'import simpleques_kg')
    sim_start = time.time()
    simpleques = get_simpleques('/apdcephfs_cq7/share_1447896/hongyiyang/data/simpleques/simpleques_test.jsonl', '/apdcephfs_cq7/share_1447896/hongyiyang/data/simpleques/simpleques_test_subgraph.jsonl')
    sim_end = time.time()
    print('simpleques time:', sim_end - sim_start)
    print('#'*10, 'import webqsp_kg')
    webqsp_start = time.time()
    webqsp = get_webqsp('/apdcephfs_cq7/share_1447896/hongyiyang/data/webqsp/webqsp_test.jsonl', '/apdcephfs_cq7/share_1447896/hongyiyang/data/webqsp/test_simple1.jsonl', '/apdcephfs_cq7/share_1447896/hongyiyang/data/webqsp/webqsp_test_com2.jsonl', '/apdcephfs_cq7/share_1447896/hongyiyang/data/webqsp')
    webqsp_end = time.time()
    print('webqsp time:', webqsp_end - webqsp_start)
    print('#'*10, 'import cwq_kg')
    cwq_start = time.time()
    cwq = get_cwq('/apdcephfs_cq7/share_1447896/hongyiyang/data/cwq/cwq_test.jsonl', '/apdcephfs_cq7/share_1447896/hongyiyang/data/cwq/test_simple1.jsonl', '/apdcephfs_cq7/share_1447896/hongyiyang/data/cwq/cwq_test_com5.jsonl', '/apdcephfs_cq7/share_1447896/hongyiyang/data/cwq')
    cwq_end = time.time()
    print('cwq time:', cwq_end - cwq_start)
    print('#'*20, 'import over !!!')

    data_class_2_data_kg = {
        'webqsp': webqsp,
        'cwq_train': cwq,
        'cwq': cwq,
        'simpleques': simpleques
    }

    data_class_2_hop = {
        'webqsp': 2,
        'cwq_train': 4,
        'cwq': 4,
        'simpleques': 1
    }

    data_path = './cwq_testdata_shot.jsonl'
    model_path = '/apdcephfs_gy2/share_303215196/hongyiyang/model/rl_14b_stage3'
    save_path = './{}_{}_infer.jsonl'
    exp_name = '14b_infer_test'
    model, tokenizer = load_model_and_tokenizer(model_path)
    get_infer_main(model, tokenizer, 1, data_path, save_path, width=3, retry=5)
