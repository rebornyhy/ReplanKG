from execute_plan import *
import jsonlines
from kg_process import *
from get_retr_plan import *
import time

PROMPT_2 = r"""
You are a knowledge graph question-answering assistant. Now you need to reason based on the given question , topic entity and the n-hop relationships around the topic entity, and produce a retrieval plan for querying the knowledge graph. 

The retrieval plan must consist of three parts:
1. One or more retrieval paths (at least one is required). Each path is structured as:
entity-relation->entity-relation->...->tail entity
(unknown entities should be represented as ‘?’ plus letter, such as “?x”).
If there are multiple paths, separate them by the newline character '\n'.
2. Optional filtering or sorting constraints, with line breaks between multiple constraints. For example:
- FILTER() (e.g., FILTER (?from < ?end); FILTER (?num < "1983-01-03"))
- ORDER BY (e.g., ORDER BY ?num LIMIT 1; ORDER BY DESC(?num) LIMIT 1 OFFSET 1)
3. The entity to be returned (this is mandatory), indicating which entity the retrieval plan is designed to return.
Separate these three parts by newline characters '\n'.

The part of n-hop relationships around the topic entity are represented by '[{{one hop relationships}}, {{two hop relationships}}, {{three hop relationships}}, ...]', for reference during inference.
Note that the relationships at each hop are only partial, not exhaustive!

Given the question: {}
and the topic entity/entities ({}): {} 
{}
please reason out the retrieval plan.

Output your reasoning process wrapped with <think></think> tags, and the retrieval plan wrapped with <plan></plan> tags, in the format:
<think>...</think><plan>...</plan>.
Do not include anything other than <think>...</think><plan>...</plan>. 
"""

PROMPT_2_shot = r"""
You are a knowledge graph question-answering assistant. Now you need to reason based on the given question , topic entity and the n-hop relationships around the topic entity, and produce a retrieval plan for querying the knowledge graph. 

The retrieval plan must consist of three parts:
1. One or more retrieval paths (at least one is required). Each path is structured as:
entity-relation->entity-relation->...->tail entity
(unknown entities should be represented as ‘?’ plus letter, such as “?x”).
If there are multiple paths, separate them by the newline character '\n'.
2. Optional filtering or sorting constraints, with line breaks between multiple constraints. For example:
- FILTER() (e.g., FILTER (?from < ?end); FILTER (?num < "1983-01-03"))
- ORDER BY (e.g., ORDER BY ?num LIMIT 1; ORDER BY DESC(?num) LIMIT 1 OFFSET 1)
3. The entity to be returned (this is mandatory), indicating which entity the retrieval plan is designed to return.
Separate these three parts by newline characters '\n'.

The part of n-hop relationships around the topic entity are represented by '[{{one hop relationships}}, {{two hop relationships}}, {{three hop relationships}}, ...]', for reference during inference.
Note that the relationships at each hop are only partial, not exhaustive!

{}

Given the question: {}
and the topic entity/entities ({}): {} 
{}
please reason out the retrieval plan.

Output your reasoning process wrapped with <think></think> tags, and the retrieval plan wrapped with <plan></plan> tags, in the format:
<think>...</think><plan>...</plan>.
Do not include anything other than <think>...</think><plan>...</plan>. 
"""

def get_topic_ent_num(topic_ent):
    if isinstance(topic_ent, str):
        num = 1
    else :
        num = len(topic_ent)
    if num > 1:
        top_str = "entities"
    else:
        top_str = "entity"
    return str(num) + ' ' + top_str

def get_topic_ent(data):
    if "parses" in data:
        data = data["parses"]["0"]
    top_str = get_topic_ent_num(data["topic_ent"])
    if isinstance(data["topic_ent"], str):
        top_ent = data["topic_ent_name"][data["topic_ent"]]
    else:
        top_ent = '; '.join([data["topic_ent_name"][e] for e in data["topic_ent"]])
    return top_str, top_ent

def get_sim_hop_str(ent_hop_dict, rel_num):
    format_str = "Part of {}-hop relationships around the {}: {}"
    ans_str = []
    for ent, hop_rel_list in ent_hop_dict.items():
        hop_num = 0
        hop_str = []
        for hop_rel in hop_rel_list:
            if hop_rel != []:
                hop_num += 1
                hop_str.append('{'+', '.join(hop_rel[:rel_num])+'}')
            else:
                break
        ans_str.append(format_str.format(hop_num, ent, hop_str))
    return '\n'.join(ans_str)

def get_ai_input_stage2(prompt, data_w_rag, width=3, few_shot=False):
    ques = data_w_rag["question"]
    if "parses" in data_w_rag:
        data_w_rag = data_w_rag["parses"]["0"]
    d_top_str, d_top = get_topic_ent(data_w_rag)
    d_sim_rel = get_sim_hop_str(data_w_rag["sim_n_hop_rels"], width)
    d_plan = data_w_rag["plan"]
    
    if few_shot == False:
        return prompt.format(ques, d_top_str, d_top, d_sim_rel)
    else:
        few_shot = data_w_rag['few_shot']
        return prompt.format(few_shot, ques, d_top_str, d_top, d_sim_rel)

def extract_rels_in_plan(plan_list):
    paths_list = []
    for p in plan_list[:-1]:
        if 'FILTER' not in p and 'ORDER' not in p:
            paths_list.append(p)
    cand_rels_list = []
    for path in paths_list:
        path_list = extract_entities_and_relations(path)
        for i in range(1, len(path_list), 2):
            #print(path_list[i])
            cand_rels_list.append(path_list[i])
    return cand_rels_list

def get_sim_rels_list(raw_data_with_graph, data_stage1, plan_list, data_name, raw_idx, mid2name, emb_model, retr_hop, sim_k):
    dd = data_stage1
    #raw_data = list(jsonlines.open(raw_data))
    ans = []
    #for dd in tqdm(data_stage1):
    if data_name == 'webqsp' and 'parses' in dd:
        d = dd['parses']['0']
    else:
        d = dd
    #if raw_idx < 1070:
    #    continue
    r_d_g = raw_data_with_graph[raw_idx]
    tri_num = len(r_d_g['subgraph'])
    start1 = time.time()
    if data_name == 'simpleques':
        recall_graph = get_recall_subgraph_trainset(d['topic_ent'], d['topic_ent_name'], r_d_g['subgraph'], d['answer_ent_name'], d['kg'], mid2name)
    else:
        kg_list = get_retrieve_plan(d).get_sparql_kg_list()
        for k_id, tri in enumerate(kg_list):
            s, r, o = tri
            if s[0:3] == 'ns:':
                s = s.replace('ns:', '')
            if o[0:3] == 'ns:':
                o = o.replace('ns:', '')
            if r[0:3] == 'ns:':
                r = r.replace('ns:', '')
            kg_list[k_id] = [s, r, o]
        if tri_num > 6000:
            recall_graph = get_recall_subgraph_trainset(d['topic_ent'], d['topic_ent_name'], r_d_g['subgraph'], d['answer_ent_name'], kg_list, mid2name)
        else:
            recall_graph = get_recall_subgraph_trainset(d['topic_ent'], d['topic_ent_name'], r_d_g['subgraph'], d['answer_ent_name'], kg_list, mid2name, True)
    end1 = time.time()
    #print('get_recall_graph',end1-start1)
    if isinstance(d['topic_ent'], str):
        t_e = [d['topic_ent']]
    else:
        t_e = d['topic_ent']
    retr_ent_hop_rel_dict = dict()
    start2 = time.time()
    for e in t_e:
        e = d['topic_ent_name'][e]
        retr_ent_hop_rel_dict[e] = []
        e_retr = e.replace('-','_')
        start3 = time.time()
        cand_dict = get_hop_rel_tree(recall_graph, e_retr, retr_hop)
        #print(cand_dict)
        infer_list = extract_rels_in_plan(plan_list)
        if len(infer_list) == 0:
            print(plan_list)
            raise ValueError('wrong plan!')
        end3 = time.time()
        #print('get_cand_list', end3-start3)
        start4 = time.time()
        for hop, cand_list in cand_dict.items():
            #print(len(cand_list))
            k_rel = top_k_similar_candidates(emb_model, cand_list, infer_list, sim_k)
            retr_ent_hop_rel_dict[e].append(k_rel)
        end4 = time.time()
        #print('get_sim_list',end4-start4)
    end2 = time.time()
    #print('all_get_cand_sim_list',end2-start2)
    #raw_idx2hop_rel[raw_idx] = retr_ent_hop_rel_dict
    d['sim_n_hop_rels'] = retr_ent_hop_rel_dict
    #print(retr_ent_hop_rel_dict)
    return retr_ent_hop_rel_dict

def get_sim_rels_list_w_inputlist(raw_data_with_graph, data_stage1, str_list, data_name, raw_idx, mid2name, emb_model, retr_hop, sim_k):
    dd = data_stage1
    #raw_data = list(jsonlines.open(raw_data))
    ans = []
    #for dd in tqdm(data_stage1):
    if data_name == 'webqsp' and 'parses' in dd:
        d = dd['parses']['0']
    else:
        d = dd
    #if raw_idx < 1070:
    #    continue
    r_d_g = raw_data_with_graph[raw_idx]
    tri_num = len(r_d_g['subgraph'])
    start1 = time.time()
    if data_name == 'simpleques':
        recall_graph = get_recall_subgraph_trainset(d['topic_ent'], d['topic_ent_name'], r_d_g['subgraph'], d['answer_ent_name'], d['kg'], mid2name)
    else:
        kg_list = get_retrieve_plan(d).get_sparql_kg_list()
        for k_id, tri in enumerate(kg_list):
            s, r, o = tri
            if s[0:3] == 'ns:':
                s = s.replace('ns:', '')
            if o[0:3] == 'ns:':
                o = o.replace('ns:', '')
            if r[0:3] == 'ns:':
                r = r.replace('ns:', '')
            kg_list[k_id] = [s, r, o]
        if tri_num > 6000:
            recall_graph = get_recall_subgraph_trainset(d['topic_ent'], d['topic_ent_name'], r_d_g['subgraph'], d['answer_ent_name'], kg_list, mid2name)
        else:
            recall_graph = get_recall_subgraph_trainset(d['topic_ent'], d['topic_ent_name'], r_d_g['subgraph'], d['answer_ent_name'], kg_list, mid2name, True)
    end1 = time.time()
    #print('get_recall_graph',end1-start1)
    if isinstance(d['topic_ent'], str):
        t_e = [d['topic_ent']]
    else:
        t_e = d['topic_ent']
    retr_ent_hop_rel_dict = dict()
    start2 = time.time()
    for e in t_e:
        e = d['topic_ent_name'][e]
        retr_ent_hop_rel_dict[e] = []
        e_retr = e.replace('-','_')
        start3 = time.time()
        cand_dict = get_hop_rel_tree(recall_graph, e_retr, retr_hop)
        #print(cand_dict)
        infer_list = str_list
        if len(infer_list) == 0:
            print(plan_list)
            raise ValueError('wrong plan!')
        end3 = time.time()
        #print('get_cand_list', end3-start3)
        start4 = time.time()
        for hop, cand_list in cand_dict.items():
            #print(len(cand_list))
            k_rel = top_k_similar_candidates(emb_model, cand_list, infer_list, sim_k)
            retr_ent_hop_rel_dict[e].append(k_rel)
        end4 = time.time()
        #print('get_sim_list',end4-start4)
    end2 = time.time()
    #print('all_get_cand_sim_list',end2-start2)
    #raw_idx2hop_rel[raw_idx] = retr_ent_hop_rel_dict
    d['sim_n_hop_rels'] = retr_ent_hop_rel_dict
    #print(retr_ent_hop_rel_dict)
    return retr_ent_hop_rel_dict
