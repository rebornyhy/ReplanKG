import numpy as np
import os
#from kg_process import *
import pickle
import jsonlines
import json
import re
from tqdm import tqdm
import copy
import networkx as nx
from typing import List

def create_graph_all(all_tri, multi = False):
    if multi == True:
        G = nx.MultiDiGraph()
        #print('Multi!')
    else:
        G = nx.DiGraph()
    for tri in all_tri:
        if len(tri) != 3:
            continue
        G.add_edge(tri[0], tri[2], relation=tri[1])
    return G

def is_valid_ai_output(text):
    """
    判别AI的输出是否符合规范：
    1. 必须严格包含一个<think>...</think>和一个<plan>...</plan>
    2. <think>必须在<plan>之前
    3. 允许<think>...</think>与<plan>...</plan>之间有任意数量换行符
    4. 除此之外不允许其他字符（包括空格）
    """
    # 允许 <think>...</think> 后跟 0 个或多个换行符，再跟 <plan>...</plan>
    pattern = r'^<think>.*?</think>\n*<plan>.*?</plan>$'
    match = re.fullmatch(pattern, text, re.DOTALL)
    if not match:
        return False

    # 额外检查仅出现一次 <think>...</think> 与一次 <plan>...</plan>
    think_count = len(re.findall(r'<think>.*?</think>', text, re.DOTALL))
    plan_count = len(re.findall(r'<plan>.*?</plan>', text, re.DOTALL))
    
    if think_count != 1 or plan_count != 1:
        return False

    # 再确保 <think> 出现在 <plan> 之前
    if text.find('<think>') > text.find('<plan>'):
        return False

    return True

def is_valid_ai_output_stage3(text):
    """
    判别AI的输出是否符合规范：
    1. 必须严格包含一个<think>...</think>和一个<answer>...</answer>
    2. <think>必须在<answer>之前
    3. 允许<think>...</think>与<answer>...</answer>之间有任意数量换行符
    4. 除此之外不允许其他字符（包括空格）
    """
    # 允许 <think>...</think> 后跟 0 个或多个换行符，再跟 <answer>...</answer>
    pattern = r'^<think>.*?</think>\n*<answer>.*?</answer>$'
    match = re.fullmatch(pattern, text, re.DOTALL)
    if not match:
        return False

    # 额外检查仅出现一次 <think>...</think> 与一次 <plan>...</plan>
    think_count = len(re.findall(r'<think>.*?</think>', text, re.DOTALL))
    plan_count = len(re.findall(r'<answer>.*?</answer>', text, re.DOTALL))
    
    if think_count != 1 or plan_count != 1:
        return False

    # 再确保 <think> 出现在 <plan> 之前
    if text.find('<think>') > text.find('<answer>'):
        return False

    return True

def is_valid_plan_path(plan_text):
    plan_text = plan_text.strip()
    
    # 合法实体：英文名词、m.前缀、g.前缀、或变量（如 ?x）
    entity_pattern = r'(?:[a-zA-Z0-9]+(?: [a-zA-Z0-9]+)*|m\.\w+|g\.\w+|\?[a-zA-Z_][a-zA-Z0-9_.\-]*)'
    
    # 关系名：仅允许字母、下划线、点
    relation_pattern = r'[a-zA-Z0-9_:.\-]+'
    
    # 只允许正向跳：-关系->实体
    hop_pattern = fr'-{relation_pattern}->{entity_pattern}'
    
    # 完整路径：起始实体 + 至少一个跳
    full_path_pattern = fr'^{entity_pattern}(?:{hop_pattern})+$'
    
    match = re.fullmatch(full_path_pattern, plan_text)
    return match is not None

def is_valid_plan_bib_path(plan_text):
    plan_text = plan_text.strip()
    
    # 实体的合法形式
    entity_pattern = r'(?:[a-zA-Z0-9]+(?: [a-zA-Z0-9]+)*|m\.\w+|g\.\w+|\?[a-zA-Z_][a-zA-Z0-9_.\-]*)'
    
    # 关系名格式
    relation_pattern = r'[a-zA-Z0-9_:.\-]+'
    
    # 一个跳跃（正向或反向）
    hop_pattern = (
        fr'(?:-{relation_pattern}->{entity_pattern}'      # 正向跳
        fr'|<-{relation_pattern}-{entity_pattern})'        # 反向跳
    )
    
    # 整体路径：以一个实体开头，跟1个或多个跳跃
    full_path_pattern = fr'^{entity_pattern}(?:{hop_pattern})+$'
    
    match = re.fullmatch(full_path_pattern, plan_text)
    return match is not None

def is_unknown_entity(s):
    """
    判断输入字符串是否是一个合法的未知实体
    形式为: ?+字母
    """
    pattern = r'^\?[a-zA-Z]+$'
    return re.fullmatch(pattern, s) is not None

def is_valid_plan_output(output_plan):
    output_plan = output_plan.strip()
    plan_list = output_plan.split('\n')
    pre_type = 'path'
    for idx, p in enumerate(plan_list):
        if idx == 0:
            if is_valid_plan_path(p) == False:
                return False
            pre_type = 'path'
            continue
        if pre_type == 'path':
            if is_valid_plan_path(p):
                pre_type = 'path'
                continue
            if 'FLITER' in p or 'ORDER' in p:
                pre_type = 'limit'
                continue
            if is_unknown_entity(p) and idx == (len(plan_list) - 1):
                return True
            return False
        if pre_type == 'limit':
            if is_valid_plan_path(p):
                return False
            if 'FLITER' in p or 'ORDER' in p:
                pre_type = 'limit'
                continue
            if is_unknown_entity(p) and idx == len(plan_list) - 1:
                return True
            return False
    return False

def is_valid_plan_path_with_explan(plan_text):
    errors = []
    plan_text = plan_text.strip()

    if "\n" in plan_text or "\r" in plan_text:
        errors.append("The path must not contain newline characters")
        return False, ';'.join(errors)

    # 1. 用正则提取所有 -关系-> 之间的实体和关系
    edge_pattern = re.compile(r'-(?P<rel>[A-Za-z0-9_:.\-]+)->')
    splits = list(edge_pattern.finditer(plan_text))

    # 2. 如果没有合法的关系连接，直接报结构错误，禁止后续实体分析
    if not splits:
        errors.append("The path format of '{}' does not conform to the structural specification of 'ent-rel->ent-rel->ent'".format(plan_text))
        return False, ';'.join(errors)

    # 3. 提取实体和关系
    entities = []
    relations = []

    first_entity = plan_text[:splits[0].start()]
    entities.append(first_entity)

    for i in range(len(splits)):
        relations.append(splits[i].group("rel"))
        start = splits[i].end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(plan_text)
        entity = plan_text[start:end]
        entities.append(entity)

    # 4. 检查实体格式
    for ent in entities:
        if ent.startswith('?'):
            if not re.fullmatch(r'\?[A-Za-z0-9_]+', ent):
                if '-' in ent or '>' in ent:
                    errors.append("The path format of '{}' does not conform to the structural specification of 'ent-rel->ent-rel->ent'".format(plan_text))
                else:
                    errors.append(f"Illegal variable entity：'{ent}'，Variables must start with ? and contain only alphabetic characters")
                
    # 5. 检查中间变量重复
    seen_vars = set()
    for ent in entities:
        if ent.startswith('?'):
            if ent in seen_vars:
                errors.append(f"Duplicate intermediate variables '{ent}' exist in one same path .")
            seen_vars.add(ent)

    return len(errors) == 0, ';'.join(errors)

def check_out_plan_path_with_explan(path):
    path = path.strip()
    general_format = is_valid_plan_path(path)
    if general_format == True:
        return is_valid_plan_path_with_explan(path)
    else:
        if is_valid_plan_bib_path(path):
            return False, "The path format of '{}' does not conform to the structural specification of 'ent-rel->ent-rel->ent'".format(path)
        else:
            return is_valid_plan_path_with_explan(path)

def check_out_whole_plan_with_explan(plan):
    plan = plan.strip()
    plan_list = plan.split('\n')
    if is_unknown_entity(plan_list[-1]) == False:
        return False, "Missing return entity, or the return entity format is incorrect (should be ? followed by letters, e.g., ?x)."
    path_list = []
    filter_list = []
    for p in plan_list:
        if p == plan_list[-1]:
            return_ent = p
            break
        if 'FILTER' in p or 'ORDER' in p:
            filter_list.append(p)
        else:
            path_list.append(p)
    if return_ent not in '\n'.join(path_list):
        return False, "The return entity must be in the path."
    for p in path_list:
        if check_out_plan_path_with_explan(p)[0] == False:
            return check_out_plan_path_with_explan(p)
    return True, ""

def extract_plan_content(output_text):
    """
    从完整的AI输出中提取 <plan>...</plan> 中的内容。
    如果未找到，返回 None。
    """
    match = re.search(r'<plan>(.*?)</plan>', output_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_entities_and_relations(plan_text):
    edge_pattern = re.compile(r'-(?P<rel>[A-Za-z0-9_:.\.]+)->')
    splits = list(edge_pattern.finditer(plan_text))

    if not splits:
        return []

    result = []

    # 提取第一个实体
    first_entity = plan_text[:splits[0].start()]
    result.append(first_entity)

    # 依次提取关系和实体
    for i in range(len(splits)):
        # 提取关系
        relation = splits[i].group("rel")
        result.append(relation)

        # 提取实体
        start = splits[i].end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(plan_text)
        entity = plan_text[start:end]
        result.append(entity)

    return result

def res_psth_str(pathlist):
    ans = []
    isent = True
    #print(pathlist[::-1])
    for p in pathlist[::-1]:
        if isent == True:
            ans.append(p)
            isent = False
        else:
            #print('.'.join(p.split('.')[-2:][::-1]))
            ans.append('.'.join(p.split('.')[-2:][::-1]))
            isent = True
    return ans

def paths_str_tree_modify(paths_str):
    def extract_entities(p_list):
        ans = []
        for i in range(0, len(p_list), 2):
            ans.append(p_list[i])
        return ans

    def reconstruct_from_list(elements):
        """
        输入: ['ent1', 'rel1', 'ent2', 'rel2', 'ent3']
        输出: 'ent1-rel1->ent2-rel2->ent3'
        """
        if not elements or len(elements) < 3 or len(elements) % 2 == 0:
            return ''
    
        result = elements[0]  # 第一个实体
        for i in range(1, len(elements), 2):
            rel = elements[i]
            ent = elements[i + 1]
            result += f"-{rel}->{ent}"
    
        return result
    ans = []
    paths_list = paths_str.split('\n')
    cur_ent_set = set()
    for p_id, path in enumerate(paths_list):
        #print(cur_ent_set)
        path_list = extract_entities_and_relations(path)
        #print(paths_list)
        if p_id == 0:
            if path_list[0][0] == '?':
                raise ValueError('invalid paths')
            ents = extract_entities(path_list)
            for e in ents:
                cur_ent_set.add(e)
            ans.append(path)
        else:
            ents = extract_entities(path_list)
            #print(ents)
            overlap = False
            overlap_ent = ''
            for e in ents:
                if e in cur_ent_set:
                    overlap = True
                    overlap_ent = e
                    break
            if overlap:
                if path_list[0] == overlap_ent:
                    ans.append(path)
                    for e in ents:
                        cur_ent_set.add(e)
                    continue
                else:
                    over_index = path_list.index(overlap_ent)
                    to_res_list = path_list[:over_index+1]
                    retain_list = path_list[over_index:]
                    res_list = res_psth_str(to_res_list)
                    retain_str = reconstruct_from_list(retain_list)
                    res_str = reconstruct_from_list(res_list)
                    ans.append(retain_str)
                    ans.append(res_str)
                    for e in ents:
                        cur_ent_set.add(e)
                    continue
            else:
                flag = False
                for e in ents:
                    if e[0] != "?" and '"' not in e:
                        overlap_ent = e
                        if path_list[0] == overlap_ent:
                            ans.append(path)
                            for e in ents:
                                cur_ent_set.add(e)
                            flag = True
                            break
                        else:
                            over_index = path_list.index(overlap_ent)
                            to_res_list = path_list[:over_index+1]
                            retain_list = path_list[over_index:]
                            res_list = res_psth_str(to_res_list)
                            retain_str = reconstruct_from_list(retain_list)
                            res_str = reconstruct_from_list(res_list)
                            ans.append(retain_str)
                            ans.append(res_str)
                            for e in ents:
                                cur_ent_set.add(e)
                            flag = True
                            break
                if flag == False:
                    raise ValueError('invalid paths')
    ret = []
    for a in ans:
        if a != '':
            ret.append(a)
    return '\n'.join(ret)

def path_logic_check(paths_list):
    def extract_entities(p_list):
        ans = []
        for i in range(0, len(p_list), 2):
            ans.append(p_list[i])
        return ans

    def reconstruct_from_list(elements):
        """
        输入: ['ent1', 'rel1', 'ent2', 'rel2', 'ent3']
        输出: 'ent1-rel1->ent2-rel2->ent3'
        """
        if not elements or len(elements) < 3 or len(elements) % 2 == 0:
            return ''
    
        result = elements[0]  # 第一个实体
        for i in range(1, len(elements), 2):
            rel = elements[i]
            ent = elements[i + 1]
            result += f"-{rel}->{ent}"
    
        return result
    ans = []
    cur_ent_set = set()
    for p_id, path in enumerate(paths_list):
        #print(cur_ent_set)
        path_list = extract_entities_and_relations(path)
        #print(paths_list)
        if p_id == 0:
            if path_list[0][0] == '?':
                return False, 'The head entity of the first retrieval path cannot be an unknown entity.'
            ents = extract_entities(path_list)
            for e in ents:
                cur_ent_set.add(e)
            ans.append(path)
        else:
            ents = extract_entities(path_list)
            #print(ents)
            overlap = False
            overlap_ent = ''
            for e in ents:
                if e in cur_ent_set:
                    overlap = True
                    overlap_ent = e
                    break
            if overlap:
                overlap_len = len(set(ents) & cur_ent_set)
                if overlap_len >= 2:
                    return False, 'There is a loop between the retrieval paths.'
                if path_list[0] == overlap_ent:
                    for e in ents:
                        cur_ent_set.add(e)
                    continue
                else:
                    for e in ents:
                        cur_ent_set.add(e)
                    continue
            else:
                flag = False
                for e in ents:
                    if e[0] != "?" and '"' not in e:
                        overlap_ent = e
                        if path_list[0] == overlap_ent:
                            ans.append(path)
                            for e_ in ents:
                                cur_ent_set.add(e_)
                            flag = True
                            break
                        else:
                            for e_ in ents:
                                cur_ent_set.add(e_)
                            flag = True
                            break
                if flag == False:
                    return False, 'Unknown entities has not appeared in the previous retrieval paths.'
    return True, ''

def clean_filter_expression_plan(filter_str):
    """
    清理FILTER表达式中的 str(...)、xsd:dateTime(...)、xsd:integer(...) 包裹
    """
    # 定义要去掉的函数名
    funcs = ['str', 'xsd:dateTime', 'xsd:integer', 'xsd:datetime', 'xsd:float']
    
    # 针对每一个函数，使用正则替换
    for func in funcs:
        # 这个正则匹配 func(...)，内部可以包含多层括号或+-*/空格等
        pattern = re.compile(rf'{func}\(\s*([^\(\)]+?)\s*\)')
        # 不断替换直到没有匹配（防止多层嵌套）
        while re.search(pattern, filter_str):
            filter_str = re.sub(pattern, r'\1', filter_str)

    #if 'EXIST' in filter_str or 'lang(' in filter_str or "FILTER (!isLiteral(?x) OR (lang(?x) = '' OR lang(?x) = 'en'))" in filter_str:
    #    return ''

    filter_str = filter_str.replace('^^xsd:dateTime', '')

    #entities = re.findall(r'ns:(?:m|g)\.[\w.]+', filter_str)
    #if entities != []:
    #    return ''
    
    return filter_str

def check_extract_modify_get_plan(raw_output):
    if is_valid_ai_output(raw_output) == False:
        return False, 'Do not include anything other than <think>...</think><plan>...</plan>.'
    plan = extract_plan_content(raw_output)
    check, explan = check_out_whole_plan_with_explan(plan)
    if check == False:
        return False, explan
    plan = plan.strip()
    plan_list = plan.split('\n')
    path_list = []
    limit_list = []
    return_ent = plan_list[-1]
    for p in plan_list:
        if p == plan_list[-1]:
            break
        if 'FILTER' in p or 'ORDER' in p:
            limit_list.append(p)
        else:
            path_list.append(p)
    logic_check, explan = path_logic_check(path_list)
    if logic_check == False:
        return False, explan
    path_list = paths_str_tree_modify('\n'.join(path_list)).split('\n')
    for l_id, limit_str in enumerate(limit_list):
        limit_list[l_id] = clean_filter_expression_plan(limit_str)
    whole_plan = []
    whole_plan += path_list
    whole_plan += limit_list
    whole_plan.append(return_ent)
    return True, '\n'.join(whole_plan)
