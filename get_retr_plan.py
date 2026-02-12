import os
import datasets
import json
from tqdm import tqdm
import jsonlines
import networkx as nx
import random
#from kg_process import *
import copy
import string
import itertools
import torch
from torch.utils.data import Dataset, DataLoader
import re
import regex
import numpy as np
from collections import defaultdict
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

def load_json_from_file(file_path):
    with open(file_path, 'r', encoding = 'utf-8') as file:
        data = json.load(file)
    return data

def dict_tojson(to_dict, json_file):
    jsonf = open(json_file, 'w', encoding='utf-8')
    json.dump(to_dict, jsonf, ensure_ascii=False)
    jsonf.close()
    
def list_dict_tojson(data, json_file):
    with open(json_file, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
        
def invert_dict(raw_dcit):
    answer = dict()
    for key, value in raw_dcit.items():
        answer[value] = key
    return answer

def get_distinct_entity(sparql):
    pattern = r"SELECT\s+DISTINCT\s+(\?\w+)"
    match = re.search(pattern, sparql, re.IGNORECASE)
    
    if match:
        variable = match.group(1)
        return variable

def count_leading_spaces(s: str) -> int:
    """
    计算字符串从头到第一个非空格字符之间的空格数。

    参数:
        s (str): 输入字符串
    
    返回:
        int: 开头连续空格的数量
    """
    count = 0
    for char in s:
        if char == ' ':
            count += 1
        else:
            break
    return count

def count_leading_entity(s: str) -> int:
    """
    计算字符串第一个非空格字符数。

    参数:
        s (str): 输入字符串
    
    返回:
        int: 数量
    """
    count = 0
    for char in s:
        if char != ' ':
            count += 1
        else:
            break
    return count

def com_sparql(spar):
    sparlist = spar.split('\n')
    infolist = []
    for s in sparlist:
        first_space = count_leading_spaces(s)
        first_ent = count_leading_entity(s[first_space:])
        second_space = count_leading_spaces(s[(first_space+first_ent):])
        cur_info = [first_space, first_ent, second_space]
        infolist.append(cur_info)
    for i in range(len(infolist)):
        if i == 0:
            continue
        if infolist[i][0] == infolist[i-1][0] + infolist[i-1][1] + infolist[i-1][2]:
            sparlist[i] = sparlist[i-1][:(infolist[i-1][0] + infolist[i-1][1] + infolist[i-1][2])] + sparlist[i][(infolist[i-1][0] + infolist[i-1][1] + infolist[i-1][2]):]
            infolist[i] = infolist[i-1]
    return '\n'.join(sparlist)

def sparql2kg(sparql):
    
    # 正则表达式匹配三元组模式
    pattern = r'(\?\w+|ns:[\w.]+)\s+(\w+:[\w.]+)\s+(\?\w+|ns:[\w.]+|"(?:[^"\\]|\\.)*")\s*[.;]'
    
    # 提取三元组
    matches = re.findall(pattern, sparql)

    #print(matches)
    
    # 格式化为三元组列表
    triples = [[subject, predicate, obj] for subject, predicate, obj in matches]

    return triples

def has_duplicate_triples(triples):
    """
    判断是否存在重复的三元组
    :param triples: List of triples (each triple is a list of 3 elements)
    :return: True if duplicate exists, False otherwise
    """
    seen = set()
    for triple in triples:
        t = tuple(triple)  # 转为不可变tuple以便存入set
        if t in seen:
            return True
        seen.add(t)
    return False

def remove_complex_filter_blocks(sparql: str) -> str:
    # 匹配 FILTER(NOT EXISTS {…} || EXISTS {…FILTER(…)} )
    # pattern = re.compile(
    #     r'''FILTER\s*\(\s*NOT\s+EXISTS\s*\{.*?\}\s*\|\|\s*EXISTS\s*\{.*?FILTER\s*\(.*?\).*?\}\s*\)\s*\.?''',
    #     re.DOTALL
    # )
    pattern = re.compile(
        r'''FILTER\s*\(\s*NOT\s+EXISTS\s*\{.*?\}[\s\r\n]*\|\|[\s\r\n]*EXISTS\s*\{.*?FILTER\s*\(.*?\).*?\}\s*\)\s*\.?''',
        re.DOTALL
    )
    # 替换为''（即删除），保留 SPARQL 其余部分
    return pattern.sub('', sparql)

def find_all_paths(triples, start, end):
    from collections import defaultdict, deque

    # 建立邻接表，key: 节点, value: (相邻节点, 三元组)
    adj = defaultdict(list)
    for s, p, o in triples:
        adj[s].append( (o, (s,p,o)) )
        adj[o].append( (s, (s,p,o)) )  # 无向图

    all_paths = []

    def dfs(current, target, visited, path_triples):
        if current == target:
            all_paths.append(list(path_triples))
            return
        for neighbor, triple in adj[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                path_triples.append(triple)
                dfs(neighbor, target, visited, path_triples)
                path_triples.pop()
                visited.remove(neighbor)

    # 启动 DFS
    visited = set([start])
    dfs(start, end, visited, [])

    return all_paths if all_paths else False

def extract_entities_from_sparql(sparql):
    # 首先去掉 FILTER (?x != ns:m.0vmt) 和 FILTER (!isLiteral...)
    sparql = sparql.replace("FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))", '')
    
    sparql_cleaned = re.sub(r'FILTER\s*\([^)]+\)', '', sparql, flags=re.DOTALL | re.IGNORECASE)

    #print(sparql_cleaned)
    
    # 提取所有 ns:实体
    #entities = re.findall(r'ns:m\.[\w.]+', sparql_cleaned)
    entities = re.findall(r'ns:(?:m|g)\.[\w.]+', sparql_cleaned)
    
    # 去重同时保持顺序
    seen = set()
    ordered_entities = []
    for entity in entities:
        if entity not in seen:
            seen.add(entity)
            ordered_entities.append(entity)

    return ordered_entities

def build_adj_list(triples):
    from collections import defaultdict
    adj = defaultdict(list)
    for triple in triples:
        s, p, o = triple
        adj[s].append((o, tuple(triple)))  # 转成tuple
        adj[o].append((s, tuple(triple)))  # 无向图
    return adj

def find_paths_from_node(start, triples):
    adj = build_adj_list(triples)
    all_paths = []

    def dfs(node, visited_nodes, used_triples, path_triples):
        visited_nodes.add(node)
        extended = False
        for neighbor, triple in adj[node]:
            if neighbor not in visited_nodes:
                if triple not in used_triples:
                    dfs(neighbor, visited_nodes.copy(), used_triples | {triple}, path_triples + [triple])
                    extended = True
        if not extended:
            all_paths.append(path_triples)

    dfs(start, set(), set(), [])

    # 为了按输入顺序输出
    result = []
    triple_tuples = [tuple(t) for t in triples]
    for path in all_paths:
        ordered_path = [list(t) for t in triple_tuples if t in path]
        result.append(ordered_path)
    return result

def remove_invalid_filter(raw_sparql):
    where_id = 0
    sparql_lines = raw_sparql.split('\n')
    for idx, sp in enumerate(sparql_lines):
        if 'WHERE' in sp:
            where_id = idx
            break
    where_id_1 = where_id + 1
    where_id_2 = where_id + 2
    if 'FILTER (' in sparql_lines[where_id_1]:
        sparql_lines[where_id_1] = ""
    if "FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))" in sparql_lines[where_id_2]:
        sparql_lines[where_id_2] = ""
    new_spar = []
    for s in sparql_lines:
        if s != "":
            new_spar.append(s)
    sparql = '\n'.join(new_spar)
    sparql = sparql.replace("FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))", '')
    return sparql

def find_unique_longest_list(lst_of_lists):
    if not lst_of_lists:
        return []

    # 计算每个子列表的长度
    lengths = [len(sublist) for sublist in lst_of_lists]
    max_length = max(lengths)

    # 找出所有长度等于最大长度的子列表索引
    max_length_indices = [i for i, length in enumerate(lengths) if length == max_length]

    # 如果只有一个这样的子列表，则返回它
    if len(max_length_indices) == 1:
        return lst_of_lists[max_length_indices[0]]
    else:
        return []

def clean_filter_expression(filter_str):
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

    if 'EXIST' in filter_str or 'lang(' in filter_str or "FILTER (!isLiteral(?x) OR (lang(?x) = '' OR lang(?x) = 'en'))" in filter_str:
        return ''

    filter_str = filter_str.replace('^^xsd:dateTime', '')

    entities = re.findall(r'ns:(?:m|g)\.[\w.]+', filter_str)
    if entities != []:
        return ''
    
    return filter_str

def get_filter_state(sparql):
    pattern = regex.compile(r'''
        FILTER
        \s*
        (?<paren>               # 命名的递归组
            \(
                (?:             # 非捕获组
                    [^()]+      # 非括号字符
                    |
                    (?&paren)   # 递归调用
                )*
            \)
        )
    [\s.;]*
    ''', regex.VERBOSE)
    matches = pattern.finditer(sparql)
    anst = []
    for m in matches:
        anst.append(m.group(0))
    ans = []
    for a in anst:
        a = clean_filter_expression(a)
        if a != '':
            ans.append(a)
    return ans

def get_limit(sparql):
    # 匹配 FILTER
    filter_pattern = re.compile(r'FILTER\s*\([^)]*\)[\s.;]*', re.DOTALL)
    
    # 匹配 ORDER BY ... LIMIT ... （允许中间跨多行）
    # order_limit_pattern = re.compile(
    #     r'(ORDER\s+BY[\s\S]*?(?:LIMIT\s+\d+))',
    #     re.IGNORECASE
    # )

    order_limit_pattern = re.compile(
        r'(ORDER\s+BY[\s\S]*?(?:LIMIT\s+\d+)(?:\s+OFFSET\s+\d+)?)',
        re.IGNORECASE
    )
    
    # 如果没有 LIMIT，只匹配单独的 ORDER BY 行
    order_only_pattern = re.compile(
        r'(ORDER\s+BY[^\n]*)',
        re.IGNORECASE
    )
    
    # 提取 FILTER
    filters = get_filter_state(sparql)
    filters = [str(f) for f in filters]
    
    # 优先提取 ORDER BY + LIMIT
    order_limits = order_limit_pattern.findall(sparql)
    
    if order_limits:
        order_results = order_limits
    else:
        # 再退而求其次提取单行 ORDER BY
        order_results = order_only_pattern.findall(sparql)

    for idx, f in enumerate(filters):
        f = f.replace('\n', '')
        while f[-1] != ')':
            f = f[:-1]
        filters[idx] = f

    for idx, f in enumerate(order_results):
        f = f.replace('\n', ' ')
        # while f[-1] != ')':
        #     f = f[:-1]
        order_results[idx] = clean_filter_expression(f)
    
        
    return filters, order_results

def find_all_longest_paths(triples):
    # 构建图：顶点 -> 出边的列表
    graph = defaultdict(list)
    # 记录所有起点和终点
    nodes_out = set()
    nodes_in = set()
    edges = []

    for s, p, o in triples:
        graph[s].append((s, p, o))
        nodes_out.add(s)
        nodes_in.add(o)
        edges.append((s, p, o))

    # 找到所有没有入边的起点（即可能的路径起点）
    possible_starts = nodes_out - nodes_in
    if not possible_starts:
        # 如果没有明显的起点，就把所有三元组的主语当作可能的起点
        possible_starts = nodes_out

    # 用 DFS 找所有无环路径
    all_paths = []

    def dfs(current_node, visited_nodes, path_edges):
        extended = False
        for s, p, o in graph.get(current_node, []):
            if o not in visited_nodes:
                dfs(o, visited_nodes | {o}, path_edges + [(s, p, o)])
                extended = True
        if not extended:
            if path_edges:
                all_paths.append(path_edges)

    # 从所有可能起点开始 DFS
    for start in possible_starts:
        dfs(start, {start}, [])

    # 如果有孤立边（例如 ?y -> ns:m.060d2）没有被遍历到，需要加上
    traversed_edges = set()
    for path in all_paths:
        traversed_edges.update(path)
    for edge in edges:
        if edge not in traversed_edges:
            all_paths.append([edge])

    return all_paths

def is_fully_connected_gen_plan(triples: List[List[str]]) -> bool:
    # 创建一个有向多重图
    G = nx.MultiDiGraph()
    
    # 添加三元组边（只用实体 s 和 o）
    for s, p, o in triples:
        G.add_edge(s, o, relation=p)

    # 转为无向图
    UG = G.to_undirected()

    # 获取所有连通子图
    connected_components = list(nx.connected_components(UG))

    # 如果只有一个连通子图，则图是连通的
    return len(connected_components) == 1

def plan_get_retre_subgraph_gen_plan(path_tri_list, res = False):
    path_list = copy.deepcopy(path_tri_list)
    ans_dict = dict()
    num = 0
    #print(path_list)
    for idx, path in enumerate(path_list):
        #print(path)
        if num == 0:
            num += 1
            cur_subgraph = dict()
            cur_subgraph['tris'] = path
            cur_subgraph['indexs'] = [idx]
            cur_subgraph['begin'] = '-'.join(path[0][:2])
            ans_dict[num] = cur_subgraph
            
        else:
            if path[0][0].startswith('?'):
                issubgraph_cur = True
                for k, v in ans_dict.items():
                    all_tris = v['tris'] + path
                    #print(all_tris)
                    if is_fully_connected_gen_plan(all_tris):
                        v['tris'] = all_tris
                        v['indexs'].append(idx)
                        ans_dict[k] = v
                        issubgraph_cur = False
                        break
                if issubgraph_cur == True:
                    num+=1
                    cur_subgraph = dict()
                    cur_subgraph['tris'] = path
                    cur_subgraph['indexs'] = [idx]
                    cur_subgraph['begin'] = '-'.join(path[0][:2])
                    ans_dict[num] = cur_subgraph
            else:
                issubgraph_cur = False
                cur_begin = '-'.join(path[0][:2])
                for k,v in ans_dict.items():
                    if cur_begin == v['begin']:
                        num+=1
                        cur_subgraph = dict()
                        cur_subgraph['tris'] = path
                        cur_subgraph['indexs'] = [idx]
                        cur_subgraph['begin'] = '-'.join(path[0][:2])
                        ans_dict[num] = cur_subgraph
                        issubgraph_cur = True
                        break
                if issubgraph_cur == True:
                    continue
                else:
                    issubgraph_cur = True
                    for k, v in ans_dict.items():
                        all_tris = v['tris'] + path
                        if is_fully_connected_gen_plan(all_tris):
                            issubgraph_cur = False
                            v['tris'] = all_tris
                            v['indexs'].append(idx)
                            ans_dict[k] = v
                            break
                    if issubgraph_cur == False:
                        continue
                    else:
                        num+=1
                        cur_subgraph = dict()
                        cur_subgraph['tris'] = path
                        cur_subgraph['indexs'] = [idx]
                        cur_subgraph['begin'] = '-'.join(path[0][:2])
                        ans_dict[num] = cur_subgraph
                        issubgraph_cur = True
    #print(ans_dict)
    ans_num = 0
    ans = dict()
    for k, v in ans_dict.items():
        ans_num+=len(v['indexs'])
        ans[k] = []
        for idx in v['indexs']:
            ans[k]+=path_tri_list[idx]
        #print(ans)
        for idx, p_l in enumerate(ans[k]):
            
            if idx > 0:
                if p_l[0].startswith('?') == False:
                    if res:
                        ans[k][idx] = res_psth_str(p_l)
                    else:
                        ans[k][idx] = p_l
    #print(ans_dict)
    #print(ans)
    #print(path_list)
    if ans_num != len(path_list):
        raise ValueError('wrong!')
    return ans

def find_all_longest_paths_new(triples):
    # 构建图
    graph = defaultdict(list)
    nodes_out = set()
    nodes_in = set()
    edges = []

    for s, p, o in triples:
        graph[s].append((s, p, o))
        nodes_out.add(s)
        nodes_in.add(o)
        edges.append((s, p, o))

    # 确定起点
    possible_starts = nodes_out - nodes_in
    if not possible_starts:
        possible_starts = nodes_out

    all_paths = []

    def dfs_collect_all(node, visited_nodes, collected_edges):
        for s, p, o in graph.get(node, []):
            if o not in visited_nodes:
                collected_edges.append((s, p, o))
                dfs_collect_all(o, visited_nodes | {o}, collected_edges)

    def split_triples(triples):
        paths = []
        current_path = []
        used_variable_subjects = set()
    
        for triple in triples:
            s, p, o = triple
            #if s.startswith('?'):
            if s in used_variable_subjects and current_path:
                # 当前变量主语已出现过，切分路径
                paths.append(current_path)
                current_path = []
                used_variable_subjects.clear()
            # 记录变量主语
            #if s.startswith('?'):
            used_variable_subjects.add(s)
            current_path.append(triple)
    
        if current_path:
            paths.append(current_path)
    
        return paths
        
    # 对每个起点，遍历所有 reachable edges，并合并成一个路径
    for start in possible_starts:
        collected_edges = []
        dfs_collect_all(start, {start}, collected_edges)
        if collected_edges:
            all_paths.append(collected_edges)

    # 补充孤立边
    traversed_edges = set()
    for path in all_paths:
        traversed_edges.update(path)
    for edge in edges:
        if edge not in traversed_edges:
            all_paths.append([edge])
    #print(all_paths)
    subgraph_path_list = []
    for cur_path_list in all_paths:
        if len(cur_path_list) == 1:
            subgraph_path_list.append(cur_path_list)
            continue
        else:
            cur_path_list_copy = copy.deepcopy(cur_path_list)
            cur_path_list_copy = [[i] for i in cur_path_list_copy]
            cur_subgraph_dict = plan_get_retre_subgraph_gen_plan(cur_path_list_copy, res = False)
            for k, g in cur_subgraph_dict.items():
                subgraph_path_list.append(g)
    
    answer = []
    for cur_new_path_list in subgraph_path_list:
        if len(cur_new_path_list) == 1:
            answer.append(cur_new_path_list)
        else:
            answer += split_triples(cur_new_path_list)
    #print(answer)
    return answer

def group_paths_by_start(paths):
    path_dict = defaultdict(list)
    for path in paths:
        if path:
            start_node = path[0][0]  # 第一个三元组的主语
            path_dict[start_node].append(path)
    return dict(path_dict)

def reorder_triples(triples, start_entity):
    from collections import defaultdict, deque

    # 建立邻接表和逆邻接表
    adj = defaultdict(list)
    rev_adj = defaultdict(list)
    edge_map = {}

    for s, p, o in triples:
        adj[s].append((o, p, False))  # False 表示正常方向
        rev_adj[o].append((s, p, True))  # True 表示逆向
        edge_map[(s, o)] = p
        edge_map[(o, s)] = roll_rale(p)  # 预先标注逆向谓词

    visited = set()
    path = []

    queue = deque()
    queue.append(start_entity)
    visited.add(start_entity)

    while queue:
        current = queue.popleft()

        # 先查正常方向
        for neighbor, pred, is_rev in adj[current]:
            if neighbor not in visited:
                path.append([current, pred, neighbor])
                visited.add(neighbor)
                queue.append(neighbor)

        # 再查逆向
        for neighbor, pred, is_rev in rev_adj[current]:
            if neighbor not in visited:
                path.append([current, roll_rale(pred), neighbor])
                visited.add(neighbor)
                queue.append(neighbor)

    return path

def triples_to_path(triples):
    # 建立从subject到(p, object)的映射
    next_map = {s: (p, o) for s, p, o in triples}
    
    # 第一个三元组的第一个实体就是起点
    current = triples[0][0]
    
    path_parts = [current]
    while current in next_map:
        p, o = next_map[current]
        path_parts.append(f"{p}->{o}")
        current = o
    
    return "-".join(path_parts)

def modify_triples(triples):
    ans = []
    for tri in triples:
        s, r, o = tri
        s = s.replace('ns:', '')
        s = s.replace('-', '_')
        o = o.replace('ns:', '')
        o = o.replace('-', '_')
        r_list = r.split('.')
        r = '.'.join(r_list[-2:])
        r = r.replace('-', '_')
        ans.append([s, r, o])
    return ans

def roll_rale(relation):
    relation_st = relation.split('.')
    relation = '.'.join(relation_st[::-1])
    return relation

class get_retrieve_plan():
    def __init__(self, data):
        self.data = data
        self.sparql = data["sparql"]

    def get_main_plan(self):
        if 'UNION' in self.sparql or 'GROUP' in self.sparql or 'COUNT' in self.sparql:
            return False
        self.cleaned_sparql = com_sparql(remove_complex_filter_blocks(self.sparql))
        self.entity_list = extract_entities_from_sparql(self.cleaned_sparql)
        self.triples = sparql2kg((self.cleaned_sparql).replace('^^xsd:dateTime', ''))
        if has_duplicate_triples(self.triples):
            return False
        self.dis_ent = get_distinct_entity(self.cleaned_sparql)
        for e in self.entity_list:
            self.path2x = find_all_paths(self.triples, e, self.dis_ent)
            if self.path2x and len(self.path2x) >= 2:
                #print(cleaned_sparql)
                #raise ValueError("数量异常！")
                return False
            if not self.path2x:
                continue
            else:
                self.path2x = [list(p) for p in self.path2x[0]]
                triples_tem = copy.deepcopy(self.triples)
                for path in self.path2x:
                    triples_tem.remove(path)
                x2end = find_paths_from_node(self.dis_ent, triples_tem)
                longest_x2end = find_unique_longest_list(x2end)
                if longest_x2end != []:
                    self.main_plan = self.path2x + longest_x2end
                else:
                    flatten_path1 = np.array(x2end[0]).flatten().tolist()
                    if 'm.01xryvm' in flatten_path1 or 'm.01mp' in flatten_path1 or 'm.01y2hnl' in flatten_path1 or 'm.081pw' in flatten_path1 or 'm.05zppz' in flatten_path1:
                        self.main_plan = self.path2x + x2end[1]
                    elif 'ns:m.01xryvm' in flatten_path1 or 'ns:m.01mp' in flatten_path1 or 'ns:m.01y2hnl' in flatten_path1 or 'ns:m.081pw' in flatten_path1 or 'ns:m.05zppz' in flatten_path1:
                        self.main_plan = self.path2x + x2end[1]
                    else:
                        self.main_plan = self.path2x + x2end[0]
                self.main_plan_start_ent = {e:self.main_plan}
                break
        return self.main_plan

    def get_other_plan(self):
        att = self.get_main_plan()
        if att == False:
            return False
        triples_tem = copy.deepcopy(self.triples)
        for path in self.main_plan:
            triples_tem.remove(list(path))
        #print(triples_tem)
        self.other_plan = find_all_longest_paths_new(triples_tem)
        #print(self.other_plan)
        #self.other_plan = [list(tri) for tri in p for p in self.other_plan]
        if self.other_plan != []:
            tolist = []
            for path in self.other_plan:
                plist = []
                for tri in path:
                    plist.append(list(tri))
                tolist.append(plist)
            self.other_plan = tolist
        self.other_plan_start_ent = group_paths_by_start(self.other_plan)
        return self.other_plan

    def get_limit_state(self):
        premove_spar = remove_invalid_filter(self.cleaned_sparql)
        filters, orders = get_limit(premove_spar)
        self.limit_state = filters + orders
        return filters, orders

    def get_raw_whole_plan(self):
        att = self.get_main_plan()
        if att == False:
            return False
        #print(self.main_plan)
        self.get_other_plan()
        #print(self.other_plan)
        self.get_limit_state()
        #print(self.limit_state)
        return [self.main_plan] + self.other_plan + self.limit_state + [self.dis_ent]

    def get_final_plan(self):
        att = self.get_raw_whole_plan()
        if att == False:
            return False
        e = list(self.main_plan_start_ent.keys())[0]
        e = e.replace('ns:', '')
        new_main_plan = modify_triples(self.main_plan)
        new_main_plan = reorder_triples(new_main_plan, e)
        #print(new_main_plan)
        new_main_plan = triples_to_path(reorder_triples(new_main_plan, e))
        self.final_main_plan = new_main_plan
        new_other_plan = []
        for path in self.other_plan:
            new_other_plan.append(triples_to_path(modify_triples(path)))
        self.final_other_plan = '\n'.join(new_other_plan)
        #print(self.final_other_plan)
        final_plan_list = []
        final_plan_list.append(self.final_main_plan)
        if self.final_other_plan != '':
            final_plan_list.append(self.final_other_plan)
        if self.limit_state != []:
            final_plan_list.append('\n'.join(self.limit_state))
        #print(final_plan_list)
        self.final_plan = '\n'.join(['\n'.join(final_plan_list), self.dis_ent])
        return self.final_plan

    def get_final_plan_name(self):
        att = self.get_raw_whole_plan()
        if att == False:
            return False
        if len(self.data["topic_ent"]) != len(self.data["topic_ent_name"]):
            return False
        e = list(self.main_plan_start_ent.keys())[0]
        e = e.replace('ns:', '')
        new_main_plan = modify_triples(self.main_plan)
        new_main_plan = reorder_triples(new_main_plan, e)
        #print(new_main_plan)
        new_main_plan = triples_to_path(reorder_triples(new_main_plan, e))
        self.final_main_plan = new_main_plan
        new_other_plan = []
        for path in self.other_plan:
            new_other_plan.append(triples_to_path(modify_triples(path)))
        self.final_other_plan = '\n'.join(new_other_plan)
        #print(self.final_other_plan)
        final_plan_list = []
        final_plan_list.append(self.final_main_plan)
        if self.final_other_plan != '':
            final_plan_list.append(self.final_other_plan)
        if self.limit_state != []:
            final_plan_list.append('\n'.join(self.limit_state))
        #print(final_plan_list)
        self.final_plan = '\n'.join(['\n'.join(final_plan_list), self.dis_ent])
        self.final_plan_name = self.final_plan
        for e in self.data["topic_ent"]:
            self.final_plan_name = self.final_plan_name.replace(e, self.data["topic_ent_name"][e])
        return self.final_plan_name

    def get_sparql_kg_list(self):
        plan = self.get_main_plan()
        if plan == False:
            return False
        return self.triples

    def get_ent_list(self):
        plan = self.get_main_plan()
        if plan == False:
            return False
        return self.entity_list
