from get_retr_plan import *
import re
import networkx as nx
from get_sim_rels_list import *
from typing import List
from collections import deque
import copy
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import unicodedata
from check_modify_plan import *

PROMPT_3 = r"""
You are a knowledge graph question-answering assistant. Now you need to reason based on the given question, topic entity/entities, and the n-hop relationships around the topic entity/entities, together with the existing retrieval plan and its execution result, to either output the final answer or propose a revised retrieval plan.

The retrieval plan must consist of three parts:
1. One or more retrieval paths (at least one is required). Each path is structured as:
entity-relation->intermediate entity-relation->...->tail entity
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
Retrieval Plan: {}
Execution Result: {}
please reason out.

Output your reasoning process wrapped with <think></think> tags, and if you propose a new retrieval plan, wrap it with <plan></plan> so the overall format is <think>...</think><plan>...</plan>, or if you return the final answer, wrap the returned entity with <answer></answer> so the overall format is <think>...</think><answer>...</answer>. 
Do not include anything other than <think>...</think><plan>...</plan> or <think>...</think><answer>...</answer>.
"""

PROMPT_3_shot = r"""
You are a knowledge graph question-answering assistant. Now you need to reason based on the given question, topic entity/entities, and the n-hop relationships around the topic entity/entities, together with the existing retrieval plan and its execution result, to either output the final answer or propose a revised retrieval plan.

The retrieval plan must consist of three parts:
1. One or more retrieval paths (at least one is required). Each path is structured as:
entity-relation->intermediate entity-relation->...->tail entity
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
Retrieval Plan: {}
Execution Result: {}
please reason out.

Output your reasoning process wrapped with <think></think> tags, and if you propose a new retrieval plan, wrap it with <plan></plan> so the overall format is <think>...</think><plan>...</plan>, or if you return the final answer, wrap the returned entity with <answer></answer> so the overall format is <think>...</think><answer>...</answer>. 
Do not include anything other than <think>...</think><plan>...</plan> or <think>...</think><answer>...</answer>.
"""

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

def group_entities_by_connected_subgraph(triples, entities):
    """
    参数：
    - triples: List[List[str, str, str]] 形式的三元组
    - entities: List[str] 要分组的实体

    返回：
    - List[List[str]]，每个子列表是同一个连通子图的实体
    """
    # 构造有向图
    if isinstance(triples[0], str):
        triples = [triples]
    G = nx.MultiDiGraph()
    for s, p, o in triples:
        G.add_edge(s, o, relation=p)
    
    # 转换为无向图，寻找连通分量
    UG = G.to_undirected()
    connected_components = list(nx.connected_components(UG))
    
    # 将实体分配到各个连通分量
    result = []
    for component in connected_components:
        # 找出在这个子图中的实体
        group = [entity for entity in entities if entity in component]
        if group:
            result.append(group)
    
    return result

def longest_undirected_path_from_entity(triples, entities):
    """
    对每个实体，计算在其所属的无向连通子图中，从它出发能达到的最长路径长度。

    参数:
        triples: List of triples (s, p, o)
        entities: List of entities to compute from

    返回:
        Dict[str, int]: 实体 => 最长路径长度（按边数）
    """
    if isinstance(triples[0], str):
        triples = [triples]
    # 构建无向图（忽略谓词方向）
    G = nx.Graph()
    for s, p, o in triples:
        G.add_edge(s, o)

    result = {}

    for entity in entities:
        if entity not in G:
            result[entity] = 0
            continue

        # BFS 计算从该实体出发的最远距离（最长路径）
        visited = set()
        queue = deque([(entity, 0)])
        max_dist = 0

        while queue:
            node, dist = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            max_dist = max(max_dist, dist)
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))

        result[entity] = max_dist

    return result

def get_hop(G, startnode, n):
    """
    获取以 startnode 为中心的 n-hop 子图。

    参数:
        G: networkx.DiGraph, 有向图
        startnode: 起始节点
        n: 跳数，表示搜索的范围

    返回:
        list: 包含 [a, relation, b] 的列表，表示 n-hop 子图
    """
    result = []
    visited = set()  # 记录已访问的节点，防止重复搜索
    current_nodes = [startnode]  # 当前跳数的节点集
    
    if isinstance(G, nx.MultiDiGraph):
        for _ in range(n):
            next_nodes = []
            for node in current_nodes:
                #if node in visited:
                #    continue
                #visited.add(node)

                # 搜索出度边
                for succ in G.successors(node):
                    if str(node)+'-'+str(succ) in visited:
                        continue
                    visited.add(str(node)+'-'+str(succ))
                    edge_data = G.get_edge_data(node, succ, default={})
                    for idx, value in edge_data.items():
                        if "relation" in value:
                            result.append([node, value["relation"], succ])
                            next_nodes.append(succ)

                # 搜索入度边
                for pred in G.predecessors(node):
                    if str(pred)+'-'+str(node) in visited:
                        continue
                    visited.add(str(pred)+'-'+str(node))
                    edge_data = G.get_edge_data(pred, node, default={})
                    for idx, value in edge_data.items():
                        if "relation" in value:
                            result.append([pred, value["relation"], node])
                            next_nodes.append(pred)

            current_nodes = list(set(next_nodes))  # 更新为下一层的节点集                  
    else:
        for _ in range(n):
            next_nodes = []
            for node in current_nodes:
                if node in visited:
                    continue
                visited.add(node)

                # 搜索出度边
                for succ in G.successors(node):
                    edge_data = G.get_edge_data(node, succ, default={})
                    if "relation" in edge_data:
                        result.append([node, edge_data["relation"], succ])
                        next_nodes.append(succ)

                # 搜索入度边
                for pred in G.predecessors(node):
                    edge_data = G.get_edge_data(pred, node, default={})
                    if "relation" in edge_data:
                        result.append([pred, edge_data["relation"], node])
                        next_nodes.append(pred)

            current_nodes = next_nodes  # 更新为下一层的节点集         

    return result

def get_hop_rel_tree(G, startnode, n):
    """
    构建n-hop BFS无环树，同时返回每一跳（每一层）的关系列表。

    参数:
        G: networkx.DiGraph 或 MultiDiGraph
        startnode: 起始节点
        n: 最大跳数
    返回:
        tree: 构建的BFS树（DiGraph）
        hop_relations: dict，key为跳数（1~n），value为该跳数所有的relation列表
    """
    #tree = nx.DiGraph()
    queue = [(startnode, 0, [startnode])]
    result = dict()
    for i in range(n):
        result[i] = []

    while queue:
        current_node, depth, path = queue.pop(0)
        if depth == n:
            continue

        for succ in G.successors(current_node):
            if succ in path:
               continue  # 跳过环

            if isinstance(G, nx.MultiDiGraph):
                edge_data = G.get_edge_data(current_node, succ, default={})
                for idx, value in edge_data.items():
                    if "relation" in value:
                        result[depth].append(value["relation"])
                                  
            else:
                edge_data = G.get_edge_data(current_node, succ, default={})
                if "relation" in edge_data:
                    result[depth].append(value["relation"])
            #print((succ, depth + 1, path + [succ]))
            queue.append((succ, depth + 1, path + [succ]))
            
    for k, v in result.items():
        result[k] = list(set(v))

    return result

def intersection_of_lists(list_of_lists):
    """
    输入: list_of_lists 是一个包含若干列表的列表
    输出: 所有列表的元素交集组成的集合
    """
    if not list_of_lists:
        return set()
    
    # 初始化交集为第一个列表的集合
    result = set(list_of_lists[0])
    
    # 依次对后面的列表取交集
    for lst in list_of_lists[1:]:
        result &= set(lst)
    
    return result

def get_recall_subgraph_list_trainset(graph, entity_dic_list):
    ans = set()
    ret = []
    for e_dict in entity_dic_list:
        raw_graph_dict = dict()
        for e, hop in e_dict.items():
            hop_g = get_hop(graph, e, hop)
            raw_graph_dict[e] = hop_g
        tri_list = []
        for k,v in raw_graph_dict.items():
            cur_list = []
            for tri in v:
                cur_list.append('￥￥'.join(tri))
            tri_list.append(cur_list)
        intersec = list(intersection_of_lists(tri_list))
        if intersec == []:
            raise Exception("NOT")
        intersec = list(set(intersec))
        for tri in intersec:
            ans.add(tri)
    for tri in ans:
        ret.append(tri.split('￥￥'))
    return ret

def mid_name(mid, topicent_dic, ansent_dic, mid2name):
    mid1 = topicent_dic.get(mid, mid)
    if mid1 != mid and mid1 != None:
        return mid1
    else:
        mid2 = ansent_dic.get(mid1, mid1)
    if mid2 != mid and mid2 != None:
        return mid2
    else:
        mid3 = mid2name.get(mid2, mid2)
        if mid3 != None:
            return mid3
        else :
            return mid

def get_recall_subgraph_trainset(topicent, topicent_dic, subgraph, ansent_dic, kg_list, mid2name, res=False):
    graph = create_graph_all(subgraph, multi = True)
    if isinstance(topicent, str):
        entities = [topicent]
    else:
        entities = topicent
    ent_group = group_entities_by_connected_subgraph(kg_list, entities)
    ent_hop = longest_undirected_path_from_entity(kg_list, entities)
    #print(kg_list, entities)
    entity_dic_list = []
    for ent_g in ent_group:
        cur_group = dict()
        for e in ent_g:
            cur_group[e] = ent_hop[e]
        entity_dic_list.append(cur_group)
    for idx, ent_dic in enumerate(entity_dic_list):
        cur_hop = []
        for k,h in ent_dic.items():
            cur_hop.append(h)
        cur_max = max(cur_hop)
        for k, h in ent_dic.items():
            ent_dic[k] = cur_max
        entity_dic_list[idx] = ent_dic
    #print(entity_dic_list)
    intersec_graph = get_recall_subgraph_list_trainset(graph, entity_dic_list)
    #print(len(intersec_graph))
    del graph
    ans = []
    for tri in intersec_graph:
        #print(tri)
        s, r, o = tri
        if s[0:3] == "ns:":
            s = s.replace("ns:", '')
            #s = s.replace("-", '_')
        s = mid_name(s, topicent_dic, ansent_dic, mid2name)
        s = s.replace('-', '_')
        r = '.'.join(r.split('.')[-2:])
        r = r.replace("-", '_')
        if o[0:3] == "ns:":
            o = o.replace("ns:", '')
            #o = o.replace("-", '_')
        #print(o)
        o = mid_name(o, topicent_dic, ansent_dic, mid2name)
        #print(o)
        o = o.replace('-', '_')
        ans.append([s, r, o])
        if res == True:
            r_res = r = '.'.join(r.split('.')[::-1])
            ans.append([o, r_res, s])
    #print(ans)
    graph = create_graph_all(ans, multi = True)
    return graph

def top_k_similar_candidates(model, cand_list: List[str], infer_list: List[str], k: int) -> List[str]:
    """
    返回与参考列表中元素相似度最高的K个候选元素。

    :param model: 一个嵌入模型，需具有encode()方法
    :param cand_list: 候选字符串列表
    :param infer_list: 参考字符串列表
    :param k: 返回相似度最高的前K个候选项
    :return: 长度为K的字符串列表，按相似度从高到低排序
    """

    # 获取嵌入
    if cand_list == [] or infer_list == []:
        return []
    cand_embs = model.encode(cand_list)  # shape: (len(cand_list), dim)
    infer_embs = model.encode(infer_list)  # shape: (len(infer_list), dim)

    # 计算候选项与所有参考项之间的相似度矩阵
    sim_matrix = cosine_similarity(cand_embs, infer_embs)  # shape: (len(cand_list), len(infer_list))

    # 对于每个候选项，取与所有参考项相似度的最大值
    max_similarities = sim_matrix.max(axis=1)  # shape: (len(cand_list),)

    # 将相似度与对应候选项打包并排序
    sorted_candidates = sorted(zip(cand_list, max_similarities), key=lambda x: x[1], reverse=True)

    # 取前k个
    top_k = [item[0] for item in sorted_candidates[:k]]
    return top_k

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

def is_fully_connected(triples: List[List[str]]) -> bool:
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

def plan_get_retre_subgraph(path_str_list, res = False):
    path_list = copy.deepcopy(path_str_list)
    for idx, p_str in enumerate(path_list):
        cur_tri_list = []
        for i in range(0, len(p_str)-2, 2):
            cur_tri_list.append(p_str[i:i+3])
        path_list[idx] = cur_tri_list
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
                for k, v in ans_dict.items():
                    all_tris = v['tris'] + path
                    if is_fully_connected(all_tris):
                        v['tris'] = all_tris
                        v['indexs'].append(idx)
                        ans_dict[k] = v
                        break
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
                        if is_fully_connected(all_tris):
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
            ans[k].append(path_str_list[idx])
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

def complete_date_string(date_str: str) -> str:
    """
    补全日期字符串，使其符合格式：YYYY-MM-DD-HH:MM
    规则：
        - 年份永远存在
        - 缺失月份 -> 01
        - 缺失日期 -> 01
        - 缺失小时和分钟 -> 08:00
    参数:
        date_str: 输入的日期字符串
    返回:
        补全后的日期字符串
    """
    # 初始化默认值
    year = None
    month = "01"
    day = "01"
    hour = "08"
    minute = "00"

    # 先拆分日期和时间
    if "-" in date_str:
        parts = date_str.split("-")
        # 第一个一定是年份
        year = parts[0]
        
        if len(parts) >= 2:
            # 判断是否是月份还是带时间的
            if ":" in parts[1]:  # 说明第二部分是时间，不是月份
                pass
            else:
                month = parts[1].zfill(2)
        
        if len(parts) >= 3:
            if ":" in parts[2]:  # 第三部分是时间
                pass
            else:
                day = parts[2].zfill(2)
        
        # 判断最后部分是否有时间
        if ":" in date_str:
            time_part = date_str.split(":")
            hour = time_part[0].split("-")[-1].zfill(2)
            minute = time_part[1].zfill(2)

    else:
        # 没有"-"，只有年份
        year = date_str

    # 组装
    return f"{year}-{month}-{day}-{hour}:{minute}"

def compare_dates_with_operator(a: str, b: str, op: str) -> bool:
    """
    比较两个完整的日期字符串（格式：YYYY-MM-DD-HH:MM）并根据比较符返回布尔值
    参数:
        a: 日期字符串 a
        b: 日期字符串 b
        op: 比较符（>, <, >=, <=, =）
    返回:
        True 或 False
    """
    def normalize_date_string(date_str: str) -> str:
        """
        如果年份不足4位，左侧补0
        输入格式：可能是 YYY-MM-DD-HH:MM
        输出格式：YYYY-MM-DD-HH:MM
        """
        return re.sub(r'^(\d{1,3})-', lambda m: m.group(1).zfill(4) + '-', date_str)

    
    fmt = "%Y-%m-%d-%H:%M"
    a = normalize_date_string(a)
    b = normalize_date_string(b)
    date_a = datetime.strptime(a, fmt)
    date_b = datetime.strptime(b, fmt)
    
    if op == ">":
        return date_a > date_b
    elif op == "<":
        return date_a < date_b
    elif op == ">=":
        return date_a >= date_b
    elif op == "<=":
        return date_a <= date_b
    elif op == "=":
        return date_a == date_b
    else:
        raise ValueError(f"不支持的比较符: {op}")
    
def numeric_disting(com_value, comed_value, op):
    # 去除前后的_符号后，若两个变量中均未有_，则作为数值比较；否则两者补全为日期字符串，再比较。
    com_value_stripped = com_value.strip('_')
    comed_value_stripped = comed_value.strip('_')

    def count_decimal_places(sci_str: str) -> int:
        """
        判断科学计数法字符串中 e 之前有几位小数
        """
        # 拿到 e 前面的部分
        before_e = sci_str.split('e')[0]
        
        # 如果包含小数点，计算小数点后的长度
        if '.' in before_e:
            return len(before_e.split('.')[1])
        else:
            return 0
    
    if "_" not in com_value_stripped and "_" not in comed_value_stripped:
        # 数值比较
        
        try:
            if 'e+' in comed_value_stripped or 'e-' in comed_value_stripped:
                if 'e+' in com_value_stripped or 'e-' in com_value_stripped:
                    num1 = float(com_value_stripped)
                    num2 = float(comed_value_stripped)
                else:
                    e_num = count_decimal_places(comed_value)           
                    e_tem = ".{}e"
                    num1 = float(format(float(com_value_stripped), e_tem.format(e_num)))
                    num2 = float(comed_value_stripped)
                    #print(com_value_stripped, e_num)
                    #print(com_value_stripped, comed_value_stripped)
            else:
                if '.' in comed_value_stripped:
                    decimal_places = len(comed_value_stripped.split('.')[1])
                    format_str = f".{decimal_places}f"
                    num1 = float(format(float(com_value_stripped), format_str))
                else:
                    num1 = float(com_value_stripped)
                num2 = float(comed_value_stripped)
        except Exception as e:
            #print(e)
            return com_value_stripped == comed_value_stripped or com_value_stripped in comed_value_stripped or comed_value_stripped in com_value_stripped

        if op == ">":
            return num1 > num2
        elif op == "<":
            return num1 < num2
        elif op == ">=":
            return num1 >= num2
        elif op == "<=":
            return num1 <= num2
        elif op == "=":
            return num1 == num2
        else:
            raise ValueError(f"不支持的比较符: {op}")
    else:
        # 日期比较
        try:
            com_value = com_value.strip('_').replace('_', '-')
            comed_value = comed_value.strip('_').replace('_', '-')
            if 'T' in com_value:
                com_value = com_value.split('T')[0]
            if 'T' in comed_value:
                comed_value = com_value.split('T')[0]
            a_completed = complete_date_string(com_value)
            b_completed = complete_date_string(comed_value)
            # print(a_completed, b_completed)
            res = compare_dates_with_operator(a_completed, b_completed, op)
            # print(res)
        except Exception as e:
            return com_value_stripped == comed_value_stripped or com_value_stripped in comed_value_stripped or comed_value_stripped in com_value_stripped
        return res
    
def no_alpha_letters(s: str) -> bool:
    """
    如果字符串中没有字母(任何语言)且没有符号/emoji，则返回 True，否则 False。
    """
    for ch in s:
        cat = unicodedata.category(ch)
        if ch.isalpha() or cat.startswith('S'):  # S = Symbol (includes emoji, currency, math symbols, etc.)
            return False
    return True

class Minitreenode:
    def __init__(self, name):
        self.name = name
        self.children = []        # 子节点列表
        self.parent = []        # 父节点列表

class Temtree:
    def __init__(self):
        self.root = None
        self.name_map = {}
        self.edges = []

    def add_node(self, name):
        if name in self.name_map:
            #print(f"节点 {name} 已存在，跳过添加。")
            return
        self.name_map[name] = Minitreenode(name)

    def add_edge(self, parent_name, relation, child_name):
        if parent_name not in self.name_map or child_name not in self.name_map:
            raise ValueError(f"父节点 {parent_name} 或子节点 {child_name} 不存在，请先添加节点。")
        parent = self.name_map[parent_name]
        child = self.name_map[child_name]
        parent.children.append(child)
        child.parent.append(parent)
        self.edges.append([parent_name, relation, child_name])


    def build_tree(self, nodes, edges, root_name):
        """
        nodes: List of (name, label)
        edges: List of (parent_name, relation, child_name)  # relation目前不使用但保留
        root_name: 树根节点名称
        """
        # 创建所有节点
        for name in nodes:
            self.name_map[name] = Minitreenode(name)
        
        # 建立树关系
        for parent_name, relation, child_name in edges:
            parent = self.name_map[parent_name]
            child = self.name_map[child_name]
            parent.children.append(child)
            child.parent.append(parent)  # 设定父指针
            self.edges.append([parent_name, relation, child_name])          
        
        self.root = self.name_map.get(root_name)    

    def get_branch_map(self):
        if len(self.name_map) == 0:
            raise ValueError('no node!')
        branch_map = dict()
        def dfs_down_get_branch(node, last_branch):
            #print(node.name)
            branch_map[node.name] = last_branch
            if len(node.children) == 1:
                dfs_down_get_branch(node.children[0], last_branch)
            elif len(node.children) > 1:
                for n in node.children:
                    dfs_down_get_branch(n, node.name)
            else:
                return
        dfs_down_get_branch(self.root, self.root.name)
        for k,v in branch_map.items():
            if v == self.root.name:
                branch_map[k] = k
        return branch_map

    def get_child_to_node(self, node_name, tar_node_name):
        node = self.name_map.get(node_name)
        tar_node = self.name_map.get(tar_node_name)
        ans = set()
        def dfs_up_get_name(node, tar_node):
            par_namelist = [i.name for i in node.parent]
            if tar_node.name in par_namelist:
                ans.add(node.name)
                return
            else:
                for par_node in node.parent:
                    dfs_up_get_name(par_node, tar_node)
        dfs_up_get_name(node, tar_node)
        return list(ans)[0]
    
class TreeNode:
    def __init__(self, name):
        self.name = name          # 节点名+类别标识
        self.children = []        # 子节点列表
        self.parent = []        # 父节点列表

class Tree:
    def __init__(self):
        self.root = None
        self.name_map = {}  # 节点名到节点对象
        self.edges = []     # 可选：记录边关系 (parent, child, relation)
        self.retre_template_tree = None

    def update_retre_template_tree(self, tri_list, root_name):
        # print(tri_list)
        self.retre_template_tree = Temtree()
        node_list = []
        edge_list = []
        for tri in tri_list:
            s, r, o = tri
            if s not in node_list:
                node_list.append(s)
            if o not in node_list:
                node_list.append(o)
            edge_list.append([s,r,o])
        self.retre_template_tree.build_tree(node_list, edge_list, root_name)
        
    def add_node(self, name):
        if name in self.name_map:
            #print(f"节点 {name} 已存在，跳过添加。")
            return
        self.name_map[name] = TreeNode(name)

    def add_edge(self, parent_name, relation, child_name):
        if parent_name not in self.name_map or child_name not in self.name_map:
            raise ValueError(f"父节点 {parent_name} 或子节点 {child_name} 不存在，请先添加节点。")
        parent = self.name_map[parent_name]
        child = self.name_map[child_name]
        parent.children.append(child)
        child.parent.append(parent)
        self.edges.append([parent_name, relation, child_name])

    def build_tree(self, nodes, edges, root_name):
        """
        nodes: List of (name)
        edges: List of (parent_name, relation, child_name)  # relation目前不使用但保留
        root_name: 树根节点名称
        """
        # 创建所有节点
        for name in nodes:
            self.name_map[name] = TreeNode(name)
        
        # 建立树关系
        for parent_name, relation, child_name in edges:
            parent = self.name_map[parent_name]
            child = self.name_map[child_name]
            parent.children.append(child)
            child.parent.append(parent)  # 设定父指针
            self.edges.append([parent_name, relation, child_name])      
        #print(root_name)
        self.root = self.name_map.get(root_name)

    def build_tris(self, tris, root_name):
        nodes = []
        edges = []
        node_set = set()
        for tri in tris:
            s, r, o = tri
            if s not in node_set:
                node_set.add(s)
                nodes.append(s)
            if o not in node_set:
                node_set.add(o)
                nodes.append(o)
            edges.append([s, r, o])
        self.build_tree(nodes, edges, root_name)

    def add_tris(self, tris):
        for tri in tris:
            s, r, o = tri
            if s not in self.name_map:
                self.add_node(s)
            if o not in self.name_map:
                self.add_node(o)
            self.name_map[s].children.append(self.name_map[o])
            self.name_map[o].parent.append(self.name_map[s])
            self.add_edge(s, r, o)
                
    def summarize_labels(self):
        """
        返回字典，键为 label，值为该 label 下的所有节点名列表
        """
        from collections import defaultdict
        summary = defaultdict(list)
        for node in self.name_map.values():
            summary[node.name.split('$￥')[1]].append(node.name.split('$￥')[0])
        return dict(summary)
    
    def filter_tree_dfs_up_label(self, node_names, stop_label):
        def dfs_up_label(node):
            if node.name.split('$￥')[1] == stop_label:
                stop_nodes.add(node.name)
                return
            else:
                keep_nodes.add(node.name)
            for parent in node.parent:  # 多个父节点
                dfs_up_label(parent)
        keep_nodes = set()
        stop_nodes = set()
        for node_name in node_names:
            node = self.name_map.get(node_name)
            if node:
                dfs_up_label(node)
        begin_label = node_names[0].split('$￥')[1]
        childlabel_to_label = self.retre_template_tree.get_child_to_node(begin_label, stop_label)
        return list(keep_nodes), list(stop_nodes), childlabel_to_label

    def filter_tree_dfs_down_label(self, node_names, label):
        def dfs_down_label(node):
            if  node.name in keep_nodes:
                return
            keep_nodes.add(node.name)
            for child in node.children:
                if child.name.split('$￥')[1] != label:
                    dfs_down_label(child)
        keep_nodes = set()
        for node_name in node_names:
            node = self.name_map.get(node_name)
            dfs_down_label(node)
        return list(keep_nodes)
    
    # def get_true_retain_node(self, target_node_names):
    #     def dfs_up_label(node):
    #         if node.label == tar_label:
    #             keep_nodes.add(node.name)
    #             return
    #         for parent in node.parent:  # 多个父节点
    #             dfs_up_label(parent)
    #     keep_nodes = set()
    #     cur_label = self.name_map.get(target_node_names[0]).label
    #     tar_label = self.filter_map[cur_label]
    #     if tar_label == cur_label:
    #         return target_node_names
    #     else:
    #         for node_name in target_node_names:
    #             node = self.name_map.get(node_name)
    #             if node:
    #                 dfs_up_label(node)
    #         return list(keep_nodes)

    def filter_tree_by_node_list(self, target_node_names):
        """
        target_node_names: 当前树某一层的部分节点名
        返回新的 Tree 对象，只保留与这些节点相关路径的节点
        """
        # 收集所有要保留的节点名（包括向上到根路径和向下到叶的路径）
        #print(target_node_names)
        keep_nodes = set()
        filter_map = self.retre_template_tree.get_branch_map()
        #print(target_node_names)
        target_node_label = self.name_map.get(target_node_names[0]).name.split('$￥')[1]
        def dfs_up(node):
            if node.name in keep_nodes:
                return
            keep_nodes.add(node.name)
            for parent in node.parent:  # 多个父节点
                dfs_up(parent)
        
        def dfs_down(node):
            if node.name in keep_nodes:
                return
            keep_nodes.add(node.name)
            for child in node.children:
                dfs_down(child)

        if target_node_label == filter_map[target_node_label]:
            for name in target_node_names:
                node = self.name_map.get(name)
                if node:
                    dfs_up(node)
                    keep_nodes.remove(node.name)
                    dfs_down(node)
        else:
            cur_label = filter_map[target_node_label]
            for name in target_node_names:
                node = self.name_map.get(name)
                if node:
                    dfs_down(node)
                keep_nodes.remove(name)
            while True:
                keep_nodes_up, stop_nodes, childlabel_to_label = self.filter_tree_dfs_up_label(target_node_names, cur_label)
                keep_nodes_down = self.filter_tree_dfs_down_label(stop_nodes, childlabel_to_label)
                keep_nodes = keep_nodes | set(keep_nodes_up) | set(keep_nodes_down)
                #print(keep_nodes)
                if cur_label == filter_map[cur_label]:
                    target_node_names = stop_nodes
                    break
                else:
                    cur_label = filter_map[cur_label]
                    target_node_names = stop_nodes
            for name in target_node_names:
                node = self.name_map.get(name)
                if node:
                    keep_nodes.remove(node.name)
                    dfs_up(node)
            #print(keep_nodes)
        # 构建新树节点
        new_nodes = []
        for node in self.name_map.values():
            if node.name in keep_nodes:
                new_nodes.append(node.name)
        #print(new_nodes)
        new_edges = []
        for tri in self.edges:
            # if node.name in keep_nodes:
            #     for child in node.children:
            #         if child.name in keep_nodes:
            #             new_edges.append([node.name, "relation", child.name])  # 关系可保留
            s, r, o = tri
            if s in keep_nodes and o in keep_nodes:
                new_edges.append([s, r, o])

        # 创建新的树
        new_tree = Tree()
        new_tree.build_tree(new_nodes, new_edges, self.root.name)
        new_tree.retre_template_tree = self.retre_template_tree
        return new_tree

    def print_tree(self, node=None, indent=0):
        if node is None:
            node = self.root
        print('  ' * indent + f"{node.name}")
        for child in node.children:
            self.print_tree(child, indent + 1)

class Retrtreenode:
    def __init__(self, name):
        self.name = name
        self.children = []        # 子节点列表
        self.parent = []        # 父节点列表

class Retrtree:
    def __init__(self):
        self.root = None
        self.name_map = {}
        self.edges = []

    def add_node(self, name):
        if name in self.name_map:
            #print(f"节点 {name} 已存在，跳过添加。")
            return
        self.name_map[name] = Retrtreenode(name)

    def add_edge(self, parent_name, relation, child_name):
        if parent_name not in self.name_map or child_name not in self.name_map:
            raise ValueError(f"父节点 {parent_name} 或子节点 {child_name} 不存在，请先添加节点。")
        parent = self.name_map[parent_name]
        child = self.name_map[child_name]
        parent.children.append(child)
        child.parent.append(parent)
        self.edges.append([parent_name, relation, child_name])


    def build_tree(self, nodes, edges, root_name):
        """
        nodes: List of (name, label)
        edges: List of (parent_name, relation, child_name)  # relation目前不使用但保留
        root_name: 树根节点名称
        """
        # 创建所有节点
        for name in nodes:
            self.name_map[name] = Retrtreenode(name)
        
        # 建立树关系
        for parent_name, relation, child_name in edges:
            parent = self.name_map[parent_name]
            child = self.name_map[child_name]
            parent.children.append(child)
            child.parent.append(parent)  # 设定父指针
            self.edges.append([parent_name, relation, child_name])          

        self.root = self.name_map.get(root_name)    

    def build_tris(self, tris, root_name):
        for tri in tris:
            s, r, o = tri
            if s not in self.name_map:
                self.name_map[s] = Retrtreenode(s)
            if o not in self.name_map:
                self.name_map[o] = Retrtreenode(o)
            self.name_map[s].children.append(self.name_map[o])
            self.name_map[o].parent.append(self.name_map[s])
            self.edges.append([s, r, o])
        self.root = self.name_map.get(root_name) 

    def filter_dfs_up(self, nodes):
        keep_node = set()
        def dfs_up(node):
            if node.name in keep_node:
                return
            keep_node.add(node.name)
            for n in node.parent:
                dfs_up(n)
        
        for node in nodes:
            node = self.name_map.get(node)
            #print(node)
            if node:
                dfs_up(node)
        return list(keep_node)
    
def execute_single_path(graph, path, begin_list, model):
    name2label = set() # 带有标签的实体名
    mid_ent = dict() # 不带有标签的实体名
    result = [] # 带有标签的实体名
    wrong_list = []
    for ent_idx in range(0, len(path)-2, 2):
        cur_rel = path[ent_idx+1]
        cur_succ = path[ent_idx+2]
        #print(cur_rel)
        # cur_succ = cur_succ.replace('-', '_')
        succ_list = []
        if ent_idx == 0:
            cur_node = begin_list
            for node in cur_node:
                name2label.add(node + '$￥' + path[ent_idx])
            mid_ent[path[ent_idx]] = cur_node
        else:
            if path[ent_idx].startswith('?'):
                cur_node = mid_ent[path[ent_idx]]
            else:
                cur_node = [path[ent_idx]]

        #### 检索逻辑 
        #next_node = []
        if cur_node == []:
            return 
        for c_n in cur_node:
            c_n_succ = []
            # print(c_n)
            for succ in graph.successors(c_n):
                # print(succ)
                edge_data = graph.get_edge_data(c_n, succ, default={})
                for idx, value in edge_data.items():
                    
                    if "relation" in value and value["relation"]==cur_rel:
                        #print(value)
                        c_n_succ.append(succ)
            #print(cur_succ)
            #print(c_n_succ)
            if cur_succ.startswith('?'):
                #print(cur_succ)
                if cur_succ not in mid_ent:
                    mid_ent[cur_succ] = []
                mid_ent[cur_succ] += c_n_succ
                succ_list+=c_n_succ
                for i in c_n_succ:
                    result.append([c_n+'$￥'+path[ent_idx], cur_rel, i+'$￥'+cur_succ])
            else:
                if '"' in cur_succ:
                    for num_n_succ in c_n_succ:
                        if numeric_disting(cur_succ.strip('"'), num_n_succ, '='):
                            mid_ent[cur_succ] = [cur_succ]
                            succ_list.append(cur_succ)
                            result.append([c_n+'$￥'+path[ent_idx], cur_rel, cur_succ+'$￥'+cur_succ])
                            break
                else:
                    if cur_succ in c_n_succ:
                        mid_ent[cur_succ] = [cur_succ]
                        succ_list.append(cur_succ)
                        result.append([c_n+'$￥'+path[ent_idx], cur_rel, cur_succ+'$￥'+cur_succ])
        
        if succ_list == []:
            all_rel = set()
            for c_n in cur_node:
                for succ in graph.successors(c_n):
                    edge_data = graph.get_edge_data(c_n, succ, default={})
                    for idx, value in edge_data.items():
                        if "relation" in value:
                            all_rel.add(value["relation"])
            
            all_rel = list(all_rel)
            #print(cur_node, all_rel)
            sim_cur_rel = top_k_similar_candidates(model, all_rel, [cur_rel], 1)
            if sim_cur_rel == []:
                wrong_list.append("Exceeding the hop count range. Unable to retrieve entity {}.".format(cur_succ))
                break
            else:
                sim_cur_rel = sim_cur_rel[0]
            #print(sim_cur_rel)
            for c_n in cur_node:
                
                c_n_succ = []
                for succ in graph.successors(c_n):
                    
                    edge_data = graph.get_edge_data(c_n, succ, default={})
                    
                    for idx, value in edge_data.items():
                        #print(sim_cur_rel)
                        if "relation" in value and value["relation"] == sim_cur_rel:
                            #print(1)
                            c_n_succ.append(succ)
    
                if cur_succ.startswith('?'):
                    if cur_succ not in mid_ent:
                        mid_ent[cur_succ] = []
                    mid_ent[cur_succ] += c_n_succ
                    succ_list+=c_n_succ
                    for i in c_n_succ:
                        result.append([c_n+'$￥'+path[ent_idx], sim_cur_rel, i+'$￥'+cur_succ])
                else:
                    if '"' in cur_succ:
                        for num_n_succ in c_n_succ:
                            if numeric_disting(cur_succ.strip('"'), num_n_succ, '='):
                                mid_ent[cur_succ] = [cur_succ]
                                succ_list.append(cur_succ)
                                result.append([c_n+'$￥'+path[ent_idx], cur_rel, cur_succ+'$￥'+cur_succ])
                                break
                    else:
                        if cur_succ in c_n_succ:
                            mid_ent[cur_succ] = [cur_succ]
                            succ_list.append(cur_succ)
                            result.append([c_n+'$￥'+path[ent_idx], sim_cur_rel, cur_succ+'$￥'+cur_succ])
                # print(c_n_succ)
            if succ_list == []:
                wrong_list.append("Entity '{}' was not retrieved.".format(cur_succ))
                break
        # print(succ_list)
        mid_ent[cur_succ] = list(set(mid_ent[cur_succ]))
        succ_list = list(set(succ_list))
        for node in succ_list:
            #if node in name2label:
                #node = node+'$￥'
                #raise ValueError('have cycle!')
            name2label.add(node+'$￥'+cur_succ)
    if len(begin_list) > 1:
        for node in begin_list:
            result.append(['start', 'test', node+'$￥'+path[0]])
        retr_tree = Retrtree()
        retr_tree.build_tris(result, 'start')
    else:
        retr_tree = Retrtree()
        retr_tree.build_tris(result, begin_list[0]+'$￥'+path[0])
    #print(wrong_list)
    if wrong_list != []:
        end_nodes = mid_ent[path[ent_idx]]
    else:
        end_nodes = mid_ent[path[-1]]
    if wrong_list != []:
        end_nodes_w_label = [i+'$￥'+path[ent_idx] for i in end_nodes]
    else:
        end_nodes_w_label = [i+'$￥'+path[-1] for i in end_nodes]
    
    keep_node = retr_tree.filter_dfs_up(end_nodes_w_label) # keep_node是带有标签的实体名的字典
    #print(keep_node)
    ans = []
    begin_af_retr = []
    for tri in result:
        s, r, o = tri
        if s == 'start':
            continue
        if s in keep_node and o in keep_node:
            ans.append([s, r, o])
            s = s.split('$￥')[0]
            if s in begin_list and s not in begin_af_retr:
                begin_af_retr.append(s)
            o = o.split('$￥')[0]
            if o in begin_list and o not in begin_af_retr:
                begin_af_retr.append(o)
    if begin_af_retr == []:
        begin_af_retr = begin_list
    return ans, begin_af_retr, wrong_list

def execute_subgraph_paths(graph, paths_list, model):
    tree = Tree()
    wrong_list = [[] for i in range(len(paths_list))]
    graph_root = paths_list[0][0]
    for p_id, p in enumerate(paths_list):
        if p_id == 0:
            if p[0].startswith('?'):
                wrong_list[p_id].append('Head node of the retrieval path {} is unknown!'.format(p))
                break
            if p[0] not in graph:
                wrong_list[p_id].append('Head node of the retrieval path {} is not in KG!'.format(p))
                break
            begin_list = [p[0]]
        else:
            mid_dict = tree.summarize_labels() # 实体名在字典中是不带标签的
            if p[0].startswith('?'):
                if p[0] not in mid_dict:
                    wrong_list[p_id].append("Entity '{}' was not retrieved.".format(p[0]))
                    continue
                begin_list = mid_dict[p[0]]
            else:
                begin_list = [p[0]]
        p_retr_ans, begin_list_afretr, cur_wrong_list = execute_single_path(graph, p, begin_list, model) # ans中的实体带标签， begin_list_afretr不带
        wrong_list[p_id] += cur_wrong_list
        if p_id == 0:
            #print(p[0])
            tree.build_tris(p_retr_ans, p[0]+'$￥'+p[0])
            #print(str_path_2_tri_path([p]), p[0], p)
            tree.update_retre_template_tree(str_path_2_tri_path([p]), p[0])
            #print(tree.retre_template_tree.root.name)
        else:
            tree.add_tris(p_retr_ans)
            tree.update_retre_template_tree(str_path_2_tri_path(paths_list[:p_id+1]), graph_root)
            if len(begin_list_afretr) != len(begin_list):
                begin_list_afretr = [i+'$￥'+p[0] for i in begin_list_afretr]
                #print(begin_list_afretr)
                tree = tree.filter_tree_by_node_list(begin_list_afretr)
    return tree, wrong_list

def str_path_2_tri_path(str_path_list):
    ans = []
    for path in str_path_list:
        for i in range(0, len(path)-2, 2):
            ans.append([path[i], path[i+1], path[i+2]])
    return ans

def execute_retr_paths_single(raw_data_with_graph, data_stage2, paths, data_name, mid2name, emb_model):
    ans = []
    raw_idx2hop_rel = dict()
    if 'simpleques' in data_name:
        recall_graph = get_recall_subgraph_trainset(data_stage2['topic_ent'], data_stage2['topic_ent_name'], raw_data_with_graph['subgraph'], data_stage2['answer_ent_name'], data_stage2['kg'], mid2name)
    else:
        kg_list = get_retrieve_plan(data_stage2).get_sparql_kg_list()
        for k_id, tri in enumerate(kg_list):
            s, r, o = tri
            if s[0:3] == 'ns:':
                s = s.replace('ns:', '')
            if o[0:3] == 'ns:':
                o = o.replace('ns:', '')
            if r[0:3] == 'ns:':
                r = r.replace('ns:', '')
            kg_list[k_id] = [s, r, o]
        tri_num = len(raw_data_with_graph['subgraph'])
        if tri_num > 6000:
            recall_graph = get_recall_subgraph_trainset(data_stage2['topic_ent'], data_stage2['topic_ent_name'], raw_data_with_graph['subgraph'], data_stage2['answer_ent_name'], kg_list, mid2name, True)
        else:
            recall_graph = get_recall_subgraph_trainset(data_stage2['topic_ent'], data_stage2['topic_ent_name'], raw_data_with_graph['subgraph'], data_stage2['answer_ent_name'], kg_list, mid2name, True)
    retr_result = execute_subgraph_paths(recall_graph, paths, emb_model)
    return retr_result

def execute_golden_test(data_subgraph, data_stage, data_name, model):
    data = list(jsonlines.open(data_stage))
    for idx, dd in tqdm(enumerate(data), total = len(data)):
        if idx == 2228:
            continue
        if data_name == 'webqsp' or data_name == 'webqsp_w_cycle' or 'webqsp' in data_name:
            d = dd['parses']['0']
        else:
            d = dd
        if 'plan' not in d:
            try:
                plan = get_retrieve_plan(d).get_final_plan_name()
            except Exception as e:
                continue
            if plan == False:
                continue
            else:
                d['plan'] = plan
        if data_name != 'cwq_train' and 'cwq_train' not in data_name:
            if len(d['answer_ent']) != len(d['answer_ent_name']):
                continue
            if len(d['topic_ent']) != len(d['topic_ent_name']):
                continue
        raw_plan_list = d['plan'].split('\n')
        raw_plan = []
        path_str_list = []
        return_ent = raw_plan_list[-1]
        if 'raw_idx' not in d:
            raw_idx = idx
        else:
            raw_idx = d['raw_idx']
        for p in raw_plan_list[:-1]:
            if 'FILTER' not in p and 'ORDER' not in p:
                raw_plan.append(p)
        for p in raw_plan:
            path_str_list.append(extract_entities_and_relations(p))
        path_str_list = [[i.replace('-', '_') for i in sublist] for sublist in path_str_list]
        try:
            plan_subgraph = plan_get_retre_subgraph(path_str_list)
        except Exception as e:
            raw_plan = '\n'.join(raw_plan)
            raw_plan = paths_str_tree_modify(raw_plan)
            raw_plan = raw_plan.split('\n')
            path_str_list = []
            for p in raw_plan:
                path_str_list.append(extract_entities_and_relations(p))
            path_str_list = [[i.replace('-', '_') for i in sublist] for sublist in path_str_list]
            plan_subgraph = plan_get_retre_subgraph(path_str_list)
        ret = dict()
        try:
            for k, p in plan_subgraph.items():
                ans_tree = execute_retr_paths_single(data_subgraph[raw_idx], d, p, data_name, mid2name, model)
                p_retr_dict = ans_tree.summarize_labels()
                ret.update(p_retr_dict)
            ret_ent_list = ret[return_ent]
        except Exception as e:
            if str(e) == 'have cycle!':
                with open('./{}_golden_plan_exe.txt'.format(data_name), 'a', encoding='utf-8') as file:
                    file.write(str(idx)+ ' ' +str(e)+'\n')
            else:
                with open('./{}_golden_plan_exe.txt'.format(data_name), 'a', encoding='utf-8') as file:
                    file.write(str(idx)+ ' ' +str(e)+'\n')
                continue
        if data_name == 'cwq_train' or 'cwq_train' in data_name:
            ans_ent_list_ = d['answer_ent']
            m_id_list = []
            for ans_dict in ans_ent_list_:
                m_id_list.append(ans_dict['answer_id'])
            # for m_idx, mid in enumerate(m_id_list):
            #     if d['topic_ent_name'].get(mid, mid) == mid:
            #         m_id_list[m_idx] = mid2name.get(mid,mid)
            #     else:
            #         m_id_list[m_idx] = d['topic_ent_name'].get(mid)
            for m_idx, mid in enumerate(m_id_list):
                m_id_list[m_idx] = mid_name(mid, d['topic_ent_name'], d['answer_ent_name'], mid2name)
            ans_ent_list = m_id_list
            ans_ent_list = [i.replace('-','_') for i in ans_ent_list]
        else:
            ans_ent_list = d['answer_ent']
            for a_id, a_ent in enumerate(ans_ent_list):
                if a_ent[:2] == 'm.' or a_ent[:2] == 'g.':
                    ans_ent_list[a_id] = d['answer_ent_name'][a_ent]
                else:
                    ans_ent_list[a_id] = a_ent
            ans_ent_list = [i.replace('-','_') for i in ans_ent_list]
        try:
            for ent in ans_ent_list:
                if no_alpha_letters(ent) == True:
                    flag = False
                    for ret_ent in ret_ent_list:
                        if numeric_disting(ent, ret_ent, '='):
                            flag = True
                            break
                    if flag == False:
                        raise ValueError(str(idx) + ' ' + 'Retr Wrong!', ret_ent_list, ans_ent_list)
                else:
                    if ent not in ret_ent_list:
                        raise ValueError(str(idx) + ' ' + 'Retr Wrong!', ret_ent_list, ans_ent_list)
        except Exception as e:
            with open('./{}_golden_plan_exe.txt'.format(data_name), 'a', encoding='utf-8') as file:
                if 'Retr Wrong!' in str(e):
                    file.write(str(e)+'\n')
                else:
                    to_write = str(idx) + ' ' + 'Retr Wrong!' + " " + '[' + ', '.join(ret_ent_list) + ']' + '[' + ', '.join(ans_ent_list) + ']'
                    file.write(to_write + '\n')
            continue

def parses_filter(filter_str):
    # 去掉 FILTER 和括号
    filter_str = filter_str.strip()
    filter_str = filter_str.replace('"', '')
    filter_str = filter_str.replace("'", '')
    if filter_str.startswith("FILTER"):
        filter_str = filter_str[len("FILTER"):].strip()
    if filter_str.startswith("(") and filter_str.endswith(")"):
        filter_str = filter_str[1:-1].strip()

    # 拆分 && 或 || 多个条件
    conditions = re.split(r'\s*&&\s*|\s*\|\|\s*', filter_str)
    if 'EXIST' in filter_str:
        return [[]]
    result = []
    for cond in conditions:
        cond = cond.strip()
        match = re.match(r"(.+?)\s*(=|!=|<=|>=|<|>)\s*(.+)", cond)
        if not match:
            result.append([])
            continue
        left, op, right = match.groups()
        # print(left, '\n', op, '\n', right)
        left = left.strip()
        right = right.strip()

        # 去掉引号
        def strip_quotes(s):
            return s[1:-1] if len(s) > 1 and (s[0] in ['"', "'"] and s[-1] in ['"', "'"]) else s
        left = strip_quotes(left)
        right = strip_quotes(right)

        # 处理 a - b < 0 或 0 < a - b
        def handle_subtraction(expr, op_sign, zero_side):
            parts = [p.strip() for p in expr.split('-')]
            #print(op_sign)
            if len(parts) == 2:
                a, b = parts
                # 确定运算符方向
                if zero_side == 'right':  # a - b < 0
                    real_op = op_sign 
                else:  # 0 < a - b
                    if '>' in op_sign:
                        real_op = op_sign.replace('>', '<')
                    else:
                        real_op = op_sign.replace('<', '>')
                    #real_op = '>' if op_sign in ['<', '<='] else '<'
                # 确定哪个是变量
                if a.startswith('?') and not b.startswith('?'):
                    return [a, b, real_op]
                elif b.startswith('?') and not a.startswith('?'):
                    if '>' in real_op:
                        return [b, a, real_op.replace('>', '<')]
                    else:
                        return [b, a, real_op.replace('<', '>')]
                    #return [b, a, '<' if real_op == '>' else '>']
                else:
                    return [a, b, real_op]
            else:
                a = ''
                b = ''
                for p_id, p in enumerate(parts):
                    if p.strip().startswith('?'):
                        if p_id == 0:
                            a = p.strip()
                            b = '-'.join(parts[1:])
                            break
                        elif p_id == len(parts)-1:
                            a = '-'.join(parts[:-1])
                            b = p.strip()
                            break
                if a == '':
                    return []
                else:
                    if zero_side == 'right':  # a - b < 0
                        real_op = op_sign 
                    else:  # 0 < a - b
                        if '>' in op_sign:
                            real_op = op_sign.replace('>', '<')
                        else:
                            real_op = op_sign.replace('<', '>')
                        #real_op = '>' if op_sign in ['<', '<='] else '<'
                    # 确定哪个是变量
                    if a.startswith('?') and not b.startswith('?'):
                        return [a, b, real_op]
                    elif b.startswith('?') and not a.startswith('?'):
                        if '>' in real_op:
                            return [b, a, real_op.replace('>', '<')]
                        else:
                            return [b, a, real_op.replace('<', '>')]
                        #return [b, a, '<' if real_op == '>' else '>']
                    else:
                        return [a, b, real_op]

        if '-' in left and right == '0':  # (a - b) op 0
            triple = handle_subtraction(left, op, 'right')
            if triple:
                result.append(triple)
                continue
            else:
                result.append([])
                continue
        if '-' in right and left == '0':  # 0 op (a - b)
            triple = handle_subtraction(right, op, 'left')
            if triple:
                result.append(triple)
                continue
            else:
                result.append([])
                continue

        # 普通情况
        if left.startswith('?') and not right.startswith('?'):
            result.append([left, right, op])
        elif right.startswith('?') and not left.startswith('?'):
            reverse_op = {'<': '>', '>': '<', '<=': '>=', '>=': '<=', '=': '=', '!=': '!='}
            result.append([right, left, reverse_op[op]])
        else:
            result.append([left, right, op])

    return result

def parses_order(order_str):
    order_str = order_str.strip()
    
    # 提取 LIMIT 数值
    limit_match = re.search(r'LIMIT\s+(\d+)', order_str, re.IGNORECASE)
    limit = int(limit_match.group(1)) if limit_match else None
    
    # 提取 OFFSET 数值
    offset_match = re.search(r'OFFSET\s+(\d+)', order_str, re.IGNORECASE)
    offset = int(offset_match.group(1)) if offset_match else None
    
    # 提取 ORDER BY 部分（在 LIMIT 或 OFFSET 之前截断）
    order_match = re.search(r'ORDER\s+BY\s+(.+?)(?:\s+LIMIT|\s+OFFSET|\s*$)', order_str, re.IGNORECASE)
    if not order_match:
        return None
    order_part = order_match.group(1).strip()
    
    # 判断排序方向和变量
    if order_part.upper().startswith("DESC"):
        var_match = re.search(r'DESC\(\s*(\?\w+)\s*\)', order_part, re.IGNORECASE)
        direction = "DESC"
    elif order_part.upper().startswith("ASC"):
        var_match = re.search(r'ASC\(\s*(\?\w+)\s*\)', order_part, re.IGNORECASE)
        direction = "ASC"
    else:
        var_match = re.search(r'(\?\w+)', order_part)
        direction = "ASC"
    
    if not var_match:
        return []
    var_name = var_match.group(1)
    
    return [var_name, direction, limit, offset]

def execute_filter(retr_node_dict, filter_str):
    filter_parse =  parses_filter(filter_str)
    #print(filter_parse)
    wrong_list = []
    after_filter_node_dict = dict()       
    for filter_tri in filter_parse:
        if filter_tri == [] or len(filter_tri) != 3:
            wrong_list.append('Parse fail! Format error in filtering constraint statements.')
            continue
        com1, com2, op = filter_tri
        if com1 not in retr_node_dict:
            wrong_list.append('Not found {}!'.format(com1))
            continue
        if com2.startswith("?"):
            if com2 not in retr_node_dict:
                wrong_list.append('Not found {}!'.format(com2))
                continue
        com1_list = copy.deepcopy(retr_node_dict[com1])
        if com2.startswith("?"):
            com2_list = copy.deepcopy(retr_node_dict[com2])
        else:
            com2_list = [com2]
        after_filter_list = []
        if len(com1_list) == 1 and len(com2_list) > 1 and com2.startswith("?") and com1.startswith("?"):
            tem_com_list = com1_list
            com1_list = com2_list
            com2_list = tem_com_list
            #print(com1_list)
            tem_com = com1
            com1 = com2
            com2 = tem_com
            if '<' in op:
                op = op.replace('<', '>')
            else:
                op = op.replace('>', '<')
        if 'XMLSchema#gYear' in com1_list:
            com1_list.remove('XMLSchema#gYear')
        if 'XMLSchema#gYear' in com2_list:
            com2_list.remove('XMLSchema#gYear')
        for com in com1_list:
            flag = True
            for comed in com2_list:
                if numeric_disting(com, comed, op) == False:
                    flag = False
                    break
            if flag == True:
                after_filter_list.append(com)
        if after_filter_list == []:
            wrong_list.append('All variables {} do not meet the condition!'.format(com1))
            continue
        cur_dict = dict()
        cur_dict[com1] = after_filter_list
        after_filter_node_dict.update(cur_dict)
    return wrong_list, after_filter_node_dict

def execute_order(retr_node_dict, order_str):
    order_parse = parses_order(order_str)
    #print(order_parse)
    wrong_list = []
    after_order_node_dict = dict()
    if len(order_parse) != 4:
        wrong_list.append('Parse fail! Format error in sorting constraint statements.')
        return wrong_list, after_order_node_dict
    if order_parse[2] != None:
        if order_parse[3] != None:
            com, types, return_num, offset_num = order_parse
        else:
            com, types, return_num, offset_num = order_parse
            offset_num = 0
    else:
        wrong_list.append('No limit number!')
        return wrong_list, after_order_node_dict
    if com not in retr_node_dict:
        wrong_list.append('Not found {}!'.format(com))
        return wrong_list, after_order_node_dict
    if retr_node_dict[com] == []:
        wrong_list.append('Not found {}!'.format(com))
        return wrong_list, after_order_node_dict
    if 'XMLSchema#gYear' in retr_node_dict[com]:
        retr_node_dict[com].remove('XMLSchema#gYear')
    if 'XMLSchema#date' in retr_node_dict[com]:
        retr_node_dict[com].remove('XMLSchema#date')
    if 'XMLSchema#gYearMonth' in retr_node_dict[com]:
        retr_node_dict[com].remove('XMLSchema#gYearMonth')
    if retr_node_dict[com] == []:
        wrong_list.append('Not found {}!'.format(com))
    if not any(ch.isdigit() for ch in retr_node_dict[com][0]) or 'm.' in retr_node_dict[com][0] or 'g.' in retr_node_dict[com][0]: 
        wrong_list.append('Variable {} cannot be sorted!'.format(com))
        return wrong_list, after_order_node_dict
    com_list = copy.deepcopy(retr_node_dict[com])
    retr_num = return_num + offset_num
    if len(com_list) < retr_num:
        after_order_node_dict[com] = com_list
        return wrong_list, after_order_node_dict
    after_order_list = []
    for i in range(retr_num):
        if types == 'ASC':
            cand_com = ''
            for cur_com in com_list:
                if cand_com == '':
                    cand_com = cur_com
                else:
                    if numeric_disting(cur_com, cand_com, '<'):
                        cand_com = cur_com
                        continue
                    else:
                        continue
            if cand_com == '':
                wrong_list.append('Sorting failed, only dates and values can be sorted!')
                return wrong_list, after_order_node_dict
            after_order_list.append(cand_com)  
            com_list.remove(cand_com)
        else:
            cand_com = ''
            for cur_com in com_list:
                if cand_com == '':
                    cand_com = cur_com
                else:
                    if numeric_disting(cur_com, cand_com, '>'):
                        cand_com = cur_com
                        continue
                    else:
                        continue
            if cand_com == '':
                wrong_list.append('Sorting failed, only dates and values can be sorted!')
                return wrong_list, after_order_node_dict
            after_order_list.append(cand_com)  
            com_list.remove(cand_com)
    after_order_list_offset = after_order_list[offset_num:]
    cur_dict = dict()
    cur_dict[com] = after_order_list_offset
    after_order_node_dict.update(cur_dict)
    return wrong_list, after_order_node_dict

def execute_whole_plan_golden_test(data_subgraph, data_stage, data_name, model, invalid_id_list = [], single_test_id = None):
    data = list(jsonlines.open(data_stage))
    for idx, dd in tqdm(enumerate(data), total = len(data)):
        if idx in invalid_id_list:
            continue
        if single_test_id != None:
            if idx != single_test_id:
                continue
        if data_name == 'webqsp' or data_name == 'webqsp_w_cycle' or 'webqsp' in data_name:
            d = dd['parses']['0']
        else:
            d = dd
        if 'plan' not in d:
            try:
                plan = get_retrieve_plan(d).get_final_plan_name()
            except Exception as e:
                continue
            if plan == False:
                continue
            else:
                d['plan'] = plan
        if data_name != 'cwq_train' and 'cwq_train' not in data_name:
            if len(d['answer_ent']) != len(d['answer_ent_name']):
                continue
            if len(d['answer_ent']) == 0:
                continue
            if len(d['topic_ent']) != len(d['topic_ent_name']):
                continue
        raw_plan_list = d['plan'].split('\n')
        #print(raw_plan_list)
        raw_plan = []
        filter_str_list = []
        order_str_list = []
        path_str_list = []
        return_ent = raw_plan_list[-1]
        if 'raw_idx' not in d:
            raw_idx = idx
        else:
            raw_idx = d['raw_idx']
        for p in raw_plan_list[:-1]:
            if 'FILTER' not in p and 'ORDER' not in p:
                raw_plan.append(p)
            if 'FILTER' in p:
                filter_str_list.append(p)
            if 'ORDER' in p:
                order_str_list.append(p)
        for p in raw_plan:
            path_str_list.append(extract_entities_and_relations(p))
        path_str_list = [[i.replace('-', '_') for i in sublist] for sublist in path_str_list]
        try:
            plan_subgraph = plan_get_retre_subgraph(path_str_list)
        except Exception as e:
            raw_plan = '\n'.join(raw_plan)
            raw_plan = paths_str_tree_modify(raw_plan)
            raw_plan = raw_plan.split('\n')
            path_str_list = []
            for p in raw_plan:
                path_str_list.append(extract_entities_and_relations(p))
            path_str_list = [[i.replace('-', '_') for i in sublist] for sublist in path_str_list]
            plan_subgraph = plan_get_retre_subgraph(path_str_list)
        ret = dict()
        try:
            ans_tree_list = []
            for k, p in plan_subgraph.items():
                ans_tree = execute_retr_paths_single(data_subgraph[raw_idx], d, p, data_name, mid2name, model)
                #ans_tree.print_tree() 
                ans_tree_list.append(ans_tree)
                p_retr_dict = ans_tree.summarize_labels()
                ret.update(p_retr_dict)
            #print(ret)
            limit_wrong_list = []
            #print(filter_str_list, order_str_list)
            for filter_str in filter_str_list:
                wrong_list, after_filter = execute_filter(ret, filter_str)
                limit_wrong_list.append(wrong_list)
                if after_filter == {}:
                    continue
                for com, com_list in after_filter.items():
                    #print(com)
                    if len(com_list) != ret[com]:
                        #print([i+'$￥'+com for i in com_list])
                        if len(ans_tree_list) == 1:                        
                            exist_nodes = ans_tree_list[0].summarize_labels()[com]
                            keep_nodes = list(set(com_list) & set(exist_nodes))
                            keep_nodes = [i+'$￥'+com for i in keep_nodes]
                            #print(keep_nodes)
                            #print(ans_tree_list[0].summarize_labels())
                            #print(ans_tree_list[0].retre_template_tree.get_branch_map())
                            ans_tree_list[0] = ans_tree_list[0].filter_tree_by_node_list(keep_nodes)
                            ret.update(ans_tree_list[0].summarize_labels())
                        else:
                            filter_tree_idx = -1
                            for tree_idx, ans_tree in enumerate(ans_tree_list):
                                if com in ans_tree.summarize_labels():
                                    filter_tree_idx = tree_idx
                            #ans_tree_list[0].print_tree()                        
                            exist_nodes = ans_tree_list[filter_tree_idx].summarize_labels()[com]
                            keep_nodes = list(set(com_list) & set(exist_nodes))
                            keep_nodes = [i+'$￥'+com for i in keep_nodes]
                            #print(keep_nodes)
                            ans_tree_list[filter_tree_idx] = ans_tree_list[filter_tree_idx].filter_tree_by_node_list(keep_nodes)
                            #ans_tree_list[0].print_tree()
                            ret.update(ans_tree_list[filter_tree_idx].summarize_labels())
            for order_str in order_str_list:
                wrong_list, after_order = execute_order(ret, order_str)
                limit_wrong_list.append(wrong_list)
                if after_order == {}:
                    continue
                for com, com_list in after_order.items():
                    #print(com)
                    if len(com_list) != ret[com]:
                        #print([i+'$￥'+com for i in com_list])
                        if len(ans_tree_list) == 1:                        
                            exist_nodes = ans_tree_list[0].summarize_labels()[com]
                            keep_nodes = list(set(com_list) & set(exist_nodes))
                            keep_nodes = [i+'$￥'+com for i in keep_nodes]
                            #print(keep_nodes)
                            ans_tree_list[0] = ans_tree_list[0].filter_tree_by_node_list(keep_nodes)
                            ret.update(ans_tree_list[0].summarize_labels())
                        else:
                            order_tree_idx = -1
                            for tree_idx, ans_tree in enumerate(ans_tree_list):
                                if com in ans_tree.summarize_labels():
                                    order_tree_idx = tree_idx
                            #ans_tree_list[0].print_tree()                        
                            exist_nodes = ans_tree_list[order_tree_idx].summarize_labels()[com]
                            keep_nodes = list(set(com_list) & set(exist_nodes))
                            keep_nodes = [i+'$￥'+com for i in keep_nodes]
                            #print(keep_nodes)
                            ans_tree_list[order_tree_idx] = ans_tree_list[order_tree_idx].filter_tree_by_node_list(keep_nodes)
                            #ans_tree_list[0].print_tree()
                            ret.update(ans_tree_list[order_tree_idx].summarize_labels())
            ret_ent_list = ret[return_ent]
            if single_test_id != None:
                return raw_plan, filter_str_list, order_str_list, ret, limit_wrong_list
        except Exception as e:
            #print(e)
            with open('./{}_whole_plan_exe.txt'.format(data_name), 'a', encoding='utf-8') as file:
                file.write(str(idx)+ ' ' +str(e)+'\n')
            continue
        if data_name == 'cwq_train' or 'cwq_train' in data_name:
            ans_ent_list_ = d['answer_ent']
            m_id_list = []
            for ans_dict in ans_ent_list_:
                m_id_list.append(ans_dict['answer_id'])
            # for m_idx, mid in enumerate(m_id_list):
            #     if d['topic_ent_name'].get(mid, mid) == mid:
            #         m_id_list[m_idx] = mid2name.get(mid,mid)
            #     else:
            #         m_id_list[m_idx] = d['topic_ent_name'].get(mid)
            for m_idx, mid in enumerate(m_id_list):
                m_id_list[m_idx] = mid_name(mid, d['topic_ent_name'], d['answer_ent_name'], mid2name)
            ans_ent_list = m_id_list
            ans_ent_list = [i.replace('-','_') for i in ans_ent_list]
        else:
            ans_ent_list = d['answer_ent']
            for a_id, a_ent in enumerate(ans_ent_list):
                if a_ent[:2] == 'm.' or a_ent[:2] == 'g.':
                    ans_ent_list[a_id] = d['answer_ent_name'][a_ent]
                else:
                    ans_ent_list[a_id] = a_ent
            ans_ent_list = [i.replace('-','_') for i in ans_ent_list]
        try:
            for ent in ans_ent_list:
                if no_alpha_letters(ent) == True:
                    flag = False
                    for ret_ent in ret_ent_list:
                        if numeric_disting(ent, ret_ent, '='):
                            flag = True
                            break
                    if flag == False:
                        raise ValueError(str(idx) + ' ' + 'Retr Wrong!', ret_ent_list, ans_ent_list)
                else:
                    if ent not in ret_ent_list:
                        raise ValueError(str(idx) + ' ' + 'Retr Wrong!', ret_ent_list, ans_ent_list)
            if 'XMLSchema#gYear' in ret_ent_list:
                ret_ent_list.remove('XMLSchema#gYear')
            if 'XMLSchema#date' in ret_ent_list:
                ret_ent_list.remove('XMLSchema#date')
            if 'XMLSchema#gYearMonth' in ret_ent_list:
                ret_ent_list.remove('XMLSchema#gYearMonth')
            if len(set(ret_ent_list)) != len(set(ans_ent_list)):
                raise ValueError(str(idx) + ' ' + 'Not exactly equal!', len(set(ret_ent_list)), len(set(ans_ent_list)), ret_ent_list, ans_ent_list)
        except Exception as e:
            with open('./{}_whole_plan_exe.txt'.format(data_name), 'a', encoding='utf-8') as file:
                if 'Retr Wrong!' in str(e) or 'Not exactly equal!' in str(e):
                    file.write(str(e)+'\n')
                else:
                    #print(ret_ent_list, ans_ent_list)
                    to_write = str(idx) + ' ' + 'Retr Wrong!' + " " + '[' + ', '.join(ret_ent_list) + ']' + '[' + ', '.join(ans_ent_list) + ']'
                    file.write(to_write + '\n')
            continue

def execute_whole_plan(data_subgraph, data_stage, data_name, whole_plan, mid2name, model):
    dd = data_stage
    wrong_dict = dict()
    if 'webqsp' in data_name:
        if 'parses' in dd:            
            d = dd['parses']['0']
        else:
            d = dd
    else:
        d = dd
    raw_plan_list = whole_plan.split('\n')
    #print(raw_plan_list)
    raw_plan = []
    filter_str_list = []
    order_str_list = []
    path_str_list = []
    return_ent = raw_plan_list[-1]
    raw_idx = d['raw_idx']
    for p in raw_plan_list[:-1]:
        if 'FILTER' not in p and 'ORDER' not in p:
            raw_plan.append(p)
        if 'FILTER' in p:
            filter_str_list.append(p)
        if 'ORDER' in p:
            order_str_list.append(p)
    for p in raw_plan:
        path_str_list.append(extract_entities_and_relations(p))
    path_str_list = [[i.replace('-', '_') for i in sublist] for sublist in path_str_list]
    plan_subgraph = plan_get_retre_subgraph(path_str_list)
    ret = dict()
    try:
        ans_tree_list = []
        for k, p in plan_subgraph.items():
            ans_tree, cur_graph_wrong_list = execute_retr_paths_single(data_subgraph[raw_idx], d, p, data_name, mid2name, model)
            #ans_tree.print_tree() 
            wrong_dict['path_sub_'+str(k)] = cur_graph_wrong_list
            ans_tree_list.append(ans_tree)
            p_retr_dict = ans_tree.summarize_labels()
            ret.update(p_retr_dict)
        #print(ret)
        limit_wrong_list = []
        #print(filter_str_list, order_str_list)
        for filter_str in filter_str_list:
            wrong_list, after_filter = execute_filter(ret, filter_str)
            wrong_dict[filter_str] = wrong_list
            limit_wrong_list.append(wrong_list)
            if after_filter == {}:
                continue
            for com, com_list in after_filter.items():
                #print(com)
                if len(com_list) != ret[com]:
                    #print([i+'$￥'+com for i in com_list])
                    if len(ans_tree_list) == 1:                        
                        exist_nodes = ans_tree_list[0].summarize_labels()[com]
                        keep_nodes = list(set(com_list) & set(exist_nodes))
                        keep_nodes = [i+'$￥'+com for i in keep_nodes]
                        #print(keep_nodes)
                        #print(ans_tree_list[0].summarize_labels())
                        #print(ans_tree_list[0].retre_template_tree.get_branch_map())
                        ans_tree_list[0] = ans_tree_list[0].filter_tree_by_node_list(keep_nodes)
                        ret.update(ans_tree_list[0].summarize_labels())
                    else:
                        filter_tree_idx = -1
                        for tree_idx, ans_tree in enumerate(ans_tree_list):
                            if com in ans_tree.summarize_labels():
                                filter_tree_idx = tree_idx
                        #ans_tree_list[0].print_tree()                        
                        exist_nodes = ans_tree_list[filter_tree_idx].summarize_labels()[com]
                        keep_nodes = list(set(com_list) & set(exist_nodes))
                        keep_nodes = [i+'$￥'+com for i in keep_nodes]
                        #print(keep_nodes)
                        ans_tree_list[filter_tree_idx] = ans_tree_list[filter_tree_idx].filter_tree_by_node_list(keep_nodes)
                        #ans_tree_list[0].print_tree()
                        ret.update(ans_tree_list[filter_tree_idx].summarize_labels())
        for order_str in order_str_list:
            wrong_list, after_order = execute_order(ret, order_str)
            wrong_dict[order_str] = wrong_list
            limit_wrong_list.append(wrong_list)
            if after_order == {}:
                continue
            for com, com_list in after_order.items():
                #print(com)
                if len(com_list) != ret[com]:
                    #print([i+'$￥'+com for i in com_list])
                    if len(ans_tree_list) == 1:                        
                        exist_nodes = ans_tree_list[0].summarize_labels()[com]
                        keep_nodes = list(set(com_list) & set(exist_nodes))
                        keep_nodes = [i+'$￥'+com for i in keep_nodes]
                        #print(keep_nodes)
                        ans_tree_list[0] = ans_tree_list[0].filter_tree_by_node_list(keep_nodes)
                        ret.update(ans_tree_list[0].summarize_labels())
                    else:
                        order_tree_idx = -1
                        for tree_idx, ans_tree in enumerate(ans_tree_list):
                            if com in ans_tree.summarize_labels():
                                order_tree_idx = tree_idx
                        #ans_tree_list[0].print_tree()                        
                        exist_nodes = ans_tree_list[order_tree_idx].summarize_labels()[com]
                        keep_nodes = list(set(com_list) & set(exist_nodes))
                        keep_nodes = [i+'$￥'+com for i in keep_nodes]
                        #print(keep_nodes)
                        ans_tree_list[order_tree_idx] = ans_tree_list[order_tree_idx].filter_tree_by_node_list(keep_nodes)
                        #ans_tree_list[0].print_tree()
                        ret.update(ans_tree_list[order_tree_idx].summarize_labels())
        #ret_ent_list = ret[return_ent]
        return raw_plan_list, ret, return_ent, wrong_dict
    except Exception as e:
        #print(e)
        raise ValueError(e)

def cal_pre_f1_score(cur_data, ret, return_ent, mid2name, data_name):
    if 'parses' in cur_data:
        d = cur_data['parses']['0']
    else:
        d = cur_data
    ret_ent_list = ret.get(return_ent, [])
    if data_name == 'cwq_train' or 'cwq_train' in data_name:
        ans_ent_list_ = d['answer_ent']
        m_id_list = []
        for ans_dict in ans_ent_list_:
            m_id_list.append(ans_dict['answer_id'])
        for m_idx, mid in enumerate(m_id_list):
            m_id_list[m_idx] = mid_name(mid, d['topic_ent_name'], d['answer_ent_name'], mid2name)
        ans_ent_list = m_id_list
        ans_ent_list = [i.replace('-','_') for i in ans_ent_list]
    else:
        ans_ent_list = d['answer_ent']
        if isinstance(ans_ent_list, str):
            ans_ent_list = [ans_ent_list]
        for a_id, a_ent in enumerate(ans_ent_list):
            if a_ent[:2] == 'm.' or a_ent[:2] == 'g.':
                ans_ent_list[a_id] = d['answer_ent_name'][a_ent]
            else:
                ans_ent_list[a_id] = a_ent
        ans_ent_list = [i.replace('-','_') for i in ans_ent_list]
    overlap_num = 0     
    for ent in ans_ent_list:
        if no_alpha_letters(ent) == True:
            for ret_ent in ret_ent_list:
                if numeric_disting(ent, ret_ent, '='):
                    if 'simple' in data_name:
                        return 1
                    overlap_num += 1
                    break
        else:
            if ent in ret_ent_list:
                if 'simple' in data_name:
                    return 1
                overlap_num += 1
    if 'XMLSchema#gYear' in ret_ent_list:
        ret_ent_list.remove('XMLSchema#gYear')
    if 'XMLSchema#date' in ret_ent_list:
        ret_ent_list.remove('XMLSchema#date')
    if 'XMLSchema#gYearMonth' in ret_ent_list:
        ret_ent_list.remove('XMLSchema#gYearMonth')
    # 计算 Precision 和 Recall
    recall = overlap_num / len(ans_ent_list) if ans_ent_list else 0
    precision = overlap_num / len(ret_ent_list) if ret_ent_list else 0
    # 计算 F1
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1

def list_modify_str(cur_list, lim_num):
    if len(cur_list) <= lim_num:
        return ' ; '.join(cur_list)
    return ' ; '.join(cur_list[:lim_num]) + ' ...'

def get_retr_result_report(raw_plan_list, ret, return_ent, wrong_dict):
    report_tem = """
The entity to be returned by the retrieval plan:
{}
The search results for other entities on the retrieval paths are:
{}
The error during the execution of the retrieval plan are:
{}
    """
    #ret_ent_res = ""
    if return_ent not in ret or ret[return_ent] == []:
        ret_ent_res = f"{return_ent} was not retrieved."
    else:
        ret_ent_res = f"{return_ent} : {list_modify_str(ret[return_ent], 2)}"
    pathslist = []
    mid_ent_name_list = []
    for p_str in raw_plan_list[:-1]:
        if 'FILTER' not in p_str and 'ORDER' not in p_str:
            pathslist.append(p_str)
    for p_str in pathslist:
        p_list = extract_entities_and_relations(p_str)
        for i in range(0, len(p_list), 2):
            if p_list[i][0] == '?' and p_list[i] != return_ent:
                mid_ent_name_list.append(p_list[i])
    mid_ent_name_list = list(set(mid_ent_name_list))
    mid_ent_res = []
    for k in mid_ent_name_list:
        if k in ret:
            mid_ent_res.append(f'{k} : {list_modify_str(ret[k], 2)}')
        else:
            mid_ent_res.append(f"{k} was not retrieved.")
    # for k, v in ret.items():
    #     if k[0] == '?' and k != return_ent:
    #         mid_ent_res.append(f'{k} : {list_modify_str(ret[k], 2)}')
    if mid_ent_res == []:
        mid_ent_res.append('No other entities in the retrieval path.')
    mid_ent_res = '\n'.join(mid_ent_res)
    paths_errors = []
    for k,v in wrong_dict.items():
        if 'path_sub_' in k:
            for p_err in v:
                if p_err == []:
                    continue
                paths_errors += p_err
    filter_errors = []
    for k,v in wrong_dict.items():
        if 'FILTER' in k:
            if v == []:
                continue
            filter_errors.append(k + ' : ' + ' '.join(v))
    order_errors = []
    for k,v in wrong_dict.items():
        if 'ORDER' in k:
            if v == []:
                continue
            order_errors.append(k + ' : ' + ' '.join(v))
    all_error = paths_errors + filter_errors + order_errors
    if all_error == []:
        all_error.append('No errors occurred!')
    all_error_str = '\n'.join(all_error)
    report_tem = report_tem.format(ret_ent_res, mid_ent_res, all_error_str)
    return report_tem

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

def get_ai_input_stage3(prompt, data_w_rag, few_shot = False):
    ques = data_w_rag["question"]
    if "parses" in data_w_rag:
        data_w_rag = data_w_rag["parses"]["0"]
    d_top_str, d_top = get_topic_ent(data_w_rag)
    d_sim_rel = get_sim_hop_str(data_w_rag["sim_n_hop_rels"], 3)
    check, whole_plan = check_extract_modify_get_plan(data_w_rag["stage2_output"])
    if check == False:
        raise ValueError('check wrong')
    d_plan = whole_plan
    d_exe_report = data_w_rag["stage2_exe_report"]
    if few_shot == False:
        return prompt.format(ques, d_top_str, d_top, d_sim_rel, d_plan, d_exe_report)
    else:
        few_shot = data_w_rag['few_shot']
        return prompt.format(ques, few_shot, d_top_str, d_top, d_sim_rel, d_plan, d_exe_report)


def get_ai_input_stage3_ref(prompt, data_w_rag, few_shot = False):
    ques = data_w_rag["question"]
    if "parses" in data_w_rag:
        data_w_rag = data_w_rag["parses"]["0"]
    d_top_str, d_top = get_topic_ent(data_w_rag)
    d_sim_rel = get_sim_hop_str(data_w_rag["sim_n_hop_rels"], 3)
    ref_dict = data_w_rag['reflection_list'][-1]
    check, whole_plan = check_extract_modify_get_plan(ref_dict["stage3_output"])
    if check == False:
        raise ValueError('check wrong')
    d_plan = whole_plan
    d_exe_report = ref_dict["stage3_exe_report"]
    if few_shot == False:
        return prompt.format(ques, d_top_str, d_top, d_sim_rel, d_plan, d_exe_report)
    else:
        few_shot = data_w_rag['few_shot']
        return prompt.format(few_shot, ques, d_top_str, d_top, d_sim_rel, d_plan, d_exe_report)
