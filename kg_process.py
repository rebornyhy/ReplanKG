import os
import datasets
import json
from tqdm import tqdm
import jsonlines
import networkx as nx
import random
import pickle
import torch
import torch
from torch.utils.data import Dataset, DataLoader
import jsonlines


def set_individual_globals(dicts):
    """动态设置每个字典为全局变量"""
    globals().update(dicts)

def invert_dict(d):
    return {v: k for k, v in d.items()}

def get_dict(input_dir, add_rev=False):
    cache_fn = os.path.join(input_dir, 'processed_data/processed.pt')
    #print(cache_fn)
    if os.path.exists(cache_fn):
        print('#' * 10, 'data read from cache file: {}!!!!'.format(cache_fn))
        with open(cache_fn, 'rb') as fp:
            ent2id, rel2id, id2name, name2id, id2ent, id2rel = pickle.load(fp)
        print('entitys_num:', len(ent2id))
        print('rels_num:', len(rel2id))
    else:
        print('Read data...')
        ent2id, id2name = {}, {}
        for line in open(os.path.join(input_dir, 'entities_id_and_name_merge.txt'), encoding="utf-8"):
            split = line.split('\t')
            ent2id[split[0].strip()] = len(ent2id)
            id2name[len(id2name)] = split[1].strip() if split[1].strip() != '-' else split[0].strip()

        rel2id = {}
        for line in open(os.path.join(input_dir, 'relations.txt')):
            rel2id[line.strip()] = len(rel2id)
        # add self relation and reverse relation
        # rel2id['<self>'] = len(rel2id)
        if add_rev:
            for line in open(os.path.join(input_dir, 'relations.txt')):
                rel2id[line.strip()+'_rev'] = len(rel2id)
        print('entitys_num:', len(ent2id))
        print('rels_num:', len(rel2id))

        id2ent = invert_dict(ent2id)
        id2rel = invert_dict(rel2id)
        name2id = invert_dict(id2name)

    if not os.path.exists(cache_fn):
        with open(cache_fn, 'wb') as fp:
            pickle.dump((ent2id, rel2id, id2name, name2id, id2ent, id2rel), fp)

    return ent2id, rel2id, id2name, name2id, id2ent, id2rel

def get_dict_w(input_dir):
    cache_fn = os.path.join(input_dir, 'processed_data/processed.pt')
    print(cache_fn)
    if os.path.exists(cache_fn):
        print('#' * 10, 'DATA read from cache file: {}!!!!'.format(cache_fn))
        with open(cache_fn, 'rb') as fp:
            ent2id, rel2id, triples = pickle.load(fp)
        print('Ent number: {}, rel number: {}, tri number: {}'.format(len(ent2id), len(rel2id), len(triples)))
    else:
        print('#' * 10, 'Read data...')

        # 图谱数据
        ent2id, id2name, count = {}, {}, 0
        for line in open(os.path.join(input_dir, 'Dic/entity.txt'), encoding="utf-8"):
            # split = line.strip().split('\t')
            # if split[0].strip() not in ent2id:
            #     ent2id[split[0].strip()] = count
            #     id2name[count] = split[1].strip()
            #     count = count + 1
            if line.strip() not in ent2id:
                ent2id[line.strip()] = count
                count = count + 1
        count = 0
        rel2id = {}
        for line in open(os.path.join(input_dir, 'Dic/relation.txt'), encoding="utf-8"):
            if line.strip() not in rel2id:
                line = line.strip()
                line = line.replace('/','.')
                rel2id[line.strip()] = count
                count = count + 1

        triples = []
        for line in open(os.path.join(input_dir, 'KG/triplets.txt'), encoding="utf-8"):
            l = line.strip().split('\t')
            s = ent2id[l[0].strip()]
            p = rel2id[l[1].strip().replace('/','.')]
            o = ent2id[l[2].strip()]
            triples.append([s, p, o])
            #p_rev = rel2id[l[1].strip()+'_reverse']
            #triples.append((o, p_rev, s))
        #triples = torch.LongTensor(triples)
        
        print('Ent number: {}, rel number: {}, tri number: {}'.format(len(ent2id), len(rel2id), len(triples)))

        with open(cache_fn, 'wb') as fp:
            pickle.dump((ent2id, rel2id, triples), fp)

    return ent2id, rel2id, triples

def find_node_all_paths(subgraph, node):
    """
    在给定的知识图谱子图中， 找到给定节点的所有路径。
    
    Find all longest relation paths starting from a given node in a knowledge graph subgraph.

    Args:
        subgraph (list of [s, r, b]): Knowledge graph subgraph represented as a list of triples.
        node: The starting node.

    Returns:
        list of list: A list of paths, where each path is a list of relations.
    """
    from collections import defaultdict

    # Build adjacency list for the graph (both directions since relationships are undirected)
    adj_list = defaultdict(list)
    for s, r, b in subgraph:
        adj_list[s].append((r, b))
        adj_list[b].append((r, s))

    def dfs(current_node, visited):
        """Perform DFS to find all longest paths."""
        visited.add(current_node)
        paths = []

        is_leaf = True
        for relation, neighbor in adj_list[current_node]:
            if neighbor not in visited:
                is_leaf = False
                for sub_path in dfs(neighbor, visited):
                    paths.append([relation] + sub_path)

        visited.remove(current_node)

        # If it's a leaf node, return an empty path as the end of the traversal
        return paths if not is_leaf else [[]]

    # Start DFS from the given node
    all_paths = dfs(node, set())

    return all_paths


def has_cycle(subgraph):
    """
    判断无向图是否存在环。
    
    参数:
        subgraph (list): 子图，元素为 [s, r, b] 的列表，表示无向图的一条边，s 和 b 为节点，r 为关系。
        
    返回:
        bool: 如果存在环则返回 True，否则返回 False。
    """
    # 并查集的 find 函数
    def find(parent, node):
        if parent[node] != node:
            parent[node] = find(parent, parent[node])
        return parent[node]
    
    # 并查集的 union 函数
    def union(parent, rank, node1, node2):
        root1 = find(parent, node1)
        root2 = find(parent, node2)
        
        if root1 != root2:
            # 按秩合并
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            elif rank[root1] < rank[root2]:
                parent[root1] = root2
            else:
                parent[root2] = root1
                rank[root1] += 1
            return False  # 未形成环
        return True  # 已形成环
    
    # 提取所有的节点
    nodes = set()
    for s, _, b in subgraph:
        nodes.add(s)
        nodes.add(b)
    
    # 初始化并查集
    parent = {node: node for node in nodes}
    rank = {node: 0 for node in nodes}
    
    # 遍历边，判断是否有环
    for s, _, b in subgraph:
        if union(parent, rank, s, b):
            return True  # 检测到环
    
    return False  # 无环


def find_paths(G, startnode_id, relation_id_list, ent2id, rel2id, id2ent, id2name, id2rel, return_name, num = None):
    """
    在知识图谱中获取给定出发节点与路径的子图
    """
    if isinstance(startnode_id, str):
        if 'm.' in startnode_id:
            startnode_id = ent2id[startnode_id]
        else:
            startnode_id = name2id[startnode_id]
    if relation_id_list != []:
        if isinstance(relation_id_list[0], str):
            relation_id_list = [rel2id[rel] for rel in relation_id_list]
            #print(relation_id_list)
    def bfs_find_paths(G, startnode, relation_list):
        """
        在有向图中，根据指定的关系列表从起始节点进行搜索，返回匹配的子图。

        参数:
            G: networkx.DiGraph, 有向图
            startnode: 起始节点
            relation_list: 包含关系类型的列表

        返回:
            list: 符合条件的子图，形式为 [[a, relation, b], ...]
        """
        result = []
        current_nodes = [startnode]  # 当前搜索到的节点集
        
        if isinstance(G, nx.MultiDiGraph):
            for relation in relation_list:
                next_nodes = []
                for node in current_nodes:
                    # 搜索出度边
                    for succ in G.successors(node):
                        edge_data = G.get_edge_data(node, succ, default={})
                        for idx, value in edge_data.items():
                            if value.get("relation") == relation:
                                result.append([node, relation, succ])
                                next_nodes.append(succ)
                                break

                    # 搜索入度边
                    for pred in G.predecessors(node):
                        edge_data = G.get_edge_data(pred, node, default={})
                        for idx, value in edge_data.items():
                            if value.get("relation") == relation:
                                result.append([pred, relation, node])
                                next_nodes.append(pred)
                                break
                        #print(next_nodes)

                current_nodes = list(set(next_nodes))  # 更新下一步的搜索节点
        else:
            for relation in relation_list:
                next_nodes = []
                for node in current_nodes:
                    # 搜索出度边
                    for succ in G.successors(node):
                        edge_data = G.get_edge_data(node, succ, default={})
                        if edge_data.get("relation") == relation:
                            result.append([node, relation, succ])
                            next_nodes.append(succ)

                    # 搜索入度边
                    for pred in G.predecessors(node):
                        edge_data = G.get_edge_data(pred, node, default={})
                        if edge_data.get("relation") == relation:
                            result.append([pred, relation, node])
                            next_nodes.append(pred)
                        #print(next_nodes)

                current_nodes = next_nodes  # 更新下一步的搜索节点            
        #print(result)

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
    
    if num == None:
        result_sub_graph = bfs_find_paths(G, startnode_id, relation_id_list)
    else:
         result_sub_graph = get_hop(G, startnode_id, num)
    answer = []
    for tri in result_sub_graph:
        if return_name == True:
            answer.append([id2ent[tri[0]], id2rel[tri[1]], id2ent[tri[2]]])
        else:
            answer.append([tri[0], tri[1], tri[2]])
    return answer

def find_paths_list(G, startnode, relation_list, num = None):
    """
    在知识图谱列表中获取给定出发节点与路径的子图
    """
    def bfs_find_paths(G, startnode, relation_list):
        """
        在有向图中，根据指定的关系列表从起始节点进行搜索，返回匹配的子图。

        参数:
            G: networkx.DiGraph, 有向图
            startnode: 起始节点
            relation_list: 包含关系类型的列表

        返回:
            list: 符合条件的子图，形式为 [[a, relation, b], ...]
        """
        result = []
        current_nodes = [startnode]  # 当前搜索到的节点集
        
        for relation in relation_list:
            next_nodes = []
            for node in current_nodes:
                # 搜索出度边
                for tri in G:
                    s, r, o = tri
                    if s == node or o == node:
                        if s == node and r == relation:
                            result.append([s, r, o])
                            next_nodes.append(o)
                        if o == node and r == relation:
                            result.append([s, r, o])
                            next_nodes.append(s)                           
                current_nodes = next_nodes  # 更新下一步的搜索节点            
        #print(result)

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
        
        for _ in range(n):
            next_nodes = []
            for node in current_nodes:
                if node in visited:
                    continue
                visited.add(node)

                # 搜索边
                for tri in G:
                    s, r, o = tri
                    if s == node or o == node:
                        if s == node:
                            result.append([s, r, o])
                            next_nodes.append(o)
                        if o == node:
                            result.append([s, r, o])
                            next_nodes.append(s)      

                current_nodes = next_nodes  # 更新为下一层的节点集         

        return result
    
    if num == None:
        result_sub_graph = bfs_find_paths(G, startnode, relation_list)
    else:
         result_sub_graph = get_hop(G, startnode_id, num)
    
    return result_sub_graph

def find_paths_mid(G, startnode_id, relation_id_list, num = None):
    """
    在知识图谱中获取给定出发节点与路径的子图
    """
    def bfs_find_paths(G, startnode, relation_list):
        """
        在有向图中，根据指定的关系列表从起始节点进行搜索，返回匹配的子图。

        参数:
            G: networkx.DiGraph, 有向图
            startnode: 起始节点
            relation_list: 包含关系类型的列表

        返回:
            list: 符合条件的子图，形式为 [[a, relation, b], ...]
        """
        result = []
        current_nodes = [startnode]  # 当前搜索到的节点集
        
        if isinstance(G, nx.MultiDiGraph):
            for relation in relation_list:
                next_nodes = []
                for node in current_nodes:
                    # 搜索出度边
                    for succ in G.successors(node):
                        edge_data = G.get_edge_data(node, succ, default={})
                        for idx, value in edge_data.items():
                            if value.get("relation") == relation:
                                result.append([node, relation, succ])
                                next_nodes.append(succ)
                                break

                    # 搜索入度边
                    for pred in G.predecessors(node):
                        edge_data = G.get_edge_data(pred, node, default={})
                        for idx, value in edge_data.items():
                            if value.get("relation") == relation:
                                result.append([pred, relation, node])
                                next_nodes.append(pred)
                                break
                        #print(next_nodes)

                current_nodes = list(set(next_nodes))  # 更新下一步的搜索节点
        else:
            for relation in relation_list:
                next_nodes = []
                for node in current_nodes:
                    # 搜索出度边
                    for succ in G.successors(node):
                        edge_data = G.get_edge_data(node, succ, default={})
                        if edge_data.get("relation") == relation:
                            result.append([node, relation, succ])
                            next_nodes.append(succ)

                    # 搜索入度边
                    for pred in G.predecessors(node):
                        edge_data = G.get_edge_data(pred, node, default={})
                        if edge_data.get("relation") == relation:
                            result.append([pred, relation, node])
                            next_nodes.append(pred)
                        #print(next_nodes)

                current_nodes = next_nodes  # 更新下一步的搜索节点            
        #print(result)

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
    
    if num == None:
        result_sub_graph = bfs_find_paths(G, startnode_id, relation_id_list)
    else:
         result_sub_graph = get_hop(G, startnode_id, num)
    answer = []
    for tri in result_sub_graph:
        answer.append([tri[0], tri[1], tri[2]])
    return answer

def extract_path_and_tails(subgraph, node, refer_rels):
    next_rel = ''
    visited = set()
    cur_node_list = [node]
    i = 0
    while(len(visited) != len(subgraph) and i < len(refer_rels)):
        next_node_list = []
        for tri in subgraph:
            if ','.join(tri) in visited:
                continue
            s, r, o = tri
            if (s in cur_node_list or o in cur_node_list) and r == refer_rels[i]:
                visited.add(','.join(tri))
                if s in cur_node_list:
                    next_node_list.append(o)
                else:
                    next_node_list.append(s)
        cur_node_list = next_node_list
        i += 1
    return i, cur_node_list

def get_data_subgraph_single(data):
    import networkx as nx
    def create_graph(all_tri):
        G = nx.DiGraph()
        for tri in all_tri:
            if len(tri) != 3:
                continue
            G.add_edge(tri[0], tri[2], relation=tri[1])
        return G
    G = create_graph(data['subgraph']['tuples'])
    return G

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

def extract_path_and_tails(subgraph, node, refer_rels):
    next_rel = ''
    visited = set()
    cur_node_list = [node]
    i = 0
    while(len(visited) != len(subgraph) and i < len(refer_rels)):
        next_node_list = []
        for tri in subgraph:
            if ','.join(str(tri)) in visited:
                continue
            s, r, o = tri
            if (s in cur_node_list or o in cur_node_list) and r == refer_rels[i]:
                visited.add(','.join(str(tri)))
                if s in cur_node_list:
                    next_node_list.append(o)
                else:
                    next_node_list.append(s)
        cur_node_list = list(set(next_node_list))
        if len(cur_node_list) == 0:
            break
        i += 1
    return i, cur_node_list

class simpleques(Dataset):
    def __init__(self, data_path, subgraph_path):
        self.data = list(jsonlines.open(data_path))
        self.subgraph = list(jsonlines.open(subgraph_path))

    def __getitem__(self,index):
        data = self.data[index]
        data['subgraph'] = self.subgraph[index]
        return data

    def __len__(self):
        return len(self.data)

def get_simpleques(data_path, subgraph_path):
    data = simpleques(data_path, subgraph_path)
    return data

def verify_webqsp_single(data):
    
    def single_kg_verify(rel_paths, topic_ent_mid, subgraph):
        error_ans = []
        if topic_ent_mid not in subgraph:
            return False, rel_paths_mid
        for idx, rel_path in enumerate(rel_paths):
            verify_subgraph = find_paths_mid(subgraph, topic_ent_mid, rel_path)
            verify_hop, tail_node_list = extract_path_and_tails(verify_subgraph, topic_ent_mid, rel_path)
            #print(verify_hop, tail_node_list, rel_paths)
            if verify_hop < len(rel_path):
                error_ans.append(rel_paths[idx])
        if len(error_ans) > 0:
            return False, error_ans
        return True, []
    
    error_dict = dict()
    retr_graph = data['subgraph']
    retr_graph = create_graph_all(retr_graph, multi = True)
    for idx, value in data['parses'].items():    
        topic_ent = value['topic_ent']
        answer_ent = value['answer_ent']
        kg = value['kg']        
        for topic_ent_key, rel_paths in value['distribution'].items():
            verify_ans, error_list = single_kg_verify(rel_paths, topic_ent_key, retr_graph)
            if verify_ans == False:
                error_dict[topic_ent_key] = error_list
    if len(error_dict) > 0:
        return False, error_dict
    
    return True, {}

class webqsp(Dataset):
    def __init__(self, data_path, sub_path, com_path, rel2id, ent2id, id2ent, id2rel):
        self.data = list(jsonlines.open(data_path))
        self.sub_graph = list(jsonlines.open(sub_path))
        self.retr_subgraph = list(jsonlines.open(com_path))
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.id2ent = id2ent
        self.id2rel = id2rel
        

    def __getitem__(self, index):
        data = self.data[index]
        graph = self.sub_graph[index]["subgraph"]
        if 'tris' in self.retr_subgraph[index]:
            retr_subgraph = self.retr_subgraph[index]
        else:
            retr_subgraph = self.retr_subgraph[index]["0"]
        subgraph, mid2name = self.combine_generate_subgraph(data, graph, retr_subgraph)
        data['mid2name'] = mid2name
        data['subgraph'] = subgraph
        return data

    def __len__(self):
        return len(self.data)
    
    def combine_generate_subgraph(self, data, graph, retr_subgraph):
        tris_set = set()
        answer = []
        for tri in graph:
            #print(tri)
            s, r, o = tri
            s = self.id2ent[s]
            o = self.id2ent[o]
            r = self.id2rel[r]
            if '-'.join([s, r, o]) not in tris_set:
                answer.append([s, r, o])
                tris_set.add('-'.join([s, r, o]))
        tem_retr_graph = retr_subgraph['tris']
        if 'mid2name' in retr_subgraph:
            mid2name = retr_subgraph['mid2name']
        else:
            mid2name = retr_subgraph['0']['mid2name']
        for tri in tem_retr_graph:
            s, r, o = tri
            s = s.replace('ns:', '')
            r = r.replace('ns:', '')
            o = o.replace('ns:', '')
            #print(s, r, o)
            #print([ent2id[s], rel2id[r], ent2id[o]])
            if '-'.join([s, r, o]) not in tris_set:
                answer.append([s, r, o])
                tris_set.add('-'.join([s, r, o]))
        return answer, mid2name    
    
def get_webqsp(data_path, sub_path, com_path, dict_path):
    ent2id, rel2id, triples = get_dict_w(dict_path)
    id2ent = invert_dict(ent2id)
    id2rel = invert_dict(rel2id)
    del triples
    #graph = create_graph_all(triples, multi = True)
    webqspdata = webqsp(data_path, sub_path, com_path, rel2id, ent2id, id2ent, id2rel)
    return webqspdata

#test_webqsp = 

class cwq(Dataset):
    def __init__(self, data_path, sub_path, com_path, rel2id, ent2id, id2ent, id2rel):
        self.data = list(jsonlines.open(data_path))
        self.sub_graph = list(jsonlines.open(sub_path))
        self.retr_subgraph = list(jsonlines.open(com_path))
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.id2ent = id2ent
        self.id2rel = id2rel

    def __getitem__(self, index):
        data = self.data[index]
        graph = self.sub_graph[index]["subgraph"]["tuples"]
        retr_subgraph = self.retr_subgraph[index]
        subgraph, mid2name = self.combine_generate_subgraph(data, graph, retr_subgraph)
        data['mid2name'] = mid2name
        data['subgraph'] = subgraph
        return data

    def __len__(self):
        return len(self.data)
    
    def combine_generate_subgraph(self, data, graph, retr_subgraph):
        tris_set = set()
        answer = []
        for tri in graph:
            #print(tri)
            s, r, o = tri
            s = self.id2ent[s]
            o = self.id2ent[o]
            r = self.id2rel[r]
            if '-'.join([s, r, o]) not in tris_set:
                answer.append([s, r, o])
                tris_set.add('-'.join([s, r, o]))
        tem_retr_graph = retr_subgraph['tris']
        mid2name = retr_subgraph['mid2name']
        for tri in tem_retr_graph:
            s, r, o = tri
            s = s.replace('ns:', '')
            r = r.replace('ns:', '')
            o = o.replace('ns:', '')
            r_list = r.split('.')
            r_per = r_list[:len(r_list)-2]
            r_res = r_list[-2:][::-1]
            r_res_list = r_per + r_res
            r_res_str = '.'.join(r_res_list)
            #print(s, r, o)
            #print([ent2id[s], rel2id[r], ent2id[o]])
            if '-'.join([s, r, o]) not in tris_set:
                answer.append([s, r, o])
                tris_set.add('-'.join([s, r, o]))
            if '-'.join([o, r_res_str, s]) not in tris_set:
                answer.append([o, r_res_str, s])
                tris_set.add('-'.join([o, r_res_str, s]))
        return answer, mid2name
    
def verify_cwq_single(data):
    
    def single_kg_verify_c(rel_paths, topic_ent_mid, subgraph):
        error_ans = []
        if topic_ent_mid not in subgraph:
            return False, rel_paths
        for idx, rel_path in enumerate(rel_paths):
            verify_subgraph = find_paths_mid(subgraph, topic_ent_mid, rel_path)
            verify_hop, tail_node_list = extract_path_and_tails(verify_subgraph, topic_ent_mid, rel_path)
            if verify_hop < len(rel_path):
                error_ans.append(rel_paths[idx])
        if len(error_ans) > 0:
            return False, error_ans
        return True, []
    
    error_dict = dict()
    retr_graph = data['subgraph']
    topic_ent = data['topic_ent']
    retr_graph = create_graph_all(retr_graph, multi = True)
    for topic_ent_mid, rel_paths in data['distribution'].items():
        verify_ans, error_list = single_kg_verify_c(rel_paths, topic_ent_mid, retr_graph)
        if verify_ans == False:
            error_dict[topic_ent_mid] = error_list
    if len(error_dict) > 0:
        return False, error_dict
    return True, {}

def get_cwq(data_path, sub_path, com_path, dict_path):
    ent2id, rel2id, id2name, name2id, id2ent, id2rel = get_dict(dict_path, add_rev=False)
    cwqdata = cwq(data_path, sub_path, com_path, rel2id, ent2id, id2ent, id2rel)
    return cwqdata
