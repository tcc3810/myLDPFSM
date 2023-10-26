import networkx as nx
from matplotlib import pylab as pl
import itertools
import math
import numpy as np
from networkx.algorithms import isomorphism
from random import choice
import time
from random import sample
import multiprocessing

def read_graph(data_file):
    file = open(data_file, "r")
    all_G = list()
    labels = list()
    max_nodes = 0

    G = nx.Graph()
    nodes_name  = file.readline()
    id = 1
    for name in nodes_name.split():
        G.add_node(id, name = name)
        id = id + 1
        if name not in labels:
            labels.append(name)
    if max_nodes < id - 1:
        max_nodes = id - 1

    while file:
        line = file.readline()
        
        if line == "EOF":
            all_G.append(G)

            break
        elif line == "EOF\n":
            all_G.append(G)

            G = nx.Graph()
            nodes_name  = file.readline()
            id = 1
            for name in nodes_name.split():
                G.add_node(id, name = name)
                id = id + 1
                if name not in labels:
                    labels.append(name)
            if max_nodes < id - 1:
                max_nodes = id - 1
        else:
            (x, y, z) = line.split()
            G.add_edge(int(x), int(y))

    file.close()
    return all_G, labels, max_nodes

def draw_graph_name(G):
    pos = nx.spring_layout(G)
    pl.show()
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos, labels = nx.get_node_attributes(G, "name"))
    nx.draw_networkx_edges(G, pos)
    pl.show()

def encode(response, domain):
    enc = np.zeros(len(domain))
    enc[response] = 1
    
    return enc

def perturb(encoded_response, epsilon):
    p = 0.5
    q = 1 / (math.exp(epsilon) + 1)
    perturb_response = np.zeros(len(encoded_response))

    for i, b in enumerate(encoded_response):
        sample = np.random.random()
        if b == 1 and sample <= p:
            perturb_response[i] = 1
        elif b == 0 and sample <= q:
            perturb_response[i] = 1

    return perturb_response

def aggregate(responses, epsilon):
    p = 0.5
    q = 1 / (math.exp(epsilon) + 1)
    
    sums = np.sum(responses, axis=0)
    n = len(responses)
    
    aggregate_response = list()
    for v in sums:
        aggregate_response.append( (v - n * q) / (p - q) )

    return aggregate_response


def generate_all_1_subgraphs(max_nodes, labels):
    all_subgraphs = list()

    diff_labels = list(itertools .combinations(labels, 2))
    for label in diff_labels:
        G = nx.Graph()
        G.add_node(1, name = label[0])
        G.add_node(2, name = label[1])
        G.add_edge(1, 2)
        all_subgraphs.append(G)
    
    same_labels = list(itertools .combinations(labels, 1))
    for label in same_labels:
        G = nx.Graph()
        G.add_node(1, name = label[0])
        G.add_node(2, name = label[0])
        G.add_edge(1, 2)
        all_subgraphs.append(G)
    
    return all_subgraphs

def generate_candidate_i_subgraphs(max_nodes, labels, i, pre_subgraphs):
    # 找到全部 candidate subgraphs
    all_candidate_subgraphs = list()
    for x, G in enumerate(pre_subgraphs):
        for y in range(x, len(pre_subgraphs)):       
            subgraphs = candidate_between_graphs(G, pre_subgraphs[y])
            # 刪除同構
            for p in range(len(subgraphs)-1, 0, -1):
                for q in range(0, p):
                    nm = isomorphism.categorical_node_match("name", "")
                    if nx.is_isomorphic(subgraphs[p], subgraphs[q], node_match = nm):
                        subgraphs.pop(p)
                        break

            for subgraph in subgraphs:
                count = 0
                for candidate_subgraph in all_candidate_subgraphs:
                    nm = isomorphism.categorical_node_match("name", "")
                    if nx.is_isomorphic(subgraph, candidate_subgraph, node_match = nm):
                        break
                    else:
                        count = count + 1
                
                if count == len(all_candidate_subgraphs):
                    all_candidate_subgraphs.append(subgraph)
    
    # 只選 i_candidate subgraphs
    candidate_subgraphs = list()
    for subgraph in all_candidate_subgraphs:
        if subgraph.number_of_edges() == i:
            candidate_subgraphs.append(subgraph)

    return candidate_subgraphs

def candidate_between_graphs(G, H):
    candidate_subgraphs = list()
    if G.number_of_edges() == 1:
        name_G = nx.get_node_attributes(G, "name")            
        name_H = nx.get_node_attributes(H, "name")

        for y in range(1, 3):
            for x in range(1, 3):
                a_name = name_G[x]
                b_name = name_G[3 - x]
                c_name = name_H[y]
                d_name = name_H[3 - y]

                # case 1: a != c
                if a_name != c_name:
                    # b == d (c->b)
                    if b_name == d_name:
                        P = G.copy()
                        P.add_node(3, name = c_name)
                        P.add_edge(3, 3 - x)
                        candidate_subgraphs.append(P)
                # case 2: a == c
                elif a_name == c_name:
                    # b != d (d->a)
                    P = G.copy()
                    P.add_node(3, name = d_name)
                    P.add_edge(3, x)
                    candidate_subgraphs.append(P)

        return candidate_subgraphs
    
    for edge_G in nx.generate_edgelist(G, data = False):
        for edge_H in nx.generate_edgelist(H, data = False):
            # 刪除 G 的一個 edge
            vec_G = [int(i) for i in edge_G.split()]
            G_bar = G.copy()
            G_bar.remove_edge(vec_G[0], vec_G[1])
            # 刪除 H 的一個 edge
            vec_H = [int(i) for i in edge_H.split()]
            H_bar = H.copy()
            H_bar.remove_edge(vec_H[0], vec_H[1])

            # 刪除尾端 edge
            if nx.number_of_isolates(G_bar) != 0 and nx.number_of_isolates(H_bar) != 0:
                # 刪除 G 孤立點
                name_G = nx.get_node_attributes(G_bar, "name")
                G_iso_pt = list(nx.isolates(G_bar))[0]
                b_name = name_G[G_iso_pt]
                b_id = G_iso_pt
                if G_iso_pt == vec_G[0]:
                    a_name = name_G[vec_G[1]]
                    a_id = vec_G[1]
                else:
                    a_name = name_G[vec_G[0]]
                    a_id = vec_G[0]
                G_bar.remove_node(G_iso_pt)
                
                # 刪除 H 孤立點
                name_H = nx.get_node_attributes(H_bar, "name")
                H_iso_pt = list(nx.isolates(H_bar))[0]
                d_name = name_H[H_iso_pt]
                d_id = H_iso_pt
                if H_iso_pt == vec_H[0]:
                    c_name = name_H[vec_H[1]]
                    c_id = vec_H[1]
                else:
                    c_name = name_H[vec_H[0]]
                    c_id = vec_H[0]
                H_bar.remove_node(H_iso_pt)

                # 同構判斷
                nm = isomorphism.categorical_node_match("name", "")
                if nx.is_isomorphic(G_bar, H_bar, node_match = nm):
                    G_copy = G.copy()
                    d_id = G_copy.number_of_nodes() + 1
                    c_id_list = list()
                    name_G_copy = nx.get_node_attributes(G_copy, "name")
                    for i in G_copy.nodes:
                        if name_G_copy[i] == c_name and i != b_id:
                            c_id_list.append(i)
                    # case 1: a != c
                    if a_name != c_name:
                        for c_id in c_id_list:
                            # (c->b)
                            if b_name == d_name:
                                P = G_copy.copy()
                                P.add_edge(c_id, b_id)
                                candidate_subgraphs.append(P)
                            # (c->d)
                            P = G_copy.copy()
                            P.add_node(d_id, name = d_name)
                            P.add_edge(c_id, d_id)
                            candidate_subgraphs.append(P)
                    # case 2: a == c
                    elif a_name == c_name:
                        # c->d
                        for c_id in c_id_list:
                            P = G_copy.copy()
                            P.add_node(d_id, name = d_name)
                            P.add_edge(c_id, d_id)
                            candidate_subgraphs.append(P)

    return candidate_subgraphs

def FO_job(t, epsilon, D):
    # encode
    B = encode(all_G[t].number_of_edges(), D)
    
    # perturb
    perturb_B = perturb(B, epsilon)

    return perturb_B

def FO(D, epsilon):
    results = list()
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes = cpu_count)
    
    for t, G in enumerate(all_G):
        result = pool.apply_async(FO_job, args = (t, epsilon,), kwds={'D' : D})
        results.append(result)
    
    pool.close()
    pool.join()
    
    responses = list()
    for result in results:
        responses.append(result.get())
    
    # aggregate
    estimate_c = aggregate(responses, epsilon)
    print("finishing FO...")

    return estimate_c

def estimate_Mg2(num_users, max_nodes, theta, epsilon1, labels, k):
    epsilon11 = epsilon1 / 2
    epsilon12 = epsilon1 / 2
    
    # calculus 1_subgraphs
    start = time.time()
    all_i_subgraphs = generate_all_1_subgraphs(max_nodes, labels)    
    end = time.time()
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("## generate_all_1_subgraphs 的執行時間：%f 秒" % (end - start))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    i = 1
    estimate_c, estimate_c_topk = PSFO(i, all_i_subgraphs, epsilon11 ,k)

    F_sub = list()
    for es in estimate_c_topk:
        if estimate_c[es] >= theta:
            F_sub.append(all_i_subgraphs[es])
    
    # calculus Mg
    D = range(1, math.comb(max_nodes, 2) + 1)
    C = FO(D, epsilon12)
    i = 1
    Mg = 1
    # theta_new
    p = 0.5
    q = 1 / (math.exp(epsilon12) + 1)
    max = (len(D) - len(D) * q) / (p - q)
    min = (0 - len(D) * q) / (p - q)
    theta_new = (theta * num_users * len(D) - min * len(D)) / (max - min)

    while i <= math.comb(max_nodes, 2):
        if i == 1:
            remain_users = num_users
        else:            
            remain_users = remain_users - C[i - 1]

        if remain_users >= theta_new:
            Mg = i
        else:
            break
        i = i + 1
    
    return Mg, F_sub

def PSFO_job(t, epsilon, subgraphs_C):
    # find the support graphs
    support_graphs = list()
    for id, sub_c in enumerate(subgraphs_C):
        GM = isomorphism.GraphMatcher(all_G[t], sub_c, node_match = lambda p,q:p['name']==q['name'])
        if GM.subgraph_is_isomorphic():
            support_graphs.append(id)

    # print(t, support_graphs)
    if len(support_graphs) > 0:
        # random_subgraph = choice(support_graphs)
        total_perturb_B = list()
        for sg in support_graphs:
            # encode
            B = encode(sg, subgraphs_C)

            # perturb
            perturb_B = perturb(B, epsilon)

            total_perturb_B.append(perturb_B)

        return total_perturb_B, len(support_graphs)
    
    elif len(support_graphs) == 0:
        total_perturb_B = list()
        perturb_B = np.zeros(len(subgraphs_C))
        total_perturb_B.append(perturb_B)
        
        return total_perturb_B, len(support_graphs)


def PSFO(i, subgraphs_C, epsilon, k):
    results = list()
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes = cpu_count)
    
    for t, G in enumerate(all_G):
        result = pool.apply_async(PSFO_job, args = (t, epsilon,), kwds={'subgraphs_C' : subgraphs_C})
        results.append(result)
    
    pool.close()
    pool.join()

    responses = list()
    count_l = 0
    for result in results:
        res = result.get()
        for r in res[0]:
            responses.append(r)
        
        count_l = count_l + res[1]
    
    global total_l
    if count_l > total_l:
        total_l = count_l
    
    # aggregate
    estimate_c = aggregate(responses, epsilon)
    estimate_c_topk = topk(estimate_c, k)

    return estimate_c, estimate_c_topk

def LDPFSM(num_users, max_nodes, theta, epsilon, labels, k, Mg):
    epsilon1 = epsilon / 2
    epsilon2 = epsilon / 2
    m, F = estimate_Mg2(num_users, max_nodes, theta, epsilon1, labels, k)

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("++ Mg: " + str(Mg))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
    i = 2
    all_i_subgraphs = generate_candidate_i_subgraphs(max_nodes, labels, i, F)
    while i <= Mg:

        if len(all_i_subgraphs) != 0:
            start = time.time()
            print("start PSFO !!! ")
            estimate_c, estimate_c_topk = PSFO(i, all_i_subgraphs, epsilon2 / (Mg - 1), k)
            
            F_sub = list()
            for es in estimate_c_topk:
                if estimate_c[es] >= theta:
                    F_sub.append(all_i_subgraphs[es])
            
            PSFO_time = time.time()
            
            for sub in F_sub:
                F.append(sub)
            print("######################################################################")
            print("## LDPFSM, i = " + str(i))
            print("## " + str(i) + "_subgraphs 的數量: " + str(len(F_sub)))
            print("## all_" + str(i) + "_subgraphs 的數量: " + str(len(all_i_subgraphs)))
            print("## PSFO 的執行時間：%f 秒" % (PSFO_time - start))

            i = i + 1

            all_i_subgraphs = generate_candidate_i_subgraphs(max_nodes, labels, i, F_sub)

            end = time.time()
            print("## generate_candidate_(i+1)_subgraphs 的執行時間：%f 秒" % (end - PSFO_time))
            print("######################################################################")
        else:
            break
        
    return F, i


def support_subgraphs_job(t, subgraphs_C):
    c = np.zeros(len(subgraphs_C))
    # print(t, c)
    for id, sub_c in enumerate(subgraphs_C):
        GM = isomorphism.GraphMatcher(all_G[t], sub_c, node_match = lambda p,q:p['name']==q['name'])
        if GM.subgraph_is_isomorphic():
            c[id] = c[id] + 1
    # print(t, c)

    return c

def support_subgraphs(i, subgraphs_C, k):
    results = list()
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes = cpu_count)
    
    for t, G in enumerate(all_G):
        result = pool.apply_async(support_subgraphs_job, args = (t,), kwds={'subgraphs_C' : subgraphs_C})
        results.append(result)
    
    pool.close()
    pool.join()      

    responses = list()
    for result in results:
        responses.append(result.get())
    aggregate_response = np.sum(responses, axis=0)
    c_topk = topk(aggregate_response, k)

    return aggregate_response, c_topk

def FSM(theta, labels, k, level):
    F = list()
    i = 1

    start = time.time()
    all_i_subgraphs = generate_all_1_subgraphs(max_nodes, labels)
    end = time.time()
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("## generate_all_1_subgraphs 的執行時間：%f 秒" % (end - start))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")

    while len(all_i_subgraphs) != 0 and i != level:
        start = time.time()

        # 計算每個的 support
        print("start support_subgraphs !!! ")
        c, c_topk = support_subgraphs(i, all_i_subgraphs, k)

        F_sub = list()
        for s in c_topk:
            if c[s] >= theta:
                F_sub.append(all_i_subgraphs[s])

        support_time = time.time()
        
        for sub in F_sub:
            F.append(sub)
        
        print("######################################################################")
        print("## FSM, i = " + str(i))
        print("## " + str(i) + "_subgraphs 的數量: " + str(len(F_sub)))
        print("## all_" + str(i) + "_subgraphs 的數量: " + str(len(all_i_subgraphs)))
        print("## support_subgraphs 的執行時間：%f 秒" % (support_time - start))

        i = i + 1

        all_i_subgraphs = generate_candidate_i_subgraphs(max_nodes, labels, i, F_sub)

        end = time.time()
        print("## generate_candidate_(i+1)_subgraphs 的執行時間：%f 秒" % (end - support_time))
        print("######################################################################")
        
    return F

def F1_score_graphs(pred_graphs, actual_graphs):
    f1_score = 0
    
    # 計算 precision, recall
    precision = 0
    recall = 0
    for pred in pred_graphs:
        for actual in actual_graphs:
            nm = isomorphism.categorical_node_match("name", "")
            if nx.is_isomorphic(pred, actual, node_match = nm):
                precision = precision + 1
                recall = recall + 1
                break
    print("intersection 的數量 : " + str(precision))
    precision = precision / len(pred_graphs)
    recall = recall / len(actual_graphs)

    if precision + recall != 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0

    return f1_score

def topk(encode_list, k):
    L = k
    if len(encode_list) <= k:
        L = len(encode_list)
    
    # 由大排到小
    encode_array = np.asarray(encode_list)
    encode_array_topk = encode_array.argsort()[-1:-1-L:-1]
    
    encode_list_topk = list()
    for index in encode_array_topk:
        encode_list_topk.append(index)
    
    return encode_list_topk

# generate the graph G
all_G, labels, max_nodes = read_graph("Cancer.dat")
total_l = 0

if __name__ == '__main__':
    
    # initial set
    epsilon = 10
    theta = 0.6
    k = 30 
    num_users = len(all_G)
    Mg = 20

    # calculus Mg
    # min_Mg = estimate_Mg(num_users, max_nodes, theta, epsilon / 2, labels)
    # for i in range(9):
    #     Mg = estimate_Mg(num_users, max_nodes, theta, epsilon / 2, labels)
    #     if Mg < min_Mg:
    #         min_Mg = Mg
    # Mg = min_Mg
    
    # calculus variance
    var = num_users * 4 * math.exp(0.5 * epsilon / 2 / (Mg-1)) / (math.exp(0.5 * epsilon / 2 / (Mg-1)) - 1) / (math.exp(0.5 * epsilon / 2 / (Mg-1)) - 1)

    # theta_FSM
    p = 0.5
    q = 1 / (math.exp(epsilon / 2 / Mg) + 1)
    max = (len(all_G) - len(all_G) * q) / (p - q)
    min = (0 - len(all_G) * q) / (p - q)
    theta_FSM = (theta * len(all_G) - min * len(all_G)) / (max - min)

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("++ epsilon_max : " + str(max))
    print("++ epsilon_min : " + str(min))
    print("++ theta_FSM : " + str(theta_FSM))
    print("++ Mg : " + str(Mg))    
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("++ epsilon: " + str(epsilon))
    print("++ theta: " + str(theta))
    print("++ top_k: " + str(k))
    print("++ labels: " + str(len(labels)))
    print("++ max_nodes: " + str(max_nodes))
    print("++ num_users: " + str(num_users))
    print("++ variance : " + str(var))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    # LDPFSM    
    print("Doing LDPFSM function...")
    LDPFSM_start = time.time()
    pred_graphs, level = LDPFSM(num_users, max_nodes, theta, epsilon, labels, k, Mg)
    LDPFSM_end = time.time()
    
    # FSM
    print("Doing FSM function...")
    FSM_start = time.time()
    actual_graphs = FSM(theta, labels, k, level)
    FSM_end = time.time()

    # calculus variance
    var = total_l * 4 * math.exp(0.5 * epsilon / 2 / (Mg-1)) / (math.exp(0.5 * epsilon / 2 / (Mg-1)) - 1) / (math.exp(0.5 * epsilon / 2 / (Mg-1)) - 1)
    
    print("######################################################################")
    print("## LDPFSM 的執行時間：%f 秒" % (LDPFSM_end - LDPFSM_start))
    print("## FSM 的執行時間：%f 秒" % (FSM_end - FSM_start))
    print("######################################################################")
    
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("++ Mg : " + str(Mg))    
    print("++ epsilon: " + str(epsilon))
    print("++ theta: " + str(theta))
    print("++ top_k: " + str(k))
    print("++ labels: " + str(len(labels)))
    print("++ max_nodes: " + str(max_nodes))
    print("++ num_users: " + str(num_users))
    print("++ variance : " + str(var))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    # accuracy
    accuracy = F1_score_graphs(pred_graphs, actual_graphs)

    print("######################################################################")
    print("numbers of FSM: " + str(len(actual_graphs)))
    print("numbers of LDPFSM: " + str(len(pred_graphs)))
    print("accuracy: " + str(accuracy))    
    print("######################################################################")

