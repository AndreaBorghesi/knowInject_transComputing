'''
Parse json file containing info on the structure of the benchmark and extract
the dependency graph

Andrea Borghesi
    University of Bologna
    2019-05-31
'''
#!/usr/bin/python3.6

import os
import numpy
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import networkx as nx

data_dir = './data'

def parse(graph, l, level=0):
    #print('line ' + str(l))
    if l == None:
        return 'P', [], None
    op = l[0]
    #print('op ' + str(op))
    if op == 'A':
        level = l[1]
        cond_path = l[2]
        v_idx = l[3]
        exp = l[4]
        lhs = exp[0]
        rhs = exp[1]
        nodes, top_node = parse_assignment(graph, v_idx, lhs, rhs, level)
    elif op == 'R':
        level = l[1]
        cond_path = l[2]
        op_type = l[3]
        v_idx = l[4]
        exp = l[5]
        lhs = exp[0]
        rhs = exp[1]
        nodes, top_node = parse_conditional_exp(graph, op_type, v_idx, lhs, rhs, 
                level)
    elif op == 'P':
        prim_type = l[1]
        nodes, top_node = parse_primitive(graph, prim_type, level)
    elif op == 'V':
        v_idx = l[1]
        nodes, top_node = parse_var(graph, v_idx, level)
    elif op == 'E':
        op_type = l[1]
        v_idx = l[2]
        lhs = l[3]
        if len(l) == 5:
            rhs = l[4]
        else:
            rhs = None
        nodes, top_node = parse_exp(graph, op_type, v_idx, lhs, rhs, level)
    elif op == 'T':
        v_idx = l[1]
        content = l[2]
        nodes, top_node = parse_temp(graph, v_idx, content, level)
    elif op == 'F':
        params = l[1]
        nodes, top_node = parse_func(graph, params, level)
    elif op == 'C':
        const_type = l[1]
        value = l[2]
        nodes, top_node = parse_const(graph, const_type, value, level)
    else:
        if len(l) == 1:  # it's a variable
            #print('\tVAR inside')
            #print('\t{}'.format(l[0]))
            _, nodes, top_node = parse(graph, l[0])
            #print('\tnodes {}'.format(nodes))
            op = 'VV'
        else:
            #print('Unexpected op {}'.format(op))
            nodes = []
            top_node = None
    return op, nodes, top_node

def parse_assignment(graph, v_idx, lhs, rhs, level):
    #print('>>>>>> PARSE ASS <<<<<<<')
    res_node = 'v{}'.format(v_idx)
    res_node_t = 't{}'.format(v_idx)
    lop, lnodes, ltop_node = parse(graph, lhs, level)
    rop, rnodes, rtop_node = parse(graph, rhs, level)
    #print('lop {} - lnodes {}'.format(lop, lnodes))
    #print('rop {} - rnodes {}'.format(rop, rnodes))
    #print('res_node {}'.format(res_node))
    nodes = lnodes
    nodes.extend(rnodes)
    nodes = list(set(nodes))
    if res_node_t in nodes:
        res_node = res_node_t
    if ltop_node != res_node and ltop_node != res_node_t:
        if ltop_node != None:
            #if(ltop_node == 't21'):
            #    print('--> Adding node from {} L'.format(ltop_node))
            graph.add_edge(ltop_node, res_node, weight=level)
    if rtop_node != res_node and rtop_node != res_node_t:
        if rtop_node != None:
            #if(rtop_node == 't21'):
            #    print('--> Adding node from {} R'.format(rtop_node))
            graph.add_edge(rtop_node, res_node, weight=level)
    return nodes, res_node

def parse_conditional_exp(graph, op_type, v_idx, lhs, rhs, level):
    node = 'v{}'.format(v_idx)
    return [], node

def parse_primitive(graph, prim_type, level):
    node = None
    return [], node

def parse_var(graph, v_idx, level):
    #print('>>>>>> PARSE VAR <<<<<<<')
    node = 'v{}'.format(v_idx)
    if node not in graph.nodes():
        #print('parse var adding node {}'.format(node))
        graph.add_node(node)
    return [node], node

def parse_exp(graph, op_type, v_idx, lhs, rhs, level):
    #print('>>>>>> PARSE EXP <<<<<<<')
    nodes = []
    lop, lnodes, ltop_node = parse(graph, lhs, level)
    rop, rnodes, rtop_node = parse(graph, rhs, level)
    nodes.extend(lnodes)
    nodes.extend(rnodes)
    nodes = list(set(nodes))
    res_node = 'v{}'.format(v_idx)
    res_node_t = 't{}'.format(v_idx)
    #print('Graph nodes {}'.format(graph.nodes))
    #print('internal nodes {}'.format(nodes))
    if res_node not in nodes and res_node_t not in nodes:
        nodes.append(res_node)
        #print('parse exp adding node {}'.format(res_node))
        graph.add_node(res_node)
    #print('lop {}'.format(lop))
    #if lop == 'V' or lop == 'T':
    #    print('\tlnodes {}'.format(lnodes))
    #    print('\tres node {}'.format(res_node))
    #    for n in lnodes:
    #        graph.add_edge(n, res_node)
    #print('rop {}'.format(rop))
    #if rop == 'V':
    #    for n in rnodes:
    #        graph.add_edge(n, res_node)
    #print(ltop_node)
    #print(rtop_node)
    #print(res_node_t)
    if res_node_t in nodes:
        res_node = res_node_t

    if ltop_node != res_node and ltop_node != res_node_t:
        if ltop_node != None:
            graph.add_edge(ltop_node, res_node, weight=level)
            #if(ltop_node == 't21'):
            #    print('--> Adding node from {} L'.format(ltop_node))
    if rtop_node != res_node and rtop_node != res_node_t:
        if rtop_node != None:
            graph.add_edge(rtop_node, res_node, weight=level)
            #if(rtop_node == 't21'):
            #    print('--> Adding node from {} R'.format(rtop_node))

    return nodes, res_node

def parse_temp(graph, v_idx, content, level):
    #print('>>>>>> PARSE TEMP <<<<<<<')
    node = 't{}'.format(v_idx)
    if node not in graph.nodes():
        #print('parse temp adding node {}'.format(node))
        graph.add_node(node)
    nodes = [node]
    #print('nodes {}'.format(nodes))
    cop, cnodes, top_node = parse(graph, content, level)
    nodes.extend(cnodes)
    #print('nodes {}'.format(nodes))
    if cop == 'V' or cop == 'VV' or cop == 'F' or cop == 'E':
        for n in cnodes:
            graph.add_edge(n, node, weight=level)
            nodes.remove(n)
            #if(n == 't21'):
            #    print('--> Adding node from {} to {}'.format(n, node))
    #print('cop {}'.format(cop))
    #print('cnodes {}'.format(cnodes))
    nodes = list(set(nodes))
    return nodes, node

def parse_func(graph, params, level):
    pop, pnodes, top_node = parse(graph, params, level)
    return pnodes, top_node

def parse_const(graph, const_type, value, level):
    return [], None

def parse_graph(benchmark):
    json_file = data_dir + '/' + benchmark + '/program_vardeps.json' 
    with open(json_file) as jfile:  
            data = json.load(jfile)
    G = nx.DiGraph()
    for l in data:
        parse(G, l)
    return G

def plot_graph(G, benchmark):
    fig = plt.figure()
    if benchmark == 'dwt':
        nx.draw(G, with_labels=True, node_size=500, alpha=.5, 
                font_weight='bold')
    elif benchmark == 'BlackScholes':
        nx.draw(G, with_labels=True, node_size=500, alpha=.5, 
                font_weight='bold')
    elif benchmark == 'Jacobi':
        nx.draw(G, with_labels=True, node_size=500, alpha=.5, 
                font_weight='bold')
    else:
        nx.draw_kamada_kawai(G, with_labels=True, node_size=500, alpha=.5, 
                font_weight='bold')
    plt.show()

def plot_graph_weightedEdges(G, benchmark):
    fig = plt.figure()
    pos = nx.layout.spring_layout(G)
    M = G.number_of_edges()
    edge_colors = range(2, M + 2)
    labels = {}    
    for node in G.nodes():
        labels[node] = node
    node_sizes = [400] * G.number_of_nodes()
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
            node_color='green', alpha=.5)
    nx.draw_networkx_labels(G, pos, labels, font_weight='bold')
    edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes,
            arrowstyle='->', arrowsize=10, edge_color=edge_colors,
            edge_cmap=plt.cm.Blues, width=2)
    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])
    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    pc.set_array(edge_colors)
    plt.colorbar(pc)
    ax = plt.gca()
    ax.set_axis_off()
    plt.show()

'''
Extract binary variable relations from the dependency graphs of the type
less-or-equal
The function returns a list composed by tuples in the form (var1, var2),
where var1 and var2 are the variables involved 
- e.g. V1 = V2+V3 ~= V1 = T4(V2) + T4(V3) --> T4 <= V1 (if T4 is not used in
  other other expressions, it would make no sense to allocate to it more bits
  than the expression result)
IN: only_temp set to true specifies that only relations involving temporary
variables must be returned 
OUT: the relations list
'''
def get_binary_leq_rels(graph, only_temp):
    rels = []
    for n in graph.nodes:
        for nn in graph.successors(n):
            if not only_temp:
                rels.append((n,nn))
            else:
                if 't' in n or 't' in nn:
                    rels.append((n,nn))
    return rels

'''
Extract temporary cast due to expression relations from the dependency graphs
- these relations state that the precision of a temporary variable introduced to
  handle two (or more) operands with different precision, must be equal to the
  minimum precision of the two operands 
- e.g. V1 = V2+V3 ~= V1 = T4(V2) + T4(V3) --> T4 = min(V2, V3)
The function returns a list composed by tuples in the form 
(var1, [var2, var3, .., varN]), denoting relations such as: 
var1 = min(var2, var3, .., varN)
'''
def get_cast_exps_rels(graph):
    rels = []
    for n in graph.nodes:
        if 't' in n and len(list(graph.predecessors(n))) > 1:
            rels.append((n, list(graph.predecessors(n))))
    return rels

'''
Parse the information related to the input benchmark and obtain the dependency
graph. Then, extract and return variables relationships
'''
def get_relations(benchmark, binRel_onlyTemp):
    G = parse_graph(benchmark)
    bin_rels = get_binary_leq_rels(G, binRel_onlyTemp)
    cast_expr_rels = get_cast_exps_rels(G)
    return bin_rels, cast_expr_rels


def getAdditionalFeatures(benchmark):
    add_feat=[]
    G = parse_graph(benchmark)
    for var in G.edges():
        add_feat.append(("var_{}".format(int(''.join(filter(str.isdigit, var[1])))), "var_{}".format(int(''.join(filter(str.isdigit, var[0]))))))
    return add_feat

def getAdjacencyMatrix(benchmark):
    G = parse_graph(benchmark)
    
    #node name remapping (0, 1, 2... instead of v0, v1, t2...)
    mapping = {}
    for node in G.nodes():
        mapping[node]=node[1]
    H = nx.relabel_nodes(G, mapping)
    
    adjacent_matrix = []
    for node in sorted(H.nodes()):
        row = listofzeros = [0] * len(H)
        for edge in H.edges(node):
            row[int(edge[1])]=1
        adjacent_matrix.append(row)
    return adjacent_matrix
    
def main(argv):
    benchmark = argv[0]
    G = parse_graph(benchmark)
    bin_rels = get_binary_leq_rels(G, False)
    bin_rels_onlyT = get_binary_leq_rels(G, True)
    cast_expr_rels = get_cast_exps_rels(G)
    #print(bin_rels)
    #print(bin_rels_onlyT)
    #print(cast_expr_rels)
    plot_graph(G, benchmark)
    #plot_graph_weightedEdges(G, benchmark)

if __name__ == '__main__':
    main(sys.argv[1:])
