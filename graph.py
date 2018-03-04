import sys
import math
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, sparse
import networkx as nx

def gen_powerlaw_graph(density, clustering):

    sources = {'NBC': -1, 'CNN': -1, 'Fox': 1, 'ABC': -0.5, 'NPR': -0.2, 'Brietbart': 1.8}

    n = 200
    m = int(n * density)
    p = clustering
    
    graph = nx.powerlaw_cluster_graph(n, m, p)

    #rank = nx.pagerank(graph)

    scepticism = np.random.normal(0, 1, n)

    matrix = nx.to_scipy_sparse_matrix(graph)

    data = {}
    data['nodes'] = [{'id': i, 'scept': scepticism[i]} for i in range(n)]
    #data['nodes'] += [{'id': name, 'group': 0.01, 'part': part} for name, part in sources.items()]
    data['links'] = []

    row, col = matrix.nonzero()

    for i in range(len(row)):
        data['links'].append({"source": int(row[i]), "target": int(col[i])})

#    for i in range(n):
#        for name, part in sources.items():
#            if 50 * rank[i] > np.random.rand():
#                data['links'].append({"source": name, "target": i, "value": 1})

    print("data_{}_{}.json".format(round(density, 4), round(clustering, 4)))
    with open("data_{}_{}.json".format(round(density, 4), round(clustering, 4)), "w") as f:
        f.write(json.dumps(data))



def generate_graph(std_dev):
    
    population = 250
    num_outlets = 10

    dist = np.random.normal(0, std_dev, population)

    def prob_edge(p1, p2):

        # mutual partisanship
        mut_part = abs(p1 - p2) / 2
        # probability that two vertices share an edge
        center, end = min(p1, p2), max(p1, p2)
        prob = 1 - stats.norm.cdf(end, loc=center, scale=1/(mut_part**2))
        return (prob > np.random.rand() * 5, mut_part)

    rows, cols, v = [], [], []

    data = {}
    data['nodes'] = [{'id': i, 'group': 1, 'part': part} for i, part in enumerate(dist)]
    data['links'] = []

    for i in range(population):
        for j in range(i + 1, population): 
            edge, length = prob_edge(dist[i], dist[j])
            if edge:
                rows.append(i)
                cols.append(j)
                v.append(length)
                data['links'].append({"source": i, "target": j, "value": length})


    with open("data.json", "w") as f:
        f.write(json.dumps(data))

#    matrix = sparse.coo_matrix((v, (rows, cols)), shape=(population, population))

#    graph = nx.Graph(matrix)
#    nx.draw_spectral(graph)
#    plt.savefig("graph.png")
#
#
#    prob_edge(dist[0], dist[1])

if __name__ == '__main__':
    for i in np.arange(0.01, 0.0225, 0.0025):
        for j in np.arange(0.5, 1, 0.1):
            gen_powerlaw_graph(i, j)
            #gen_powerlaw_graph(float(sys.argv[1]), float(sys.argv[2]))

    #generate_graph(int(sys.argv[1]))