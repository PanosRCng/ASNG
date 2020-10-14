import itertools
import random

import numpy as np
import networkx as nx
from sklearn.cluster import AgglomerativeClustering



RANDOM_SCALE_FREE_GRAPH_NEW_NODE_EDGES = 3




class ASNG:


    def __init__(self):
        self.__G = nx.Graph()
        self.__cliques = {}
        self.__node_merges = {}




    def generate_graph(self, cliques_min_size, cliques_max_size, number_of_cliques):

        self.__cliques = self.__generate_cliques(min_size=cliques_min_size, max_size=cliques_max_size, k=number_of_cliques)

        self.__impose_scale_free_degree_distribution()

        cliques_merges = self.__cliques_merges()

        ClusterTree = self.__generate_cluster_tree()

        probability_vectors = self.__probability_vectors(ClusterTree)

        pairwise_merges = self.__pairwise_merges(cliques_merges, probability_vectors)

        self.__add_tree_to_graph(ClusterTree)

        self.__merge_cliques(pairwise_merges)

        return self.__G



    def __generate_cliques(self, min_size, max_size, k):

        cliques = {}

        for i in range(k):

            cliques[i] = []

            clique = self.__generate_clique( random.randint(min_size, max_size) )

            for node in nx.nodes(clique):
                clique.nodes[node]['clique'] = i

            self.__G = nx.union(self.__G, clique, rename=('A-', 'B-'))

        self.__G = nx.convert_node_labels_to_integers(self.__G)

        for node in nx.nodes(self.__G):
            cliques[self.__G.nodes[node]['clique']].append(node)

        return cliques


    def __impose_scale_free_degree_distribution(self):

        n = nx.number_of_nodes(self.__G)

        random_scale_free_G = nx.barabasi_albert_graph(n, m=RANDOM_SCALE_FREE_GRAPH_NEW_NODE_EDGES)

        for g_node, r_node in self.__associate_nodes(self.__G, random_scale_free_G).items():

            if self.__G.degree[g_node] < random_scale_free_G.degree[r_node]:
                self.__G.nodes[g_node]['sfDeg'] = random_scale_free_G.degree[r_node]
            else:
                self.__G.nodes[g_node]['sfDeg'] = self.__G.degree[g_node]


    def __generate_clique(self, size):

        G = nx.empty_graph(size)

        if size > 1:

            edges = itertools.combinations(range(0, size), 2)

            G.add_edges_from(edges)

        return G


    def __associate_nodes(self, graph_a, graph_b):

        associations = {}

        a_nodes = list(graph_a.nodes)
        b_nodes = list(graph_b.nodes)

        used_a = []
        used_b = []

        while len(associations) < len(a_nodes):

            pick_a = random.choice(a_nodes)
            pick_b = random.choice(b_nodes)

            if (pick_a in used_a) or (pick_b in used_b):
                continue

            associations[pick_a] = pick_b
            used_a.append(pick_a)
            used_b.append(pick_b)

        return associations


    def __cliques_merges(self):

        clique_merges = {}

        self.__node_merges = self.__calculate_node_merges()

        for clique, nodes in self.__cliques.items():
            clique_merges[clique] = sum( [self.__node_merges[node] for node in nodes] )

        return clique_merges


    def __generate_cluster_tree(self):

        ClusterTree =  nx.empty_graph()

        points = np.random.rand(len(self.__cliques), 2)

        model = AgglomerativeClustering(affinity='euclidean', linkage='complete', distance_threshold=0, n_clusters=None, compute_full_tree=True)
        model = model.fit(points)

        links = dict(enumerate(model.children_, model.n_leaves_))
        root_node = max(links.keys())

        for node in range(root_node, -1, -1):
            ClusterTree.add_node(node)

        for node in range(root_node, -1, -1):

            if node not in links.keys():
                continue

            for ch_node in links[node]:
                ClusterTree.add_edge(node, ch_node)

        return ClusterTree


    def __probability_vectors(self, ClusterTree):

        probability_vectors = []

        for src_clique in self.__cliques:

            prob_vector = []

            for dest_clique in self.__cliques:

                if src_clique == dest_clique:
                    prob_vector.append(0)
                    continue

                prob_vector.append( self.__merge_probability(ClusterTree, src_node=src_clique, dest_node=dest_clique) )

            probability_vectors.append(prob_vector)

        return probability_vectors


    def __pairwise_merges(self, cliques_merges, probability_vectors):

        pairwise_merges = []

        for p in probability_vectors:

            pv = [int(merges * p[clique]) for clique, merges in cliques_merges.items()]

            pairwise_merges.append(pv)

        return pairwise_merges


    def __add_tree_to_graph(self, ClusterTree):

        n_offset = nx.number_of_nodes(self.__G) - len(self.__cliques)

        for node in nx.nodes(ClusterTree):

            if node < len(self.__cliques):
                attach_node = random.choice(self.__cliques[node])
            else:
                attach_node = node + n_offset
                self.__G.add_node(attach_node)

            for n_node in nx.neighbors(ClusterTree, node):

                if n_node < len(self.__cliques):
                    continue

                self.__G.add_edge(attach_node, n_node + n_offset)


    def __merge_cliques(self, pairwise_merges):

        for clique_a in self.__cliques:
            for clique_b in self.__cliques:

                if clique_a is clique_b:
                    continue

                while pairwise_merges[clique_a][clique_b] > 0:

                    pairwise_merges[clique_a][clique_b] += -1

                    self.__merge(clique_a, clique_b)



    def __merge(self, clique_a, clique_b):

        n_u = self.__select_one_node_rand(clique_a)

        if n_u is None:
            return

        self.__node_merges[n_u] += - 1

        n_v = self.__select_one_node_prob(clique_b)

        if n_v is None:
            return

        self.__merging_nodes(n_u, n_v)


    def __calculate_node_merges(self):

        node_merges = {}

        avg_node_degree = self.__average_node_degree()

        for node in nx.nodes(self.__G):
            node_merges[node] = int(self.__G.nodes[node]['sfDeg'] / avg_node_degree)

        return node_merges


    def __average_node_degree(self):

        number_of_nodes = nx.number_of_nodes(self.__G)

        if number_of_nodes == 0:
            return 0

        sum_degree = sum( [self.__G.degree[node] for node in nx.nodes(self.__G)] )

        return int( sum_degree / number_of_nodes )


    def __merge_probability(self, ClusterTree, src_node, dest_node):

        prob = 1.0

        for node in nx.shortest_path(ClusterTree, source=src_node, target=dest_node)[1:]:

            forward_branches = ClusterTree.degree[node]

            prob = prob * (1 / forward_branches)

        return round(prob, 2)


    def __select_one_node_rand(self, clique):

        to_select = [node for node in self.__cliques[clique] if self.__node_merges[node] > 0]

        if len(to_select) == 0:
            return None

        return random.choice(to_select)


    def __select_one_node_prob(self, clique):

        to_select = [node for node in self.__cliques[clique]]

        if len(to_select) == 0:
            return None

        sfDegs = [self.__G.nodes[node]['sfDeg'] for node in to_select]

        max_index = sfDegs.index(max(sfDegs))

        return to_select[max_index]


    def __merging_nodes(self, node_a, node_b):

        neighbors = [nn for nn in nx.neighbors(self.__G, node_b)]

        edges = [(node_a, neighbor) for neighbor in set(neighbors)]
        self.__G.add_edges_from(edges)

        return


