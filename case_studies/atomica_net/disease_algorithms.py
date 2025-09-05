import scipy
import numpy as np
from collections import defaultdict
import networkx as nx
from tqdm import tqdm
from scipy import sparse

def neighborhood_approach(G, seed_genes):
    """
    Implement the Neighborhood approach for predicting protein annotations.
    
    Parameters:
    G (nx.Graph): The protein-protein interaction network
    seed_genes (set): Set of genes known to be associated with the disease
    theta (float): Threshold for prediction (default: 0.5)
    
    Returns:
    set: Set of proteins predicted to be associated with the disease
    """
    
    predicted_proteins = dict()
    
    for protein in G.nodes():
        if protein not in seed_genes:
            neighbors = set(G.neighbors(protein))
            if neighbors:
                disease_associated_neighbors = neighbors.intersection(seed_genes)
                predicted_proteins[protein] = len(disease_associated_neighbors) / len(neighbors)
    
    return predicted_proteins

def random_walk_with_restart(G, A, seed_genes, r=0.75, tol=1e-6, max_iter=1000):
    """
    Perform Random Walk with Restart on a network.
    
    Parameters:
    G (nx.Graph): The input graph
    A (np.ndarray): Adjacency matrix of the PPI network
    seed_genes (list): List of seed genes (starting points for the random walk)
    r (float): Probability of restarting the walk
    tol (float): Convergence tolerance
    max_iter (int): Maximum number of iterations
    
    Returns:
    numpy.ndarray: Steady-state probabilities for each node
    """
    # Convert NetworkX graph to SciPy sparse matrix
    # A = nx.to_scipy_sparse_matrix(G, format='csr')
    A = sparse.csr_matrix(A)
    
    # Normalize the adjacency matrix
    D_inv = sparse.diags(1.0 / A.sum(axis=1).A1)
    W = D_inv @ A
    
    # Initialize the probability vector
    n = A.shape[0]
    p0 = np.zeros(n)
    for gene in seed_genes:
        if gene in G:
            p0[list(G.nodes()).index(gene)] = 1 / len(seed_genes)
    
    # Perform the random walk
    p = p0.copy()
    for _ in range(max_iter):
        p_next = (1 - r) * (W @ p) + r * p0
        if np.linalg.norm(p_next - p) < tol:
            return p_next
        p = p_next
    
    return p

# def random_walk(G, A, seed_genes, r=0.75, theta=0.01, max_iter=1000, tol=1e-6):
#     """
#     Perform random walk with restart on a PPI network.
    
#     Parameters:
#     G (nx.Graph): The PPI network
#     A (np.ndarray): Adjacency matrix of the PPI network
#     seed_genes (list): List of seed genes (starting points for the random walk)
#     r (float): Probability of returning to the seed genes
#     theta (float): Threshold for prediction
#     max_iter (int): Maximum number of iterations
#     tol (float): Convergence tolerance
    
#     Returns:
#     dict: Dictionary of genes and their visitation probabilities
#     """
    
#     # Initialize probability vector
#     print("Initializing probability vector...")
#     p0 = np.zeros(len(G))
#     for gene in seed_genes:
#         if gene in G:
#             p0[list(G.nodes()).index(gene)] = 1 / len(seed_genes)

#     # Normalize initial vector
#     p0 = p0 / np.sum(p0)
    
#     # Get adjacency matrix
#     print("Getting adjacency matrix...")
#     # A = nx.to_numpy_array(G)
    
#     # Normalize adjacency matrix
#     print("Normalizing adjacency matrix...")
#     D = np.diag(1 / np.sum(A, axis=1))
#     W = D.dot(A)
    
#     # Perform random walk
#     print("Performing random walk...")
#     p_prev = p0
#     for _ in tqdm(range(max_iter), total=max_iter, desc='Random Walk'):
#         p_next = (1 - r) * W.dot(p_prev) + r * p0
        
#         # Check for convergence
#         if np.linalg.norm(p_next - p_prev) < tol:
#             break
        
#         p_prev = p_next
    
#     # Get results
#     results = dict(zip(G.nodes(), p_next))
    
#     # Filter results based on theta
#     predictions = {gene: prob for gene, prob in results.items() if prob > theta}
    
#     return predictions

# ================================================================================
def compute_all_gamma_ln(N):
    """
    precomputes all logarithmic gammas 
    """
    gamma_ln = {}
    for i in range(1, N + 1):
        gamma_ln[i] = scipy.special.gammaln(i)

    return gamma_ln

# =============================================================================
def logchoose(n, k, gamma_ln):
    if n - k + 1 <= 0:
        return scipy.inf
    lgn1 = gamma_ln[n + 1]                                                      
    lgk1 = gamma_ln[k + 1]                                                      
    lgnk1 = gamma_ln[n - k + 1]                                                    
    return lgn1 - (lgnk1 + lgk1)

# =============================================================================
def gauss_hypergeom(x, r, b, n, gamma_ln):
    return np.exp(logchoose(r, x, gamma_ln) +
                  logchoose(b, n - x, gamma_ln) -
                  logchoose(r + b, n, gamma_ln))

# =============================================================================
def pvalue(kb, k, N, s, gamma_ln):
    """                                                                        
    -------------------------------------------------------------------        
    Computes the p-value for a node that has kb out of k links to              
    seeds, given that there's a total of s seeds in a network of N nodes.       
                                                                               
    p-val = \sum_{n=kb}^{k} HypergemetricPDF(n,k,N,s)                          
    -------------------------------------------------------------------        
    """                                                                        
    p = 0.0                                                                    
    for n in range(kb, k + 1):
        if n > s:
            break
        prob = gauss_hypergeom(n, s, N - s, k, gamma_ln)
        # print(prob)                                                           
        p += prob

    return min(p, 1)

# =============================================================================
def get_neighbors_and_degrees(G):
    neighbors, all_degrees = {}, {}
    for node in G.nodes():
        nn = set(G.neighbors(node))
        neighbors[node] = nn
        all_degrees[node] = G.degree(node)

    return neighbors, all_degrees

# =============================================================================
# Reduce number of calculations
# =============================================================================
def reduce_not_in_cluster_nodes(all_degrees, neighbors, G, not_in_cluster, cluster_nodes, alpha): 
    reduced_not_in_cluster = {}                                                        
    kb2k = defaultdict(dict)
    for node in not_in_cluster:
        k = all_degrees[node]                                                          
        kb = 0                                                                         
        # Going through all neighbors and counting the number of module neighbors
        for neighbor in neighbors[node]:
            if neighbor in cluster_nodes:
                kb += 1
        
        # Adding weights to the edges connected to seeds
        k += (alpha - 1) * kb
        kb += (alpha - 1) * kb
        kb2k[kb][k] = node

    # Going to choose the node with the largest kb, given k
    k2kb = defaultdict(dict)
    for kb, k2node in kb2k.items():
        min_k = min(k2node.keys())
        node = k2node[min_k]
        k2kb[min_k][kb] = node
                                                                                       
    for k, kb2node in k2kb.items():
        max_kb = max(kb2node.keys())
        node = kb2node[max_kb]
        reduced_not_in_cluster[node] = (max_kb, k)

    return reduced_not_in_cluster

#======================================================================================
#   C O R E    A L G O R I T H M
#======================================================================================
def diamond_iteration_of_first_X_nodes(G, S, X, alpha):
    
    """
    Parameters:                                                                     
    ----------                                                                      
    - G:     graph
    - S:     seeds 
    - X:     the number of iterations, i.e only the first X genes will be pulled in
    - alpha: seeds weight

    Returns:                                                                        
    --------
    
    - added_nodes: ordered list of nodes in the order by which they
      are agglomerated. Each entry has 4 info:

      * name : dito
      * k    : degree of the node
      * kb   : number of +1 neighbors
      * p    : p-value at agglomeration
    """
    
    N = G.number_of_nodes()

    added_nodes = []

    # ------------------------------------------------------------------
    # Setting up dictionaries with all neighbor lists
    # and all degrees
    # ------------------------------------------------------------------
    neighbors, all_degrees = get_neighbors_and_degrees(G)

    # ------------------------------------------------------------------
    # Setting up the initial set of nodes in the cluster
    # ------------------------------------------------------------------
    
    cluster_nodes = set(S)
    not_in_cluster = set()
    s0 = len(cluster_nodes)
    
    s0 += (alpha - 1) * s0
    N += (alpha - 1) * s0
    
    # ------------------------------------------------------------------
    # Precompute the logarithmic gamma functions
    # ------------------------------------------------------------------
    gamma_ln = compute_all_gamma_ln(N + 1)
    
    # ------------------------------------------------------------------
    # Setting the initial set of nodes not in the cluster
    # ------------------------------------------------------------------
    for node in cluster_nodes:
        not_in_cluster |= neighbors[node]
    not_in_cluster -= cluster_nodes

    # ------------------------------------------------------------------
    #
    # M A I N     L O O P 
    #
    # ------------------------------------------------------------------

    all_p = {}

    while len(added_nodes) < X:    

        # ------------------------------------------------------------------
        #
        # Going through all nodes that are not in the cluster yet and
        # record k, kb, and p 
        #
        # ------------------------------------------------------------------ 
        info = {}
    
        pmin = 10
        next_node = None
        reduced_not_in_cluster = reduce_not_in_cluster_nodes(all_degrees,
                                                             neighbors, G,
                                                             not_in_cluster,
                                                             cluster_nodes, alpha)
        
        for node, kbk in reduced_not_in_cluster.items():
            # Getting the p-value of this kb, k combination and save it in all_p, so computing it only once!
            kb, k = kbk
            if (k, kb, s0) not in all_p:
                all_p[(k, kb, s0)] = pvalue(kb, k, N, s0, gamma_ln)

            p = all_p[(k, kb, s0)]

            # recording the node with smallest p-value
            if p < pmin:
                pmin = p
                next_node = node
    
            info[node] = (k, kb, p)

        # ---------------------------------------------------------------------
        # Adding the node with the smallest p-value to the list of agglomerated nodes
        # --------------------------------------------------------------------- 
        if next_node is None:
            break

        added_nodes.append((next_node,
                            info[next_node][0],
                            info[next_node][1],
                            info[next_node][2]))

        # Updating the list of cluster nodes and s0
        cluster_nodes.add(next_node)
        s0 = len(cluster_nodes)
        not_in_cluster |= (neighbors[next_node] - cluster_nodes)
        not_in_cluster.remove(next_node)

    return added_nodes

# ===========================================================================
#
#   M A I N    D I A M O n D    A L G O R I T H M
# 
# ===========================================================================
def DIAMOnD(G_original, seed_genes, max_number_of_added_nodes, alpha, outfile=None):
    """
    Runs the DIAMOnD algorithm

    Input:
    ------
     - G_original :
             The network
     - seed_genes : 
             a set of seed genes 
     - max_number_of_added_nodes:
             after how many added nodes should the algorithm stop
     - alpha:
             given weight to the seeds
     - outfile:
             filename for the output generated by the algorithm,
             if not given the program will name it 'first_x_added_nodes.txt'

     Returns:
     --------
      - added_nodes: A list with 4 entries at each element:
            * name : name of the node
            * k    : degree of the node
            * kb   : number of neighbors that are part of the module (at agglomeration)
            * p    : connectivity p-value at agglomeration
    """
    
    # 1. Throwing away the seed genes that are not in the network
    all_genes_in_network = set(G_original.nodes())
    seed_genes = set(seed_genes)
    disease_genes = seed_genes & all_genes_in_network

    if len(disease_genes) != len(seed_genes):
        print(f"DIAMOnD(): ignoring {len(seed_genes - all_genes_in_network)} of {len(seed_genes)} seed genes that are not in the network")
 
    # 2. Agglomeration algorithm. 
    added_nodes = diamond_iteration_of_first_X_nodes(G_original,
                                                     disease_genes,
                                                     max_number_of_added_nodes, alpha)
    # 3. Saving the results
    if outfile is not None:
        with open(outfile, 'w') as fout:
            fout.write('#rank\tDIAMOnD_node\n')
            rank = 0
            for DIAMOnD_node_info in added_nodes:
                rank += 1
                DIAMOnD_node = DIAMOnD_node_info[0]
                fout.write(f'{rank}\t{DIAMOnD_node}\n')

    return added_nodes