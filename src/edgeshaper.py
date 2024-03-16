### Module implementing EdgeSHAPer algorithm ###
### Author: Andrea Mastropietro © All rights reserved ###

import torch
import torch.nn.functional as F

import numpy as np
from numpy.random import default_rng

import matplotlib.pyplot as plt
# from tqdm import tqdm
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Draw
from src.rdkit_heatmaps import mapvalues2mol
from src.rdkit_heatmaps.utils import transform2png

###EdgeSHAPer as a class###

class Edgeshaper():
    """ EdgeSHAPer class for Shaplay value approximation for edge importance in GNNs
        
            P (float, optional): probablity of an edge to exist in the random graph. Defaults to the original graph density.
            M (int): number of Monte Carlo sampling steps to perform.
            target_class (int, optional): index of the class prediction to explain. Defaults to class 0. None if the model is a regression model.
            deviation (float, optional): deviation gap from sum of Shapley values and predicted output prob. If ```None```, M sampling
                steps are computed, otherwise the procedure stops when the desired approxiamtion ```deviation``` is reached. However, after M steps
                the procedure always terminates.
            log_odds (bool, optional). Default is ```False```. If ```True```, use log odds instead of probabilty as Value for Shaply values approximation.
            seed (float, optional): seed used for the random number generator.
            device (string, optional): states if using ```cuda``` or ```cpu```.
        Returns:
                list: list of Shapley values for the edges computed by EdgeSHAPer. The order of the edges is the same as in ```E```.
    """

    def __init__(self, model, x, edge_index, edge_weight = None, device = "cpu"):
        """ Args:
            model (Torch.NN.Module): Torch GNN model used.
            x (tensor): 2D tensor containing node features of the graph to explain
            edge_index (tensor): edge index of the graph to be explained.
            edge_weight (tensor): weights of the edges.
            device (string, optional): states if using ```cuda``` or ```cpu```.
        """
        super(Edgeshaper, self).__init__()
        # torch.manual_seed(12345)
        self.model = model
        self.model.to(device)
        self.x = x.to(device)
        self.edge_index = edge_index.to(device)
        self.edge_weight = edge_weight.to(device) if edge_weight is not None else None
        self.device = device


        self.phi_edges = None
        
        self.target_class = None
        self.explained = False

        self.pertinent_positive_set = None
        self.minimal_top_k_set = None
        self.fidelity = None
        self.infidelity = None
        self.trustuworthiness = None

        self.original_pred_prob = None

    def explain(self, M = 100, target_class = 0, P = None, deviation = None, log_odds = False, seed = None):
        """ Compute edge importance using EdgeSHAPer algorithm.
            M (int): number of Monte Carlo sampling steps to perform.
            target_class (int, optional): index of the class prediction to explain. Defaults to class 0. None if the model is a regression model.
            P (float, optional): probablity of an edge to exist in the random graph. Defaults to the original graph density.
            deviation (float, optional): deviation gap from sum of Shapley values and predicted output prob. If ```None```, M sampling
                steps are computed, otherwise the procedure stops when the desired approxiamtion ```deviation``` is reached. However, after M steps
                the procedure always terminates.
            log_odds (bool, optional). Default is ```False```. If ```True```, use log odds instead of probabilty as Value for Shapley values approximation.
            seed (float, optional): seed used for the random number generator.
        Returns:
                list: list of Shapley values for the edges computed by EdgeSHAPer. The order of the edges is the same as in ```self.edge_index```.
        """

        if deviation is not None:
            return self.explain_with_deviation(M = M, target_class = target_class, P = P, deviation = deviation, log_odds = log_odds, seed = seed, device=self.device)

        if target_class is None:
            print("No target class specified. Regression model assumed.")

        E = self.edge_index
        rng = default_rng(seed = seed)
        self.model.eval()
        phi_edges = []

        num_nodes = self.x.shape[0]
        num_edges = E.shape[1]
        
        if P == None:
            max_num_edges = num_nodes*(num_nodes-1)
            graph_density = num_edges/max_num_edges
            P = graph_density

        for j in tqdm(range(num_edges)):
            marginal_contrib = 0
            for i in range(M):
                E_z_mask = rng.binomial(1, P, num_edges)
                E_mask = torch.ones(num_edges)
                pi = torch.randperm(num_edges)

                E_j_plus_index = torch.ones(num_edges, dtype=torch.int)
                E_j_minus_index = torch.ones(num_edges, dtype=torch.int)
                selected_edge_index = np.where(pi == j)[0].item()
                for k in range(num_edges):
                    if k <= selected_edge_index:
                        E_j_plus_index[pi[k]] = E_mask[pi[k]]
                    else:
                        E_j_plus_index[pi[k]] = E_z_mask[pi[k]]

                for k in range(num_edges):
                    if k < selected_edge_index:
                        E_j_minus_index[pi[k]] = E_mask[pi[k]]
                    else:
                        E_j_minus_index[pi[k]] = E_z_mask[pi[k]]


                #we compute marginal contribs
                
                # with edge j
                retained_indices_plus = torch.LongTensor(torch.nonzero(E_j_plus_index).tolist()).to(self.device).squeeze()
                E_j_plus = torch.index_select(E, dim = 1, index = retained_indices_plus)
                edge_weight_j_plus = None

                if self.edge_weight is not None:
                    edge_weight_j_plus = torch.index_select(self.edge_weight, dim = 0, index = retained_indices_plus)
                
                batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
                
                # out = self.model(self.x, E_j_plus, batch=batch, edge_weight=edge_weight_j_plus)
                out = model.infer(batch, altered_edges= E_j_plus, altered_edge_attr = edge_weight_j_plus)
                
                out_prob = None
                V_j_plus = None
                
                if target_class is not None:
                    if not log_odds:
                        out_prob = F.softmax(out, dim = 1)
                    else:
                        out_prob = out #out prob variable now containts log_odds

                    V_j_plus = out_prob[0][target_class].item()
                else:
                    
                    out_prob = out #out prob variable now containts the regression output
                
                    V_j_plus = out_prob[0][0].item()

                # without edge j
                retained_indices_minus = torch.LongTensor(torch.nonzero(E_j_minus_index).tolist()).to(self.device).squeeze()
                E_j_minus = torch.index_select(E, dim = 1, index = retained_indices_minus)
                edge_weight_j_minus = None

                if self.edge_weight is not None:
                    edge_weight_j_minus = torch.index_select(self.edge_weight, dim = 0, index = retained_indices_minus)

                batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
                # out = self.model(self.x, E_j_minus, batch=batch, edge_weight=edge_weight_j_minus)
                out = model.infer(batch, altered_edges= E_j_minus, altered_edge_attr = edge_weight_j_minus)

                V_j_minus = None
                
                if target_class is not None:
                    if not log_odds:
                        out_prob = F.softmax(out, dim = 1)
                    else:
                        out_prob = out #out prob variable now containts log_odds

                    V_j_minus = out_prob[0][target_class].item()
                else:
                    out_prob = out
                
                    V_j_minus = out_prob[0][0].item()

                marginal_contrib += (V_j_plus - V_j_minus)

            phi_edges.append(marginal_contrib/M)

        self.target_class = target_class
        self.explained = True    
        self.phi_edges = phi_edges
        return phi_edges


    def explain_with_deviation(self, M = 100, target_class = 0, P = None, deviation = None, log_odds = False, seed = None, device = "cpu"):
        
        if target_class is None:
            print("No target class specified. Regression model assumed.")
            
        rng = default_rng(seed = seed)
        self.model.eval()
        batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
        E = self.edge_index
        out = self.model(self.x, E, batch=batch, edge_weight=self.edge_weight)
        out_prob_real = F.softmax(out, dim = 1)[0][target_class].item() if target_class is not None else out[0][0].item()

        num_nodes = self.x.shape[0]
        num_edges = E.shape[1]

        phi_edges = []
        phi_edges_current = [0] * num_edges
        
        if P == None:
            max_num_edges = num_nodes*(num_nodes-1)
            graph_density = num_edges/max_num_edges
            P = graph_density

        for i in tqdm(range(M)):
        
            for j in range(num_edges):
                
                
                E_z_mask = rng.binomial(1, P, num_edges)
                E_mask = torch.ones(num_edges)
                pi = torch.randperm(num_edges)

                E_j_plus_index = torch.ones(num_edges, dtype=torch.int)
                E_j_minus_index = torch.ones(num_edges, dtype=torch.int)
                selected_edge_index = np.where(pi == j)[0].item()
                for k in range(num_edges):
                    if k <= selected_edge_index:
                        E_j_plus_index[pi[k]] = E_mask[pi[k]]
                    else:
                        E_j_plus_index[pi[k]] = E_z_mask[pi[k]]

                for k in range(num_edges):
                    if k < selected_edge_index:
                        E_j_minus_index[pi[k]] = E_mask[pi[k]]
                    else:
                        E_j_minus_index[pi[k]] = E_z_mask[pi[k]]


                #we compute marginal contribs
                
                # with edge j
                retained_indices_plus = torch.LongTensor(torch.nonzero(E_j_plus_index).tolist()).to(device).squeeze()
                E_j_plus = torch.index_select(E, dim = 1, index = retained_indices_plus)
                edge_weight_j_plus = None

                if self.edge_weight is not None:
                    edge_weight_j_plus = torch.index_select(self.edge_weight, dim = 0, index = retained_indices_plus)

                batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
                
                #out = self.model(self.x, E_j_plus, batch=batch, edge_weight=edge_weight_j_plus)
                out = model.infer(batch, altered_edges= E_j_plus, altered_edge_attr = edge_weight_j_plus)
                out_prob = None

                V_j_plus = None
                
                if target_class is not None:
                    if not log_odds:
                        out_prob = F.softmax(out, dim = 1)
                    else:
                        out_prob = out #out prob variable now containts log_odds

                    V_j_plus = out_prob[0][target_class].item()
                else:
                    
                    out_prob = out
                
                    V_j_plus = out_prob[0][0].item()

                # without edge j
                retained_indices_minus = torch.LongTensor(torch.nonzero(E_j_minus_index).tolist()).to(device).squeeze()
                E_j_minus = torch.index_select(E, dim = 1, index = retained_indices_minus)
                edge_weight_j_minus = None

                if self.edge_weight is not None:
                    edge_weight_j_minus = torch.index_select(self.edge_weight, dim = 0, index = retained_indices_minus)

                batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
                #out = self.model(self.x, E_j_minus, batch=batch, edge_weight=edge_weight_j_minus)
                out = model.infer(batch, altered_edges= E_j_minus, altered_edge_attr = edge_weight_j_minus)

                V_j_minus = None
                
                if target_class is not None:
                    if not log_odds:
                        out_prob = F.softmax(out, dim = 1)
                    else:
                        out_prob = out #out prob variable now containts log_odds

                    V_j_minus = out_prob[0][target_class].item()
                else:
                    out_prob = out
                
                    V_j_minus = out_prob[0][0].item()

                phi_edges_current[j] += (V_j_plus - V_j_minus)

            
            phi_edges = [elem / (i+1) for elem in phi_edges_current]
            # print(sum(phi_edges))
            if abs(out_prob_real - sum(phi_edges)) <= deviation:
                break

        self.phi_edges = phi_edges
        self.target_class = target_class
        self.explained = True    
        return phi_edges

    def compute_original_predicted_probability(self):
        self.model.eval()
        batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
        out_log_odds = self.model(self.x, self.edge_index, batch=batch, edge_weight=self.edge_weight)
        out_prob = F.softmax(out_log_odds, dim = 1)
        original_pred_prob = out_prob[0][self.target_class].item()

        self.original_pred_prob = original_pred_prob

    def compute_pertinent_positive_set(self, verbose = False):
        assert(self.explained) #make sure that the explanation has been computed
        
        if self.target_class is None:
            raise Exception("Minimal informative sets are not defined for regression problems.")

        if self.original_pred_prob is None:
            self.compute_original_predicted_probability()

        self.model.eval()
        infidelity = 1 #None it was none, last edit since it remains none if the class does not change, so we put 1
        important_edges_ranking = np.argsort(-np.array(self.phi_edges))
        for i in range(1, important_edges_ranking.shape[0]+1):
            reduced_edge_index = torch.index_select(self.edge_index, dim = 1, index = torch.LongTensor(important_edges_ranking[0:i]).to(self.device))
            batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
            out = self.model(self.x, reduced_edge_index, batch=batch)
            out_prob = F.softmax(out, dim = 1)
            # print(out_prob)
            predicted_class = torch.argmax(out_prob[0]).item()
            if (predicted_class == self.target_class):
                
                
                pred_prob = out_prob[0][self.target_class].item()
                infidelity = self.original_pred_prob-pred_prob

                if verbose:
                    print("FID- using pertinent positive set: ", infidelity)
                break

        self.pertinent_positive_set = reduced_edge_index
        self.infidelity = infidelity
        return reduced_edge_index, infidelity

    #for legacy with old code, we keep the mispelled function which is now deprecated but calls the correct one
    def compute_pertinent_positivite_set(self, verbose = False):
        print("WARNING: compute_pertinent_positivite_set is now deprecated, use compute_pertinent_positive_set instead.")
        return self.compute_pertinent_positive_set(verbose)

    def compute_minimal_top_k_set(self, verbose = False):
        assert(self.explained) #make sure that the explanation has been computed
        
        if self.target_class is None:
            raise Exception("Minimal informative sets are not defined for regression problems.")
            
        if self.original_pred_prob is None:
            self.compute_original_predicted_probability()

        self.model.eval()
        fidelity = 0 #it was None, last edit since it remains none if the class does not change, so we put 0
        pertinent_set_indices = []
        pertinent_set_edge_index = None
        important_edges_ranking = np.argsort(-np.array(self.phi_edges))
        for i in range(important_edges_ranking.shape[0]):
            index_of_edge_to_remove = important_edges_ranking[i]
            pertinent_set_indices.append(index_of_edge_to_remove)

            reduced_edge_index = torch.index_select(self.edge_index, dim = 1, index = torch.LongTensor(important_edges_ranking[i:]).to(self.device))
            
            # all nodes belong to same graph
            batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
            out = self.model(self.x, reduced_edge_index, batch=batch)
            out_prob = F.softmax(out, dim = 1)
            # print(out_prob)
            predicted_class = torch.argmax(out_prob[0]).item()

            if predicted_class != self.target_class:
                pred_prob = out_prob[0][self.target_class].item()
                fidelity = self.original_pred_prob - pred_prob
                if verbose:
                    print("FID+ using minimal top-k set: ", fidelity)
                break

        pertinent_set_edge_index = torch.index_select(self.edge_index, dim = 1, index = torch.LongTensor(pertinent_set_indices).to(self.device))
        
        self.minimal_top_k_set = pertinent_set_edge_index
        self.fidelity = fidelity

        return pertinent_set_edge_index, fidelity


    def compute_trustworthiness(self, verbose = False):
        ''' 
        Computes the Trustworthiness (TW) of the explanation. The TW metric is defined as an harmonic mean of the fidelity and reverse of the infidelity. 
        '''

        assert(self.explained) #make sure that the explanation has been computed

        assert(self.fidelity is not None) #make sure that the fidelity has been computed
        assert(self.infidelity is not None) #make sure that the infidelity has been computed

        TW = None
        
        if self.fidelity+(1-self.infidelity) == 0:
            TW = 0
        else:
            TW = 2* ((self.fidelity*(1-self.infidelity))/(self.fidelity+(1-self.infidelity)))

        self.trustworthiness = TW

        if verbose:
            print("Trustworthiness: ", self.trustuworthiness)

        return self.trustworthiness

    def visualize_molecule_explanations(self, smiles, save_path=None, pertinent_positive = False, minimal_top_k = False):
        assert(self.explained) #make sure that the explanation has been computed

        img_expl = None
        img_pert_pos = None
        img_min_top_k = None

        edge_index = self.edge_index.to("cpu")

        test_mol = Chem.MolFromSmiles(smiles)
        test_mol = Draw.PrepareMolForDrawing(test_mol)

        num_bonds = len(test_mol.GetBonds())

        rdkit_bonds = {}

        for i in range(num_bonds):
            init_atom = test_mol.GetBondWithIdx(i).GetBeginAtomIdx()
            end_atom = test_mol.GetBondWithIdx(i).GetEndAtomIdx()
            
            rdkit_bonds[(init_atom, end_atom)] = i

        rdkit_bonds_phi = [0]*num_bonds
        for i in range(len(self.phi_edges)):
            phi_value = self.phi_edges[i]
            init_atom = edge_index[0][i].item()
            end_atom = edge_index[1][i].item()
            
            if (init_atom, end_atom) in rdkit_bonds:
                bond_index = rdkit_bonds[(init_atom, end_atom)]
                rdkit_bonds_phi[bond_index] += phi_value
            if (end_atom, init_atom) in rdkit_bonds:
                bond_index = rdkit_bonds[(end_atom, init_atom)]
                rdkit_bonds_phi[bond_index] += phi_value

        plt.clf()
        canvas = mapvalues2mol(test_mol, None, rdkit_bonds_phi, atom_width=0.2, bond_length=0.5, bond_width=0.5) #TBD: only one direction for edges? bonds weights is wrt rdkit bonds order?
        img_expl = transform2png(canvas.GetDrawingText())

        if save_path is not None:
            img_expl.save(save_path + "/" + "EdgeSHAPer_explanations_heatmap.png", dpi = (300,300))

        if pertinent_positive:
            if self.pertinent_positive_set is None:
                self.compute_pertinent_positivite_set()

            rdkit_bonds_phi_pertinent_set = [0]*num_bonds
            pertinent_set_edge_index = self.pertinent_positive_set
            for i in range(pertinent_set_edge_index.shape[1]):
                
                init_atom = pertinent_set_edge_index[0][i].item()
                end_atom = pertinent_set_edge_index[1][i].item()
                
                
                if (init_atom, end_atom) in rdkit_bonds:
                    bond_index = rdkit_bonds[(init_atom, end_atom)]
                    if rdkit_bonds_phi_pertinent_set[bond_index] == 0:
                        rdkit_bonds_phi_pertinent_set[bond_index] += rdkit_bonds_phi[bond_index]
                if (end_atom, init_atom) in rdkit_bonds:
                    bond_index = rdkit_bonds[(end_atom, init_atom)]
                    if rdkit_bonds_phi_pertinent_set[bond_index] == 0:
                        rdkit_bonds_phi_pertinent_set[bond_index] += rdkit_bonds_phi[bond_index]

            plt.clf()
            canvas = mapvalues2mol(test_mol, None, rdkit_bonds_phi_pertinent_set, atom_width=0.2, bond_length=0.5, bond_width=0.5)
            img_pert_pos = transform2png(canvas.GetDrawingText())

            if save_path is not None:

                img_pert_pos.save(save_path + "/" + "EdgeSHAPer_pertinent_positive_set_heatmap.png", dpi = (300,300))

        if minimal_top_k:
            if self.minimal_top_k_set is None:
                self.compute_minimal_top_k_set()

            rdkit_bonds_phi_pertinent_set = [0]*num_bonds
            pertinent_set_edge_index = self.minimal_top_k_set
            for i in range(pertinent_set_edge_index.shape[0]):
                
                init_atom = pertinent_set_edge_index[0][i].item()
                end_atom = pertinent_set_edge_index[1][i].item()
                
                
                if (init_atom, end_atom) in rdkit_bonds:
                    bond_index = rdkit_bonds[(init_atom, end_atom)]
                    if rdkit_bonds_phi_pertinent_set[bond_index] == 0:
                        rdkit_bonds_phi_pertinent_set[bond_index] += rdkit_bonds_phi[bond_index]
                if (end_atom, init_atom) in rdkit_bonds:
                    bond_index = rdkit_bonds[(end_atom, init_atom)]
                    if rdkit_bonds_phi_pertinent_set[bond_index] == 0:
                        rdkit_bonds_phi_pertinent_set[bond_index] += rdkit_bonds_phi[bond_index]

            plt.clf()
            canvas = mapvalues2mol(test_mol, None, rdkit_bonds_phi_pertinent_set, atom_width=0.2, bond_length=0.5, bond_width=0.5)
            img_min_top_k = transform2png(canvas.GetDrawingText())

            if save_path is not None:
                img_min_top_k.save(save_path + "/" + "EdgeSHAPer_minimal_top_k_set_heatmap.png", dpi = (300,300))

        plt.clf()
        return img_expl, img_pert_pos, img_min_top_k    


    


###EdgeSHAPer as a function###
def edgeshaper(model, x, E, M = 100, target_class = 0, P = None, deviation = None, log_odds = False, seed = 42, edge_weight = None, device = "cpu", batch=None):
    """ Compute Shapley values approximation for edge importance in GNNs
        Args:
            model (Torch.NN.Module): Torch GNN model used.
            x (tensor): 2D tensor containing node features of the graph to explain
            E (tensor): edge index of the graph to be explained.
            P (float, optional): probablity of an edge to exist in the random graph. Defaults to the original graph density.
            M (int): number of Monte Carlo sampling steps to perform.
            target_class (int, optional): index of the class prediction to explain. Defaults to class 0.
            deviation (float, optional): deviation gap from sum of Shapley values and predicted output prob. If ```None```, M sampling
                steps are computed, otherwise the procedure stops when the desired approxiamtion ```deviation``` is reached. However, after M steps
                the procedure always terminates.
            log_odds (bool, optional). Default is ```False```. If ```True```, use log odds instead of probabilty as Value for Shaply values approximation.
            seed (float, optional): seed used for the random number generator.
            device (string, optional): states if using ```cuda``` or ```cpu```.
        Returns:
                list: list of Shapley values for the edges computed by EdgeSHAPer. The order of the edges is the same as in ```E```.
        """
    
    if deviation != None:
        return edgeshaper_deviation(model, x, E, M = M, target_class = target_class, P = P, deviation = deviation, log_odds = log_odds, seed = seed, edge_weight = edge_weight, device=device)


    rng = default_rng(seed = seed)
    model.eval()
    phi_edges = []

    num_nodes = x.shape[0]
    num_edges = E.shape[1]
    
    if P == None:
        max_num_edges = num_nodes*(num_nodes-1)
        graph_density = num_edges/max_num_edges
        P = graph_density

    for j in tqdm(range(num_edges)):
        marginal_contrib = 0
        for i in range(M):
            E_z_mask = rng.binomial(1, P, num_edges)
            E_mask = torch.ones(num_edges)
            pi = torch.randperm(num_edges)

            E_j_plus_index = torch.ones(num_edges, dtype=torch.int)
            E_j_minus_index = torch.ones(num_edges, dtype=torch.int)
            selected_edge_index = np.where(pi == j)[0].item()
            for k in range(num_edges):
                if k <= selected_edge_index:
                    E_j_plus_index[pi[k]] = E_mask[pi[k]]
                else:
                    E_j_plus_index[pi[k]] = E_z_mask[pi[k]]

            for k in range(num_edges):
                if k < selected_edge_index:
                    E_j_minus_index[pi[k]] = E_mask[pi[k]]
                else:
                    E_j_minus_index[pi[k]] = E_z_mask[pi[k]]

            #we compute marginal contribs
            # with edge j
            retained_indices_plus = torch.LongTensor(torch.nonzero(E_j_plus_index).tolist()).to(device).squeeze()
            E_j_plus = torch.index_select(E, dim = 1, index = retained_indices_plus)
            edge_weight_j_plus = None

            if edge_weight is not None:
                edge_weight_j_plus = torch.index_select(edge_weight, dim = 0, index = retained_indices_plus)

            # batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
            
            out = model.infer(batch, altered_edges= E_j_plus, altered_edge_attr = edge_weight_j_plus)
            out_prob = None

            V_j_plus = None
                
            if target_class is not None:
                if not log_odds:
                    out_prob = F.softmax(out, dim = 1)
                else:
                    out_prob = out #out prob variable now containts log_odds

                V_j_plus = out_prob[0][target_class].item()
            else:
                
                out_prob = out
            
                #V_j_plus = out_prob[0][0].item()
                V_j_plus = out_prob.item()

            # without edge j
            retained_indices_minus = torch.LongTensor(torch.nonzero(E_j_minus_index).tolist()).to(device).squeeze()
            E_j_minus = torch.index_select(E, dim = 1, index = retained_indices_minus)
            edge_weight_j_minus = None

            if edge_weight is not None:
                edge_weight_j_minus = torch.index_select(edge_weight, dim = 0, index = retained_indices_minus)

            #batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
            out = model.infer(batch, altered_edges= E_j_minus, altered_edge_attr = edge_weight_j_minus)

            V_j_minus = None
                
            if target_class is not None:
                if not log_odds:
                    out_prob = F.softmax(out, dim = 1)
                else:
                    out_prob = out #out prob variable now containts log_odds

                V_j_minus = out_prob[0][target_class].item()
            else:
                out_prob = out
            
                V_j_minus = out_prob.item()

            marginal_contrib += (V_j_plus - V_j_minus)

        phi_edges.append(marginal_contrib/M)
        
    return phi_edges


def edgeshaper_deviation(model, x, E, M = 100, target_class = 0, P = None, deviation = None, log_odds = False, seed = 42, edge_weight = None, device = "cpu"):

    rng = default_rng(seed = seed)
    model.eval()
    batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
    out = model(x, E, batch=batch, edge_weight = edge_weight)
    #out = model(batch)
    out_prob_real = F.softmax(out, dim = 1)[0][target_class].item() if target_class is not None else out[0][0].item()

    num_nodes = x.shape[0]
    num_edges = E.shape[1]

    phi_edges = []
    phi_edges_current = [0] * num_edges
    
    if P == None:
        max_num_edges = num_nodes*(num_nodes-1)
        graph_density = num_edges/max_num_edges
        P = graph_density

    for i in tqdm(range(M)):
    
        for j in range(num_edges):
            
            
            E_z_mask = rng.binomial(1, P, num_edges)
            E_mask = torch.ones(num_edges)
            pi = torch.randperm(num_edges)

            E_j_plus_index = torch.ones(num_edges, dtype=torch.int)
            E_j_minus_index = torch.ones(num_edges, dtype=torch.int)
            selected_edge_index = np.where(pi == j)[0].item()
            for k in range(num_edges):
                if k <= selected_edge_index:
                    E_j_plus_index[pi[k]] = E_mask[pi[k]]
                else:
                    E_j_plus_index[pi[k]] = E_z_mask[pi[k]]

            for k in range(num_edges):
                if k < selected_edge_index:
                    E_j_minus_index[pi[k]] = E_mask[pi[k]]
                else:
                    E_j_minus_index[pi[k]] = E_z_mask[pi[k]]


            #we compute marginal contribs
            
            # with edge j
            retained_indices_plus = torch.LongTensor(torch.nonzero(E_j_plus_index).tolist()).to(device).squeeze()
            E_j_plus = torch.index_select(E, dim = 1, index = retained_indices_plus)

            edge_weight_j_plus = None

            if edge_weight is not None:
                edge_weight_j_plus = torch.index_select(edge_weight, dim = 0, index = retained_indices_plus)

            batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
            
            out = model(x, E_j_plus, batch=batch, edge_weight = edge_weight_j_plus)
            out_prob = None

            V_j_plus = None
                
            if target_class is not None:
                if not log_odds:
                    out_prob = F.softmax(out, dim = 1)
                else:
                    out_prob = out #out prob variable now containts log_odds

                V_j_plus = out_prob[0][target_class].item()
            else:
                out_prob = out
            
                V_j_plus = out_prob[0][0].item()

            # without edge j
            retained_indices_minus = torch.LongTensor(torch.nonzero(E_j_minus_index).tolist()).to(device).squeeze()
            E_j_minus = torch.index_select(E, dim = 1, index = retained_indices_minus)

            edge_weight_j_minus = None

            if edge_weight is not None:
                edge_weight_j_minus = torch.index_select(edge_weight, dim = 0, index = retained_indices_minus)

            batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
            out = model(x, E_j_minus, batch=batch, edge_weight = edge_weight_j_minus)

            V_j_minus = None
                
            if target_class is not None:
                if not log_odds:
                    out_prob = F.softmax(out, dim = 1)
                else:
                    out_prob = out #out prob variable now containts log_odds

                V_j_minus = out_prob[0][target_class].item()
            else:
                out_prob = out
            
                V_j_minus = out_prob[0][0].item()

            phi_edges_current[j] += (V_j_minus - V_j_minus)

        
        phi_edges = [elem / (i+1) for elem in phi_edges_current]
        # print(sum(phi_edges))
        if abs(out_prob_real - sum(phi_edges)) <= deviation:
            break
             
    return phi_edges