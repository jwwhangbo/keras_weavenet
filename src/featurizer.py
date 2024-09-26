from typing import List, Optional, Tuple
from rdkit import Chem
import numpy as np


def onehot_encoding(x: int, allowable_set: list) -> np.ndarray:
    # Create an array of zeros
    enc = np.zeros(len(allowable_set))
    # Get the index of the element
    enc[allowable_set.index(x)] = 1
    return enc

class SimpleFeaturizer:
    def __init__(self, max_dist: int, max_atoms: Optional[int] = None, max_pairs: Optional[int] = None):
        self.max_dist = max_dist
        self.max_atoms = max_atoms
        self.max_pairs = max_pairs
        
    def featurize(self, mol_list: List[Chem.rdchem.Mol]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        atom_features_list = []
        pair_features_list = []
        atom_to_pair_list = []

        for mol in mol_list:
            atom_features, pair_features, atom_to_pair = self._featurize(mol)
            atom_features_list.append(atom_features)
            pair_features_list.append(pair_features)
            atom_to_pair_list.append(atom_to_pair)
        
        if not self.max_atoms: 
            max_atoms = max([atom_features.shape[0] for atom_features in atom_features_list])
        if not self.max_pairs:
            max_pairs = max([pair_features.shape[0] for pair_features in pair_features_list])

        for i in range(len(atom_features_list)):
            atom_features_list[i] = np.pad(atom_features_list[i], ((0, max_atoms - atom_features_list[i].shape[0]), (0, 0)))
            pair_features_list[i] = np.pad(pair_features_list[i], ((0, max_pairs - pair_features_list[i].shape[0]), (0, 0)))
            atom_to_pair_list[i] = np.pad(atom_to_pair_list[i], ((0, 0), (0, max_pairs - atom_to_pair_list[i].shape[1])))

        atom_features = np.stack(atom_features_list)
        pair_features = np.stack(pair_features_list)
        atom_to_pair = np.stack(atom_to_pair_list)

        return atom_features, pair_features, atom_to_pair
        
    def _featurize(self, mol: Chem.rdchem.Mol) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Get atom features
        atom_features = [(atom.GetIdx(), onehot_encoding(
            atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H'])) for atom in mol.GetAtoms()]
        atom_features.sort(key=lambda x: x[0])
        idx, features = list(zip(*atom_features))
        atom_features = np.array(features, dtype=np.float32)

        def get_bond_type_value(bond):
            BondType = bond.GetBondType()
            if BondType == BondType.SINGLE:
                return 1
            elif BondType == BondType.DOUBLE:
                return 2
            elif BondType == BondType.TRIPLE:
                return 3
            elif BondType == BondType.AROMATIC:
                return 4
            else:
                return 0  # No bond or other types

        # def get_pair_features(mol):
        num_atoms = mol.GetNumAtoms()

        # Initialize pair features as a list of (bond_type, topological_distance) tuples
        pair_features = []
        atom_to_pair = []

        # Compute topological distance matrix (shortest path in terms of number of bonds)
        topo_dist_matrix = Chem.GetDistanceMatrix(mol)

        # Loop over all pairs of atoms
        for i in range(num_atoms):
            for j in range(i+1, num_atoms):
                # Get the bond type (if there's a bond between atom i and atom j)
                bond = mol.GetBondBetweenAtoms(i, j)
                bond_type = 0
                if bond:
                    bond_type = get_bond_type_value(bond)

                # Get the topological distance between the atoms
                topological_distance = int(topo_dist_matrix[i, j])
                if topological_distance > self.max_dist:
                    topological_distance = 0 # If the distance is too large, set it to 0
                # Store the pair feature (bond type, topological distance)
                pair_features.append([bond_type, topological_distance])

                atom_to_pair.append([i, j])

        pair_features = np.array(pair_features, dtype=np.float32)
        atom_to_pair = np.array(atom_to_pair, dtype=np.int32)
        atom_to_pair = atom_to_pair.T

        return atom_features, pair_features, atom_to_pair
