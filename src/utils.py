from rdkit import Chem
import numpy as np

class SMILESFingerprint:
    def __init__(self, smiles) -> None:
        self.smiles = smiles
        self.maxNumHeavyAtoms = self.getMaxNumHeavyAtoms()
        self.unique_atoms, self.unique_bonds = self.getUniqueElements()
        self.atom_mapping = self.generateAtomMapping()
        self.bond_mapping = self.generateBondMapping()

    def getMaxNumHeavyAtoms(self) -> int:
        heavyAtomNums = []
        for smile in self.smiles:
            molecule = Chem.MolFromSmiles(smile)
            numHeavyAtoms = molecule.GetNumHeavyAtoms()
            heavyAtomNums.append(numHeavyAtoms)
        numHeavyAtoms = max(heavyAtomNums)
        return numHeavyAtoms
    
    def generateNumAtomsDict(self) -> int:
        numAtomsDict = {}
        for smile in self.smiles:
            molecule = Chem.MolFromSmiles(smile)
            numAtoms = molecule.GetNumAtoms()
            numAtomsDict[smile] = numAtoms
        return numAtomsDict
            
    
    def getUniqueElements(self) -> tuple[set, set]:
        unique_atoms = set()
        unique_bonds = set()

        for smile in self.smiles:
            molecule = Chem.MolFromSmiles(smile)
            if molecule is None:
                continue

            # Extract unique atoms
            for atom in molecule.GetAtoms():
                unique_atoms.add(atom.GetSymbol())

            # Extract unique bonds
            for bond in molecule.GetBonds():
                unique_bonds.add(bond.GetBondType().name)

        return unique_atoms, unique_bonds
    
    def generateAtomMapping(self) -> dict:
        atom_mapping = {}
        for n, atom in enumerate(self.unique_atoms):
            atom_mapping[atom] = n
            atom_mapping[n] = atom
        return atom_mapping
    
    def generateBondMapping(self) -> dict:
        bond_mapping = {}
        for n, bond in enumerate(self.unique_bonds):
            bond_mapping[bond] = n
            bond_mapping[n] = bond
        return bond_mapping
    
    def getTensorsFromSMILES(self, smiles: list[str]):
        adjacency_tensor, feature_tensor = [], []
        BOND_DIM = len(self.unique_bonds) + 1
        NUM_ATOMS = self.maxNumHeavyAtoms
        ATOM_DIM = len(self.unique_atoms) + 1
        for smile in smiles:
            adjacency = np.zeros((BOND_DIM, NUM_ATOMS, NUM_ATOMS), "float32")
            features = np.zeros((NUM_ATOMS, ATOM_DIM), "float32")

            molecule = Chem.MolFromSmiles(smile)

            for atom in molecule.GetAtoms():
                i = atom.GetIdx()
                atom_type = self.atom_mapping[atom.GetSymbol()]
                features[i] = np.eye(ATOM_DIM)[atom_type]

                # loop over one-hop neighbors
                for neighbor in atom.GetNeighbors():
                    j = neighbor.GetIdx()
                    bond = molecule.GetBondBetweenAtoms(i, j)
                    bond_type_idx = self.bond_mapping[bond.GetBondType().name]
                    adjacency[bond_type_idx, [i, j], [j, i]] = 1

            # Where no bond, add 1 to last channel (indicating "non-bond")
            # Notice: channels-first
            adjacency[-1, np.sum(adjacency, axis=0) == 0] = 1

            # Where no atom, add 1 to last column (indicating "non-atom")
            features[np.where(np.sum(features, axis=1) == 0)[0], -1] = 1
            adjacency_tensor.append(adjacency)
            feature_tensor.append(features)

        return np.array(adjacency_tensor), np.array(feature_tensor)

