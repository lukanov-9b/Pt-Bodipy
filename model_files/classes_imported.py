# Impotring.
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem, MolFromSmiles


class GetSolventParams(BaseEstimator, TransformerMixin):
    
    def __init__(self) -> None:
        SOLVENT_PARAMS = pd.read_csv('model_files/solvent_params.csv')
        SOLVENT_PARAMS.index = SOLVENT_PARAMS['smiles']
        SOLVENT_PARAMS = SOLVENT_PARAMS.drop(['smiles', 'solvent'], axis=1)
        self.params = SOLVENT_PARAMS      
    
    def fit(self, features, target=None):
        return self

    def transform(self, features, target=None):
        return np.array([self.params.loc[x, :].tolist() for x in features])
    
class GetFgp(BaseEstimator, TransformerMixin):
    
    def __init__(self) -> None:
        pass   
    
    def fit(self, features, target=None):
        return self

    def transform(self, features, target=None):
        return np.array([list(AllChem.GetMorganFingerprintAsBitVect(MolFromSmiles(x), 3, nBits=2048)) for x in features])
    

# Kernel function to generate weights (kNN).
def get_weights(distances:np.array, param:float=10) -> np.array:
    weights = []
    for item in distances:
        weight_for_item = [np.exp(-2*(x/param)**2) for x in item]
        weights.append(weight_for_item)

    return np.array(weights)/((2*np.pi)**(0.5))