import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder

class DataPipline:
    
    def __init__(self,filepath,targetColumn,test_size =.2,random_state = 42):

        self.filepath = filepath
        self.targetColumn = targetColumn
        self.test_size = test_size
        self.random_state = random_state

        self.X_train = None 
        self.X_test = None 
        self.y_train = None 
        self.y_test = None 
    
    def loadData(self):
        self.df = pd.read_csv(self.filepath)
        return self
    
   

    def preprocess(self):
        self.df = self.df.dropna(subset=['ca','thal'])

        self.df = pd.get_dummies(self.df, columns=['cp', 'thal', 'slope', 'restecg'], drop_first=True)
        return self 
        
