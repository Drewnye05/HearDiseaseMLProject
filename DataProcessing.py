import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder

class DataPipeline:
    
    def __init__(self,filepath,targetColumn,test_size =.2,random_state = 42):

        self.filepath = filepath
        self.targetColumn = targetColumn
        self.test_size = test_size
        self.random_state = random_state

        self.X_train = None 
        self.X_test = None 
        self.y_train = None 
        self.y_test = None 
        self.scalar = StandardScaler()
    
    def loadData(self):
        self.df = pd.read_csv(self.filepath)
        return self
    
   

    def preprocess(self, binary = False):
        self.df = self.df.dropna(subset=['ca','thal'])

        if binary:
            self.df[self.targetColumn]=(self.df[self.targetColumn]> 0).astype(int)

        # self.df = pd.get_dummies(self.df, columns=['cp', 'thal', 'slope', 'restecg'], drop_first=True)
        cat_cols = self.df.select_dtypes(include=['object','category']).columns
        self.df = pd.get_dummies(self.df,columns=cat_cols,drop_first=True)
        return self 
        
    def splitData(self):
        X = self.df.drop(columns=[self.targetColumn])
        y = self.df[self.targetColumn]

        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X,y,train_size=self.test_size,random_state=self.random_state)

        return self 
    
    def scaleFeatures(self):
        self.X_train = self.scalar.fit_transform(self.X_train)
        self.X_test = self.scalar.transform(self.X_test)
        return self
    
    def run (self,binary=False):
        self.loadData()
        self.preprocess(binary=binary)
        self.splitData()
        self.scaleFeatures()
        print(self.df['num'].value_counts())
        return self
    def getData(self):
        return self.X_train, self.y_train, self.X_test, self.y_test