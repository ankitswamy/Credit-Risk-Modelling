# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency,f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support,f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from itertools import product
import xgboost as xgb
import warnings
import pickle
warnings.filterwarnings('ignore')




class CreditModel:
  def __init__(self,internal_data,external_data,testing_data):
    self.internal_data = internal_data
    self.external_data = external_data
    self.testing_data = testing_data 

  def data_collection(self):
    # creating the copy of this data so nothing changes 
    self.internal= self.internal_data.copy()
    self.external = self.external_data.copy()

  def data_preprocessing(self): 
    #remove nulls from internal data  
    self.internal = self.internal.loc[self.internal['Age_Oldest_TL'] != -99999]


  # deleting unwanted columns where null values are more than 10000 
  # Cleans external dataset by removing columns/rows with excessive missing values.
  # Returns a merged dataset of internal and external data.  
    columns_to_be_removed = []
    for i in self.external.columns:
      if self.external[self.external[i] == -99999].shape[0] > 10000:
        columns_to_be_removed.append(i)

    self.external  = self.external.drop(columns_to_be_removed,axis=1)   

  # deleting all those rows which have null values present
    for i in self.external.columns:
      self.external = self.external.loc[self.external[i] != -99999]

  #Checking common column names 
    for i in list(self.external.columns):
      if i in list(self.internal.columns):
        print('Common column thorugh which we can merge data is :',i)    

  #now both datasets have common column PROSPECTID so we will merge them 
  #Merge the two dataframe based on inner join so that no nulls are present
    dataset = pd.merge(self.internal,self.external, how="inner",left_on=["PROSPECTID"],right_on=["PROSPECTID"])

    return dataset       

     
  def chi_square(self,dataset):
    # Chi-Square Test 
    for i in ['MARITALSTATUS',
              'EDUCATION',
              'GENDER',
              'last_prod_enq2',
              'first_prod_enq2']:
      chi2,pval,_,_ =chi2_contingency(pd.crosstab(dataset[i],dataset['Approved_Flag']))
      print(i,'----',pval)
      # Ab Sabhi ka pval value jo hai vo less than 0.05 to fir hamm sabhi ko accept kar lenge

    
  def vif_selector(self,dataset):
    # Selecting all numerical columns for VIF other than PROSPECTID   
    #Selecting numerical columns 
    self.numerical_columns = []
    for i in dataset.columns:
      if dataset[i].dtype != 'object' and i not in ["PROSPECTID","Approved_Flag"]:
        self.numerical_columns.append(i)

    # VIF CHECK and Selecting those columns which pass VIF Check
    VIFdataset = dataset[self.numerical_columns]
    totalColumns = len(self.numerical_columns)
    self.columns_to_be_kept = []
    column_index = 0

    for i in range(0,totalColumns):
      if len(VIFdataset.columns) <= column_index:
        break
      col_name = VIFdataset.columns[column_index]
      if VIFdataset[col_name].std()==0: # constant column
        VIFdataset = VIFdataset.drop(col_name, axis=1)
        continue 
      try:
        VIF = variance_inflation_factor(VIFdataset,column_index)
        print(f"{column_index} --- {VIF}")
        if np.isnan(VIF) or np.isinf(VIF) or VIF>6:
          VIFdataset = VIFdataset.drop(col_name, axis =1)
        else:
          column_index +=1
          self.columns_to_be_kept.append(col_name)
      except:
        # If VIF fails, drop 
        VIFdataset = VIFdataset.drop(col_name, axis=1)
    return self.columns_to_be_kept         
 
      
    # Annova Test
  def AnnovaTest(self,dataset):  
    self.columns_to_be_kept_numerical = []
    for i in self.columns_to_be_kept:
      a = list(dataset[i])
      b = list(dataset['Approved_Flag'])

      groupP1 = [value for value, group in zip(a,b) if group == 'P1']
      groupP2 = [value for value, group in zip(a,b) if group == 'P2']
      groupP3 = [value for value, group in zip(a,b) if group == 'P3']
      groupP4 = [value for value, group in zip(a,b) if group == 'P4']

      groups = [g for g in [groupP1,groupP2,groupP3,groupP4] if len(g)>=2]
      if len(groups) < 2:
        continue 

      f_statistics, pvalue = f_oneway(*groups)

      if pvalue <=0.05:
        self.columns_to_be_kept_numerical.append(i) 
    return self.columns_to_be_kept_numerical    

  def perform_eda_and_feature_selection(self,dataset):
    #Printing categorical columns
    for i in dataset.columns:
      if dataset[i].dtype == 'object':
        print('Categorical columns : [',i," ]",end = " ")   
    

    #calling feature selection functions for work to be done
    self.chi_square(dataset)
    self.vif_selector(dataset)
    self.AnnovaTest(dataset)

    #final dataset after feature selection
    finalColumns = self.columns_to_be_kept_numerical + ['MARITALSTATUS','EDUCATION','GENDER','last_prod_enq2', 'first_prod_enq2']
    dataset = dataset[finalColumns + ['Approved_Flag']]

    return dataset          

  def encode_features(self,dataset):

    #First Lets deal with Education column because it is an ordinal column so we have give ranking or ordering to column values
    dataset.loc[dataset['EDUCATION'] == 'SSC',['EDUCATION']]              = 1
    dataset.loc[dataset['EDUCATION'] == '12TH',['EDUCATION']]             = 2
    dataset.loc[dataset['EDUCATION'] == 'GRADUATE',['EDUCATION']]         = 3
    dataset.loc[dataset['EDUCATION'] == 'UNDER GRADUATE',['EDUCATION']]   = 3
    dataset.loc[dataset['EDUCATION'] == 'POST-GRADUATE',['EDUCATION']]    = 4
    dataset.loc[dataset['EDUCATION'] == 'OTHERS',['EDUCATION']]           = 1
    dataset.loc[dataset['EDUCATION'] == 'PROFESSIONAL',['EDUCATION']]     = 3
    dataset['EDUCATION'] = dataset['EDUCATION'].astype(int)

    #Other than Education column, for all columns we can do one-hot encoding 
    dataset['EDUCATION'].value_counts()
    dataset['EDUCATION'] = dataset['EDUCATION'].astype(int)
    dataset.info()
    datasetEncoded = pd.get_dummies(dataset, columns=['MARITALSTATUS','GENDER','last_prod_enq2', 'first_prod_enq2'],dtype=int)

    return datasetEncoded
  
  def train_random_forest(self,x_train,y_train,x_test,y_test):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    self.accuracy_randomforest = accuracy_score(y_test,y_pred)
    print("RandomForest Results")
    print("accuracy score :",self.accuracy_randomforest)
    precision, recall, f1_vals, _ = precision_recall_fscore_support(y_test,y_pred)
    for i, v in enumerate(['p1','p2','p3','p4']):
      print(f"Class{v}")
      print(f"Precision: {precision[i]}")
      print(f"recall: {recall[i]}")
      print(f"F1_score: {f1_vals[i]}")
      print()
      
  def train_xgboost(self,x_train,y_train,x_test,y_test):
    model = xgb.XGBClassifier(objective='multi:softmax',num_class=4)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    self.accuracy_xgboost = accuracy_score(y_test,y_pred)
    print("XGBOOST Results")
    print("accuracy score :",self.accuracy_xgboost)
    precision, recall, f1_vals, _ = precision_recall_fscore_support(y_test,y_pred)
    for i, v in enumerate(['p1','p2','p3','p4']):
      print(f"Class{v}")
      print(f"Precision: {precision[i]}")
      print(f"recall: {recall[i]}")
      print(f"F1_score: {f1_vals[i]}")
      print()

  def train_decision_tree(self,x_train,y_train,x_test,y_test):
    model = DecisionTreeClassifier(max_depth=20,min_samples_split=10)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    self.accuracy_decisiontree = accuracy_score(y_test,y_pred)
    print("Decision Tree Results")
    print("accuracy score:",self.accuracy_decisiontree)
    precision, recall, f1_vals, _ = precision_recall_fscore_support(y_test,y_pred)
    for i, v in enumerate(['p1','p2','p3','p4']):
      print(f"Class{v}")
      print(f"Precision: {precision[i]}")
      print(f"recall: {recall[i]}")
      print(f"F1_Score: {f1_vals[i]}")
      print()   

  def train_models(self,datasetEncoded):
    self.labelencoder = LabelEncoder()
    y = self.labelencoder.fit_transform(datasetEncoded['Approved_Flag'])
    x = datasetEncoded.drop(['Approved_Flag'],axis=1)
    

    

    #train test splitting 
    self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    self.train_random_forest(self.x_train,self.y_train,self.x_test,self.y_test) 
    self.train_xgboost(self.x_train,self.y_train,self.x_test,self.y_test)
    self.train_decision_tree(self.x_train,self.y_train,self.x_test,self.y_test) 

    #Apply Standard Scaler and train once again after Scaling 
    columns_to_be_scaled = ['Age_Oldest_TL','Age_Newest_TL','time_since_recent_payment','max_recent_level_of_deliq','recent_level_of_deliq','time_since_recent_enq','NETMONTHLYINCOME','Time_With_Curr_Empr']

    for i in columns_to_be_scaled:
      column_data = datasetEncoded[i].values.reshape(-1,1)
      scaler = StandardScaler()
      scaled_column = scaler.fit_transform(column_data)
      datasetEncoded[i] = scaled_column

    x = datasetEncoded.drop(['Approved_Flag'],axis=1)
    #train test splitting 
    self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    self.train_random_forest(self.x_train,self.y_train,self.x_test,self.y_test) 
    self.train_xgboost(self.x_train,self.y_train,self.x_test,self.y_test)
    self.train_decision_tree(self.x_train,self.y_train,self.x_test,self.y_test)
     

  def best_model_to_tune(self):
    self.model_results = {
      'RandomForest': self.accuracy_randomforest,
      'XGBOOST': self.accuracy_xgboost,
      'DecisionTree': self.accuracy_decisiontree
    }    
    self.best_model_name = max(self.model_results, key = self.model_results.get)  
    print(self.best_model_name)
    # we got XGBOOST as the best model i already checked for that 
  
  def tune_hyperparameters(self):
    index = 0
    results_list = []
    # Defining parameter grids for each model 
    parameterGrids = {
      'XGBOOST': {
      'colsample_bytree': [0.1,0.3,0.5, 0.7, 0.9],
      'learning_rate': [0.001, 0.01, 0.1],
      'max_depth': [3, 5, 8, 10],
      'alpha': [1, 10], 
      'n_estimators': [10,50,100],
      'subsample': [0.7, 0.9, 1.0]
      },

      'RandomForest': {
      'n_estimators': [100, 200, 300],
      'max_depth': [10, 15, 20, None],
      'min_samples_split': [2, 5, 10],
      'min_samples_leaf': [1, 2, 4],
      'max_features': ['sqrt', 'log2']
      },

      'DecisionTree': {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
      }

    }
    # Selecting which module performed best without hyperparameter tuning 
    grid = parameterGrids[self.best_model_name]

    parameter_combination = list(product(*grid.values()))
    

    for parameters in parameter_combination:
      index +=1 
      parameters_dictionary = dict(zip(grid.keys(),parameters))

      if self.best_model_name =="XGBOOST":
        model = xgb.XGBClassifier(objective = 'multi:softmax',num_class = 4,**parameters_dictionary)
      elif self.best_model_name =="RandomForest":
        model = RandomForestClassifier(random_state=42,**parameters_dictionary)
      elif self.best_model_name =="DecisionTree":
        model = DecisionTreeClassifier(random_state=42,**parameters_dictionary) 
      # Train the model
      model.fit(self.x_train,self.y_train)

      #Evaluate the model 
      train_pred = model.predict(self.x_train)
      test_pred = model.predict(self.x_test)
      train_acc = accuracy_score(self.y_train,train_pred)
      test_acc = accuracy_score(self.y_test,test_pred)
      train_f1 = f1_score(self.y_train,train_pred, average = 'macro')
      test_f1 = f1_score(self.y_test,test_pred, average = 'macro')

      results_list.append({
        'Combination':index,
        'Train Accuracy': train_acc,
        'Test Accuracy':test_acc,
        'Train F1':train_f1,
        'Test F1': test_f1,
        **parameters_dictionary
      })
    # Converting results into a DataFrame
    results = pd.DataFrame(results_list)
    print(results)
    #Select the best parameters (highest test accuracy)
    results['Generalization Gap'] = abs(results['Train F1'] - results['Test F1'])
    results['Combined Score'] = results['Test F1'] - 0.2 * results['Generalization Gap']
    best_row = results.loc[results['Combined Score'].idxmax()]

    self.best_values = best_row.to_dict()

    exclude_keys = ['Test Accuracy', 'Train Accuracy', 'Train F1', 'Test F1', 'Generalization Gap' ,'Combination','Combined Score']
    params = {
      k:v for k,v in self.best_values.items() if k not in exclude_keys
    }
    int_params = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf']
    for key in int_params:
      if key in params:
        params[key] = int(params[key])

    print("\n Best Parameters Found:")
    print(params)

     # Re-train best model with best parameters
    if self.best_model_name == "XGBOOST":
        self.final_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4, **params)
        self.final_model.fit(self.x_train,self.y_train)
        y_pred = self.final_model.predict(self.x_test)
        self.accuracy_xgboost = accuracy_score(self.y_test,y_pred)
        print("XGBOOST Results")
        print("accuracy score :",self.accuracy_xgboost)
        precision, recall, f1_vals, _ = precision_recall_fscore_support(self.y_test,y_pred)
        for i, v in enumerate(['p1','p2','p3','p4']):
          print(f"Class{v}")
          print(f"Precision: {precision[i]}")
          print(f"recall: {recall[i]}")
          print(f"F1_score: {f1_vals[i]}")
          print()
    elif self.best_model_name == "RandomForest":
        self.final_model = RandomForestClassifier(random_state=42, **params)
        self.final_model.fit(self.x_train,self.y_train)
        y_pred = self.final_model.predict(self.x_test)
        self.accuracy_randomforest = accuracy_score(self.y_test,y_pred)
        print("Random Forest Results")
        print("accuracy score :",self.accuracy_randomforest)
        precision, recall, f1_vals, _ = precision_recall_fscore_support(self.y_test,y_pred)
        for i, v in enumerate(['p1','p2','p3','p4']):
          print(f"Class{v}")
          print(f"Precision: {precision[i]}")
          print(f"recall: {recall[i]}")
          print(f"F1_score: {f1_vals[i]}")
          print()
    elif self.best_model_name == "DecisionTree":
        self.final_model = DecisionTreeClassifier(random_state=42, **params)
        self.final_model.fit(self.x_train,self.y_train)
        y_pred = self.final_model.predict(self.x_test)
        self.accuracy_decisiontree = accuracy_score(self.y_test,y_pred)
        print("Decision Tree Results")
        print("accuracy score :",self.accuracy_decisiontree)
        precision, recall, f1_vals, _ = precision_recall_fscore_support(self.y_test,y_pred)
        for i, v in enumerate(['p1','p2','p3','p4']):
          print(f"Class{v}")
          print(f"Precision: {precision[i]}")
          print(f"recall: {recall[i]}")
          print(f"F1_score: {f1_vals[i]}")
          print() 
    self.final_model.fit(self.x_train, self.y_train)
    print("\n✅ Final tuned model trained successfully!") 

    with open(os.path.join(os.getcwd(), "best_model.pkl"),"wb") as f:
      pickle.dump(self.final_model,f)

  def model_evaluation(self,model,data):
    unseen_data = self.encode_features(data)  
    prediction = pd.Series(model.predict(unseen_data))
    print(prediction.value_counts())  # It is performing somewhat nearby to original data if we compare value counts 
    unseen_data['Approved_Flag']=prediction       
    unseen_data.to_csv('test_predictions.csv',index=False)
    print("Predictions saved to test_predictions.csv") 


  def data_pipeline(self):
    self.data_collection()
    data = self.data_preprocessing()
    dataset = self.perform_eda_and_feature_selection(data)
    datasetEncoded = self.encode_features(dataset)
    self.train_models(datasetEncoded)
    self.best_model_to_tune()
    self.tune_hyperparameters()
    self.model_evaluation(self.final_model,self.testing_data)
    print("All tasks ran succesfully")
    return self.final_model

def main():
  try:
    internal_data = pd.read_excel("/home/ankitmaan/Credit Risk Modelling/Datasets/case_study1.xlsx")
  except:
    raise FileNotFoundError("File Path of internal data is wrong, put it once again") 
  try:
    external_data= pd.read_excel("/home/ankitmaan/Credit Risk Modelling/Datasets/case_study2.xlsx")
  except:
    raise FileNotFoundError("File Path is wrong of external data, put it once again")
  # two datasets one is internal from the bank and one is cibil dataset(external dataset)
  try:
    testing_data = pd.read_excel("/home/ankitmaan/Credit Risk Modelling/Datasets/Unseen_Dataset.xlsx")
  except:
    raise FileNotFoundError("File Path of testing data is wrong put it once again") 
  #this is testing data to test whether our model is performing great or not
  model = CreditModel(internal_data,external_data,testing_data)
  model.data_pipeline()

if __name__ == "__main__":
  main()  

