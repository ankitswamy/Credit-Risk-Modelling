import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from itertools import product
import xgboost as xgb
import warnings
import pickle
warnings.filterwarnings('ignore')


class CreditModel:
    def __init__(self, internal_data, external_data, testing_data):
        self.internal_data = internal_data
        self.external_data = external_data
        self.testing_data = testing_data
        self.scalers = {}  # Store scalers for each column

    def data_collection(self):
        self.internal = self.internal_data.copy()
        self.external = self.external_data.copy()

    def data_preprocessing(self):
        # Remove nulls from internal data
        self.internal = self.internal.loc[self.internal['Age_Oldest_TL'] != -99999]
        
        # Clean external dataset
        columns_to_be_removed = []
        for i in self.external.columns:
            if self.external[self.external[i] == -99999].shape[0] > 10000:
                columns_to_be_removed.append(i)
        
        self.external = self.external.drop(columns_to_be_removed, axis=1)
        
        # Remove rows with null values
        for i in self.external.columns:
            self.external = self.external.loc[self.external[i] != -99999]
        
        # Merge datasets
        dataset = pd.merge(self.internal, self.external, how="inner", 
                          left_on=["PROSPECTID"], right_on=["PROSPECTID"])
        
        print(f"✅ Data preprocessing complete. Dataset shape: {dataset.shape}")
        print(f"Target distribution:\n{dataset['Approved_Flag'].value_counts()}")
        
        return dataset

    def chi_square(self, dataset):
        print("\n📊 Chi-Square Test Results:")
        categorical_cols = ['MARITALSTATUS', 'EDUCATION', 'GENDER', 
                           'last_prod_enq2', 'first_prod_enq2']
        
        for col in categorical_cols:
            if col in dataset.columns:
                chi2, pval, _, _ = chi2_contingency(pd.crosstab(dataset[col], dataset['Approved_Flag']))
                print(f"  {col}: p-value = {pval:.6f} {'✓ Significant' if pval < 0.05 else '✗ Not significant'}")

    def vif_selector(self, dataset):
        print("\n📊 VIF Selection:")
        # Select numerical columns
        self.numerical_columns = []
        for col in dataset.columns:
            if dataset[col].dtype != 'object' and col not in ["PROSPECTID", "Approved_Flag"]:
                self.numerical_columns.append(col)
        
        print(f"Initial numerical columns: {len(self.numerical_columns)}")
        
        # VIF check
        VIFdataset = dataset[self.numerical_columns].copy()
        totalColumns = len(self.numerical_columns)
        self.columns_to_be_kept = []
        column_index = 0
        
        for i in range(totalColumns):
            if len(VIFdataset.columns) <= column_index:
                break
            
            col_name = VIFdataset.columns[column_index]
            
            # Check for constant column
            if VIFdataset[col_name].std() == 0:
                print(f"  Dropping {col_name}: constant column")
                VIFdataset = VIFdataset.drop(col_name, axis=1)
                continue
            
            try:
                VIF = variance_inflation_factor(VIFdataset.values, column_index)
                
                if np.isnan(VIF) or np.isinf(VIF) or VIF > 6:
                    print(f"  Dropping {col_name}: VIF = {VIF:.2f}")
                    VIFdataset = VIFdataset.drop(col_name, axis=1)
                else:
                    self.columns_to_be_kept.append(col_name)
                    column_index += 1
            except:
                print(f"  Dropping {col_name}: VIF calculation failed")
                VIFdataset = VIFdataset.drop(col_name, axis=1)
        
        print(f"Columns after VIF: {len(self.columns_to_be_kept)}")
        return self.columns_to_be_kept

    def AnnovaTest(self, dataset):
        print("\n📊 ANOVA Test:")
        self.columns_to_be_kept_numerical = []
        
        for col in self.columns_to_be_kept:
            a = list(dataset[col])
            b = list(dataset['Approved_Flag'])
            
            # Group data by target class
            groupP1 = [value for value, group in zip(a, b) if group == 'P1']
            groupP2 = [value for value, group in zip(a, b) if group == 'P2']
            groupP3 = [value for value, group in zip(a, b) if group == 'P3']
            groupP4 = [value for value, group in zip(a, b) if group == 'P4']
            
            groups = [g for g in [groupP1, groupP2, groupP3, groupP4] if len(g) >= 2]
            
            if len(groups) < 2:
                continue
            
            f_statistics, pvalue = f_oneway(*groups)
            
            if pvalue <= 0.05:
                self.columns_to_be_kept_numerical.append(col)
                print(f"  {col}: p-value = {pvalue:.6f} ✓")
        
        print(f"Columns after ANOVA: {len(self.columns_to_be_kept_numerical)}")
        return self.columns_to_be_kept_numerical

    def perform_eda_and_feature_selection(self, dataset):
        print("\n🔍 Feature Selection Process:")
        
        # Chi-square test
        self.chi_square(dataset)
        
        # VIF test
        self.vif_selector(dataset)
        
        # ANOVA test
        self.AnnovaTest(dataset)
        
        # Final columns
        finalColumns = self.columns_to_be_kept_numerical + [
            'MARITALSTATUS', 'EDUCATION', 'GENDER', 
            'last_prod_enq2', 'first_prod_enq2'
        ]
        
        dataset = dataset[finalColumns + ['Approved_Flag']]
        print(f"\n✅ Final dataset shape: {dataset.shape}")
        
        return dataset

    def encode_features(self, dataset):
        print("\n🔧 Encoding Features:")
        
        # Encode EDUCATION (ordinal)
        education_mapping = {
            'SSC': 1,
            '12TH': 2,
            'GRADUATE': 3,
            'UNDER GRADUATE': 3,
            'POST-GRADUATE': 4,
            'OTHERS': 1,
            'PROFESSIONAL': 3
        }
        
        dataset['EDUCATION'] = dataset['EDUCATION'].map(education_mapping)
        dataset['EDUCATION'] = dataset['EDUCATION'].fillna(1).astype(int)
        
        # One-hot encoding for other categorical variables
        datasetEncoded = pd.get_dummies(
            dataset, 
            columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'],
            dtype=int
        )
        
        print(f"Encoded dataset shape: {datasetEncoded.shape}")
        print(f"Columns: {list(datasetEncoded.columns)}")
        
        return datasetEncoded

    def train_random_forest(self, x_train, y_train, x_test, y_test):
        print("\n🌲 Training Random Forest...")
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        self.accuracy_randomforest = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {self.accuracy_randomforest:.4f}")
        
        precision, recall, f1_vals, _ = precision_recall_fscore_support(y_test, y_pred)
        for i, v in enumerate(['P1', 'P2', 'P3', 'P4']):
            print(f"  Class {v}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1_vals[i]:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")

    def train_xgboost(self, x_train, y_train, x_test, y_test):
        print("\n🚀 Training XGBoost...")
        model = xgb.XGBClassifier(objective='multi:softmax', num_class=4, random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        self.accuracy_xgboost = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {self.accuracy_xgboost:.4f}")
        
        precision, recall, f1_vals, _ = precision_recall_fscore_support(y_test, y_pred)
        for i, v in enumerate(['P1', 'P2', 'P3', 'P4']):
            print(f"  Class {v}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1_vals[i]:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")

    def train_decision_tree(self, x_train, y_train, x_test, y_test):
        print("\n🌳 Training Decision Tree...")
        model = DecisionTreeClassifier(max_depth=20, min_samples_split=10, random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        self.accuracy_decisiontree = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {self.accuracy_decisiontree:.4f}")
        
        precision, recall, f1_vals, _ = precision_recall_fscore_support(y_test, y_pred)
        for i, v in enumerate(['P1', 'P2', 'P3', 'P4']):
            print(f"  Class {v}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1_vals[i]:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")

    def train_models(self, datasetEncoded):
        print("\n🎯 Training Models:")
        
        # Label encoding
        self.labelencoder = LabelEncoder()
        y = self.labelencoder.fit_transform(datasetEncoded['Approved_Flag'])
        x = datasetEncoded.drop(['Approved_Flag'], axis=1)
        
        print(f"Label mapping: {dict(zip(self.labelencoder.classes_, self.labelencoder.transform(self.labelencoder.classes_)))}")
        
        # Train-test split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train set: {self.x_train.shape}, Test set: {self.x_test.shape}")
        print(f"Train target distribution: {np.bincount(self.y_train)}")
        print(f"Test target distribution: {np.bincount(self.y_test)}")
        
        # Train without scaling
        print("\n--- Training WITHOUT Scaling ---")
        self.train_random_forest(self.x_train, self.y_train, self.x_test, self.y_test)
        self.train_xgboost(self.x_train, self.y_train, self.x_test, self.y_test)
        self.train_decision_tree(self.x_train, self.y_train, self.x_test, self.y_test)
        
        # Apply scaling
        print("\n--- Applying Standard Scaling ---")
        columns_to_be_scaled = [
            'Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment',
            'max_recent_level_of_deliq', 'recent_level_of_deliq',
            'time_since_recent_enq', 'NETMONTHLYINCOME', 'Time_With_Curr_Empr'
        ]
        
        # Store scalers for later use
        for col in columns_to_be_scaled:
            if col in datasetEncoded.columns:
                scaler = StandardScaler()
                column_data = datasetEncoded[col].values.reshape(-1, 1)
                scaled_column = scaler.fit_transform(column_data)
                datasetEncoded[col] = scaled_column.flatten()
                self.scalers[col] = scaler  # Store scaler
                print(f"  Scaled {col}: mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f}")
        
        # Re-split data
        x = datasetEncoded.drop(['Approved_Flag'], axis=1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Store feature names for later
        self.feature_names = list(x.columns)
        print(f"\nFinal feature names: {self.feature_names}")
        
        # Train with scaling
        print("\n--- Training WITH Scaling ---")
        self.train_random_forest(self.x_train, self.y_train, self.x_test, self.y_test)
        self.train_xgboost(self.x_train, self.y_train, self.x_test, self.y_test)
        self.train_decision_tree(self.x_train, self.y_train, self.x_test, self.y_test)

    def best_model_to_tune(self):
        self.model_results = {
            'RandomForest': self.accuracy_randomforest,
            'XGBOOST': self.accuracy_xgboost,
            'DecisionTree': self.accuracy_decisiontree
        }
        
        self.best_model_name = max(self.model_results, key=self.model_results.get)
        print(f"\n🏆 Best model: {self.best_model_name} (Accuracy: {self.model_results[self.best_model_name]:.4f})")
        return self.best_model_name

    def tune_hyperparameters(self):
        print("\n🔧 Hyperparameter Tuning:")
        
        results_list = []
        
        # Parameter grids
        parameterGrids = {
            'XGBOOST': {
                'colsample_bytree': [0.7, 0.9],
                'learning_rate': [0.01, 0.1],
                'max_depth': [5, 8],
                'alpha': [1, 10],
                'n_estimators': [100, 200],
                'subsample': [0.7, 0.9]
            },
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [15, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'DecisionTree': {
                'max_depth': [15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2],
                'criterion': ['gini', 'entropy']
            }
        }
        
        grid = parameterGrids[self.best_model_name]
        parameter_combinations = list(product(*grid.values()))
        
        print(f"Testing {len(parameter_combinations)} parameter combinations...")
        
        for idx, parameters in enumerate(parameter_combinations):
            parameters_dict = dict(zip(grid.keys(), parameters))
            
            # Create model
            if self.best_model_name == "XGBOOST":
                model = xgb.XGBClassifier(
                    objective='multi:softmax', 
                    num_class=4, 
                    random_state=42,
                    **parameters_dict
                )
            elif self.best_model_name == "RandomForest":
                model = RandomForestClassifier(random_state=42, **parameters_dict)
            elif self.best_model_name == "DecisionTree":
                model = DecisionTreeClassifier(random_state=42, **parameters_dict)
            
            # Train and evaluate
            model.fit(self.x_train, self.y_train)
            
            train_pred = model.predict(self.x_train)
            test_pred = model.predict(self.x_test)
            
            train_acc = accuracy_score(self.y_train, train_pred)
            test_acc = accuracy_score(self.y_test, test_pred)
            train_f1 = f1_score(self.y_train, train_pred, average='macro')
            test_f1 = f1_score(self.y_test, test_pred, average='macro')
            
            results_list.append({
                'Combination': idx + 1,
                'Train Accuracy': train_acc,
                'Test Accuracy': test_acc,
                'Train F1': train_f1,
                'Test F1': test_f1,
                **parameters_dict
            })
            
            if (idx + 1) % 10 == 0:
                print(f"  Tested {idx + 1}/{len(parameter_combinations)} combinations...")
        
        # Analyze results
        results = pd.DataFrame(results_list)
        results['Generalization Gap'] = abs(results['Train F1'] - results['Test F1'])
        results['Combined Score'] = results['Test F1'] - 0.2 * results['Generalization Gap']
        
        best_row = results.loc[results['Combined Score'].idxmax()]
        self.best_values = best_row.to_dict()
        
        # Extract parameters
        exclude_keys = ['Test Accuracy', 'Train Accuracy', 'Train F1', 'Test F1', 
                       'Generalization Gap', 'Combination', 'Combined Score']
        params = {k: v for k, v in self.best_values.items() if k not in exclude_keys}
        
        # Convert to int where needed
        int_params = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf']
        for key in int_params:
            if key in params:
                params[key] = int(params[key])
        
        print(f"\n🎯 Best Parameters: {params}")
        print(f"Best Test F1: {best_row['Test F1']:.4f}")
        print(f"Best Test Accuracy: {best_row['Test Accuracy']:.4f}")
        
        # Train final model
        if self.best_model_name == "XGBOOST":
            self.final_model = xgb.XGBClassifier(
                objective='multi:softmax', 
                num_class=4, 
                random_state=42,
                **params
            )
        elif self.best_model_name == "RandomForest":
            self.final_model = RandomForestClassifier(random_state=42, **params)
        elif self.best_model_name == "DecisionTree":
            self.final_model = DecisionTreeClassifier(random_state=42, **params)
        
        self.final_model.fit(self.x_train, self.y_train)
        
        # Final evaluation
        y_pred = self.final_model.predict(self.x_test)
        final_acc = accuracy_score(self.y_test, y_pred)
        final_f1 = f1_score(self.y_test, y_pred, average='macro')
        
        print(f"\n✅ Final Model Performance:")
        print(f"  Accuracy: {final_acc:.4f}")
        print(f"  F1 Score: {final_f1:.4f}")
        
        precision, recall, f1_vals, _ = precision_recall_fscore_support(self.y_test, y_pred)
        for i, v in enumerate(['P1', 'P2', 'P3', 'P4']):
            print(f"  Class {v}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1_vals[i]:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix:\n{cm}")
        
        # Save model
        model_path = os.path.join(os.getcwd(), "best_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.final_model, f)
        print(f"\n💾 Model saved to: {model_path}")

    def model_evaluation(self, model, data):
        print("\n📊 Evaluating on Test Data:")
        
        # Encode features
        unseen_data = self.encode_features(data)
        
        # Make predictions
        predictions = model.predict(unseen_data)
        prediction_classes = self.labelencoder.inverse_transform(predictions)
        
        print(f"Prediction distribution:\n{pd.Series(prediction_classes).value_counts()}")
        
        # Save predictions
        unseen_data['Approved_Flag'] = prediction_classes
        output_path = 'test_predictions.csv'
        unseen_data.to_csv(output_path, index=False)
        print(f"✅ Predictions saved to: {output_path}")

    def data_pipeline(self):
        print("=" * 60)
        print("🚀 Credit Risk Model Training Pipeline")
        print("=" * 60)
        
        self.data_collection()
        data = self.data_preprocessing()
        dataset = self.perform_eda_and_feature_selection(data)
        datasetEncoded = self.encode_features(dataset)
        self.train_models(datasetEncoded)
        self.best_model_to_tune()
        self.tune_hyperparameters()
        self.model_evaluation(self.final_model, self.testing_data)
        
        print("\n" + "=" * 60)
        print("✅ Pipeline completed successfully!")
        print("=" * 60)
        
        return self.final_model


def main():
    # Update these paths to your actual file locations
    internal_path = "/home/ankitmaan/Credit Risk Modelling/Datasets/case_study1.xlsx"
    external_path = "/home/ankitmaan/Credit Risk Modelling/Datasets/case_study2.xlsx"
    testing_path = "/home/ankitmaan/Credit Risk Modelling/Datasets/Unseen_Dataset.xlsx"
    
    try:
        print("📂 Loading datasets...")
        internal_data = pd.read_excel(internal_path)
        print(f"  ✓ Internal data: {internal_data.shape}")
    except:
        raise FileNotFoundError(f"❌ Internal data not found at: {internal_path}")
    
    try:
        external_data = pd.read_excel(external_path)
        print(f"  ✓ External data: {external_data.shape}")
    except:
        raise FileNotFoundError(f"❌ External data not found at: {external_path}")
    
    try:
        testing_data = pd.read_excel(testing_path)
        print(f"  ✓ Testing data: {testing_data.shape}")
    except:
        raise FileNotFoundError(f"❌ Testing data not found at: {testing_path}")
    
    # Create and run model
    model = CreditModel(internal_data, external_data, testing_data)
    final_model = model.data_pipeline()
    
    return final_model


if __name__ == "__main__":
    main()