# import os
# import pickle
# import pandas as pd
# import numpy as np
# from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
# from werkzeug.utils import secure_filename
# from flask_wtf import FlaskForm
# from flask_wtf.csrf import CSRFProtect
# from wtforms import StringField, IntegerField, FloatField, SelectField, FileField, SubmitField, validators
# from wtforms.validators import DataRequired, NumberRange, Length
# import io
# import csv
# import json
# import warnings
# warnings.filterwarnings('ignore')

# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
# app.config['UPLOAD_FOLDER'] = 'uploads'

# # Initialize CSRF protection
# csrf = CSRFProtect(app)

# # Create upload folder if it doesn't exist
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Load the trained model
# try:
#     with open('best_model.pkl', 'rb') as f:
#         model = pickle.load(f)
#     print("✅ Model loaded successfully!")
# except FileNotFoundError:
#     print("❌ Model file 'best_model.pkl' not found!")
#     model = None

# # Define feature columns based on the model structure
# FEATURE_COLUMNS = [
#     'pct_tl_open_L6M', 'pct_tl_closed_L6M', 'Tot_TL_closed_L12M', 'pct_tl_closed_L12M',
#     'Tot_Missed_Pmnt', 'CC_TL', 'Home_TL', 'PL_TL', 'Secured_TL', 'Unsecured_TL', 'Other_TL',
#     'Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment', 'max_recent_level_of_deliq',
#     'num_deliq_6_12mts', 'num_times_60p_dpd', 'num_std_12mts', 'num_sub', 'num_sub_6mts',
#     'num_sub_12mts', 'num_dbt', 'num_dbt_12mts', 'num_lss', 'recent_level_of_deliq',
#     'CC_enq_L12m', 'PL_enq_L12m', 'time_since_recent_enq', 'enq_L3m', 'NETMONTHLYINCOME',
#     'Time_With_Curr_Empr', 'CC_Flag', 'PL_Flag', 'pct_PL_enq_L6m_of_ever', 'pct_CC_enq_L6m_of_ever',
#     'HL_Flag', 'GL_Flag', 'EDUCATION', 'MARITALSTATUS_Married', 'MARITALSTATUS_Single',
#     'GENDER_F', 'GENDER_M', 'last_prod_enq2_AL', 'last_prod_enq2_CC', 'last_prod_enq2_ConsumerLoan',
#     'last_prod_enq2_HL', 'last_prod_enq2_PL', 'last_prod_enq2_others', 'first_prod_enq2_AL',
#     'first_prod_enq2_CC', 'first_prod_enq2_ConsumerLoan', 'first_prod_enq2_HL', 'first_prod_enq2_PL',
#     'first_prod_enq2_others'
# ]

# # Class labels mapping
# CLASS_LABELS = {0: 'P1', 1: 'P2', 2: 'P3', 3: 'P4'}

# class CreditRiskForm(FlaskForm):
#     # Numerical fields with user-friendly descriptions
#     pct_tl_open_L6M = FloatField('New Credit Accounts Opened (Last 6 Months)', 
#                                 validators=[NumberRange(min=0, max=1, message="Must be between 0 and 1")],
#                                 description="What percentage of your credit accounts were opened in the last 6 months? (0.0 = none, 1.0 = all)")
#     pct_tl_closed_L6M = FloatField('Credit Accounts Closed (Last 6 Months)', 
#                                   validators=[NumberRange(min=0, max=1, message="Must be between 0 and 1")],
#                                   description="What percentage of your credit accounts were closed in the last 6 months?")
#     Tot_TL_closed_L12M = IntegerField('Total Credit Accounts Closed (Last 12 Months)', 
#                                      validators=[NumberRange(min=0, message="Must be non-negative")],
#                                      description="How many credit accounts (loans, cards) have you closed in the past year?")
#     pct_tl_closed_L12M = FloatField('Credit Accounts Closed (Last 12 Months)', 
#                                    validators=[NumberRange(min=0, max=1, message="Must be between 0 and 1")],
#                                    description="What percentage of your credit accounts were closed in the last 12 months?")
#     Tot_Missed_Pmnt = IntegerField('Total Missed Payments', 
#                                   validators=[NumberRange(min=0, message="Must be non-negative")],
#                                   description="How many times have you missed payments on any credit account?")
#     CC_TL = IntegerField('Number of Credit Cards', 
#                         validators=[NumberRange(min=0, message="Must be non-negative")],
#                         description="How many credit cards do you currently have?")
#     Home_TL = IntegerField('Number of Home Loans', 
#                           validators=[NumberRange(min=0, message="Must be non-negative")],
#                           description="How many home loans or mortgages do you currently have?")
#     PL_TL = IntegerField('Number of Personal Loans', 
#                         validators=[NumberRange(min=0, message="Must be non-negative")],
#                         description="How many personal loans do you currently have?")
#     Secured_TL = IntegerField('Number of Secured Loans', 
#                              validators=[NumberRange(min=0, message="Must be non-negative")],
#                              description="How many loans do you have that are backed by collateral (like home, car)?")
#     Unsecured_TL = IntegerField('Number of Unsecured Loans', 
#                                validators=[NumberRange(min=0, message="Must be non-negative")],
#                                description="How many loans do you have that are not backed by any collateral?")
#     Other_TL = IntegerField('Number of Other Credit Accounts', 
#                            validators=[NumberRange(min=0, message="Must be non-negative")],
#                            description="How many other types of credit accounts do you have?")
#     Age_Oldest_TL = IntegerField('Age of Oldest Credit Account (months)', 
#                                 validators=[NumberRange(min=0, message="Must be non-negative")],
#                                 description="How many months ago did you open your oldest credit account?")
#     Age_Newest_TL = IntegerField('Age of Newest Credit Account (months)', 
#                                 validators=[NumberRange(min=0, message="Must be non-negative")],
#                                 description="How many months ago did you open your newest credit account?")
#     time_since_recent_payment = IntegerField('Days Since Last Payment', 
#                                             validators=[NumberRange(min=0, message="Must be non-negative")],
#                                             description="How many days ago did you make your most recent payment on any credit account?")
#     max_recent_level_of_deliq = IntegerField('Highest Recent Payment Delay', 
#                                             validators=[NumberRange(min=0, message="Must be non-negative")],
#                                             description="What is the highest number of days you were late on any recent payment?")
#     num_deliq_6_12mts = IntegerField('Payment Delays (6-12 Months Ago)', 
#                                      validators=[NumberRange(min=0, message="Must be non-negative")],
#                                      description="How many times were you late on payments between 6-12 months ago?")
#     num_times_60p_dpd = IntegerField('Severe Payment Delays (60+ Days)', 
#                                     validators=[NumberRange(min=0, message="Must be non-negative")],
#                                     description="How many times have you been 60+ days late on payments?")
#     num_std_12mts = IntegerField('Good Payment History (Last 12 Months)', 
#                                 validators=[NumberRange(min=0, message="Must be non-negative")],
#                                 description="How many accounts have you maintained with good payment history in the last year?")
#     num_sub = IntegerField('Number of High-Risk Accounts', 
#                           validators=[NumberRange(min=0, message="Must be non-negative")],
#                           description="How many accounts do you have that are considered high-risk or subprime?")
#     num_sub_6mts = IntegerField('High-Risk Accounts (Last 6 Months)', 
#                                validators=[NumberRange(min=0, message="Must be non-negative")],
#                                description="How many high-risk accounts have you had in the last 6 months?")
#     num_sub_12mts = IntegerField('High-Risk Accounts (Last 12 Months)', 
#                                  validators=[NumberRange(min=0, message="Must be non-negative")],
#                                  description="How many high-risk accounts have you had in the last 12 months?")
#     num_dbt = IntegerField('Number of Debt Settlement Accounts', 
#                           validators=[NumberRange(min=0, message="Must be non-negative")],
#                           description="How many accounts have you settled for less than the full amount owed?")
#     num_dbt_12mts = IntegerField('Debt Settlement Accounts (Last 12 Months)', 
#                                 validators=[NumberRange(min=0, message="Must be non-negative")],
#                                 description="How many debt settlements have you had in the last 12 months?")
#     num_lss = IntegerField('Number of Written-Off Accounts', 
#                           validators=[NumberRange(min=0, message="Must be non-negative")],
#                           description="How many accounts have been written off as losses by lenders?")
#     recent_level_of_deliq = IntegerField('Current Payment Delay Status', 
#                                         validators=[NumberRange(min=0, message="Must be non-negative")],
#                                         description="What is your current payment delay status on any account?")
#     CC_enq_L12m = IntegerField('Credit Card Applications (Last 12 Months)', 
#                               validators=[NumberRange(min=0, message="Must be non-negative")],
#                               description="How many times have you applied for credit cards in the last year?")
#     PL_enq_L12m = IntegerField('Personal Loan Applications (Last 12 Months)', 
#                               validators=[NumberRange(min=0, message="Must be non-negative")],
#                               description="How many times have you applied for personal loans in the last year?")
#     time_since_recent_enq = IntegerField('Days Since Last Credit Application', 
#                                         validators=[NumberRange(min=0, message="Must be non-negative")],
#                                         description="How many days ago did you last apply for any type of credit?")
#     enq_L3m = IntegerField('Credit Applications (Last 3 Months)', 
#                           validators=[NumberRange(min=0, message="Must be non-negative")],
#                           description="How many times have you applied for credit in the last 3 months?")
#     NETMONTHLYINCOME = IntegerField('Monthly Income (After Tax)', 
#                                    validators=[NumberRange(min=0, message="Must be non-negative")],
#                                    description="What is your monthly take-home income after taxes?")
#     Time_With_Curr_Empr = IntegerField('Time with Current Employer (months)', 
#                                       validators=[NumberRange(min=0, message="Must be non-negative")],
#                                       description="How many months have you been working with your current employer?")
    
#     # Binary flags with user-friendly descriptions
#     CC_Flag = SelectField('Do you have Credit Cards?', 
#                          choices=[(0, 'No'), (1, 'Yes')], 
#                          coerce=int,
#                          description="Do you currently have any credit cards?")
#     PL_Flag = SelectField('Do you have Personal Loans?', 
#                          choices=[(0, 'No'), (1, 'Yes')], 
#                          coerce=int,
#                          description="Do you currently have any personal loans?")
#     HL_Flag = SelectField('Do you have Home Loans?', 
#                          choices=[(0, 'No'), (1, 'Yes')], 
#                          coerce=int,
#                          description="Do you currently have any home loans or mortgages?")
#     GL_Flag = SelectField('Do you have Gold Loans?', 
#                          choices=[(0, 'No'), (1, 'Yes')], 
#                          coerce=int,
#                          description="Do you currently have any gold loans?")
    
#     # Percentage fields with user-friendly descriptions
#     pct_PL_enq_L6m_of_ever = FloatField('Personal Loan Applications (Last 6 Months)', 
#                                        validators=[NumberRange(min=0, max=1, message="Must be between 0 and 1")],
#                                        description="What percentage of your total personal loan applications were made in the last 6 months?")
#     pct_CC_enq_L6m_of_ever = FloatField('Credit Card Applications (Last 6 Months)', 
#                                        validators=[NumberRange(min=0, max=1, message="Must be between 0 and 1")],
#                                        description="What percentage of your total credit card applications were made in the last 6 months?")
    
#     # Categorical fields with user-friendly descriptions
#     EDUCATION = SelectField('Education Level', 
#                           choices=[(1, 'High School/SSC/Others'), (2, '12th Grade'), (3, 'Graduate/Under Graduate/Professional'), (4, 'Post Graduate')], 
#                           coerce=int,
#                           description="What is your highest level of education completed?")
    
#     MARITALSTATUS_Married = SelectField('Are you Married?', 
#                                        choices=[(0, 'No'), (1, 'Yes')], 
#                                        coerce=int,
#                                        description="Are you currently married?")
#     MARITALSTATUS_Single = SelectField('Are you Single?', 
#                                       choices=[(0, 'No'), (1, 'Yes')], 
#                                       coerce=int,
#                                       description="Are you currently single (not married)?")
    
#     GENDER_F = SelectField('Are you Female?', 
#                           choices=[(0, 'No'), (1, 'Yes')], 
#                           coerce=int,
#                           description="Are you female?")
#     GENDER_M = SelectField('Are you Male?', 
#                           choices=[(0, 'No'), (1, 'Yes')], 
#                           coerce=int,
#                           description="Are you male?")
    
#     # Product enquiry fields with user-friendly descriptions
#     last_prod_enq2_AL = SelectField('Last Application: Auto Loan', 
#                                    choices=[(0, 'No'), (1, 'Yes')], 
#                                    coerce=int,
#                                    description="Was your most recent credit application for an auto loan?")
#     last_prod_enq2_CC = SelectField('Last Application: Credit Card', 
#                                    choices=[(0, 'No'), (1, 'Yes')], 
#                                    coerce=int,
#                                    description="Was your most recent credit application for a credit card?")
#     last_prod_enq2_ConsumerLoan = SelectField('Last Application: Consumer Loan', 
#                                              choices=[(0, 'No'), (1, 'Yes')], 
#                                              coerce=int,
#                                              description="Was your most recent credit application for a consumer loan?")
#     last_prod_enq2_HL = SelectField('Last Application: Home Loan', 
#                                    choices=[(0, 'No'), (1, 'Yes')], 
#                                    coerce=int,
#                                    description="Was your most recent credit application for a home loan?")
#     last_prod_enq2_PL = SelectField('Last Application: Personal Loan', 
#                                    choices=[(0, 'No'), (1, 'Yes')], 
#                                    coerce=int,
#                                    description="Was your most recent credit application for a personal loan?")
#     last_prod_enq2_others = SelectField('Last Application: Other Type', 
#                                        choices=[(0, 'No'), (1, 'Yes')], 
#                                        coerce=int,
#                                        description="Was your most recent credit application for some other type of loan?")
    
#     first_prod_enq2_AL = SelectField('First Application: Auto Loan', 
#                                      choices=[(0, 'No'), (1, 'Yes')], 
#                                      coerce=int,
#                                      description="Was your first credit application for an auto loan?")
#     first_prod_enq2_CC = SelectField('First Application: Credit Card', 
#                                     choices=[(0, 'No'), (1, 'Yes')], 
#                                     coerce=int,
#                                     description="Was your first credit application for a credit card?")
#     first_prod_enq2_ConsumerLoan = SelectField('First Application: Consumer Loan', 
#                                               choices=[(0, 'No'), (1, 'Yes')], 
#                                               coerce=int,
#                                               description="Was your first credit application for a consumer loan?")
#     first_prod_enq2_HL = SelectField('First Application: Home Loan', 
#                                      choices=[(0, 'No'), (1, 'Yes')], 
#                                      coerce=int,
#                                      description="Was your first credit application for a home loan?")
#     first_prod_enq2_PL = SelectField('First Application: Personal Loan', 
#                                     choices=[(0, 'No'), (1, 'Yes')], 
#                                     coerce=int,
#                                     description="Was your first credit application for a personal loan?")
#     first_prod_enq2_others = SelectField('First Application: Other Type', 
#                                         choices=[(0, 'No'), (1, 'Yes')], 
#                                         coerce=int,
#                                         description="Was your first credit application for some other type of loan?")
    
#     submit = SubmitField('Predict Credit Risk')

# class BatchUploadForm(FlaskForm):
#     batch_file = FileField('Upload Data File (CSV, Excel, or JSON)', validators=[DataRequired()])
#     submit = SubmitField('Upload and Predict')

# def validate_form_data(form_data):
#     """Validate form data and return cleaned data"""
#     cleaned_data = {}
    
#     for field in FEATURE_COLUMNS:
#         if field in form_data:
#             value = form_data[field]
#             if value == '' or value is None:
#                 cleaned_data[field] = 0
#             else:
#                 try:
#                     # Convert to appropriate type
#                     if field in ['pct_tl_open_L6M', 'pct_tl_closed_L6M', 'pct_tl_closed_L12M', 
#                                'pct_PL_enq_L6m_of_ever', 'pct_CC_enq_L6m_of_ever']:
#                         cleaned_data[field] = float(value)
#                     else:
#                         cleaned_data[field] = int(value)
#                 except (ValueError, TypeError):
#                     cleaned_data[field] = 0
#         else:
#             cleaned_data[field] = 0
    
#     return cleaned_data

# def prepare_prediction_data(data):
#     """Prepare data for prediction in the correct format"""
#     # Create DataFrame with all features in the correct order
#     df = pd.DataFrame([data])
    
#     # Ensure all columns are present and in correct order
#     for col in FEATURE_COLUMNS:
#         if col not in df.columns:
#             df[col] = 0
    
#     # Reorder columns to match model expectations
#     df = df[FEATURE_COLUMNS]
    
#     return df

# def read_data_file(file, filename):
#     """Read data from different file formats (CSV, Excel, JSON)"""
#     try:
#         # Get file extension
#         file_extension = filename.lower().split('.')[-1]
        
#         if file_extension == 'csv':
#             # Read CSV file
#             df = pd.read_csv(file)
            
#         elif file_extension in ['xlsx', 'xls']:
#             # Read Excel file
#             df = pd.read_excel(file)
            
#         elif file_extension == 'json':
#             # Read JSON file
#             file.seek(0)  # Reset file pointer
#             json_data = json.load(file)
            
#             # Handle different JSON structures
#             if isinstance(json_data, list):
#                 # List of dictionaries
#                 df = pd.DataFrame(json_data)
#             elif isinstance(json_data, dict):
#                 if 'data' in json_data:
#                     # JSON with 'data' key
#                     df = pd.DataFrame(json_data['data'])
#                 elif 'records' in json_data:
#                     # JSON with 'records' key
#                     df = pd.DataFrame(json_data['records'])
#                 else:
#                     # Single record as dictionary
#                     df = pd.DataFrame([json_data])
#             else:
#                 raise ValueError("Invalid JSON structure")
                
#         else:
#             raise ValueError(f"Unsupported file format: {file_extension}")
            
#         return df
        
#     except Exception as e:
#         raise ValueError(f"Error reading file: {str(e)}")

# def validate_and_prepare_batch_data(df):
#     """Validate and prepare batch data for prediction"""
#     # Check if DataFrame is empty
#     if df.empty:
#         raise ValueError("File is empty or contains no data")
    
#     # Validate required columns
#     missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
#     if missing_cols:
#         raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
#     # Select only required columns and fill missing values
#     df_processed = df[FEATURE_COLUMNS].fillna(0)
    
#     # Convert data types
#     for col in FEATURE_COLUMNS:
#         if col in ['pct_tl_open_L6M', 'pct_tl_closed_L6M', 'pct_tl_closed_L12M', 
#                    'pct_PL_enq_L6m_of_ever', 'pct_CC_enq_L6m_of_ever']:
#             # Percentage fields
#             df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
#             # Ensure values are between 0 and 1
#             df_processed[col] = df_processed[col].clip(0, 1)
#         else:
#             # Integer fields
#             df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0).astype(int)
    
#     return df_processed

# @app.route('/')
# def index():
#     form = CreditRiskForm()
#     batch_form = BatchUploadForm()
    
#     # Add descriptions as data attributes for JavaScript
#     for field_name, field in form._fields.items():
#         if hasattr(field, 'description') and field.description:
#             field.render_kw = field.render_kw or {}
#             field.render_kw['data-description'] = field.description
    
#     return render_template('index.html', form=form, batch_form=batch_form)

# @app.route('/predict', methods=['POST'])
# def predict():
#     if model is None:
#         return jsonify({'error': 'Model not loaded'}), 500
    
#     try:
#         # Get form data
#         form_data = request.form.to_dict()
        
#         # Validate and clean data
#         cleaned_data = validate_form_data(form_data)
        
#         # Prepare data for prediction
#         prediction_data = prepare_prediction_data(cleaned_data)
        
#         # Make prediction
#         prediction = model.predict(prediction_data)[0]
#         prediction_class = CLASS_LABELS[prediction]
        
#         # Get prediction probabilities if available
#         probabilities = None
#         if hasattr(model, 'predict_proba'):
#             proba = model.predict_proba(prediction_data)[0]
#             probabilities = {
#                 CLASS_LABELS[i]: float(prob) for i, prob in enumerate(proba)
#             }
        
#         return jsonify({
#             'prediction': prediction_class,
#             'probabilities': probabilities,
#             'success': True
#         })
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/predict_batch', methods=['POST'])
# def predict_batch():
#     if model is None:
#         return jsonify({'error': 'Model not loaded'}), 500
    
#     try:
#         # Check if file was uploaded
#         if 'batch_file' not in request.files:
#             return jsonify({'error': 'No file uploaded'}), 400
        
#         file = request.files['batch_file']
#         if file.filename == '':
#             return jsonify({'error': 'No file selected'}), 400
        
#         # Check file extension
#         file_extension = file.filename.lower().split('.')[-1]
#         if file_extension not in ['csv', 'xlsx', 'xls', 'json']:
#             return jsonify({'error': 'File must be CSV, Excel (.xlsx/.xls), or JSON format'}), 400
        
#         # Read data file
#         df = read_data_file(file, file.filename)
        
#         # Validate and prepare data
#         prediction_data = validate_and_prepare_batch_data(df)
        
#         # Make predictions
#         predictions = model.predict(prediction_data)
#         prediction_classes = [CLASS_LABELS[pred] for pred in predictions]
        
#         # Add predictions to dataframe
#         df['predicted_class'] = prediction_classes
        
#         # Get probabilities if available
#         if hasattr(model, 'predict_proba'):
#             probabilities = model.predict_proba(prediction_data)
#             for i, class_label in CLASS_LABELS.items():
#                 df[f'probability_{class_label}'] = probabilities[:, i]
        
#         # Determine output format based on input format
#         if file_extension == 'json':
#             # Return JSON response
#             output_data = df.to_dict('records')
#             return jsonify({
#                 'success': True,
#                 'predictions': output_data,
#                 'total_records': len(df)
#             })
#         else:
#             # Return CSV response for CSV and Excel files
#             output = io.StringIO()
#             df.to_csv(output, index=False)
#             output.seek(0)
            
#             # Create response
#             response = app.response_class(
#                 output.getvalue(),
#                 mimetype='text/csv',
#                 headers={'Content-Disposition': f'attachment; filename=predictions_{file_extension}.csv'}
#             )
            
#             return response
        
#     except ValueError as e:
#         return jsonify({'error': str(e)}), 400
#     except Exception as e:
#         return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

# @app.route('/health')
# def health():
#     return jsonify({'status': 'healthy', 'model_loaded': model is not None})

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)

import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from wtforms import IntegerField, FloatField, SelectField, FileField, SubmitField
from wtforms.validators import DataRequired, NumberRange
import io
import json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
try:
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Model classes: {model.classes_ if hasattr(model, 'classes_') else 'N/A'}")
except FileNotFoundError:
    print("❌ Model file 'best_model.pkl' not found!")
    model = None
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model = None

# Define feature columns based on the model structure
FEATURE_COLUMNS = [
    'pct_tl_open_L6M', 'pct_tl_closed_L6M', 'Tot_TL_closed_L12M', 'pct_tl_closed_L12M',
    'Tot_Missed_Pmnt', 'CC_TL', 'Home_TL', 'PL_TL', 'Secured_TL', 'Unsecured_TL', 'Other_TL',
    'Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment', 'max_recent_level_of_deliq',
    'num_deliq_6_12mts', 'num_times_60p_dpd', 'num_std_12mts', 'num_sub', 'num_sub_6mts',
    'num_sub_12mts', 'num_dbt', 'num_dbt_12mts', 'num_lss', 'recent_level_of_deliq',
    'CC_enq_L12m', 'PL_enq_L12m', 'time_since_recent_enq', 'enq_L3m', 'NETMONTHLYINCOME',
    'Time_With_Curr_Empr', 'CC_Flag', 'PL_Flag', 'pct_PL_enq_L6m_of_ever', 'pct_CC_enq_L6m_of_ever',
    'HL_Flag', 'GL_Flag', 'EDUCATION', 'MARITALSTATUS_Married', 'MARITALSTATUS_Single',
    'GENDER_F', 'GENDER_M', 'last_prod_enq2_AL', 'last_prod_enq2_CC', 'last_prod_enq2_ConsumerLoan',
    'last_prod_enq2_HL', 'last_prod_enq2_PL', 'last_prod_enq2_others', 'first_prod_enq2_AL',
    'first_prod_enq2_CC', 'first_prod_enq2_ConsumerLoan', 'first_prod_enq2_HL', 'first_prod_enq2_PL',
    'first_prod_enq2_others'
]

# Class labels mapping - Adjusted based on your model
CLASS_LABELS = {0: 'P1', 1: 'P2', 2: 'P3', 3: 'P4'}

# Approval mapping: P1 and P2 are typically approved, P3 and P4 are rejected
APPROVAL_MAPPING = {
    'P1': 'Approved',
    'P2': 'Approved', 
    'P3': 'Rejected',
    'P4': 'Rejected'
}

class CreditRiskForm(FlaskForm):
    # Your existing form fields (keeping them as is)
    pct_tl_open_L6M = FloatField('New Credit Accounts Opened (Last 6 Months)', 
                                validators=[NumberRange(min=0, max=1)],
                                description="What percentage of your credit accounts were opened in the last 6 months?")
    pct_tl_closed_L6M = FloatField('Credit Accounts Closed (Last 6 Months)', 
                                  validators=[NumberRange(min=0, max=1, message="Must be between 0 and 1")],
                                  description="What percentage of your credit accounts were closed in the last 6 months?")
    Tot_TL_closed_L12M = IntegerField('Total Credit Accounts Closed (Last 12 Months)', 
                                     validators=[NumberRange(min=0, message="Must be non-negative")],
                                     description="How many credit accounts (loans, cards) have you closed in the past year?")
    pct_tl_closed_L12M = FloatField('Credit Accounts Closed (Last 12 Months)', 
                                   validators=[NumberRange(min=0, max=1, message="Must be between 0 and 1")],
                                   description="What percentage of your credit accounts were closed in the last 12 months?")
    Tot_Missed_Pmnt = IntegerField('Total Missed Payments', 
                                  validators=[NumberRange(min=0, message="Must be non-negative")],
                                  description="How many times have you missed payments on any credit account?")
    CC_TL = IntegerField('Number of Credit Cards', 
                        validators=[NumberRange(min=0, message="Must be non-negative")],
                        description="How many credit cards do you currently have?")
    Home_TL = IntegerField('Number of Home Loans', 
                          validators=[NumberRange(min=0, message="Must be non-negative")],
                          description="How many home loans or mortgages do you currently have?")
    PL_TL = IntegerField('Number of Personal Loans', 
                        validators=[NumberRange(min=0, message="Must be non-negative")],
                        description="How many personal loans do you currently have?")
    Secured_TL = IntegerField('Number of Secured Loans', 
                             validators=[NumberRange(min=0, message="Must be non-negative")],
                             description="How many loans do you have that are backed by collateral (like home, car)?")
    Unsecured_TL = IntegerField('Number of Unsecured Loans', 
                               validators=[NumberRange(min=0, message="Must be non-negative")],
                               description="How many loans do you have that are not backed by any collateral?")
    Other_TL = IntegerField('Number of Other Credit Accounts', 
                           validators=[NumberRange(min=0, message="Must be non-negative")],
                           description="How many other types of credit accounts do you have?")
    Age_Oldest_TL = IntegerField('Age of Oldest Credit Account (months)', 
                                validators=[NumberRange(min=0, message="Must be non-negative")],
                                description="How many months ago did you open your oldest credit account?")
    Age_Newest_TL = IntegerField('Age of Newest Credit Account (months)', 
                                validators=[NumberRange(min=0, message="Must be non-negative")],
                                description="How many months ago did you open your newest credit account?")
    time_since_recent_payment = IntegerField('Days Since Last Payment', 
                                            validators=[NumberRange(min=0, message="Must be non-negative")],
                                            description="How many days ago did you make your most recent payment on any credit account?")
    max_recent_level_of_deliq = IntegerField('Highest Recent Payment Delay', 
                                            validators=[NumberRange(min=0, message="Must be non-negative")],
                                            description="What is the highest number of days you were late on any recent payment?")
    num_deliq_6_12mts = IntegerField('Payment Delays (6-12 Months Ago)', 
                                     validators=[NumberRange(min=0, message="Must be non-negative")],
                                     description="How many times were you late on payments between 6-12 months ago?")
    num_times_60p_dpd = IntegerField('Severe Payment Delays (60+ Days)', 
                                    validators=[NumberRange(min=0, message="Must be non-negative")],
                                    description="How many times have you been 60+ days late on payments?")
    num_std_12mts = IntegerField('Good Payment History (Last 12 Months)', 
                                validators=[NumberRange(min=0, message="Must be non-negative")],
                                description="How many accounts have you maintained with good payment history in the last year?")
    num_sub = IntegerField('Number of High-Risk Accounts', 
                          validators=[NumberRange(min=0, message="Must be non-negative")],
                          description="How many accounts do you have that are considered high-risk or subprime?")
    num_sub_6mts = IntegerField('High-Risk Accounts (Last 6 Months)', 
                               validators=[NumberRange(min=0, message="Must be non-negative")],
                               description="How many high-risk accounts have you had in the last 6 months?")
    num_sub_12mts = IntegerField('High-Risk Accounts (Last 12 Months)', 
                                 validators=[NumberRange(min=0, message="Must be non-negative")],
                                 description="How many high-risk accounts have you had in the last 12 months?")
    num_dbt = IntegerField('Number of Debt Settlement Accounts', 
                          validators=[NumberRange(min=0, message="Must be non-negative")],
                          description="How many accounts have you settled for less than the full amount owed?")
    num_dbt_12mts = IntegerField('Debt Settlement Accounts (Last 12 Months)', 
                                validators=[NumberRange(min=0, message="Must be non-negative")],
                                description="How many debt settlements have you had in the last 12 months?")
    num_lss = IntegerField('Number of Written-Off Accounts', 
                          validators=[NumberRange(min=0, message="Must be non-negative")],
                          description="How many accounts have been written off as losses by lenders?")
    recent_level_of_deliq = IntegerField('Current Payment Delay Status', 
                                        validators=[NumberRange(min=0, message="Must be non-negative")],
                                        description="What is your current payment delay status on any account?")
    CC_enq_L12m = IntegerField('Credit Card Applications (Last 12 Months)', 
                              validators=[NumberRange(min=0, message="Must be non-negative")],
                              description="How many times have you applied for credit cards in the last year?")
    PL_enq_L12m = IntegerField('Personal Loan Applications (Last 12 Months)', 
                              validators=[NumberRange(min=0, message="Must be non-negative")],
                              description="How many times have you applied for personal loans in the last year?")
    time_since_recent_enq = IntegerField('Days Since Last Credit Application', 
                                        validators=[NumberRange(min=0, message="Must be non-negative")],
                                        description="How many days ago did you last apply for any type of credit?")
    enq_L3m = IntegerField('Credit Applications (Last 3 Months)', 
                          validators=[NumberRange(min=0, message="Must be non-negative")],
                          description="How many times have you applied for credit in the last 3 months?")
    NETMONTHLYINCOME = IntegerField('Monthly Income (After Tax)', 
                                   validators=[NumberRange(min=0, message="Must be non-negative")],
                                   description="What is your monthly take-home income after taxes?")
    Time_With_Curr_Empr = IntegerField('Time with Current Employer (months)', 
                                      validators=[NumberRange(min=0, message="Must be non-negative")],
                                      description="How many months have you been working with your current employer?")
    
    # Binary flags with user-friendly descriptions
    CC_Flag = SelectField('Do you have Credit Cards?', 
                         choices=[(0, 'No'), (1, 'Yes')], 
                         coerce=int,
                         description="Do you currently have any credit cards?")
    PL_Flag = SelectField('Do you have Personal Loans?', 
                         choices=[(0, 'No'), (1, 'Yes')], 
                         coerce=int,
                         description="Do you currently have any personal loans?")
    HL_Flag = SelectField('Do you have Home Loans?', 
                         choices=[(0, 'No'), (1, 'Yes')], 
                         coerce=int,
                         description="Do you currently have any home loans or mortgages?")
    GL_Flag = SelectField('Do you have Gold Loans?', 
                         choices=[(0, 'No'), (1, 'Yes')], 
                         coerce=int,
                         description="Do you currently have any gold loans?")
    
    # Percentage fields with user-friendly descriptions
    pct_PL_enq_L6m_of_ever = FloatField('Personal Loan Applications (Last 6 Months)', 
                                       validators=[NumberRange(min=0, max=1, message="Must be between 0 and 1")],
                                       description="What percentage of your total personal loan applications were made in the last 6 months?")
    pct_CC_enq_L6m_of_ever = FloatField('Credit Card Applications (Last 6 Months)', 
                                       validators=[NumberRange(min=0, max=1, message="Must be between 0 and 1")],
                                       description="What percentage of your total credit card applications were made in the last 6 months?")
    
    # Categorical fields with user-friendly descriptions
    EDUCATION = SelectField('Education Level', 
                          choices=[(1, 'High School/SSC/Others'), (2, '12th Grade'), (3, 'Graduate/Under Graduate/Professional'), (4, 'Post Graduate')], 
                          coerce=int,
                          description="What is your highest level of education completed?")
    
    MARITALSTATUS_Married = SelectField('Are you Married?', 
                                       choices=[(0, 'No'), (1, 'Yes')], 
                                       coerce=int,
                                       description="Are you currently married?")
    MARITALSTATUS_Single = SelectField('Are you Single?', 
                                      choices=[(0, 'No'), (1, 'Yes')], 
                                      coerce=int,
                                      description="Are you currently single (not married)?")
    
    GENDER_F = SelectField('Are you Female?', 
                          choices=[(0, 'No'), (1, 'Yes')], 
                          coerce=int,
                          description="Are you female?")
    GENDER_M = SelectField('Are you Male?', 
                          choices=[(0, 'No'), (1, 'Yes')], 
                          coerce=int,
                          description="Are you male?")
    
    # Product enquiry fields with user-friendly descriptions
    last_prod_enq2_AL = SelectField('Last Application: Auto Loan', 
                                   choices=[(0, 'No'), (1, 'Yes')], 
                                   coerce=int,
                                   description="Was your most recent credit application for an auto loan?")
    last_prod_enq2_CC = SelectField('Last Application: Credit Card', 
                                   choices=[(0, 'No'), (1, 'Yes')], 
                                   coerce=int,
                                   description="Was your most recent credit application for a credit card?")
    last_prod_enq2_ConsumerLoan = SelectField('Last Application: Consumer Loan', 
                                             choices=[(0, 'No'), (1, 'Yes')], 
                                             coerce=int,
                                             description="Was your most recent credit application for a consumer loan?")
    last_prod_enq2_HL = SelectField('Last Application: Home Loan', 
                                   choices=[(0, 'No'), (1, 'Yes')], 
                                   coerce=int,
                                   description="Was your most recent credit application for a home loan?")
    last_prod_enq2_PL = SelectField('Last Application: Personal Loan', 
                                   choices=[(0, 'No'), (1, 'Yes')], 
                                   coerce=int,
                                   description="Was your most recent credit application for a personal loan?")
    last_prod_enq2_others = SelectField('Last Application: Other Type', 
                                       choices=[(0, 'No'), (1, 'Yes')], 
                                       coerce=int,
                                       description="Was your most recent credit application for some other type of loan?")
    
    first_prod_enq2_AL = SelectField('First Application: Auto Loan', 
                                     choices=[(0, 'No'), (1, 'Yes')], 
                                     coerce=int,
                                     description="Was your first credit application for an auto loan?")
    first_prod_enq2_CC = SelectField('First Application: Credit Card', 
                                    choices=[(0, 'No'), (1, 'Yes')], 
                                    coerce=int,
                                    description="Was your first credit application for a credit card?")
    first_prod_enq2_ConsumerLoan = SelectField('First Application: Consumer Loan', 
                                              choices=[(0, 'No'), (1, 'Yes')], 
                                              coerce=int,
                                              description="Was your first credit application for a consumer loan?")
    first_prod_enq2_HL = SelectField('First Application: Home Loan', 
                                     choices=[(0, 'No'), (1, 'Yes')], 
                                     coerce=int,
                                     description="Was your first credit application for a home loan?")
    first_prod_enq2_PL = SelectField('First Application: Personal Loan', 
                                    choices=[(0, 'No'), (1, 'Yes')], 
                                    coerce=int,
                                    description="Was your first credit application for a personal loan?")
    first_prod_enq2_others = SelectField('First Application: Other Type', 
                                        choices=[(0, 'No'), (1, 'Yes')], 
                                        coerce=int,
                                        description="Was your first credit application for some other type of loan?")
    

    # ... (include all other fields from your original code)
    submit = SubmitField('Predict Credit Risk')

class BatchUploadForm(FlaskForm):
    batch_file = FileField('Upload Data File (CSV, Excel, or JSON)', validators=[DataRequired()])
    submit = SubmitField('Upload and Predict')

def validate_form_data(form_data):
    """Validate form data and return cleaned data"""
    cleaned_data = {}
    
    for field in FEATURE_COLUMNS:
        if field in form_data:
            value = form_data[field]
            if value == '' or value is None:
                cleaned_data[field] = 0
            else:
                try:
                    if field in ['pct_tl_open_L6M', 'pct_tl_closed_L6M', 'pct_tl_closed_L12M', 
                               'pct_PL_enq_L6m_of_ever', 'pct_CC_enq_L6m_of_ever']:
                        cleaned_data[field] = float(value)
                    else:
                        cleaned_data[field] = int(value)
                except (ValueError, TypeError):
                    cleaned_data[field] = 0
        else:
            cleaned_data[field] = 0
    
    return cleaned_data

def prepare_prediction_data(data):
    """Prepare data for prediction in the correct format"""
    df = pd.DataFrame([data])
    
    # Ensure all columns are present and in correct order
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match model expectations
    df = df[FEATURE_COLUMNS]
    
    return df

def read_data_file(file, filename):
    """Read data from different file formats (CSV, Excel, JSON)"""
    try:
        file_extension = filename.lower().split('.')[-1]
        
        if file_extension == 'csv':
            df = pd.read_csv(file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(file, engine='openpyxl' if file_extension == 'xlsx' else 'xlrd')
        elif file_extension == 'json':
            file.seek(0)
            json_data = json.load(file)
            
            if isinstance(json_data, list):
                df = pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                if 'data' in json_data:
                    df = pd.DataFrame(json_data['data'])
                elif 'records' in json_data:
                    df = pd.DataFrame(json_data['records'])
                else:
                    df = pd.DataFrame([json_data])
            else:
                raise ValueError("Invalid JSON structure")
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
        return df
        
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")

def validate_and_prepare_batch_data(df):
    """Validate and prepare batch data for prediction"""
    if df.empty:
        raise ValueError("File is empty or contains no data")
    
    # Check for missing columns
    missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_cols))}")
    
    # Select only required columns and fill missing values
    df_processed = df[FEATURE_COLUMNS].copy()
    df_processed = df_processed.fillna(0)
    
    # Convert data types
    for col in FEATURE_COLUMNS:
        if col in ['pct_tl_open_L6M', 'pct_tl_closed_L6M', 'pct_tl_closed_L12M', 
                   'pct_PL_enq_L6m_of_ever', 'pct_CC_enq_L6m_of_ever']:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
            df_processed[col] = df_processed[col].clip(0, 1)
        else:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0).astype(int)
    
    return df_processed

@app.route('/')
def index():
    form = CreditRiskForm()
    batch_form = BatchUploadForm()
    
    for field_name, field in form._fields.items():
        if hasattr(field, 'description') and field.description:
            field.render_kw = field.render_kw or {}
            field.render_kw['data-description'] = field.description
    
    return render_template('index.html', form=form, batch_form=batch_form)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        form_data = request.form.to_dict()
        cleaned_data = validate_form_data(form_data)
        prediction_data = prepare_prediction_data(cleaned_data)
        
        # Make prediction
        prediction = model.predict(prediction_data)[0]
        prediction_class = CLASS_LABELS.get(prediction, f'P{prediction+1}')
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(prediction_data)[0]
            probabilities = {
                CLASS_LABELS.get(i, f'P{i+1}'): float(prob) for i, prob in enumerate(proba)
            }
            
            # Debug: Print probabilities
            print(f"Prediction: {prediction_class}")
            print(f"Probabilities: {probabilities}")
        
        return jsonify({
            'prediction': prediction_class,
            'probabilities': probabilities,
            'success': True
        })
        
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Check if file was uploaded
        if 'batch_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['batch_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        file_extension = file.filename.lower().split('.')[-1]
        if file_extension not in ['csv', 'xlsx', 'xls', 'json']:
            return jsonify({'error': 'File must be CSV, Excel (.xlsx/.xls), or JSON format'}), 400
        
        # Read data file
        df_original = read_data_file(file, file.filename)
        
        # Validate and prepare data for prediction
        df_for_prediction = validate_and_prepare_batch_data(df_original)
        
        # Make predictions
        predictions = model.predict(df_for_prediction)
        prediction_classes = [CLASS_LABELS.get(pred, f'P{pred+1}') for pred in predictions]
        
        # Create output dataframe with original data + Approved Flag
        df_output = df_original.copy()
        
        # Add Approved Flag column
        df_output['Approved Flag'] = [APPROVAL_MAPPING.get(pred_class, 'Rejected') 
                                       for pred_class in prediction_classes]
        
        # Debug: Print unique predictions
        print(f"Unique predictions: {set(prediction_classes)}")
        print(f"Prediction distribution: {pd.Series(prediction_classes).value_counts().to_dict()}")
        
        # Return based on input format
        if file_extension == 'json':
            output_data = df_output.to_dict('records')
            return jsonify({
                'success': True,
                'predictions': output_data,
                'total_records': len(df_output)
            })
        else:
            # Create CSV output
            output = io.StringIO()
            df_output.to_csv(output, index=False)
            output.seek(0)
            
            # Create response with proper headers
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={
                    'Content-Disposition': f'attachment; filename=predictions_output.csv',
                    'Content-Type': 'text/csv; charset=utf-8'
                }
            )
        
    except ValueError as e:
        print(f"ValueError in predict_batch: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Unexpected error in predict_batch: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/health')
def health():
    model_info = {
        'status': 'healthy',
        'model_loaded': model is not None
    }
    
    if model is not None:
        model_info['model_type'] = str(type(model).__name__)
        if hasattr(model, 'classes_'):
            model_info['model_classes'] = model.classes_.tolist()
    
    return jsonify(model_info)

@app.route('/debug_model')
def debug_model():
    """Debug endpoint to check model details"""
    if model is None:
        return jsonify({'error': 'Model not loaded'})
    
    info = {
        'model_type': str(type(model).__name__),
        'has_predict_proba': hasattr(model, 'predict_proba'),
        'has_classes': hasattr(model, 'classes_')
    }
    
    if hasattr(model, 'classes_'):
        info['classes'] = model.classes_.tolist()
    
    if hasattr(model, 'n_classes_'):
        info['n_classes'] = model.n_classes_
        
    return jsonify(info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
