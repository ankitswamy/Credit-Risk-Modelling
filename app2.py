import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from wtforms import IntegerField, FloatField, SelectField, FileField, SubmitField
from wtforms.validators import DataRequired, NumberRange
from sklearn.preprocessing import StandardScaler
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
    if hasattr(model, 'classes_'):
        print(f"Model classes: {model.classes_}")
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

# Columns that need to be scaled (same as in training)
COLUMNS_TO_SCALE = [
    'Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment', 
    'max_recent_level_of_deliq', 'recent_level_of_deliq', 
    'time_since_recent_enq', 'NETMONTHLYINCOME', 'Time_With_Curr_Empr'
]

# Class labels mapping - This should match your LabelEncoder from training
# The model predicts 0,1,2,3 which map to P1,P2,P3,P4
CLASS_LABELS = {0: 'P1', 1: 'P2', 2: 'P3', 3: 'P4'}

# Approval mapping
APPROVAL_MAPPING = {
    'P1': 'Approved',
    'P2': 'Approved', 
    'P3': 'Rejected',
    'P4': 'Rejected'
}

class CreditRiskForm(FlaskForm):
    # Numerical fields with user-friendly descriptions
    pct_tl_open_L6M = FloatField('New Credit Accounts Opened (Last 6 Months)', 
                                validators=[NumberRange(min=0, max=1)],
                                description="What percentage of your credit accounts were opened in the last 6 months?")
    pct_tl_closed_L6M = FloatField('Credit Accounts Closed (Last 6 Months)', 
                                  validators=[NumberRange(min=0, max=1)],
                                  description="What percentage of your credit accounts were closed in the last 6 months?")
    Tot_TL_closed_L12M = IntegerField('Total Credit Accounts Closed (Last 12 Months)', 
                                     validators=[NumberRange(min=0)],
                                     description="How many credit accounts have you closed?")
    pct_tl_closed_L12M = FloatField('Credit Accounts Closed (Last 12 Months)', 
                                   validators=[NumberRange(min=0, max=1)],
                                   description="What percentage closed in last 12 months?")
    Tot_Missed_Pmnt = IntegerField('Total Missed Payments', 
                                  validators=[NumberRange(min=0)],
                                  description="How many missed payments?")
    CC_TL = IntegerField('Number of Credit Cards', 
                        validators=[NumberRange(min=0)],
                        description="How many credit cards?")
    Home_TL = IntegerField('Number of Home Loans', 
                          validators=[NumberRange(min=0)],
                          description="How many home loans?")
    PL_TL = IntegerField('Number of Personal Loans', 
                        validators=[NumberRange(min=0)],
                        description="How many personal loans?")
    Secured_TL = IntegerField('Number of Secured Loans', 
                             validators=[NumberRange(min=0)],
                             description="How many secured loans?")
    Unsecured_TL = IntegerField('Number of Unsecured Loans', 
                               validators=[NumberRange(min=0)],
                               description="How many unsecured loans?")
    Other_TL = IntegerField('Number of Other Credit Accounts', 
                           validators=[NumberRange(min=0)],
                           description="How many other credit accounts?")
    Age_Oldest_TL = IntegerField('Age of Oldest Credit Account (months)', 
                                validators=[NumberRange(min=0)],
                                description="Age of oldest account in months?")
    Age_Newest_TL = IntegerField('Age of Newest Credit Account (months)', 
                                validators=[NumberRange(min=0)],
                                description="Age of newest account in months?")
    time_since_recent_payment = IntegerField('Days Since Last Payment', 
                                            validators=[NumberRange(min=0)],
                                            description="Days since last payment?")
    max_recent_level_of_deliq = IntegerField('Highest Recent Payment Delay', 
                                            validators=[NumberRange(min=0)],
                                            description="Highest payment delay?")
    num_deliq_6_12mts = IntegerField('Payment Delays (6-12 Months Ago)', 
                                     validators=[NumberRange(min=0)],
                                     description="Payment delays 6-12 months ago?")
    num_times_60p_dpd = IntegerField('Severe Payment Delays (60+ Days)', 
                                    validators=[NumberRange(min=0)],
                                    description="Times 60+ days late?")
    num_std_12mts = IntegerField('Good Payment History (Last 12 Months)', 
                                validators=[NumberRange(min=0)],
                                description="Good payment accounts?")
    num_sub = IntegerField('Number of High-Risk Accounts', 
                          validators=[NumberRange(min=0)],
                          description="High-risk accounts?")
    num_sub_6mts = IntegerField('High-Risk Accounts (Last 6 Months)', 
                               validators=[NumberRange(min=0)],
                               description="High-risk in last 6 months?")
    num_sub_12mts = IntegerField('High-Risk Accounts (Last 12 Months)', 
                                 validators=[NumberRange(min=0)],
                                 description="High-risk in last 12 months?")
    num_dbt = IntegerField('Number of Debt Settlement Accounts', 
                          validators=[NumberRange(min=0)],
                          description="Debt settlements?")
    num_dbt_12mts = IntegerField('Debt Settlement Accounts (Last 12 Months)', 
                                validators=[NumberRange(min=0)],
                                description="Debt settlements last 12 months?")
    num_lss = IntegerField('Number of Written-Off Accounts', 
                          validators=[NumberRange(min=0)],
                          description="Written-off accounts?")
    recent_level_of_deliq = IntegerField('Current Payment Delay Status', 
                                        validators=[NumberRange(min=0)],
                                        description="Current payment delay?")
    CC_enq_L12m = IntegerField('Credit Card Applications (Last 12 Months)', 
                              validators=[NumberRange(min=0)],
                              description="CC applications last year?")
    PL_enq_L12m = IntegerField('Personal Loan Applications (Last 12 Months)', 
                              validators=[NumberRange(min=0)],
                              description="PL applications last year?")
    time_since_recent_enq = IntegerField('Days Since Last Credit Application', 
                                        validators=[NumberRange(min=0)],
                                        description="Days since last application?")
    enq_L3m = IntegerField('Credit Applications (Last 3 Months)', 
                          validators=[NumberRange(min=0)],
                          description="Applications last 3 months?")
    NETMONTHLYINCOME = IntegerField('Monthly Income (After Tax)', 
                                   validators=[NumberRange(min=0)],
                                   description="Monthly income?")
    Time_With_Curr_Empr = IntegerField('Time with Current Employer (months)', 
                                      validators=[NumberRange(min=0)],
                                      description="Months with current employer?")
    
    # Binary flags
    CC_Flag = SelectField('Do you have Credit Cards?', 
                         choices=[(0, 'No'), (1, 'Yes')], 
                         coerce=int)
    PL_Flag = SelectField('Do you have Personal Loans?', 
                         choices=[(0, 'No'), (1, 'Yes')], 
                         coerce=int)
    HL_Flag = SelectField('Do you have Home Loans?', 
                         choices=[(0, 'No'), (1, 'Yes')], 
                         coerce=int)
    GL_Flag = SelectField('Do you have Gold Loans?', 
                         choices=[(0, 'No'), (1, 'Yes')], 
                         coerce=int)
    
    # Percentage fields
    pct_PL_enq_L6m_of_ever = FloatField('Personal Loan Applications (Last 6 Months)', 
                                       validators=[NumberRange(min=0, max=1)],
                                       description="% of PL applications in last 6 months?")
    pct_CC_enq_L6m_of_ever = FloatField('Credit Card Applications (Last 6 Months)', 
                                       validators=[NumberRange(min=0, max=1)],
                                       description="% of CC applications in last 6 months?")
    
    # Categorical fields
    EDUCATION = SelectField('Education Level', 
                          choices=[(1, 'High School/SSC/Others'), (2, '12th Grade'), 
                                   (3, 'Graduate/Under Graduate/Professional'), (4, 'Post Graduate')], 
                          coerce=int)
    
    MARITALSTATUS_Married = SelectField('Are you Married?', 
                                       choices=[(0, 'No'), (1, 'Yes')], 
                                       coerce=int)
    MARITALSTATUS_Single = SelectField('Are you Single?', 
                                      choices=[(0, 'No'), (1, 'Yes')], 
                                      coerce=int)
    
    GENDER_F = SelectField('Are you Female?', 
                          choices=[(0, 'No'), (1, 'Yes')], 
                          coerce=int)
    GENDER_M = SelectField('Are you Male?', 
                          choices=[(0, 'No'), (1, 'Yes')], 
                          coerce=int)
    
    # Product enquiry fields - Last application
    last_prod_enq2_AL = SelectField('Last Application: Auto Loan', 
                                   choices=[(0, 'No'), (1, 'Yes')], 
                                   coerce=int)
    last_prod_enq2_CC = SelectField('Last Application: Credit Card', 
                                   choices=[(0, 'No'), (1, 'Yes')], 
                                   coerce=int)
    last_prod_enq2_ConsumerLoan = SelectField('Last Application: Consumer Loan', 
                                             choices=[(0, 'No'), (1, 'Yes')], 
                                             coerce=int)
    last_prod_enq2_HL = SelectField('Last Application: Home Loan', 
                                   choices=[(0, 'No'), (1, 'Yes')], 
                                   coerce=int)
    last_prod_enq2_PL = SelectField('Last Application: Personal Loan', 
                                   choices=[(0, 'No'), (1, 'Yes')], 
                                   coerce=int)
    last_prod_enq2_others = SelectField('Last Application: Other Type', 
                                       choices=[(0, 'No'), (1, 'Yes')], 
                                       coerce=int)
    
    # Product enquiry fields - First application
    first_prod_enq2_AL = SelectField('First Application: Auto Loan', 
                                     choices=[(0, 'No'), (1, 'Yes')], 
                                     coerce=int)
    first_prod_enq2_CC = SelectField('First Application: Credit Card', 
                                    choices=[(0, 'No'), (1, 'Yes')], 
                                    coerce=int)
    first_prod_enq2_ConsumerLoan = SelectField('First Application: Consumer Loan', 
                                              choices=[(0, 'No'), (1, 'Yes')], 
                                              coerce=int)
    first_prod_enq2_HL = SelectField('First Application: Home Loan', 
                                     choices=[(0, 'No'), (1, 'Yes')], 
                                     coerce=int)
    first_prod_enq2_PL = SelectField('First Application: Personal Loan', 
                                    choices=[(0, 'No'), (1, 'Yes')], 
                                    coerce=int)
    first_prod_enq2_others = SelectField('First Application: Other Type', 
                                        choices=[(0, 'No'), (1, 'Yes')], 
                                        coerce=int)
    
    submit = SubmitField('Predict Credit Risk')

class BatchUploadForm(FlaskForm):
    batch_file = FileField('Upload Data File (CSV, Excel, or JSON)', validators=[DataRequired()])
    submit = SubmitField('Upload and Predict')

def apply_scaling(df):
    """Apply StandardScaler to specified columns (mimics training preprocessing)"""
    df_scaled = df.copy()
    
    for col in COLUMNS_TO_SCALE:
        if col in df_scaled.columns:
            # Create and fit scaler for this column
            scaler = StandardScaler()
            column_data = df_scaled[col].values.reshape(-1, 1)
            scaled_values = scaler.fit_transform(column_data)
            df_scaled[col] = scaled_values.flatten()
    
    return df_scaled

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
    """Prepare data for prediction with scaling"""
    df = pd.DataFrame([data])
    
    # Ensure all columns are present
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns
    df = df[FEATURE_COLUMNS]
    
    # Apply scaling
    df = apply_scaling(df)
    
    return df

def read_data_file(file, filename):
    """Read data from different file formats"""
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
        raise ValueError("File is empty")
    
    # Check for missing columns
    missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing_cols))}")
    
    # Select and process columns
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
    
    # Apply scaling - CRITICAL STEP
    df_processed = apply_scaling(df_processed)
    
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
        
        print(f"Raw prediction: {prediction}, Mapped class: {prediction_class}")
        
        # Get probabilities
        probabilities = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(prediction_data)[0]
            probabilities = {
                CLASS_LABELS.get(i, f'P{i+1}'): float(prob) for i, prob in enumerate(proba)
            }
            print(f"Probabilities: {probabilities}")
        
        return jsonify({
            'prediction': prediction_class,
            'probabilities': probabilities,
            'success': True
        })
        
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Check file upload
        if 'batch_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['batch_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        file_extension = file.filename.lower().split('.')[-1]
        if file_extension not in ['csv', 'xlsx', 'xls', 'json']:
            return jsonify({'error': 'File must be CSV, Excel, or JSON'}), 400
        
        # Read original data
        df_original = read_data_file(file, file.filename)
        print(f"Original data shape: {df_original.shape}")
        
        # Prepare data for prediction (with scaling)
        df_for_prediction = validate_and_prepare_batch_data(df_original)
        print(f"Processed data shape: {df_for_prediction.shape}")
        
        # Make predictions
        predictions = model.predict(df_for_prediction)
        print(f"Raw predictions: {predictions}")
        print(f"Unique predictions: {np.unique(predictions, return_counts=True)}")
        
        prediction_classes = [CLASS_LABELS.get(pred, f'P{pred+1}') for pred in predictions]
        print(f"Prediction distribution: {pd.Series(prediction_classes).value_counts().to_dict()}")
        
        # Create output dataframe
        df_output = df_original.copy()
        df_output['Approved Flag'] = [APPROVAL_MAPPING.get(pred_class, 'Rejected') 
                                       for pred_class in prediction_classes]
        
        # Add probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df_for_prediction)
            for i, class_label in CLASS_LABELS.items():
                df_output[f'Probability_{class_label}'] = probabilities[:, i]
        
        # Return based on format
        if file_extension == 'json':
            output_data = df_output.to_dict('records')
            return jsonify({
                'success': True,
                'predictions': output_data,
                'total_records': len(df_output)
            })
        else:
            # CSV output
            output = io.StringIO()
            df_output.to_csv(output, index=False)
            output.seek(0)
            
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={
                    'Content-Disposition': f'attachment; filename=predictions_output.csv',
                    'Content-Type': 'text/csv; charset=utf-8'
                }
            )
        
    except ValueError as e:
        print(f"ValueError: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
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
    """Debug endpoint"""
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