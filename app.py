
import os
import pandas as pd
import numpy as np
import joblib
import os
import pandas as pd
import numpy as np
import joblib
import cv2
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import io

app = Flask(__name__)

# Ensure we are in the script's directory for relative paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MODEL_PATH = os.path.join('Model', 'arrhythmia_model.joblib')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    print(f"Warning: Model not found at {MODEL_PATH}. Prediction will fail.")

# Class Names Mapping (1-based index from dataset matches list index + 1)
CLASS_NAMES = [
    "Normal", 
    "Ischemic changes (CAD)", 
    "Old Anterior Myocardial Infraction",
    "Old Inferior Myocardial Infraction",
    "Sinus tachycardy", 
    "Sinus bradycardy", 
    "Ventricular Premature Contraction (PVC)",
    "Supraventricular Premature Contraction",
    "Left Bundle branch block",
    "Right Bundle branch block",
    "1. Degree AtrioVentricular block",
    "2. Degree AV block",
    "3. Degree AV block",
    "Left Ventricule hypertrophy",
    "Atrial Fibrillation or Flutter",
    "Others"
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Read CSV
            # Expecting a file without header, or handling it if provided?
            # typically these files don't have headers based on the training data.
            # We'll read without header.
            df = pd.read_csv(filepath, header=None)
            
            # Validation: Check feature count
            # Model expects 279 features. 
            if df.shape[1] == 2:
                return jsonify({'error': 'Invalid Format: Uploaded file contains raw signal data (2 columns). This model requires 279 CALCULATED clinical features (e.g., Age, Sex, QRS Duration...), not raw waveforms.'}), 400
            
            if df.shape[1] == 280:
                # Assuming last column is target, drop it
                data = df.iloc[:, :-1]
            elif df.shape[1] == 279:
                data = df
            else:
                return jsonify({'error': f'Invalid feature count: {df.shape[1]}. Expected 279 or 280 columns.'}), 400
            
            # Preprocessing 
            # 1. Replace '?' with NaN (as done in training)
            data = data.replace('?', np.nan)
            
            # 2. Drop column 13 (index 13) as done in training logic
            # However, the pipeline's Imputer might expect 278 columns if we dropped it before training?
            # Let's check the training script history.
            # In train_model.py:
            # df.drop(columns=[13], inplace=True)
            # pipeline.fit(X, y)
            # So the pipeline expects 279 - 1 = 278 features INPUT.
            # BUT: Pipeline steps are Applied sequentially. 
            # Wait, the df.drop was done BEFORE the pipeline. 
            # So the input to pipeline.predict() MUST ALREADY HAVE COLUMN 13 REMOVED.
            
            if 13 in data.columns:
                 data.drop(columns=[13], inplace=True)
            else:
                # If dataframe was re-indexed or headerless, indices are 0..278. 
                # Dropping column at index 13 is strictly required.
                # Since we read with header=None, columns are 0,1,2...
                data.drop(columns=[13], inplace=True)

            # Check for batch prediction or single prediction
            predictions = model.predict(data)
            
            results = []
            for pred in predictions:
                # pred is 1-16
                class_name = CLASS_NAMES[pred - 1] if 1 <= pred <= 16 else "Unknown"
                results.append({'class_id': int(pred), 'class_name': class_name})
            
            # Return first result for single file upload usually
            return jsonify({'success': True, 'prediction': results[0]})

        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
                
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/convert', methods=['POST'])
def convert_ecg():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Simple Digitization Logic (Prototype)
            # 1. Read Image
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Could not read image file.")

            # 2. Thresholding (Assume dark signal on light background)
            # Invert so signal is bright, background is dark
            _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
            
            # 3. Find column-wise maximums (simplified signal tracing)
            # Find the first 'white' pixel in each column to represent the wave
            height, width = thresh.shape
            signal = []
            
            for x in range(width):
                col = thresh[:, x]
                # Find indices of non-zero pixels
                indices = np.nonzero(col)[0]
                if len(indices) > 0:
                    # Take the mean position of the line thickness as the signal value
                    # Invert Y axis because image 0,0 is top-left
                    y_val = height - np.mean(indices)
                    signal.append(y_val)
                else:
                    # If no signal found in column, use NaN or interpolate (using 0 for now)
                    signal.append(0)
            
            # 4. Create CSV
            df = pd.DataFrame({'sample_index': range(len(signal)), 'amplitude': signal})
            
            output = io.BytesIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            return send_file(
                output,
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'{os.path.splitext(filename)[0]}_digitized.csv'
            )

        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
                
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
