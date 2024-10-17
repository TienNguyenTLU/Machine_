from flask import Flask, render_template, request
import joblib as jb
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
app = Flask(__name__)

# Tải mô hình đã huấn luyện
models = {}
model_names = ['linear', 'lasso', 'nn', 'stacking']
for model_name in model_names:
    with open(f'{model_name}.pkl', 'rb') as file:
        models[model_name] = jb.load(file)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Lấy dữ liệu từ form
        age = int(request.form['age'])
        Experience = int(request.form['experience'])
        Education = float(request.form['education'])
        Location = float(request.form['location'])
        gender = request.form['gender']
        Job_Title = float(request.form['job_title'])
        model_selection = request.form['model']

        # Tạo DataFrame từ input
        input_data = np.array([[ Experience, Education, Job_Title, Location]])
        input_data_new = pd.DataFrame(input_data)
        # Dự đoán
        prediction = models[model_selection].predict(input_data_new)[0]

        # Hiển thị kết quả
        return render_template('index.html', prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)
