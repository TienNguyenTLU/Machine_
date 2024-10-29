from flask import Flask, render_template, request
import joblib as jb
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
        Age = int(request.form['age'])
        Experience = int(request.form['experience'])
        Education = (request.form['education'])
        Location = (request.form['location'])
        gender = request.form['gender']
        Job_Title = (request.form['job_title'])
        model_selection = request.form['model']

        # Tạo DataFrame từ input
        input_data = pd.DataFrame([[ Experience, Education, Job_Title, Location, Age]], columns=["Experience", "Education", "Job_Title", "Location", "Age"])
        train_data = pd.DataFrame({
            'Experience': [1,4,5,6],
            'Education': ['Bachelor', 'Master', 'PhD', 'High School'],
            'Job_Title': ['Engineer', 'Manager', 'Analyst','Director'],
            'Location': ['Suburban', 'Urban', 'Rural','Rural'],
            'Age' : [26,27,29,30]
        })
        input_table = pd.concat([input_data,train_data], axis=0)
        categorical_features = ['Education', 'Location', 'Job_Title']
        numeric_features = ['Experience', 'Age']
        preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)]
        )
        X=input_table
        input_data_scaler = preprocessor.fit_transform(X)
        df = pd.DataFrame(input_data_scaler).head(1)
        prediction = models[model_selection].predict(df)
        # Hiển thị kết quả
        return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
