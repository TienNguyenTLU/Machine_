from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np
from starlette.responses import HTMLResponse

# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# Khởi tạo thư mục chứa templates
templates = Jinja2Templates(directory="templates")

# Tải mô hình đã huấn luyện
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Endpoint trang chủ trả về giao diện HTML
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint xử lý form và dự đoán lương
@app.post("/predict", response_class=HTMLResponse)
async def predict_salary(
    request: Request,
    education: str = Form(...),
    experience: int = Form(...),
    location: str = Form(...),
    job_title: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...)
):
    # Chuyển đổi dữ liệu đầu vào thành định dạng mô hình yêu cầu (One-Hot Encoding)
    input_features = np.array([[age, experience, 0, 0, 0, 0, 0, 0, 0, 0]])

    # Mã hóa các giá trị phân loại thành dạng one-hot
    if education == 'Bachelor':
        input_features[0][2] = 1
    elif education == 'Master':
        input_features[0][3] = 1
    elif education == 'PhD':
        input_features[0][4] = 1
    elif education == 'High School':
        input_features[0][5] = 1

    if location == 'Urban':
        input_features[0][6] = 1
    elif location == 'Suburban':
        input_features[0][7] = 1
    elif location == 'Rural':
        input_features[0][8] = 1

    if gender == 'Male':
        input_features[0][9] = 1
    else:
        input_features[0][9] = 0

    # Dự đoán mức lương
    prediction = model.predict(input_features)[0]

    # Render lại trang HTML với kết quả dự đoán
    return templates.TemplateResponse("index.html", {"request": request, "prediction": round(prediction, 2)})
