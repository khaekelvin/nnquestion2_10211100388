# Loan Default Prediction System

## Description
A machine learning web application that predicts loan default probability using neural networks. The system analyzes various customer parameters to assist financial institutions in risk assessment.

Link :   [Loan Default Prediction System](https://nnquestion210211100388-baadpyygjlfxskpafa56qq.streamlit.app/)

## Loan Model Prediction

### Step 1
<img width="1680" alt="Screenshot 2024-12-25 at 9 54 34 PM" src="https://github.com/user-attachments/assets/492e0af4-45d3-4559-8418-27c170249d78" />



### Step 2
<img width="1680" alt="Screenshot 2024-12-25 at 9 54 49 PM" src="https://github.com/user-attachments/assets/a22a7706-5f2f-4f92-98ee-418e369bce44" />





### Step 3


<img width="1679" alt="Screenshot 2024-12-25 at 9 55 01 PM" src="https://github.com/user-attachments/assets/7478da2a-dfdc-401b-b552-94e34935c098" />



```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate 

# Install dependencies
pip install -r requirements.txt


The model uses the Loandataset.csv which includes the following features:

loan_amount
rate_of_interest
Interest_rate_spread
Upfront_charges
term_in_months
property_value
income
Credit_Score
Status (target variable: 0 = no default, 1 = default)

### Project Structure
nnquestion2_10211100388

├── README.md
├── requirements.txt
├── app.py        
├── train.py      
├── data/         
├── models/        
├── static/        
├── templates/     
└── utils/         


### Usage

Place the dataset:
Train the model:
Run the application:
Access the web interface at: http://127.0.0.1:5000
Technologies Used
Python 3.10
TensorFlow/Keras
Flask
Scikit-learn
Pandas
NumPy
HTML/CSS/JavaScript


## Author
Name: Kelvin Sungzie Duobu Index Number: 10211100388
