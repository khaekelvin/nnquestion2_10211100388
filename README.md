# Loan Default Prediction System

## Description
A machine learning web application that predicts loan default probability using neural networks. The system analyzes various customer parameters to assist financial institutions in risk assessment.

## Installation

```bash
# Clone the repository
git clone [repository-url]

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