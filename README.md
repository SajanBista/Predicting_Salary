# Predicting Salary Using Polynomial Regression for my first assignment of fuse machine ai fellowship where I am trying to implement 12 factor app

This project is a machine learning web application built with FastAPI which predicts salary based on their level or position using Polynomial Regression.

It allows users to:
- Upload a dataset
- Train the model using different polynomial degrees
- Automatically select the best model based on R² score
- Predict salary for a custom input level


## Project Overview

Polynomial Regression is a form of regression analysis in which the relationship between the independent variable X and the dependent variable Y is modeled as an nth-degree polynomial.

This app demonstrates how polynomial regression can model non-linear salary data more effectively than simple linear regression.


## How It Works
1. Dataset Upload: Upload a CSV file containing at least one input feature (eg., Level) and one output (e.g., Salary).
2. Model Training: The app trains models using polynomial degrees from 2 to 5. It automatically selects the one with the best R² score.
3. Prediction: Enter a custom level (e.g., 6.5) and the app will return the predicted salary using the best-trained model.



## Technologies Used                  
FastAPI for High-performance Python web framework
scikit-learn : ML model training and evaluation 
other pandas, numpy etc are not mentioned but are in requirement.txt



## Example Dataset Format

Your CSV file should have the following format:
Position| Level | Salary  |

## To note
The app assumes the last column is the target (Salary), and features are in between the first and last columns.





### 1. Clone the Repository

''terminal:

git clone https://github.com/SajanBista/Predicting_Salary.git

cd Predicting_Salary
