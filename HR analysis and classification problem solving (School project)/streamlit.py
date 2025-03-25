import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


st.title("Employee Attrition Prediction App")

model = joblib.load('best_rf_model.pkl')
train_columns = [
    'BusinessTravel_Non-Travel', 'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
    'Department_Human Resources', 'Department_Research & Development', 'Department_Sales',
    'EducationField_Human Resources', 'EducationField_Life Sciences', 'EducationField_Marketing',
    'EducationField_Medical', 'EducationField_Other', 'EducationField_Technical Degree',
    'Gender_Female', 'Gender_Male', 'JobRole_Healthcare Representative', 'JobRole_Human Resources',
    'JobRole_Laboratory Technician', 'JobRole_Manager', 'JobRole_Manufacturing Director',
    'JobRole_Research Director', 'JobRole_Research Scientist', 'JobRole_Sales Executive',
    'JobRole_Sales Representative', 'MaritalStatus_Divorced', 'MaritalStatus_Married',
    'MaritalStatus_Single', 'Age', 'DistanceFromHome', 'Education', 'JobLevel', 'MonthlyIncome',
    'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'JobInvolvement', 'PerformanceRating', 'EnvironmentSatisfaction', 'JobSatisfaction',
    'WorkLifeBalance', 'JoursTravailles', 'HeuresParJour', 'PlusLongueAbsence', 'TravailHorsHoraires'
]

default_values = {
    'Age': 51,
    'BusinessTravel': 'Travel_Rarely',
    'Department': 'Sales',
    'DistanceFromHome': 6,
    'Education': 2,
    'EducationField': 'Life Sciences',
    'EmployeeID': 1,
    'Gender': 'Female',
    'JobLevel': 1,
    'JobRole': 'Healthcare Representative',
    'MaritalStatus': 'Married',
    'MonthlyIncome': 131160,
    'NumCompaniesWorked': 1.0,
    'PercentSalaryHike': 11,
    'StockOptionLevel': 0,
    'TotalWorkingYears': 1.0,
    'TrainingTimesLastYear': 6,
    'YearsAtCompany': 1,
    'YearsSinceLastPromotion': 0,
    'YearsWithCurrManager': 0,
    'JobInvolvement': 3,
    'PerformanceRating': 3,
    'EnvironmentSatisfaction': 3.0,
    'JobSatisfaction': 4.0,
    'WorkLifeBalance': 2.0,
    'JoursTravailles': 232,
    'HeuresParJour': 7.37,
    'PlusLongueAbsence': 3,
    'TravailHorsHoraires': 0
}

input_data = {}
for col, default in default_values.items():
    if isinstance(default, int) or isinstance(default, float):
        input_data[col] = st.number_input(f"Enter value for {col}", value=default)
    else:
        input_data[col] = st.text_input(f"Enter value for {col}", value=default)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])

    categorical_features = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']
    numerical_features = ['Age', 'DistanceFromHome', 'Education', 'JobLevel', 'MonthlyIncome', 'NumCompaniesWorked',
                          'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
                          'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'JobInvolvement',
                          'PerformanceRating', 'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance',
                          'JoursTravailles', 'HeuresParJour', 'PlusLongueAbsence', 'TravailHorsHoraires']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    X_test = preprocessor.fit_transform(input_df)

    X_test_df = pd.DataFrame(X_test, columns=preprocessor.get_feature_names_out())
    X_test_df = X_test_df.reindex(columns=train_columns, fill_value=0)

    y_pred = model.predict(X_test_df)
    y_proba = model.predict_proba(X_test_df)[:, 1]

    st.subheader("Prediction Results : ")
    attrition_label = "Yes" if y_pred[0] == 1 else "No"
    attrition_color = "red" if y_pred[0] == 1 else "green"
    st.markdown(f"<h3 style='color: {attrition_color};'>Predicted Attrition: {attrition_label}</h3>", unsafe_allow_html=True)

