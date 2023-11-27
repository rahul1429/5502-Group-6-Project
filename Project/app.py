import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

data = pd.read_excel('3000_salary.xlsx')

# Drop missing values
data_clean = data.dropna()

# Resample the data
resampled_data = data_clean.groupby('SubTitle', group_keys=False).apply(lambda x: x.sample(max(len(x), 700), replace=True))

# Convert 'SubTitle' and 'Location' to one-hot encoded variables
resampled_data = pd.get_dummies(resampled_data, columns=['SubTitle', 'Location'])

# Load the pickled model
with open('salary_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


# Set background image using CSS
background_image_path = 'D:/5502/Project/background.jpeg'
st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url('{background_image_path}');
            background-size: cover;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.title('Salary Prediction App')

# User input fields
subtitle = st.selectbox('Select Your Job Title', options=data['SubTitle'].unique())
years_of_experience = st.slider('Years of Experience', 0, 50, 5)
location = st.selectbox('Select Your Location', options=data['Location'].unique())

# Feature engineering if needed
# For example, you might want to add more features based on user inputs
columns_list = resampled_data.drop(['Annual_salary', 'Title', 'Company', 'Salary'], axis=1).columns


def createInputArray(Years, title, location):
    input_array = np.zeros(348)
    col1 = 'SubTitle_' + title
    col2 = 'Location_' + location

    for i in range(len(columns_list)):
        if columns_list[i] == 'Years of Experience':
            input_array[i] = Years


        if col1 == columns_list[i]:
            input_array[i] = 1

        if col2 == columns_list[i]:
            input_array[i] = 1
    return input_array


# Make predictions
inputs = createInputArray(years_of_experience, subtitle, location)

prediction = model.predict([inputs])[0] 

years_range = range(max(0, years_of_experience - 3), years_of_experience + 4)
# Convert to Pandas Series
years_range = pd.Series(list(years_range), name='Years of Experience')

salaries = []

for year in years_range:
    input_array1 = createInputArray(year, subtitle, location)
    # input_array1 = input_array.reshape(1, -1)
    salary = model.predict([input_array1])
    salaries.append(salary)


# Predict button
if st.button('Predict Salary'):
    # Display the prediction
    st.write(f"""### Predicted Annual Salary: $ {round(prediction, 2)}""")
    # Plotting the graph using seaborn
    plt.plot(years_range, salaries, marker='o')
    plt.title('Salaries for Different Years of Experience')
    plt.xlabel('Years of Experience')
    plt.ylabel('Predicted Annual Salary')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
