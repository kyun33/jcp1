import streamlit as st
# make models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
#read data
import pandas as pd
#make plot
import matplotlib.pyplot as plt
import seaborn as sns
import time

st.title('Classification Model Dashboard')

st.sidebar.header("Upload Your Data")


file = st.sidebar.file_uploader("Upload a CSV", type = ["CSV"])



if file:
    df = pd.read_csv(file)
    st.write("Dataset preview:", df.head())
    #st.write(df.shape)
    #st.dataframe(df)
    target_column = st.sidebar.selectbox("Select Target Column", df.columns)
    X_cols = [col_name for col_name in df.columns if col_name != target_column]
    X = df[X_cols]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=420)
    
    model_selection = st.sidebar.selectbox("Select Model", ["Logisitic Regression", "Decision Tree", "Support Vector Machine"])

    if model_selection == "Logistic Regression":
        model = LogisticRegression()
    elif model_selection == "Decision Tree":
        model = DecisionTreeClassifier(random_state=420)
    else:
        model = SVC()

button_clicked = st.sidebar.button("Train")
if button_clicked:
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    time_to_train = end - start
    st.write(f"{time_to_train:.04f} seconds to train") 
    predicted = model.predict(X_test)
    score = accuracy_score(y_test,predicted)
    st.write("Score", score)

    confusionmatrix = confusion_matrix(y_test, predicted)

    fig, ax = plt.subplots()
    sns.heatmap(confusionmatrix, annot= True, fmt='d', cmap = 'Blues', ax=ax)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    st.pyplot(fig)


