import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === Streamlit Page Config ===
st.set_page_config(page_title="xAPI Dashboard", layout="wide")
st.title("📊 xAPI Student Performance Dashboard")

# === Load or Upload CSV ===
csv_path = "xAPI-Edu-Data.csv"

if not os.path.exists(csv_path):
    uploaded_file = st.file_uploader("📂 Upload the xAPI-Edu-Data CSV file", type=["csv"])
    if uploaded_file is not None:
        with open(csv_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ File uploaded and saved as 'xAPI-Edu-Data.csv'. Please refresh the page.")
        st.stop()
    else:
        st.warning("📁 Please upload the CSV file to continue.")
        st.stop()
else:
    st.info("✅ Found: xAPI-Edu-Data.csv — Loading data...")
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)

    # === Encode Categorical Data ===
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # === Normalize Selected Features ===
    scale_cols = ['raisedhands', 'VisITedResources', 'Discussion']
    df[scale_cols] = StandardScaler().fit_transform(df[scale_cols])

    # === Sidebar Filters ===
    st.sidebar.header("🔍 Filter Data")
    gender_filter = st.sidebar.multiselect("Gender", options=df['gender'].unique(), default=df['gender'].unique())
    class_filter = st.sidebar.multiselect("Class", options=df['Class'].unique(), default=df['Class'].unique())
    filtered_df = df[(df['gender'].isin(gender_filter)) & (df['Class'].isin(class_filter))]

    # === Define X and y ===
    X = filtered_df.drop("Class", axis=1)
    y = filtered_df["Class"]

    # === Train the Model ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # === Key Insights ===
    st.header("🔑 Key Insights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📚 Avg. Raised Hands", f"{filtered_df['raisedhands'].mean():.2f}")
    with col2:
        st.metric("🧑‍💻 Avg. Visited Resources", f"{filtered_df['VisITedResources'].mean():.2f}")
    with col3:
        class_counts = filtered_df['Class'].value_counts()
        st.metric("🎓 Most Common Class", f"{class_counts.idxmax()} ({class_counts.max()})")

    # === Data Summary ===
    st.header("📊 Data Summary")
    st.write(filtered_df.describe())
    st.dataframe(filtered_df.head())

    # === Correlation Heatmap ===
    st.header("📈 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(filtered_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # === Model Accuracy ===
    st.header("🧠 Model Accuracy")
    st.success(f"🎯 Accuracy on Filtered Data: {acc:.2%}")

    # === Live Prediction ===
    st.header("📍 Predict Student Class (Live Input)")
    with st.form("predict_form"):
        st.subheader("📝 Enter Student Details")

        input_data = {}
        for col in X.columns:
            if df[col].dtype in ['int64', 'float64']:
                input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))
            else:
                input_data[col] = st.selectbox(f"{col}", options=sorted(df[col].unique()))

        predict_btn = st.form_submit_button("🔮 Predict")

        if predict_btn:
            input_df = pd.DataFrame([input_data])

            # Encode categorical features
            for col in input_df.select_dtypes(include='object').columns:
                le = LabelEncoder()
                le.fit(df[col])
                input_df[col] = le.transform(input_df[col])

            # Scale numeric features
            input_df[scale_cols] = StandardScaler().fit(df[scale_cols]).transform(input_df[scale_cols])

            prediction = model.predict(input_df)[0]
            st.success(f"🧠 Predicted Class: **{prediction}**")

    # === Add / Delete Student Records ===
    st.header("✏️ Update Student Information")
    action = st.radio("Choose Action", ["Add", "Delete"], horizontal=True)

    if action == "Add":
        with st.form("add_student_form"):
            gender = st.selectbox("Gender", df['gender'].unique())
            nationality = st.selectbox("Nationality", df['NationalITy'].unique())
            stage = st.selectbox("Stage", df['StageID'].unique())
            grade = st.selectbox("Grade", df['GradeID'].unique())
            raised = st.slider("Raised Hands", 0, 100)
            visited = st.slider("Visited Resources", 0, 100)
            discussion = st.slider("Discussion", 0, 100)
            class_label = st.selectbox("Class", df['Class'].unique())

            submitted = st.form_submit_button("➕ Add Student")

            if submitted:
                new_row = pd.DataFrame([{
                    "gender": gender,
                    "NationalITy": nationality,
                    "StageID": stage,
                    "GradeID": grade,
                    "raisedhands": raised,
                    "VisITedResources": visited,
                    "Discussion": discussion,
                    "Class": class_label
                }])
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(csv_path, index=False)
                st.success("✅ Student added. Please refresh to see changes.")

    elif action == "Delete":
        student_index = st.number_input("Enter index to delete", 0, len(df) - 1)
        if st.button("🗑️ Delete Student"):
            df = df.drop(student_index).reset_index(drop=True)
            df.to_csv(csv_path, index=False)
            st.success("🧹 Student deleted. Please refresh to see changes.")