import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import base64

# Set page configuration
st.set_page_config(page_title="Salary Prediction Pro", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
    
    body {
        font-family: 'Montserrat', sans-serif;
        color: #f0f0f0;
    }

    .reportview-container {
        background: linear-gradient(to bottom right, #0f2027, #203a43, #2c5364);
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    .stButton>button {
        color: #ffffff;
        background: linear-gradient(to right, #ff4b4b, #ff7b7b);
        border-radius: 25px;
        border: none;
        padding: 12px 28px;
        font-size: 16px;
        font-weight: 600;
        box-shadow: 0 4px 15px 0 rgba(255, 10, 80, 0.4);
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(255, 10, 80, 0.5);
    }
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 600;
    }
    .stExpander {
        border: 1px solid #777;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.05);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .stImage > img {
        border-radius: 10px;
    }
    .prediction-card {
        background: linear-gradient(to right, #6a11cb, #2575fc);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 20px 0 rgba(0,0,0,0.2);
    }
    .prediction-card h3 {
        color: white;
        font-size: 24px;
        font-weight: 600;
    }
    .prediction-card p {
        font-size: 36px;
        font-weight: 700;
        margin-top: 10px;
    }
    /* Ensure text inside widgets on the main page is light */
    .st-emotion-cache-q8sbsg p, .st-emotion-cache-16txtl3 {
        color: #f0f0f0;
    }
    /* Ensure text inside widgets in the sidebar is dark */
    .sidebar .st-emotion-cache-q8sbsg p, .sidebar .st-emotion-cache-16txtl3 {
        color: #31333F;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.image("https://placehold.co/150x150/ff4b4b/FFFFFF?text=S-AI", use_container_width=True)
    st.title("Salary-AI Pro")
    st.write("Upload your dataset and let the machine learning model predict salary brackets.")
    
    uploaded_file = st.file_uploader("Upload your CSV", type="csv")
    
    st.subheader("⚙️ Model Hyperparameters")
    n_estimators = st.slider("Number of Estimators", 100, 500, 200, 100)
    learning_rate = st.selectbox("Learning Rate", [0.01, 0.05, 0.1, 0.2], index=2)
    max_depth = st.slider("Max Depth", 3, 10, 4, 1)

    train_button = st.button("🚀 Train Model")

# --- Main Panel ---
st.image("https://placehold.co/1200x300/31333F/FFFFFF?text=Salary+Prediction+Dashboard", use_container_width=True)


if 'best_model' not in st.session_state:
    st.session_state.best_model = None

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)

    # Display data preview in an expander
    with st.expander("📊 Data Preview", expanded=True):
        st.dataframe(data.head())

    # --- Data Cleaning and Preprocessing ---
    st.subheader("🧹 Data Cleaning & Preprocessing")
    
    with st.spinner("Cleaning and preprocessing data..."):
        # Replace '?' with NaN and drop rows with missing essential data
        data.replace('?', np.nan, inplace=True)
        data.dropna(subset=['workclass', 'occupation', 'native-country'], inplace=True)

        # Filter data based on outlier analysis
        data = data[(data['age'] <= 75) & (data['age'] >= 17)]
        data = data[(data['educational-num'] <= 16) & (data['educational-num'] >= 5)]
        if 'education' in data.columns:
            data = data.drop(columns=['education'])

        # Reset index after dropping rows
        data.reset_index(drop=True, inplace=True)
    st.success("Data cleaned and preprocessed successfully.")

    # --- Exploratory Data Analysis (EDA) ---
    st.subheader("📈 Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Distribution of Income Classes**")
        fig1 = px.histogram(data, x='income', color='income', title='Distribution of Income Classes', color_discrete_sequence=px.colors.qualitative.Pastel, template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.write("**Age Distribution**")
        fig2 = px.histogram(data, x='age', nbins=30, title='Age Distribution', color_discrete_sequence=['skyblue'], template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    # --- Model Training ---
    if train_button:
        st.subheader("🤖 Model Training")
        with st.spinner("Training model... This may take a while."):
            # Separate features (X) and target (y)
            X = data.drop('income', axis=1)
            y = data['income']

            # Identify feature types
            categorical_features = X.select_dtypes(include=['object']).columns
            numerical_features = X.select_dtypes(include=np.number).columns

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # Create a preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ],
                remainder='passthrough'
            )

            # Define the full model pipeline with SMOTE
            pipeline_with_smote = ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('classifier', GradientBoostingClassifier(random_state=42))
            ])

            # Define the hyperparameter grid from sidebar inputs
            param_grid = {
                'classifier__n_estimators': [n_estimators],
                'classifier__learning_rate': [learning_rate],
                'classifier__max_depth': [max_depth]
            }

            # Set up and run GridSearchCV
            grid_search = GridSearchCV(pipeline_with_smote, param_grid, cv=3, n_jobs=-1, scoring='accuracy', verbose=2)
            grid_search.fit(X_train, y_train)

            st.session_state.best_model = grid_search.best_estimator_

            st.success("Model training complete!")

            # --- Model Evaluation ---
            st.subheader("🎯 Model Evaluation")
            
            eval_col1, eval_col2 = st.columns(2)
            
            with eval_col1:
                st.metric("Best cross-validation accuracy", f"{grid_search.best_score_:.4f}")
                
                y_pred = st.session_state.best_model.predict(X_test)

                st.metric("Test Set Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")

                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred))

            with eval_col2:
                # Confusion Matrix Visualization
                st.write("**Confusion Matrix**")
                plt.style.use('dark_background')
                cm = confusion_matrix(y_test, y_pred)
                fig3, ax3 = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=st.session_state.best_model.classes_, yticklabels=st.session_state.best_model.classes_, ax=ax3)
                ax3.set_title('Confusion Matrix', fontsize=16)
                ax3.set_ylabel('Actual')
                ax3.set_xlabel('Predicted')
                st.pyplot(fig3)

            # --- Save Model ---
            st.subheader("💾 Save Model")
            model_file = "tuned_salary_model_pipeline.pkl"
            joblib.dump(st.session_state.best_model, model_file)
            st.success(f"✅ Saved tuned model pipeline to '{model_file}'")
            
            with open(model_file, "rb") as f:
                st.download_button("Download Model", f, file_name=model_file)

    st.markdown("<hr>", unsafe_allow_html=True)
    # --- Real-time Prediction ---
    if st.session_state.best_model:
        with st.expander("🔮 Real-time Prediction", expanded=True):
            
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            
            with pred_col1:
                age = st.number_input("Age", 17, 75, 30)
                workclass = st.selectbox("Workclass", data['workclass'].unique())
                fnlwgt = st.number_input("Fnlwgt", 0, 1000000, 150000)
            
            with pred_col2:
                educational_num = st.slider("Educational Number", 5, 16, 10)
                marital_status = st.selectbox("Marital Status", data['marital-status'].unique())
                occupation = st.selectbox("Occupation", data['occupation'].unique())

            with pred_col3:
                relationship = st.selectbox("Relationship", data['relationship'].unique())
                race = st.selectbox("Race", data['race'].unique())
                gender = st.selectbox("Gender", data['gender'].unique())
                capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
                capital_loss = st.number_input("Capital Loss", 0, 10000, 0)
                hours_per_week = st.number_input("Hours per Week", 1, 99, 40)
                native_country = st.selectbox("Native Country", data['native-country'].unique())

            if st.button("✨ Predict Salary"):
                input_data = pd.DataFrame({
                    'age': [age], 'workclass': [workclass], 'fnlwgt': [fnlwgt], 
                    'educational-num': [educational_num], 'marital-status': [marital_status], 
                    'occupation': [occupation], 'relationship': [relationship], 'race': [race], 
                    'gender': [gender], 'capital-gain': [capital_gain], 'capital-loss': [capital_loss], 
                    'hours-per-week': [hours_per_week], 'native-country': [native_country]
                })
                
                prediction = st.session_state.best_model.predict(input_data)
                prediction_proba = st.session_state.best_model.predict_proba(input_data)
                confidence = np.max(prediction_proba) * 100

                st.markdown(f"""
                <div class="prediction-card">
                    <h3>Predicted Salary Bracket</h3>
                    <p>{prediction[0]}</p>
                    <p style="font-size: 16px; font-weight: 400;">Confidence: {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

else:
    st.info("👈 Please upload a CSV file using the sidebar to begin.")

