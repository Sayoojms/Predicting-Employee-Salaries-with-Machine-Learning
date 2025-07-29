# Predicting Employee Salaries with Machine Learning

This project uses machine learning techniques to predict employee salaries based on key features such as experience, test scores, and interview performance. It demonstrates the complete ML pipeline-from data preprocessing and visualization to model training and evaluation
employees given key features. This can be useful for:

*HR professionals to estimate fair compensation.

*Job seekers to benchmark their expected salary.

*Companies to ensure pay equity.

📂 Project Structure
 
 Employee Salary Prediction.ipynb - jupyter notebooks
 
 tuned_salary_model_pipeline.pkl  - Models
 
 income_prediction_app.py         - streamkit application file
 
 requirements.txt                 - dependies
 




🛠️ Technologies Used


This project leverages a suite of powerful Python libraries for data science:

*Core Libraries: Pandas & NumPy for data manipulation.

*Data Visualization: Matplotlib & Seaborn for creating insightful plots.

*Machine Learning: Scikit-learn for building the modeling pipeline and evaluation.

*Imbalanced Data: Imbalanced-learn for the SMOTE resampling technique.

*Model Saving: Joblib for serializing the final model pipeline.



📊 The Dataset

The project uses the "Adult Census Income" dataset, which contains demographic and employment-related information for a collection of individuals. 
The target variable is income, categorized into two classes.



🤖 Machine Learning Workflow



The project follows a systematic workflow to ensure a high-quality, reproducible model.

🧹 Data Cleaning: Loaded the dataset, handled missing values (?), filtered outliers, and removed redundant features.

🎨 Exploratory Data Analysis (EDA): Visualized feature distributions, correlations, and relationships to uncover key insights, most notably the significant class imbalance in the income variable.

🔧 Pipeline Construction: Built an automated pipeline using ColumnTransformer to scale numerical features and one-hot encode categorical features.

⚖️ Handling Imbalance: Integrated SMOTE (Synthetic Minority Over-sampling Technique) into the pipeline to create synthetic samples for the minority class (>50K), preventing model bias.

🧠 Model Training & Tuning: Chose a Gradient Boosting Classifier and optimized its performance using GridSearchCV to find the best hyperparameters.

✅ Evaluation & Saving: Assessed the final model on a held-out test set and saved the entire trained pipeline using joblib for future use.





🏆 Model Performance


The final tuned model achieved strong and balanced performance on the test data.

✅ Cross-Validation Accuracy: 84.63%

🎯 Test Set Accuracy: 84.67%

The classification report shows that the model is both precise in its predictions for the majority class and effective at identifying individuals in the minority class, thanks to the SMOTE balancing.





💡 Key Takeaways



The Importance of Resampling: Using SMOTE was crucial. It significantly improved the model's ability to correctly predict individuals earning >$50K (recall of 79%), which a naive model would have struggled with.

Insightful Features: EDA clearly showed that features like occupation, education, and age are powerful predictors of income.

Power of Pipelines: Automating the workflow with scikit-learn pipelines ensures consistency, prevents data leakage, and makes the model easily deployable.





🔮 Future Improvements



While the current model is robust, there are several avenues for future work:

🚀 Explore Advanced Models: Test more powerful algorithms like XGBoost or LightGBM, which often provide a performance edge.

🧩 Advanced Feature Engineering: Create interaction terms (e.g., age * hours-per-week) to potentially capture more nuanced data patterns.

🌐 Model Deployment: Package the saved model into a REST API using a framework like Flask or FastAPI to serve real-time predictions.











