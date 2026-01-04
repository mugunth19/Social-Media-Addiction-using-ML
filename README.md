# Social Media Addiction Prediction

This project aims to predict social media addiction based on various student-related features. The 'addicted_score' was transformed into a binary classification problem, where a score of 7 or higher indicates 'addicted' and lower than 7 indicates 'not addicted'.

## Data Source

The dataset used for this project is `Students Social Media Addiction.csv`, obtained from KaggleHub (`adilshamim8/social-media-addiction-vs-relationships`).

## Data Preprocessing

1.  **Column Renaming**: All column names were converted to lowercase for consistency.
2.  **Irrelevant Feature Removal**: The `student_id` column was dropped as it does not contribute to the prediction.
3.  **Missing Value Check**: The dataset was checked for missing values, and none were found.
4.  **One-Hot Encoding**: Categorical features were converted into numerical format using `DictVectorizer`.
5.  **Feature Scaling**: Numerical features were scaled using `StandardScaler` to normalize their ranges.

## Exploratory Data Analysis (EDA)

Key insights from the EDA included:

*   **Age Distribution**: The age distribution of students was visualized.
*   **Addiction Score Distribution**: The distribution of the `addicted_score` was examined, and a log transformation was applied to observe its distribution.
*   **Mutual Information**: `conflicts_over_social_media`, `mental_health_score`, `avg_daily_usage_hours`, and `sleep_hours_per_night` showed high mutual information with the `addicted_score`.
*   **Correlation Matrix**: A heatmap of numerical features revealed strong positive correlations between `avg_daily_usage_hours` and `conflicts_over_social_media` with `addicted_score`, and strong negative correlations between `sleep_hours_per_night` and `mental_health_score` with `addicted_score`.

## Model Training

The preprocessed data was split into training, validation, and test sets. Two classification models were trained:

1.  **Logistic Regression**
2.  **Random Forest Classifier**

## Model Evaluation

Both models were evaluated on the validation and test sets using classification reports and AUC scores. The results showed excellent performance:

### Logistic Regression
*   **Validation Set**: Accuracy: 1.00, AUC Score: 1.00
*   **Test Set**: Accuracy: 0.99, AUC Score: 0.999

### Random Forest
*   **Validation Set**: Accuracy: 1.00, AUC Score: 1.00
*   **Test Set**: Accuracy: 0.99, AUC Score: 0.999

Both models performed exceptionally well, indicating strong predictive capabilities on this dataset.

## Saved Models

The trained Logistic Regression model (`logistic_regression_model.pkl`) and the `StandardScaler` (`scaler.pkl`) have been saved using `joblib` for future use.