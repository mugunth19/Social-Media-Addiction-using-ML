FROM public.ecr.aws/lambda/python:3.11

WORKDIR /var/task

# Update pip and install runtime dependencies directly. Install numpy first
# and prefer binary wheels to avoid compiling from source in the image.
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefer-binary numpy scikit-learn joblib pandas

# Copy model artifacts and handler
COPY logistic_regression_model.pkl scaler.pkl dict_vectorizer.pkl lambda.py ./

# Lambda handler
CMD ["lambda.lambda_handler"]
