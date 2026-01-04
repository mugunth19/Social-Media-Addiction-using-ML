import json
import os
import joblib
import traceback

# Load artifacts from the package root so they can be included in the Lambda bundle
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "logistic_regression_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
DV_PATH = os.path.join(BASE_DIR, "dict_vectorizer.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    dv = joblib.load(DV_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")


def lambda_handler(event, context):
    """AWS Lambda handler (single-request).

    Expects `event['body']` to be a JSON object with the same fields used
    by the `predict.py` Pydantic model. Returns a single JSON prediction.
    """
    try:
        body = event.get("body")
        if not body:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Missing request body"}),
            }

        payload = json.loads(body)
        if not isinstance(payload, dict):
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Request body must be a JSON object"}),
            }

        # Transform and predict
        X = dv.transform([payload])
        X_scaled = scaler.transform(X)
        pred = int(model.predict(X_scaled)[0])

        # Build result using the model's prediction only.
        result = {
            "prediction": pred,
            "addiction_status": "Addicted" if pred == 1 else "Not Addicted",
        }

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result),
        }

    except Exception as exc:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(exc), "trace": traceback.format_exc()}),
        }
