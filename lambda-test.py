#!/usr/bin/env python3
import os
import sys
import json
import argparse

try:
    import requests
except Exception:
    print("Missing dependency: requests. Install with: pip install requests")
    sys.exit(1)

DEFAULT_PAYLOAD = {
    "age": 25,
    "gender": "Male",
    "academic_level": "Undergraduate",
    "avg_daily_usage_hours": 3.5,
    "most_used_platform": "Instagram",
    "sleep_hours_per_night": 7,
    "mental_health_score": 5,
    "conflicts_over_social_media": 0,
    "affects_academic_performance": "No",
    "relationship_status": "Single",
}


def load_payload(args):
    if args.payload_file:
        with open(args.payload_file, "r", encoding="utf-8") as f:
            return json.load(f)
    if args.data:
        return json.loads(args.data)
    return DEFAULT_PAYLOAD


def main():
    parser = argparse.ArgumentParser(description="Send a test POST to your Lambda API Gateway endpoint")
    parser.add_argument("url", nargs="?", help="Full API Gateway URL (or set LAMBDA_URL env var)")
    parser.add_argument("--payload-file", "-f", help="Path to JSON file with payload")
    parser.add_argument("--data", "-d", help="Inline JSON payload string")
    parser.add_argument("--show", action="store_true", help="Print the request payload before sending")
    args = parser.parse_args()

    url = "https://zmkcs7373e.execute-api.ap-south-1.amazonaws.com/development/smaddiction-predict/predict"

    payload = load_payload(args)

    if args.show:
        print("Request payload:")
        print(json.dumps(payload, indent=2))

    # API Gateway wraps the JSON body in a 'body' field
    api_payload = {"body": json.dumps(payload)}

    try:
        resp = requests.post(url, json=api_payload, timeout=20)
    except Exception as e:
        print(f"Request failed: {e}")
        sys.exit(3)

    print(f"Status: {resp.status_code}")
    # Try to print JSON response, fall back to text
    try:
        print(json.dumps(resp.json(), indent=2))
    except Exception:
        print(resp.text)


if __name__ == "__main__":
    main()
