#!/usr/bin/env python3
import argparse
import json
import sys

import requests


def main():
    parser = argparse.ArgumentParser(description="Send a sample request to the /predict endpoint")
    parser.add_argument("--url", default="http://localhost:8000/predict", help="Prediction endpoint URL")
    args = parser.parse_args()

    payload = {
        "age": 21,
        "gender": "Female",
        "academic_level": "University",
        "avg_daily_usage_hours": 5.5,
        "most_used_platform": "Instagram",
        "sleep_hours_per_night": 6,
        "mental_health_score": 65,
        "conflicts_over_social_media": 3,
        "affects_academic_performance": "Yes",
        "relationship_status": "Single"
    }

    try:
        resp = requests.post(args.url, json=payload, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print("Request failed:", e)
        sys.exit(1)

    try:
        data = resp.json()
        print(json.dumps(data, indent=2))
    except ValueError:
        print("Non-JSON response:\n", resp.text)


if __name__ == "__main__":
    main()
