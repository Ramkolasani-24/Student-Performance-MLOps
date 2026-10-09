import json
import sys

MINIMUM_MACRO_F1 = 0.45

with open("artifacts/metrics.json", "r") as file:
    metrics = json.load(file)

macro_f1 = metrics["macro_f1"]

print("Model macro F1:", round(macro_f1, 4))
print("Required macro F1:", MINIMUM_MACRO_F1)

if macro_f1 < MINIMUM_MACRO_F1:
    print("QUALITY GATE FAILED")
    print("Model performance is below the threshold.")
    sys.exit(1)

print("QUALITY GATE PASSED")
print("Model performance meets the required threshold.")
sys.exit(0)
