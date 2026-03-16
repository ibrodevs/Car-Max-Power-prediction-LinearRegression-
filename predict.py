import pickle

import numpy as np
import pandas as pd

def read_float(prompt: str) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            return float(raw)
        except ValueError:
            print("Invalid input: please enter a number.")

def read_yes_no(prompt: str) -> bool:
    while True:
        raw = input(prompt).strip().lower()
        if raw in {"yes", "y"}:
            return True
        if raw in {"no", "n"}:
            return False
        print("Invalid input: please enter a word 'yes' or 'no' (not a number).")

def main() -> None:
    try:
        with open("model.pkl", "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("model.pkl not found. Run train.py first to train and save the model.")
        return

    model = data["model"]
    feature_names = data["feature_names"]

    print("Enter values for prediction:")
    print("Target to predict: Max Power")

    while True:
        values = [read_float(f"{name}: ") for name in feature_names]
        x_input = pd.DataFrame(np.array([values], dtype=float), columns=feature_names)
        prediction = model.predict(x_input)[0]
        print(f"Predicted Max Power: {prediction:.2f}")

        if not read_yes_no("Do you want to make another prediction? (yes/no): "):
            break

if __name__ == "__main__":
    main()
