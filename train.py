import re
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def num(x):
    m = re.search(r"([0-9]+\.?[0-9]*)", str(x))
    return float(m.group(1)) if m else None

def main() -> None:
    df = pd.read_csv("car details v4.csv")
    for c in ["Engine", "Max Power", "Max Torque"]:
        df[c + "_num"] = df[c].map(num)

    cols = ["Price", "Year", "Kilometer", "Length", "Width", "Height",
            "Seating Capacity", "Fuel Tank Capacity", "Engine_num", "Max Torque_num", "Max Power_num"]
    df = df[cols].dropna()

    X = df.drop(columns=["Max Power_num"])
    y = df["Max Power_num"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    with open("model.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_names": X.columns.tolist()}, f)

    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    print(f"R2 (accuracy): {r2 * 100:.2f}%")
    print("Model saved to model.pkl")

    plt.scatter(y_test, pred, alpha=0.6)
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    plt.plot(lims, lims, "r-", linewidth=2)  # linear reference line y=x
    plt.xlabel("Actual Max Power")
    plt.ylabel("Predicted Max Power")
    plt.title("Linear Regression (Actual vs Predicted)")
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
