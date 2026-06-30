import joblib
import pandas as pd


# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load("models/best_model.pkl")


# ==========================================================
# UI FUNCTIONS
# ==========================================================

def print_banner():

    print("\n" + "=" * 70)
    print("            HEART DISEASE PREDICTION SYSTEM")
    print("                Machine Learning Edition")
    print("=" * 70)


def print_goodbye():

    print("\n" + "=" * 70)
    print("Thank you for using the Heart Disease Prediction System!")
    print("=" * 70)


# ==========================================================
# INPUT FUNCTIONS
# ==========================================================

def get_int(prompt, minimum=None, maximum=None):

    while True:

        try:

            value = int(input(prompt))

            if minimum is not None and value < minimum:
                print(f"Enter a value >= {minimum}\n")
                continue

            if maximum is not None and value > maximum:
                print(f"Enter a value <= {maximum}\n")
                continue

            return value

        except ValueError:

            print("Please enter a valid integer.\n")


def get_float(prompt, minimum=None, maximum=None):

    while True:

        try:

            value = float(input(prompt))

            if minimum is not None and value < minimum:
                print(f"Enter a value >= {minimum}\n")
                continue

            if maximum is not None and value > maximum:
                print(f"Enter a value <= {maximum}\n")
                continue

            return value

        except ValueError:

            print("Please enter a valid number.\n")


# ==========================================================
# RISK LEVEL
# ==========================================================

def get_risk_level(confidence):

    if confidence >= 90:
        return "VERY HIGH"

    elif confidence >= 75:
        return "HIGH"

    elif confidence >= 50:
        return "MODERATE"

    else:
        return "LOW"


# ==========================================================
# RECOMMENDATION
# ==========================================================

def print_recommendation(prediction):

    print("\nRecommendation")
    print("-" * 30)

    if prediction == 1:

        print("• Consult a qualified healthcare professional.")
        print("• Maintain a healthy diet.")
        print("• Exercise regularly.")
        print("• Monitor blood pressure and cholesterol.")
        print("• Follow your doctor's advice.")

    else:

        print("• Continue maintaining a healthy lifestyle.")
        print("• Exercise regularly.")
        print("• Eat a balanced diet.")
        print("• Schedule routine health check-ups.")


# ==========================================================
# PATIENT SUMMARY
# ==========================================================

def print_patient_summary(patient):

    print("\n" + "=" * 70)
    print("                     PATIENT SUMMARY")
    print("=" * 70)

    print(f"Age                     : {patient['age']} Years")

    print(
        f"Sex                     : {'Male' if patient['sex']==1 else 'Female'}"
    )

    print(f"Chest Pain Type         : {patient['cp']}")

    print(f"Resting Blood Pressure  : {patient['trestbps']} mmHg")

    print(f"Cholesterol             : {patient['chol']} mg/dL")

    print(
        f"Fasting Blood Sugar     : {'Yes' if patient['fbs']==1 else 'No'}"
    )

    print(f"Resting ECG             : {patient['restecg']}")

    print(
        f"Maximum Heart Rate      : {patient['thalach']} bpm"
    )

    print(
        f"Exercise Angina         : {'Yes' if patient['exang']==1 else 'No'}"
    )

    print(f"Oldpeak                 : {patient['oldpeak']}")

    print(f"Slope                   : {patient['slope']}")

    print(f"Major Vessels           : {patient['ca']}")

    print(f"Thal                    : {patient['thal']}")


# ==========================================================
# MAIN PREDICTION
# ==========================================================

def predict_patient():

    print("\nPlease enter the patient's information.\n")

    age = get_int("Age (29-77): ", 29, 77)

    sex = get_int("Sex (1=Male, 0=Female): ", 0, 1)

    cp = get_int("Chest Pain Type (0-3): ", 0, 3)

    trestbps = get_int("Resting Blood Pressure: ", 80, 250)

    chol = get_int("Cholesterol: ", 100, 700)

    fbs = get_int("Fasting Blood Sugar (>120 mg/dl) (1/0): ", 0, 1)

    restecg = get_int("Resting ECG (0-2): ", 0, 2)

    thalach = get_int("Maximum Heart Rate: ", 50, 250)

    exang = get_int("Exercise Induced Angina (1/0): ", 0, 1)

    oldpeak = get_float("Oldpeak: ", 0, 10)

    slope = get_int("Slope (0-2): ", 0, 2)

    ca = get_int("Number of Major Vessels (0-4): ", 0, 4)

    thal = get_int("Thal (0-3): ", 0, 3)

    patient = {

        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal

    }

    df = pd.DataFrame([patient])

    prediction = model.predict(df)[0]

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(df)[0]

        confidence = probabilities[prediction] * 100

    else:

        confidence = 100

    print_patient_summary(patient)

    print("\n" + "=" * 70)
    print("              HEART DISEASE PREDICTION REPORT")
    print("=" * 70)

    print("\nPrediction")
    print("-" * 30)

    if prediction == 1:

        print("Heart Disease Detected")

    else:

        print("No Heart Disease Detected")

    print("\nConfidence")
    print("-" * 30)
    print(f"{confidence:.2f}%")

    print("\nRisk Level")
    print("-" * 30)
    print(get_risk_level(confidence))

    print_recommendation(prediction)

    print("\nDisclaimer")
    print("-" * 30)
    print("This prediction is generated by a Machine Learning model.")
    print("It should NOT be used as a substitute for professional")
    print("medical diagnosis or treatment.")

    print("=" * 70)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print_banner()

    while True:

        predict_patient()

        choice = input("\nWould you like to predict another patient? (Y/N): ")

        if choice.lower() != "y":
            break

    print_goodbye()


if __name__ == "__main__":
    main()