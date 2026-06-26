from flask import Flask, render_template, request
import pandas as pd
import pickle


app = Flask(__name__)


model = pickle.load(open("churn.model", "rb"))

MODEL_COLUMNS = [
    'SeniorCitizen', 'MonthlyCharges', 'TotalCharges',
    'gender_Female', 'gender_Male',
    'Partner_No', 'Partner_Yes',
    'Dependents_No', 'Dependents_Yes',
    'PhoneService_No', 'PhoneService_Yes',
    'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_No', 'PaperlessBilling_Yes',
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    'tenure_group_1 - 12', 'tenure_group_13 - 24', 'tenure_group_25 - 36',
    'tenure_group_37 - 48', 'tenure_group_49 - 60', 'tenure_group_61 - 72',
]


def make_tenure_group(tenure):
    bins = list(range(1, 80, 12))   # [1, 13, 25, 37, 49, 61, 73]
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    for i in range(len(bins) - 1):
        if bins[i] <= tenure < bins[i + 1]:
            return labels[i]
    return labels[-1]  # fallback for edge values


def build_model_input(form):

    row = {col: 0 for col in MODEL_COLUMNS}

    row["SeniorCitizen"] = int(form["SeniorCitizen"])
    row["MonthlyCharges"] = float(form["MonthlyCharges"])
    row["TotalCharges"] = float(form["TotalCharges"])

    tenure = int(form["tenure"])
    tenure_group = make_tenure_group(tenure)
    row[f"tenure_group_{tenure_group}"] = 1

    categorical_fields = [
        ("gender", "gender"),
        ("Partner", "Partner"),
        ("Dependents", "Dependents"),
        ("PhoneService", "PhoneService"),
        ("MultipleLines", "MultipleLines"),
        ("InternetService", "InternetService"),
        ("OnlineSecurity", "OnlineSecurity"),
        ("OnlineBackup", "OnlineBackup"),
        ("DeviceProtection", "DeviceProtection"),
        ("TechSupport", "TechSupport"),
        ("StreamingTV", "StreamingTV"),
        ("StreamingMovies", "StreamingMovies"),
        ("Contract", "Contract"),
        ("PaperlessBilling", "PaperlessBilling"),
        ("PaymentMethod", "PaymentMethod"),
    ]

    for field_name, prefix in categorical_fields:
        chosen_value = form[field_name]          
        column_name = f"{prefix}_{chosen_value}"  
        if column_name in row:
            row[column_name] = 1

    input_df = pd.DataFrame([row], columns=MODEL_COLUMNS)
    return input_df


@app.route("/", methods=["GET"])
def home():
    return render_template("home.html", prediction_text="")


@app.route("/predict", methods=["POST"])
def predict():
    
    input_df = build_model_input(request.form)

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1] 

    if prediction == 1:
        result_text = (
            f"This customer is LIKELY TO CHURN "
            f"(confidence: {probability * 100:.1f}%)"
        )
    else:
        result_text = (
            f"This customer is LIKELY TO STAY "
            f"(confidence: {(1 - probability) * 100:.1f}%)"
        )

    return render_template("home.html", prediction_text=result_text)


if __name__ == "__main__":
    app.run(debug=True)
