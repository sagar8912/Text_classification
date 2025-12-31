from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained pipeline model (TF-IDF + LinearSVC)
model = joblib.load("text_classifier_linearsvc.pkl")


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Safely get input text
    user_text = request.form.get("text", "").strip()

    if user_text == "":
        return render_template(
            "index.html",
            error="Please enter some text before predicting."
        )

    # IMPORTANT: pass text as LIST
    prediction = model.predict([user_text])[0]

    return render_template(
        "result.html",
        original_text=user_text,
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)
