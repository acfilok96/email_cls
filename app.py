from flask import Flask, render_template, redirect, url_for, request
app=Flask(__name__)
import pickle

cv_path = "cv.pkl"
cls_path = "mulnb.pkl"

def prediction(text, cv_path, cls_path):
    text = str(text)
    
    cv_model = pickle.load(open(cv_path, 'rb'))
    numeric_text = cv_model.transform([text])

    cls_model = pickle.load(open(cls_path, 'rb'))
    pred = cls_model.predict(numeric_text)

    return pred[0]


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result", methods=["POST", "GET"])
def result_function():
    if request.method == "POST":
        text = request.form["messege"]
        pred_t = prediction(text, cv_path, cls_path)
        result = " Harm "
        if pred_t == 1:
            result = " Spam "
        return render_template("index.html", msg = result)

if __name__ == "__main__":
    app.run(debug=True)

