from flask import Flask, render_template, url_for, request
import stock_prediction as model

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    if(request.method == 'POST'):
        input_date = request.form['dateinput']
        result = str(model.make_prediction(input_date))
        return render_template("result.html", result=result, input_date=input_date)
    else:
        return render_template("index.html")


if(__name__ == "__main__"):
    app.run(debug=True)
