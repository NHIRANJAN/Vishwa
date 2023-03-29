import numpy as np

from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load('main.pkl')
@app.route('/')
def home():
   return render_template('index.html')

# creating the Decorators so that we can link our pages
@app.route('/index',methods=['POST','GET'])
def index():
    # collecting the data from html
    if request.method == "POST":
        ext=int(request.form['external'])
        hours =int(request.form['content'])
        final_hours =[np.array(hours)]
        prediction=model.predict([final_hours])
        output = np.round(prediction[0], 2)
        if output<ext:
            observation="Great, Keep it up"
        else:
            observation="Good, But Still need Improvement"
        return render_template('index.html', pedicted="\nExternal Marks Scored should be {} for the internal mark of {}".format((output),(hours)),external="\n Your External Mark is {}".format((ext)),observation="\n Observation : {}".format((observation)))
        #return render_template('index.html', external="\n Your External Mark is {}".format((ext)))
if __name__ == "__main__":
    app.run(debug=True)

