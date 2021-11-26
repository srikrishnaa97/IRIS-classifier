from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open('IRIS_classifier_model','rb') as f:
    model = pickle.load(f)

@app.route('/',methods=['POST','GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        prediction = []
        try: 
            sepal_len = float(request.form['sepal_len'])
            sepal_wid = float(request.form['sepal_width'])
            petal_len = float(request.form['petal_len'])
            petal_wid = float(request.form['petal_width'])

        except:
            prediction = [-999]
        else:
            prediction = model.predict([[sepal_len,sepal_wid,petal_len,petal_wid]])
            
        finally:
            return render_template('index.html',prediction=prediction)

    else:
        return render_template('index.html')
        

if __name__ == '__main__':
    app.run(debug=True)