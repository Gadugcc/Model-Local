from flask import Flask, render_template, request
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        input_x = request.form.get('input_x')

        if file and input_x:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            
            df = pd.read_csv(filepath,sep=';')
            if 'x' in df.columns and 'y' in df.columns:
                X = df[['x']].values
                y = df['y'].values

                model = LinearRegression()
                model.fit(X, y)

                x_val = np.array([[float(input_x)]])
                prediction = model.predict(x_val)[0]
            else:
                prediction = 'Arquivo deve conter colunas "x" e "y".'
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
