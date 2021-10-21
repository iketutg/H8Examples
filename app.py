import flask
from flask.templating import render_template
import numpy as np
import pickle
from matplotlib.figure import Figure
from io import BytesIO
import base64
import pandas as pd 
import plotly
import plotly.express as px
import json 



app = flask.Flask(__name__, template_folder='templates')

model = pickle.load(open('model/model_classifier.pkl', 'rb'))

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI IKG
    '''
    int_features = [int(x) for x in flask.request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = {0: 'not placed', 1: 'placed'}

    return flask.render_template('main.html', prediction_text='Student must be {} to workplace'.format(output[prediction[0]]))

## Contoh mengenerate images file d
@app.route("/plot")
def helloplot():
    # Generate the figure **without using pyplot**.
    fig = Figure()
    ax = fig.subplots()
    ax.plot([1, 2])
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"

## contoh menggunakan https://www.chartjs.org/
@app.route("/chart")
def hellochart():
    data = [
        ("01-10-2021",1500),
        ("02-10-2021",1200),
        ("03-10-2021",1300),
        ("04-10-2021",1450),
        ("05-10-2021",800),
        ("06-10-2021",1320),
        ("07-10-2021",1520),
        ("08-10-2021",1610),
        ("09-10-2021",1700),
        ("10-10-2021",1200),
        ("10-10-2021",1000),
    ]
    labels = [x[0] for x in data]
    values = [y[1] for y in data]
    return render_template("graph.html",labels=labels, values=values)


labels = [
    'JAN', 'FEB', 'MAR', 'APR',
    'MAY', 'JUN', 'JUL', 'AUG',
    'SEP', 'OCT', 'NOV', 'DEC'
]

values = [
    967.67, 1190.89, 1079.75, 1349.19,
    2328.91, 2504.28, 2873.83, 4764.87,
    4349.29, 6458.30, 9907, 16297
]

colors = [
    "#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA",
    "#ABCDEF", "#DDDDDD", "#ABCABC", "#4169E1",
    "#C71585", "#FF4500", "#FEDCBA", "#46BFBD"]

@app.route('/bar')
def bar():
    bar_labels=labels
    bar_values=values
    return render_template('bar_chart.html', title='Harga Bitcoin Bulanan $', max=17000, labels=bar_labels, values=bar_values)

@app.route('/line')
def line():
    line_labels=labels
    line_values=values
    return render_template('line_chart.html', title='Harga Bitcoin Bulanan $', max=17000, labels=line_labels, values=line_values)

@app.route('/pie')
def pie():
    pie_labels = labels
    pie_values = values
    return render_template('pie_chart.html', title='Harga Bitcoin Bulanan $', max=17000, set=zip(pie_values, pie_labels, colors))

@app.route('/plotly1')
def plotly1():
    df = pd.DataFrame({
      'Fruit': ['Apples', 'Oranges', 'Bananas', 'Apples', 'Oranges', 'Bananas'],
      'Amount': [4, 1, 2, 2, 4, 5],
      'City': ['SF', 'SF', 'SF', 'Montreal', 'Montreal', 'Montreal']
    })
    fig = px.bar(df, x='Fruit', y='Amount', color='City',    barmode='group')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('plotly.html', graphJSON=graphJSON)

@app.route('/plotly2')
def plotly2():
    df = px.data.gapminder().query("year == 2007").query("continent == 'Europe'")
    df.loc[df['pop'] < 2.e6, 'country'] = 'Other countries' # Represent only large countries
    fig = px.pie(df, values='pop', names='country', title='Population of European continent')

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('plotly.html', graphJSON=graphJSON)

@app.route('/')
def main():
    return(flask.render_template('main.html'))

if __name__ == '__main__':
    app.run(debug=True)


    