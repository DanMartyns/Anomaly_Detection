import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
from plotly.subplots import make_subplots

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

x = [str(a) for a in np.arange(1,53)]
# ETC
# y = [82.78, 84.33, 86.08, 82.71, 83.55, 82.22, 83.97, 81.59, 86.38,86.18, 82.47, 87.54, 84.06, 83.90, 84.38, 83.04, 86.66, 87.32, 83.21, 87.66, 83.59, 85.26, 84.22, 86.05, 
# 85.16,86.31, 86.05,85.61, 86.00, 85.92, 86.63, 85.50, 85.32, 84.95, 86.07, 85.21, 84.54, 84.96, 84.50, 86.65, 84.12, 85.15, 84.59, 85.27, 83.89, 85.99, 85.32, 85.23, 84.48, 84.09, 84.93, 84.12]

# PCA 
# y = [82.76, 84.67, 82.34, 82.91, 82.37, 83.25, 82.81, 82.59, 82.05, 82.21, 83.01, 83.04, 82.69, 84.20, 84.29, 83.72, 83.58, 82.79, 83.11, 82.25, 82.58, 83.93, 83.69, 83.50, 
# 83.43, 82.44, 83.78, 82.39, 85.15, 83.20, 85.28, 83.06, 82.73, 83.17, 84.63, 83.99, 83.60, 83.54, 85.71, 84.31, 84.30, 83.98, 84.33, 84.84, 84.46, 85.53, 82.07, 82.15, 82.19, 82.14, 82.17, 82.19 ] + [0]*0

# ETC
# x = [1,5,10,15,20,25,30,35,40,45]
# y = [73.51, 100, 100, 100, 100, 99, 100, 100, 100, 98]

# ETC simplificado
x = [1,2,3,4,5]
y = [83.20,75.23,76.54,100,100]

# PCA 
# xx = [1,5,10,15,20,25,30,35,40,45]
# yy = [75.49, 100, 90.86, 93.93, 99.68, 100, 100, 100, 100, 100]

# PCA simplificado
# x = [36,37,38,39,40]
# y = [83.90,0,0,0,0,0]

# ETC with PCA
# x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# y = [71.22, 60.84, 85.84, 85.28, 88.92, 83.73, 83.77, 86.63, 86.03, 83.70, 83.88, 83.30, 83.18, 82.88, 83.93, 85.07, 80.16, 86.83,83.02,84.99]

data = {'NrFeatures': x, 'F1Score': y}
df = pd.DataFrame(data=data)
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x = df['NrFeatures'], y = df['F1Score'], mode='lines+markers', name='PCA over ETC', marker_color='#10739E', marker_line_color='#10739E'), row=1, col=1)
# fig.add_trace(go.Scatter(x = df['NrFeatures'], y = df['F1Score'], mode='lines+markers', name='ETC', marker_color='#10739E', marker_line_color='#10739E'), row=1, col=1)
# fig.add_trace(go.Scatter(x = df['NrFeaturesPCA'], y = df['F1ScorePCA'], mode='lines+markers', name='PCA'), row=1, col=1)  

# Customize aspect
# fig.add_annotation(x=16, y=76.92,
#             text="76.92%",
#             showarrow=True,
#             arrowhead=1)
# fig.add_annotation(x=17, y=78.39,
#             text="78.39%",
#             showarrow=True,
#             arrowhead=1)
# fig.add_annotation(x=18, y=80.12,
#             text="80.12%",
#             showarrow=True,
#             arrowhead=1)
# fig.add_annotation(x=19, y=82.25,
#             text="82.25%",
#             showarrow=True,
#             arrowhead=1)
# fig.add_annotation(x=20, y=86.89,
#             text="86.90%",
#             showarrow=True,
#             arrowhead=1)
# fig.add_annotation(x=40, y=85.79,
#             text="max = 85.79%",
#             showarrow=True,
#             arrowhead=1)
# fig.add_annotation(x=20, y=87.02,
#             text="max = 87.02%",
#             showarrow=True,
#             arrowhead=1)

# fig.add_annotation(x=5, y=88.92,
#             text="max = 88.92%",
#             showarrow=True,
#             arrowhead=1)
fig.update_yaxes(range=[0,105])
fig.update_xaxes(dtick=1, row=1, col=1)
fig.update_layout(
    xaxis_title="Number of Features",
    yaxis_title="F1-Score",
    font_family="Courier New",
    font_size=15,
    title={
        'text': "F1-Score applying PCA over ETC",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    autosize = False,
    width= 800,
    height= 800
    )

app.layout = html.Div(children=[

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=False, port=8053)




