import numpy as np
import pandas as pa
import os.path as path
import plotly.express as px
import plotly.graph_objects as go
import statistics as sta


def print_hi():
    current_directory = path.curdir
    data_file = path.join(current_directory, 'data\\deaths.csv')
    test_data = pa.read_csv(data_file, delimiter=';')
    reg_result = sta.calculate_regression(test_data, 'nurses', 'deaths')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_data['nurses'], y=test_data['deaths'], mode='markers'))
    fig.add_trace(go.Scatter(
        x=test_data['nurses'],
        y=reg_result['a'] * test_data['nurses'] + reg_result['b'],
        mode='lines',
        name=f'Regression: r2:{reg_result["r2"]:.2f} and r:{reg_result["r"]:.2f}'
    ))

    fig.update_layout(title='Test Regression', xaxis_title="Nurse count", yaxis_title='Deaths')
    fig.show()
    print(reg_result['sp_r'])


if __name__ == '__main__':
    print_hi()
