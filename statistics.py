import pandas as pa
import math


def calculate_regression(data, x, y):
    result_data = pa.DataFrame()
    x_average = data[x].mean()
    y_average = data[y].mean()
    result_data['rank_x'] = data[x].rank()
    result_data['rank_y'] = data[y].rank()
    result_data['d2'] = (result_data['rank_x'] - result_data['rank_y']) ** 2
    result_data['avg_diff_x'] = data[x] - x_average
    result_data['avg_diff_y'] = data[y] - y_average
    result_data['diff_mul'] = result_data['avg_diff_x'] * result_data['avg_diff_y']
    result_data['avg_diff_x_sqr'] = result_data['avg_diff_x'] ** 2
    result_data['avg_diff_y_sqr'] = result_data['avg_diff_y'] ** 2
    r = result_data['diff_mul'].sum() / (math.sqrt(result_data['avg_diff_x_sqr'].sum()) *
                                         math.sqrt(result_data['avg_diff_y_sqr'].sum()))

    a = result_data['diff_mul'].sum() / result_data['avg_diff_x_sqr'].sum()
    b = y_average - a * x_average
    result_data['expected'] = a * data[x] + b
    r2 = ((result_data['expected'] - y_average) ** 2).sum() / ((data[y] - y_average) ** 2).sum()
    sp_r = 1 - ((6 * result_data['d2'].sum()) /
                (result_data['rank_x'].count() * (result_data['rank_x'].count() ** 2 - 1)))
    return {'a': a, 'b': b, 'r2': r2, 'r': r, 'sp_r': sp_r}
