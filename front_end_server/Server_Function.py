import dash_html_components as html


def generate_table(dataframe, max_rows=10, max_columns=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns[:max_columns]])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns[:max_columns]
            ]) for i in range(min(len(dataframe), max_rows))
        ]),
    ], className='data_table')


def parameter_process(param, default_value):
    if param is None or param == '':
        return default_value
    return int(param)
