import pathlib
import uuid
from collections import defaultdict

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import requests
from dash.dependencies import Input, Output, State
from typing import Union

app = dash.Dash(
    __name__,
    title='Bert-Net',
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

server = app.server
app.config.suppress_callback_exceptions = True
global_dict = defaultdict(dict)

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()


def description_card():
    """
    控制面板描述卡
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.Img(src=app.get_asset_url("logo.png"), className='banner'),
            html.Div(
                id="intro",
                children="harvest knowledge from pretrained language models",
            ),
        ],
    )


operator_list = ['+', '-', '*', '/']


def generate_control_card():
    """
    :return: A Div containing control options.
    """
    return html.Div(
        id="control-card",
        children=[
            html.H6('Settings'),
            html.P('Task Type'),
            html.Div(
                dcc.Dropdown(
                    id="task-type",
                    options=['RC','All Entity'],
                    value='RC',
                    clearable=False,
                )
            ),
            html.Br(),
            html.P('Model Type'),
            html.Div(
                dcc.Checklist(
                    id="model-type",
                    options=[
                        {'label': '  BERT-base', 'value': 'bert-base'},
                        {'label': '  BERT-large', 'value': 'bert-large'},
                        {'label': '  DistilBERT', 'value': 'DistilBERT'},
                        {'label': '  RoBERTa-base', 'value': 'roberta-base'},
                        {'label': '  RoBERTa-large', 'value': 'roberta-large'},
                        ],
                    value=['bert-base'],
                )
            ),
            html.Br(),
            html.P("prompt diversity (similarity threshold)"),
            html.Div(dcc.Slider(0, 100, marks=None, value=10, id="diversity",
                                tooltip={"placement": "bottom", "always_visible": True})),
            html.Br(),
            html.P('top-k entity pairs'),
            html.Div(dcc.Slider(0, 100, marks=None, value=10, id="top_k",
                                tooltip={"placement": "bottom", "always_visible": True})),
            html.Br(),
            html.P("number of prompts"),
            html.Div(dcc.Slider(0, 20, marks=None, value=10, id="top_p",
                                tooltip={"placement": "bottom", "always_visible": True})),
            html.Br(),
            dcc.Interval(id="interval", interval=5000, n_intervals=0),
            html.Div(
                id="start-btn-outer",
                children=html.Button(id="start-btn", children="Start", n_clicks=0),
            ),
        ],
    )


def server_layout():
    # global layout
    session_id = str(uuid.uuid4())
    print('session_id', session_id)
    return html.Div(
        id="app-container",
        children=[
            html.Div(
                id="left-column",
                # className="four columns",
                className="three columns",
                children=[description_card(), generate_control_card()]
                         + [
                             html.Div(
                                 ["initial child"], id="output-clientside", style={"display": "none"}
                             )
                         ],
            ),
            # Right column
            html.Div(
                id="right-column",
                className="nine columns",
            ),
            html.Div(session_id, id='session-id', style={'display': 'none'}),
        ],
    )


@app.callback([Output('right-column', 'children'),
               Output('start-btn', 'n_clicks')],
              Input('task-type', 'value'))
def result_bar(task_type):
    # 生成右边面板
    if task_type == 'All Entity':
        return html.Div(children=[  # 注意修改CSS
            html.Div(
                id="input_relation_card",
                children=[
                    html.B("Entity"),
                    html.Hr(),
                    dcc.Textarea(
                        id='Seed-Entity',
                        placeholder='Enter Entity...',
                        value='',
                        style={'width': '100%', 'margin-top': '5px', 'height': '100px'}
                    ),
                ],
            ),
            html.Div(id='Input-Relation'),
            html.Div(
                id="result_card",
                children=[
                    html.B("Final Result"),
                    html.Hr(),
                    html.Div(
                        id='result-state',
                    ),
                ],
            )]), 0
    else:
        return html.Div(children=[  # 注意修改CSS
            html.Div(
                id="input_relation_card",
                children=[
                    html.B("Input Relation"),
                    html.Hr(),
                    dcc.Textarea(
                        id='Input-Relation',
                        placeholder='Enter Relationships...',
                        value='',
                        style={'width': '100%', 'margin-top': '5px', 'height': '100px'}
                    ),
                ],
            ),
            html.Div(
                id="seed_entity_card",
                children=[
                    html.B("Seed Entity"),
                    html.Hr(),
                    dcc.Textarea(
                        id='Seed-Entity',
                        placeholder='Enter Seed Entities...',
                        value='',
                        style={'width': '100%', 'margin-top': '5px'}
                    ),
                ],
            ),
            html.Div(
                id="result_card",
                children=[
                    html.B("Final Result"),
                    html.Hr(),
                    html.Div(
                        id='result-state',
                    ),
                ],
            )]), 0


def generate_all_entities(entity):
    return entity


def generate_final_result(var_list=None):
    # 生成可视化的结果界面
    if var_list is None:
        var_list = [
            {
                'head': 'A subway station',
                'relation': f'AtLocation',
                'tail': 'New York',
                'weight': '0.111',
            },
            {
                'head': 'A subway station',
                'relation': f'AtLocation',
                'tail': 'New York',
                'weight': '0.111',
            }
        ]
    children = []
    for v in var_list:
        children.append(html.Div(
            children=[html.B(v['head'],
                             style={
                                 'background-color': 'lightseagreen',
                                 'font-size': '15px'
                             }
                             ),
                      html.Span(' '),
                      html.B(v['relation'], style={
                          'background-color': 'midnightblue',
                          'color': 'white',
                          'font-size': '15px'
                      }),
                      html.Span(' '),
                      html.B(v['weight'], style={
                          'background-color': '#018282',
                          'color': 'white',
                          'font-size': '15px'
                      }),
                      html.Span(' '),
                      html.B(v['tail'],
                             style={
                                 'background-color': 'lightseagreen',
                                 'font-size': '15px'
                             }
                             ),
                      html.Br()]
        ))
    return children


app.layout = server_layout()


@app.callback(Output('result-state', 'children'),
              [
                  Input('start-btn', 'n_clicks'),
                  Input("interval", "n_intervals"),
              ],
              [
                  State('session-id', 'children'),
                  State('task-type', 'value'),
                  State('model-type', 'value'),
                  State('diversity', 'value'),
                  State('top_k', 'value'),
                  State('top_p', 'value'),
                  State('Input-Relation', 'value'),
                  State('Seed-Entity', 'value'),
              ])
def update_output(n_clicks, interval, session_id, task_type, model_type, diversity, top_k, top_p, input_relation, seed_entity):
    print('output session_id', session_id)
    if n_clicks > 0:
        if global_dict[session_id]['n_clicks'] == n_clicks:
            if task_type == 'All Entity':
                url = 'http://127.0.0.1:8000/all_relation'
                x = requests.post(url, json=global_dict[session_id]['info'])
                return generate_table(x.json())
            else:
                # result = send_request(global_dict[session_id]['info'])
                info = global_dict[session_id]['info']
                x = requests.get(f"http://127.0.0.1:8000/predict/{model_type}/{info['relation']}/{info['prompt']}")
                result = x.json()
                all_results = [{
                    'Head': r[0][0],
                    'Relation': info['relation'],
                    'Tail': r[0][1],
                    'Weight': r[1],
                } for r in result]
                print(pd.DataFrame(all_results))
                # return html.Div(generate_final_result(all_results))
                return generate_table(all_results)
        else:
            global_dict[session_id]['n_clicks'] = n_clicks
            if task_type == 'All Entity':
                url = 'http://127.0.0.1:8000/all_relation'
                info = {
                    'entity': seed_entity
                }
                global_dict[session_id]['info'] = info
                x = requests.post(url, json=info)
                return generate_table(x.json())
            else:
                info = {
                    'relation': input_relation,
                    'prompt': seed_entity,
                }
                global_dict[session_id]['info'] = info
                # result = send_request(info)
                info = global_dict[session_id]['info']
                x = requests.get(f"http://127.0.0.1:8000/predict/{info['relation']}/{info['prompt']}")
                result = x.json()
                print('result', result)
                all_results = [{
                    'Head': r[0][0],
                    'Relation': info['relation'],
                    'Tail': r[0][1],
                    'Weight': r[1],
                } for r in result]
                # return html.Div(generate_final_result(all_results))
                return generate_table(all_results)
    else:
        global_dict[session_id]['n_clicks'] = 0
        if task_type == 'All Entity':
            return html.Div('Please Input Data!')
        else:
            return html.Div('Please Input Data!')


def generate_table(data: Union[dict, list]):
    from dash import html
    column_per_row = 4
    all_table = []
    if isinstance(data, list):
        data = pd.DataFrame.from_dict(data)
    else:
        data = pd.DataFrame.from_dict(data, orient='index').T
    print(data)
    for column in range(0, len(data.columns), column_per_row):
        table_header = [
            html.Thead(html.Tr([html.Th(x, style={'text-align': 'center'})
                                for x in data.columns[column:column + column_per_row]]))
        ]

        all_rows = []
        for r in range(len(data)):
            row_data = [html.Td(r, style={'text-align': 'center'})
                        for r in list(data.iloc[r, column: column + column_per_row])]
            row1 = html.Tr(row_data)
            all_rows.append(row1)

        table_body = [html.Tbody(all_rows)]
        all_table.extend(table_header + table_body)
    table = dbc.Table(all_table, bordered=True, id='entity_table')
    return table


if __name__ == "__main__":
    app.run_server(debug=True)
    # app.run_server()
    # app.run_server(host='0.0.0.0')
