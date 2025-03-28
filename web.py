from dash import Dash, html, dcc, Input, Output
import pickle

# 파일에서 Figure를 불러오는 함수 정의
def load_figure(system):
    filename = f'./data/results/{system}.pkl'
    with open(filename, 'rb') as f:
        fig = pickle.load(f)
    return fig

# Dash 앱 생성
app = Dash(__name__)

# 앱 레이아웃 정의: 제목, 드롭다운, 그래프 컴포넌트 추가
app.layout = html.Div([
    html.H1("GNSS Positioning Visualization"),
    dcc.Dropdown(
        id="gnss-system",
        options=[
            {"label": "GPS", "value": "gps"},
            {"label": "BeiDou", "value": "beidou"},
            {"label": "Galileo", "value": "galileo"}
        ],
        value="gps",  # 기본 선택값
        clearable=False
    ),
    dcc.Graph(id="graph-display")
])

# 콜백 함수: 드롭다운의 선택값이 바뀔 때마다 해당 파일을 불러와서 그래프 업데이트
@app.callback(
    Output("graph-display", "figure"),
    Input("gnss-system", "value")
)
def update_graph(selected_system):
    return load_figure(selected_system)

# 서버 실행 (포트 8050에서 실행)
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)