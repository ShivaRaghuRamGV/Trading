import pandas as pd
from datetime import datetime
from dash import Dash, html, dcc, Input, Output
import plotly.express as px

# 1. Load data — adjust file paths / URLs
df_vix = pd.read_csv('india_vix_history.csv', parse_dates=['Date'])
df_nifty = pd.read_csv('nifty_history.csv', parse_dates=['Date'])

# 2. Preprocess / merge
df = pd.merge(df_nifty[['Date','Close']], df_vix[['Date','Close']], on='Date', suffixes=('_nifty','_vix'))
df = df.sort_values('Date').dropna()

# 3. Compute monthly returns and isolate January
df['Nifty_Return'] = df['Close_nifty'].pct_change() * 100
df_month = df.resample('M', on='Date').last().reset_index()
df_month['Return_%'] = df_month['Close_nifty'].pct_change() * 100

jan_df = df_month[df_month['Date'].dt.month == 1].copy()
# Add VIX at start of month:
first_of_month = df.set_index('Date').resample('M').first().reset_index()
first_of_month = first_of_month[first_of_month['Date'].dt.month == 1][['Date','Close_vix']]
first_of_month = first_of_month.rename(columns={'Close_vix':'VIX_start'})
jan_df = jan_df.merge(first_of_month, on='Date')

# 4. Setup Dash
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Nifty vs India VIX — January Analysis"),
    dcc.Graph(id='time-series'),
    dcc.Graph(id='vix-vs-return'),
    dcc.Dropdown(
        id='year-range',
        options=[{'label': str(y), 'value': y} for y in jan_df['Date'].dt.year],
        multi=True,
        value=list(jan_df['Date'].dt.year)
    )
])

@app.callback(
    Output('vix-vs-return', 'figure'),
    Input('year-range', 'value')
)
def update_scatter(selected_years):
    dff = jan_df[jan_df['Date'].dt.year.isin(selected_years)]
    fig = px.scatter(dff,
                     x='VIX_start',
                     y='Return_%',
                     text=dff['Date'].dt.year,
                     title="Start-Jan VIX vs Jan Return of Nifty")
    fig.update_traces(textposition='top center')
    fig.update_layout(xaxis_title='VIX at start of January',
                      yaxis_title='Jan Return (%)')
    return fig

@app.callback(
    Output('time-series', 'figure'),
    Input('year-range', 'value')
)
def update_ts(selected_years):
    # For simplicity, show full overlay — can refine per selection
    fig = px.line(df, x='Date', y=['Close_nifty','Close_vix'],
                  title="Nifty and India VIX History",
                  labels={'value':'Index / VIX', 'variable':'Legend'})
    return fig

if __name__ == '__main__':
    app.run(debug=True)