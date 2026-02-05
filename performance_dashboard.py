# dashboard/performance_dashboard.py

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import plotly.graph_objs as go
import plotly.io as pio
import uvicorn
import pandas as pd
from typing import Dict
import threading

# Shared modules (assumed to exist)
from shared_state import shared_state
from performance_monitor import performance_monitor

app = FastAPI(title="Octivault Trader - Performance Dashboard")

class KPIData(BaseModel):
    pnl: Dict[str, float]
    win_rate: Dict[str, float]
    roi: Dict[str, float]
    sharpe: Dict[str, float]

def generate_bar_chart(data_dict, title, yaxis_title):
    df = pd.DataFrame(list(data_dict.items()), columns=["Label", "Value"])
    fig = go.Figure([go.Bar(x=df["Label"], y=df["Value"])])
    fig.update_layout(title=title, yaxis_title=yaxis_title)
    return pio.to_html(fig, full_html=False)

@app.get("/", response_class=HTMLResponse)
async def index():
    pnl = performance_monitor.get_pnl_by_symbol()
    win_rate = performance_monitor.get_win_rate_by_agent()
    roi = performance_monitor.get_roi_by_agent()
    sharpe = performance_monitor.get_sharpe_by_agent()

    pnl_chart = generate_bar_chart(pnl, "PnL per Symbol", "PnL (USDT)")
    win_chart = generate_bar_chart(win_rate, "Win Rate by Agent", "Win %")
    roi_chart = generate_bar_chart(roi, "ROI by Agent", "% ROI")
    sharpe_chart = generate_bar_chart(sharpe, "Sharpe Ratio by Agent", "Sharpe Ratio")

    html_content = f"""
    <html>
        <head>
            <title>Octivault Performance Dashboard</title>
        </head>
        <body>
            <h1>📈 Octivault Trader - Performance Dashboard</h1>
            <div>{pnl_chart}</div>
            <div>{win_chart}</div>
            <div>{roi_chart}</div>
            <div>{sharpe_chart}</div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Optional: Run in thread when imported elsewhere
def run_dashboard():
    uvicorn.run("dashboard.performance_dashboard:app", host="0.0.0.0", port=8050, reload=False)

if __name__ == "__main__":
    run_dashboard()
