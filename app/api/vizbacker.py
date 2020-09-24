from fastapi import APIRouter, HTTPException
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

router = APIRouter()


@router.get('/vizbacker')
async def visual():
    # load in airbnb dataset
    DATA_PATH = 'https://raw.githubusercontent.com/Air-BnB-2-BW/data-science/master/data/%24_per_backer.csv'
    df = pd.read_csv(DATA_PATH, index_col=0)

    x = ['$0-25', '$25-50', '$50-75', '$75-100', '$100-125', '$125-150', '$150-175', '$175-200', '$200+']
    y = [93, 100, 78, 37, 30, 12, 5, 7, 33]

    fig = go.Figure(data=[go.Bar(x=x, y=y)])

    # Customize aspect
    fig.update_traces(marker_color='rgb(28,186,28)', marker_line_color='rgb(11,74,11)',
                      marker_line_width=4.5, opacity=0.6)

    fig.update_layout(title_text='Pledge Amount From Backers of Successful Kickstarter Projects')
    fig.show()
    return fig.to_json()
