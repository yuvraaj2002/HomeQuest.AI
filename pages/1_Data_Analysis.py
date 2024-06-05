import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as ps
import plotly.express as px
from plotly.subplots import make_subplots
import statsmodels.api as sm
import plotly.figure_factory as ff
import warnings


st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    # padding-left: 2rem;
                    # padding-right:2rem;
                }
                .block-button{
                    padding: 10px; 
                    width: 100%;
                    background-color: #c4fcce;
                }
        </style>
        """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_dataframe():
    df = pd.read_csv('Artifacts/Data.csv')
    df.drop(['store room'], axis=1, inplace=True)
    df['sector'] = df['sector'].astype(object)
    return df


def univariate_analysis():
    st.title("Introductory Univariate Analysis")


def multivariate_analysis(data):
    st.title("Multivariate Analysis")

    col1, col2 = st.columns(spec=(1, 1), gap="small")
    with col1:
        # Container
        container = st.container()

        # Correlation between property price and built-up area
        with container:
            fig1 = px.scatter(data, x='built_up_area', y='price',
                              title='Correlation between Property Price and Built-up Area')
            st.plotly_chart(fig1)

        # Effect of the number of bedrooms on property price
        with container:
            fig2 = px.violin(data, x='bedRoom', y='price', box=True, points="all",
                             title='Effect of Number of Bedrooms on Property Price')
            st.plotly_chart(fig2)

        # Relationship between the number of bathrooms and property price
        with container:
            fig3 = px.box(data, x='bathroom', y='price', points="all",
                          title='Relationship between Number of Bathrooms and Property Price')
            st.plotly_chart(fig3)

        # Impact of having a balcony on property price
        with container:
            fig4 = px.violin(data, x='balcony', y='price', box=True, points="all",
                             title='Impact of Having a Balcony on Property Price')
            st.plotly_chart(fig4)

        # Effect of the age of possession on property price
        with container:
            fig5 = px.scatter(data, x='agePossession', y='price', title='Effect of Age of Possession on Property Price')
            st.plotly_chart(fig5)

    with col2:

        # Container
        container = st.container()

        # Correlation between the presence of a servant room and property price
        with container:
            fig6 = px.bar(data, x='servant room', y='price', barmode='group',
                          title='Correlation between Servant Room and Property Price')
            st.plotly_chart(fig6)

        # Influence of furnishing type on property price
        with container:
            fig7 = px.violin(data, x='furnishing_type', y='price',
                             title='Influence of Furnishing Type on Property Price')
            st.plotly_chart(fig7)

        # Impact of luxury category on property price
        with container:
            fig8 = px.box(data, x='luxury_category', y='price',
                          title='Impact of Luxury Category on Property Price')
            st.plotly_chart(fig8)

        # Effect of floor category on property price
        with container:
            fig9 = px.box(data, x='floor_category', y='price',
                          title='Effect of Floor Category on Property Price')
            st.plotly_chart(fig9)

        # Difference in property price between flat and independent house types
        with container:
            fig10 = px.box(data, x='Property_Type', y='price',
                           title='Difference in Property Price between Flat and Independent House Types')
            st.plotly_chart(fig10)


def visualizations():

    # Calling the functions for loading the dataset and also the basic details
    df = load_dataframe()

    # Calling the function for adding univariate analysis
    univariate_analysis()


    multivariate_analysis(df)




visualizations()