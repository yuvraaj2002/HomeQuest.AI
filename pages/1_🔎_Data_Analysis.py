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



def univariate_analysis(df):
    st.markdown(
        "<h2 style='text-align: left; font-size: 40px; '>Introductory Analysis</h1>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(spec=(2, 1), gap="small")
    with col1:
        st.dataframe(df.head(8))
    with col2:
        st.markdown(
            "<p style='font-size: 17px; text-align: left;background-color:#CEFCBA;padding:1rem;'>Welcome to the Univariate Analysis Module! When it comes to understanding data, focusing on one thing at a time is key. It's all about looking closely at one variable at a time, helping us uncover important patterns and insights. Think of it as the first step in exploring data, like peeling back layers to reveal what's underneath and discover the secrets hidden within your data!.</p>",
            unsafe_allow_html=True,
        )
        with st.expander(label = "What is the overall dimensionality of the dataset ?"):
            st.write(df.shape,"Which means there are around 3630 rows and 12 features")
        with st.expander(label = "What's the count of categorical/numerical features in our data ?"):
            st.write("Out of the 12 features, 10 are categorical—6 ordinal and 2 nominal—while 2 are continuous numerical.")


    # Plotting the bubble plot
    sector_counts = df['sector'].value_counts().reset_index()
    sector_counts.columns = ['sector', 'count']

    # Create bubble plot
    fig = px.scatter(sector_counts, x='sector', y='count', size='count',
                     labels={'sector': 'Sector', 'count': 'Count'},
                     title='Understanding the frequency of sectors',
                     size_max=50,color_discrete_sequence=['#44844c'])

    # Update layout
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)


    # Plotting the pie charts
    pie_col1, pie_col2, pie_col3,pie_col4 = st.columns(spec=(1, 1, 1,1), gap="large")
    with pie_col1:
        # Plot pie chart for employment_type
        fig1 = px.pie(df, names='luxury_category', title='Distribution of Luxury Category')
        st.plotly_chart(fig1, use_container_width=True)

    with pie_col2:
        # Plot pie chart for required_experience
        fig2 = px.pie(df, names='floor_category', title='Distribution of Floor Category')
        st.plotly_chart(fig2, use_container_width=True)

    with pie_col3:
        # Plot pie chart for required_education
        fig3 = px.pie(df, names='Property_Type', title='Distribution of Property type feature')
        st.plotly_chart(fig3, use_container_width=True)

    with pie_col4:
        # Plot pie chart for required_education
        fig3 = px.pie(df, names='agePossession', title='Distribution of agePossession feature')
        st.plotly_chart(fig3, use_container_width=True)



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
    univariate_analysis(df)

    # Calling function fo
    multivariate_analysis(df)




visualizations()