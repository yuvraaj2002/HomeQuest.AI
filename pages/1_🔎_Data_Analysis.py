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
            "<p style='font-size: 17px; text-align: left;background-color:#abdbfc;padding:1rem;'>Welcome to the Univariate Analysis Module! When it comes to understanding data, focusing on one thing at a time is key. It's all about looking closely at one variable at a time, helping us uncover important patterns and insights. Think of it as the first step in exploring data, like peeling back layers to reveal what's underneath and discover the secrets hidden within your data!.</p>",
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
                     size_max=50, color_discrete_sequence=['#086ccc'])

    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    # Plotting the pie charts
    pie_col1, pie_col2, pie_col3, pie_col4 = st.columns(spec=(1, 1, 1, 1), gap="large")

    with pie_col1:
        # Plot pie chart for employment_type
        fig1 = px.pie(df, names='luxury_category', title='Distribution of Luxury Category')
        fig1.update_layout(height=380, width=380)  # Adjust size here
        st.plotly_chart(fig1, use_container_width=True)

    with pie_col2:
        # Plot pie chart for required_experience
        fig2 = px.pie(df, names='floor_category', title='Distribution of Floor Category')
        fig2.update_layout(height=380, width=380)  # Adjust size here
        st.plotly_chart(fig2, use_container_width=True)

    with pie_col3:
        # Plot pie chart for required_education
        fig3 = px.pie(df, names='Property_Type', title='Distribution of Property type feature')
        fig3.update_layout(height=380, width=380)  # Adjust size here
        st.plotly_chart(fig3, use_container_width=True)

    with pie_col4:
        # Plot pie chart for required_education
        fig4 = px.pie(df, names='agePossession', title='Distribution of agePossession feature')
        fig4.update_layout(height=380, width=380)  # Adjust size here
        st.plotly_chart(fig4, use_container_width=True)


    # Plotting the bar plots
    bar_col1, bar_col2, bar_col3, bar_col4 = st.columns(spec=(1, 1, 1, 1), gap="large")
    with bar_col1:
        # Plot bar plot for bedRoom
        fig1 = px.bar(df, x=df['bedRoom'].value_counts().index, y=df['bedRoom'].value_counts().values,
                      title='Distribution of Bedrooms')
        fig1.update_layout(height=380, width=380)  # Adjust size here
        st.plotly_chart(fig1, use_container_width=True)

    with bar_col2:
        # Plot bar plot for bathroom
        fig2 = px.bar(df, x=df['bathroom'].value_counts().index, y=df['bathroom'].value_counts().values,
                      title='Distribution of Bathrooms')
        fig2.update_layout(height=380, width=380)  # Adjust size here
        st.plotly_chart(fig2, use_container_width=True)

    with bar_col3:
        # Plot bar plot for balcony
        fig3 = px.bar(df, x=df['balcony'].value_counts().index, y=df['balcony'].value_counts().values,
                      title='Distribution of Balconies')
        fig3.update_layout(height=380, width=380)  # Adjust size here
        st.plotly_chart(fig3, use_container_width=True)

    with bar_col4:
        # Plot bar plot for furnishing_type
        fig4 = px.bar(df, x=df['furnishing_type'].value_counts().index, y=df['furnishing_type'].value_counts().values,
                      title='Distribution of Furnishing Type')
        fig4.update_layout(height=380, width=380)  # Adjust size here
        st.plotly_chart(fig4, use_container_width=True)



def multivariate_analysis(data):
    st.title("Multivariate Analysis")
    st.markdown(
        "<p style='font-size: 20px; text-align: left;'>Multivariate analysis is a statistical approach that examines data sets with multiple variables simultaneously, contrasting with univariate analysis, which focuses on single variables. By considering the relationships between two or more variables, it unveils complex data structures and patterns. In predicting house prices, multivariate analysis proves invaluable, as it delves into the interdependencies among factors like location, size, amenities, and market trends. Rather than scrutinizing variables in isolation, this method enables a comprehensive understanding of how various factors collectively influence house prices, enhancing predictive accuracy and guiding informed decision-making in real estate transactions.</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(spec=(1, 1), gap="small")
    with col1:
        st.markdown(
            "<p style='font-size:17px; background-color: #abdbfc; padding: 0.5rem'><strong>Question 1:</strong> What is the relationship between property type and price?</p>",
            unsafe_allow_html=True)
        fig1 = px.box(data, x='Property_Type', y='price', title='Property Type vs Price')
        st.plotly_chart(fig1)

    with col1:
        st.markdown(
            "<p style='font-size:17px; background-color: #abdbfc; padding: 0.5rem'><strong>Question 2:</strong> How does the number of bedrooms, bathrooms, and balconies collectively affect the price?</p>",
            unsafe_allow_html=True)
        fig2 = px.scatter_3d(data, x='bedRoom', y='bathroom', z='balcony', color='price',
                             title='Bedrooms, Bathrooms, Balconies vs Price', color_continuous_scale='viridis')
        st.plotly_chart(fig2)


    with col1:
        st.markdown(
            "<p style='font-size:17px; background-color: #abdbfc; padding: 0.5rem'><strong>Question 3:</strong> What is the impact of the age/possession status of the property on its price when controlling for other variables?</p>",
            unsafe_allow_html=True)
        fig3 = px.violin(data, x='agePossession', y='price', box=True, points='all',
                         title='Age/Possession Status vs Price')
        st.plotly_chart(fig3)


    with col1:
        st.markdown(
            "<p style='font-size:17px; background-color: #abdbfc; padding: 0.5rem'><strong>Question 4:</strong> Does built-up area influence the price differently across different property types?</p>",
            unsafe_allow_html=True)
        fig4 = px.scatter(data, x='built_up_area', y='price', color='Property_Type', facet_col='Property_Type',
                          title='Built-up Area vs Price by Property Type')
        st.plotly_chart(fig4)



    with col2:
        st.markdown(
            "<p style='font-size:17px; background-color: #abdbfc; padding: 0.5rem'><strong>Question 5:</strong> What is the combined effect of luxury category and furnishing type on the price?</p>",
            unsafe_allow_html=True)
        fig5 = px.line(data, x='luxury_category', y='price', color='furnishing_type', markers=True,
                       title='Luxury Category and Furnishing Type vs Price')
        st.plotly_chart(fig5)


    with col2:
        st.markdown(
            "<p style='font-size:17px; background-color: #abdbfc; padding: 0.5rem'><strong>Question 6:</strong> How do floor categories impact house prices for different property types and built-up areas?</p>",
            unsafe_allow_html=True)
        fig6 = px.box(data, x='floor_category', y='price', color='Property_Type',
                      title='Floor Categories vs Price by Property Type', facet_col='Property_Type')
        st.plotly_chart(fig6)


    with col2:
        st.markdown(
            "<p style='font-size:17px; background-color: #abdbfc; padding: 0.5rem'><strong>Question 7:</strong>Is there a significant interaction between the number of balconies and the age of the property on the price?</p>",
            unsafe_allow_html=True)
        fig7 = px.line(data, x='balcony', y='price', color='agePossession', markers=True,
                       title='Number of Balconies and Age of Property vs Price')
        st.plotly_chart(fig7)


    with col2:
        st.markdown(
            "<p style='font-size:17px; background-color: #abdbfc; padding: 0.5rem'><strong>Question 8:</strong> How does the presence of a servant room influence the price across different property types and sectors?</p>",
            unsafe_allow_html=True)
        fig8 = px.box(data, x='servant room', y='price', color='Property_Type',
                      title='Servant Room Presence vs Price by Property Type')
        st.plotly_chart(fig8)


    # Get the counts of each sector
    sector_counts = data['sector'].value_counts()
    sectors_to_keep = sector_counts[sector_counts > 40].index
    filtered_data = data[data['sector'].isin(sectors_to_keep)]
    pivot_table = filtered_data.pivot_table(values='price', index='sector', columns='Property_Type',
                                            aggfunc='mean').reset_index()
    pivot_table = pivot_table.set_index('sector').T
    fig = px.imshow(pivot_table, title='Sector and Property Type vs Price', aspect='auto')
    fig.update_layout(width=1500)
    st.plotly_chart(fig)



def visualizations():

    # Calling the functions for loading the dataset and also the basic details
    df = load_dataframe()

    # Calling the function for adding univariate analysis
    univariate_analysis(df)

    # Calling function fo
    multivariate_analysis(df)




visualizations()