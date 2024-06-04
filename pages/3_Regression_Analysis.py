import streamlit as st
import pickle

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 0.5rem;
                    padding-bottom: 0rem;
                    # padding-left: 2rem;
                    # padding-right:2rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)



def regression_analysis_ui():

    st.title("Regression Analysis")
    Guideline_text = "<p style='font-size: 20px;padding-bottom:1rem;'>Regression analysis is a statistical method used to determine the structure of a relationship between variables. It is a powerful tool for uncovering associations between variables observed in data. In the context of our AI application, regression analysis will help us to predict house prices based on various parameters such as location, size, and amenities. By analyzing the relationships between these parameters and the house price, we can identify the most significant factors that affect the price and make more accurate predictions. This will enable our application to provide users with more reliable and personalized house price estimates, ultimately enhancing the overall user experience</p>"
    st.markdown(Guideline_text, unsafe_allow_html=True)

    col1,col2 = st.columns(spec=(1,1), gap="small")
    with col1:
        st.image('Artifacts/Regresion_Analysis.png')
    with col2:
        st.markdown(
            "<p style='font-size: 19px; text-align: left;background-color:#fbe0b4;padding:1rem;'>We begin by assessing the overall significance of the model. This evaluation, using the F-test, determines whether the independent variables, collectively, have a statistically significant relationship with the dependent variable.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size: 19px; text-align: left;background-color:#fbe0b4;padding:1rem;'>Next, we will delve into the strength of the association between the independent and dependent variables. This analysis focuses on the goodness-of-fit of the model, which indicates how well the model captures the variation in the dependent variable explained by the independent variables. To assess this, we will employ two key metrics: R-squared and Adjusted R-squared.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size: 19px; text-align: left;background-color:#fbe0b4;padding:1rem;'>After evaluating the overall model fit, we analyze each independent variable to understand its influence on the dependent variable. We assess the relationship's magnitude using the coefficient and determine its statistical significance through a t-test and p-value. Additionally, we construct confidence intervals for each coefficient to estimate the true population value and its precision.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size: 19px; text-align: left;background-color:#fbe0b4;padding:1rem;'>Beyond evaluating the fit and individual coefficients, it's crucial to verify whether the data adheres to the underlying assumptions of linear regression. These assumptions ensure the validity of the model's inferences. Common tests employed for this purpose include the Omnibus and Jarque-Bera tests.</p>",
            unsafe_allow_html=True,
        )


    st.write("***")


regression_analysis_ui()