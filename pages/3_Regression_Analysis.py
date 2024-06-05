import streamlit as st
import pickle
import plotly.graph_objs as go

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


def conduct_f_test():
    st.title("F test for Overall significance")
    st.markdown(
        "<p style='font-size: 19px; text-align: left;'>"
        "The F-test for overall significance is a statistical test used to determine whether a linear regression model is statistically significant, meaning it provides a better fit to the data than just using the mean of the dependent variable i.e. no independent variable at all. "
        "Now in order to conduct this test, first we need to set the <b>null hypothesis</b> and <b>alternative hypothesis</b>. "
        "<b>Null hypothesis (H0):</b> All regression coefficients (except the intercept) are equal to zero (β1 = β2 = ... = βk = 0), meaning that none of the independent variables contribute significantly to the explanation of the dependent variable's variation. "
        "<b>Alternative hypothesis (H1):</b> At least one regression coefficient is not equal to zero, indicating that at least one independent variable contributes significantly to the explanation of the dependent variable's variation."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='font-size: 19px; text-align: left;padding-bottom:1rem;'>"
        "After formulating the hypotheses, the F-statistic is calculated to assess overall model significance by comparing the explained variance (Mean Squared Regression, MSR) to the unexplained variance (Mean Squared Error, MSE). MSR is obtained by dividing the Explained Sum of Squares (ESS) by the degrees of freedom for the model (number of independent variables, k), while MSE is derived from the Residual Sum of Squares (RSS) divided by the degrees of freedom for residuals (n - k - 1, where n is the number of data points). This comparison determines if the model's explanatory power is statistically significant. The statistical significance of the F-statistic is then assessed by calculating the p-value using the F-distribution or statistical software. By comparing the p-value to a pre-defined significance level (α, typically 0.05), we determine if we can reject the null hypothesis. If the p-value is less than α, we reject the null hypothesis, indicating that at least one independent variable significantly predicts the dependent variable and the overall model is statistically significant. If the p-value is greater than or equal to α, we fail to reject the null hypothesis, suggesting that the independent variables do not significantly contribute to the prediction, and the overall model may not be statistically significant.</p>",
        unsafe_allow_html=True,
    )
    col1,col2 = st.columns(spec=(1,2), gap="large")
    with col1:
        st.image("Artifacts/F-distribution_pdf.svg.png")
    with col2:
        st.write("")
        st.markdown(
            "<p style='font-size: 19px; text-align: left;background-color:#CEFCBA;padding:1rem;'>Degree of Freedom of model 11, and degree of freedom of residual 2892.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size: 19px; text-align: left;background-color:#CEFCBA;padding:1rem;'>MSR value is 867.66 and MSE value is 1.0047</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size: 19px; text-align: left;background-color:#CEFCBA;padding:1rem;'>F Statsitic Value 863.3</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size: 19px; text-align: left;background-color:#CEFCBA;padding:1rem;'>F Statsitic Value 863.3</p>",
            unsafe_allow_html=True,
        )


    st.write("")


def goodness_of_fit():
    st.title("Evaluating the goodness of fit")
    st.markdown(
        "<p style='font-size: 19px; text-align: left;'>"
        "Goodness of fit is a statistical concept that describes how well a model fits a set of observations. In the context of regression analysis, it measures how well the estimated regression line or curve approximates the real data points. Assessing the goodness of fit helps determine the reliability of the predictions made by the model.To evaluate the goodness of fit in our regression analysis, we employ two distinct methodologies: statistical assessment and visualization techniques. In terms of statistical measures, we rely on the R² score and adjusted R² score to quantify the model's explanatory power and its adaptation to the number of predictors. Complementing these statistical metrics, our visual analysis involves the scrutiny of residual plots and Q-Q plots. These visualizations allow us to discern any patterns or deviations from expected behaviors in the residuals, providing valuable insights into the model's performance and its adherence to underlying assumptions. By integrating both statistical rigor and visual diagnostics, we ensure a comprehensive evaluation of the goodness of fit, facilitating informed decisions regarding the efficacy and reliability of our regression model.</p>",
        unsafe_allow_html=True,
    )

    # Create columns in Streamlit with a specified gap
    col1, col2 = st.columns(spec=(1, 1), gap="large")

    # Values and categories to be plotted
    r2_values = [0.854, 0.896]
    mae_values = [0.47,0.43]
    categories = ['Before Tuning', 'After Tuning']

    # Function to create a horizontal bar plot
    def create_bar_plot(values, categories, title):
        fig = go.Figure(go.Bar(
            x=values,
            y=categories,
            orientation='h',
            marker=dict(color=['#1f77b4', '#ff7f0e']),  # Colors for the bars
            width=0.6  # Adjust the bar width
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Values',
            yaxis_title='Categories',
            yaxis=dict(tickmode='linear'),
            height=400,
            width=500
        )
        return fig

    with col1:
        fig1 = create_bar_plot(r2_values, categories, 'R2 Score Comparison Before and After Hyper-Parameter Tuning')
        st.plotly_chart(fig1)

    with col2:
        fig2 = create_bar_plot(mae_values, categories, 'MAE value Comparison Before and After Hyper-Parameter Tuning')
        st.plotly_chart(fig2)







def independent_variables_analysis():
    st.title("Statistical significance of individual variables and confidence interval")
    st.markdown(
        "<p style='font-size: 19px; text-align: left;'>"
        "Statistical significance and confidence intervals are key concepts in regression analysis, offering insights into the reliability of estimated coefficients. By examining p-values, we determine if independent variables have a significant impact on the dependent variable. A low p-value (typically < 0.05) suggests significance, while a high one implies insignificance. Confidence intervals provide a range of likely values for population parameters, like regression coefficients. For instance, a 95% confidence interval indicates that we're 95% confident the true coefficient lies within it. Together, these tools illuminate variable relationships, aiding in informed decision-making regarding their importance and influence on the outcome..</p>",
        unsafe_allow_html=True,
    )

    # Data extracted from the regression summary
    variables = ['const', 'Builtup Area', 'Balcony', 'Age Possesion', 'Luxury Category', 'Floor Category','Property Type', 'Property Type', 'Sector', 'Bedroom', 'Servant Room', 'Furnishing Type']
    coefficients = [0.1585, 0.7818, 0.0804, -0.0629, 0.0011, 0.0181, 0.3633, 0.0402, 0.2036, 0.0587, -0.1049, 0.0273]
    ci_lower = [0.134, 0.740, 0.063, -0.081, -0.011, 0.006, 0.336, -0.008, 0.158, 0.044, -0.119, 0.017]
    ci_upper = [0.183, 0.823, 0.097, -0.045, 0.013, 0.031, 0.391, 0.089, 0.249, 0.074, -0.091, 0.038]

    # Create the plot
    fig = go.Figure()

    # Add coefficients as points
    fig.add_trace(go.Scatter(
        x=variables,
        y=coefficients,
        mode='markers',
        marker=dict(color='blue', size=10),
        name='Coefficients'
    ))

    # Add confidence intervals as error bars
    fig.add_trace(go.Scatter(
        x=variables,
        y=ci_lower,
        mode='lines',
        line=dict(color='black', width=1),
        name='Lower CI'
    ))
    fig.add_trace(go.Scatter(
        x=variables,
        y=ci_upper,
        mode='lines',
        line=dict(color='black', width=1),
        name='Upper CI'
    ))

    # Adding error bars
    fig.update_traces(error_y=dict(
        type='data',
        symmetric=False,
        array=[u - c for u, c in zip(ci_upper, coefficients)],
        arrayminus=[c - l for c, l in zip(coefficients, ci_lower)],
        thickness=1.5,
        width=5,
        color='black'
    ))

    # Update layout
    fig.update_layout(
        title='Coefficient Plot with 95% Confidence Intervals',
        xaxis_title='Variables',
        yaxis_title='Coefficient Value',
        xaxis=dict(tickmode='linear'),
        width=1500,
        height=700,
        showlegend=False
    )

    st.plotly_chart(fig)




def regression_analysis_ui():

    st.title("Regression Analysis")
    Guideline_text = "<p style='font-size: 20px;padding-bottom:1rem;'>Regression analysis is a statistical method used to determine the structure of a relationship between variables. It is a powerful tool for uncovering associations between variables observed in data. In the context of our AI application, regression analysis will help us to predict house prices based on various parameters such as location, size, and amenities. By analyzing the relationships between these parameters and the house price, we can identify the most significant factors that affect the price and make more accurate predictions. This will enable our application to provide users with more reliable and personalized house price estimates, ultimately enhancing the overall user experience</p>"
    st.markdown(Guideline_text, unsafe_allow_html=True)


    col1,col2 = st.columns(spec=(1,1), gap="small")
    with col1:
        st.image('Artifacts/Regresion_Analysis.png')
    with col2:
        st.markdown(
            "<p style='font-size: 19px; text-align: left;background-color:#CEFCBA;padding:1rem;'>We begin by assessing the overall significance of the model. This evaluation, using the F-test, determines whether the independent variables, collectively, have a statistically significant relationship with the dependent variable.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size: 19px; text-align: left;background-color:#CEFCBA;padding:1rem;'>Next, we will delve into the strength of the association between the independent and dependent variables. This analysis focuses on the goodness-of-fit of the model, which indicates how well the model captures the variation in the dependent variable explained by the independent variables. To assess this, we will employ two key metrics: R-squared and Adjusted R-squared.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size: 19px; text-align: left;background-color:#CEFCBA;padding:1rem;'>After evaluating the overall model fit, we analyze each independent variable's influence on the dependent variable. We use the coefficient to assess the relationship's magnitude and a t-test and p-value to determine its statistical significance. Additionally, we construct confidence intervals for each coefficient to estimate the true population value and its precision.</p>",
            unsafe_allow_html=True,
        )


    st.write("***")

    # Calling function for the F test for overall significance
    conduct_f_test()

    # Calling function for goodness of fit
    goodness_of_fit()

    # Calling function for independnet variable significance
    independent_variables_analysis()



regression_analysis_ui()