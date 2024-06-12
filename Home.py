import streamlit as st
# from pages.Price_Prediction import Price_Prediction_Page
# from pages.Recommendation_System import Recommendation_System_Page
# from pages.Loan_Eligibility import load_eligibility_Page

# Set page configuration
st.set_page_config(
    page_title="HomeQuest.AI",
    page_icon="🏠",
    layout="wide",
)

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 0rem;
                    # padding-left: 2rem;
                    # padding-right:2rem;
                }
                .top-margin{
                    margin-top: 4rem;
                    margin-bottom:2rem;
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


# Main page function
def main_page():
    Overview_col, Img_col = st.columns(spec=(1.4, 1), gap="large")

    with Overview_col:

        # Content for main page
        st.markdown(
            "<h1 style='text-align: left; font-size: 70px; '>HomeQuest.AI🏡</h1>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown("""
            <p style='font-size: 23px; text-align: left;'>
                The AI-Powered Home Finder revolutionizes Gurgaon's house-hunting with sophisticated AI algorithms. Seamlessly streamlining the process, it offers tailored recommendations based on individual preferences and needs. Three integrated modules enhance the user experience.
            </p>
        """, unsafe_allow_html=True)

        st.write("")
        st.markdown("""
            <div>
                <ul>
                    <li><p style='font-size: 22px; text-align: left;'><strong><em>The Data Analysis Module:</em></strong> Harnessing comprehensive data analytics, this module uncovers valuable insights from real estate datasets, enabling informed decision-making with actionable intelligence.</p></li>
                    <li><p style='font-size: 22px; text-align: left;'><strong><em>The Price Prediction Module:</em></strong> Employs machine learning to deliver precise property value estimations based on user preferences, facilitating confident real estate decisions.</p></li>
                    <li><p style='font-size: 22px; text-align: left;'><strong><em>The Regression Analysis Module:</em></strong> Utilizes advanced regression models to identify key factors influencing property prices, enabling accurate predictions accounting for complex data relationships.</p></li>
                    <li><p style='font-size: 22px; text-align: left;'><strong><em>The Loan Eligibility Module:</em></strong> Seamlessly integrated, assesses user eligibility for loans by analyzing financial profiles, aiding in effective financial planning for real estate transactions.</p></li>
                    <li><p style='font-size: 22px; text-align: left;'><strong><em>The Recommendation System Module:</em></strong> Utilizes advanced recommendation algorithms to match user preferences with tailored property suggestions, streamlining the property search process with personalized recommendations.</p></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with Img_col:
        st.write("")
        st.markdown("<div class='top-margin'> </div>", unsafe_allow_html=True)
        st.image("Artifacts/Banner.jpg")
        st.write("")

        social_col1, social_col2, social_col3,social_col4 = st.columns(spec=(1, 1, 1,1), gap="large")
        with social_col1:
            st.link_button("Github👨‍💻", use_container_width=True, url="https://github.com/yuvraaj2002")

        with social_col2:
            st.link_button("Linkedin🧑‍💼",use_container_width=True,url = "https://www.linkedin.com/in/yuvraj-singh-a4430a215/")

        with social_col3:
            st.link_button("Twitter🧠",use_container_width=True,url = "https://twitter.com/Singh_yuvraaj1")

        with social_col4:
            st.link_button("Blogs✒️", use_container_width=True, url="https://yuvraj01.hashnode.dev/")


# page_names_to_funcs = {
#     "Project Overview": main_page,
#     "Property Price Prediction🗃️": Price_Prediction_Page,
#     "Recommendation Engine": Recommendation_System_Page,
#     "Loan Eligibility Module": load_eligibility_Page,
# }
# selected_page = st.sidebar.selectbox("Select Module", list(page_names_to_funcs.keys()))
# page_names_to_funcs[selected_page]()
main_page()