import streamlit as st
import mlflow
import pickle
import pandas as pd

def main():
    html_temp = """
    <style>
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    tbody th {display:none}
    .blank {display:none}
    h1 {
    text-align: center;
    }
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 450px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 450px;
    }
    div.block-container{top:-90px;}
    </style>
    """

    st.markdown(html_temp,unsafe_allow_html=True)
    # loaded_model = pickle.load(open('xgboost_98.pkl', 'rb'))

    df = pd.read_csv('Ahmedabad_rent.csv')

    st.sidebar.title('Ahmedabad Rent Prediction')
    seller_type = st.sidebar.selectbox('Seller Type',options=['Owner','Agent','Builder'])
    bedroom = st.sidebar.selectbox('No. of bedroom',options=['1','2','3','4','5'])
    layout_type = st.sidebar.selectbox('Layout type',options=['BHK','RK'])
    property_type= st.sidebar.selectbox('Property type',options=['Apartment','Studio Apartment','Independent House','Independent Floor','Villa','Penthouse'])
    area = st.sidebar.text_input('Area')
    furnish_type = st.sidebar.selectbox('Furnish type',options=['Furnished','Semi-Furnished','Unfurnished'])
    locality = st.sidebar.selectbox('Locality',options=df['locality'])
    bathroom = st.sidebar.selectbox('No. of Bathrooms',options=['1','2','3','4','5'])
main()