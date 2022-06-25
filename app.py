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

    if st.sidebar.button('Submit'):
        col = ['seller_type','bedroom','layout_type','property_type','area','furnish_type','bathroom']
        res_df = pd.DataFrame(columns=col)  
        res_df = res_df.append({
            'seller_type':seller_type,
            'bedroom':bedroom,
            'layout_type':layout_type,
            'property_type':property_type,
            'area':area,
            'furnish_type':furnish_type,
            'bathroom':bathroom
        },ignore_index=True)

        df2=pd.get_dummies(df["locality"])
        col_df = pd.DataFrame(columns=df2.columns)
        if locality:
            col_df = col_df.append({f'{locality}':1},ignore_index=True)
            col_df = col_df.fillna(0)

        res_df = pd.concat([res_df,col_df],axis=1)
        
        final_model = pickle.load(open('kmeans.pkl', 'rb'))

        prediction=final_model.predict(res_df)

        clus_df = pd.DataFrame(columns=['0','1','2','3'])
        clus_df = clus_df.append({f'{prediction}':1},ignore_index=True)
        clus_df.fillna(0)
        
        res_df = pd.concat([res_df,clus_df],axis=1)
        
        loaded_model = pickle.load(open('xgboost.pkl', 'rb'))

        res = loaded_model.predict(res)

        st.success(f'The expected price is between {res-2000} - {res+2000}')
        with mlflow.start_run() as run:
            params = {
                'seller_type':seller_type,
                'bedroom':bedroom,
                'layout_type':layout_type,
                'property_type':property_type,
                'area':area,
                'furnish_type':furnish_type,
                'bathroom':bathroom,
                'locality':locality,
                'predicted price':res
                }
            mlflow.log_params(params)

main()