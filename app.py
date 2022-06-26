import streamlit as st
import mlflow
import pickle
import pandas as pd
from PIL import Image
image = Image.open('Capture.PNG')

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
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
    }
    div.block-container{top:-50px;}
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
    locality = st.sidebar.selectbox('Locality',options=df['locality'].unique())
    bathroom = st.sidebar.selectbox('No. of Bathrooms',options=['1','2','3','4','5'])

    d1 = {'Owner':0,'Agent':1,'Builder':2}
    d2 = {'BHK':0,'RK':1}
    d3 = {'Apartment':0,'Studio Apartment':1,'Independent House':2,'Independent Floor':3,'Villa':4,'Penthouse':5}
    d4 = {'Furnished':0,'Semi-Furnished':1,'Unfurnished':2}

    if st.sidebar.button('Submit'):
        col = ['seller_type','bedroom','layout_type','property_type','area','furnish_type','bathroom']
        res_df = pd.DataFrame(columns=col)  
        res_df = res_df.append({
            'seller_type':d1[seller_type],
            'bedroom':int(bedroom),
            'layout_type':d2[layout_type],
            'property_type':d3[property_type],
            'area':int(area),
            'furnish_type':d4[furnish_type],
            'bathroom':int(bathroom)
        },ignore_index=True)

        print(type(d1[seller_type]))

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
        res_df = res_df.drop(columns=['[0]'])
        
        loaded_model = pickle.load(open('xgboost.pkl', 'rb'))

        res_df['seller_type'] = res_df['seller_type'].astype(int)
        res_df['bedroom'] = res_df['seller_type'].astype(int)
        res_df['layout_type'] = res_df['seller_type'].astype(int)
        res_df['property_type'] = res_df['seller_type'].astype(int)
        res_df['area'] = res_df['seller_type'].astype(int)
        res_df['furnish_type'] = res_df['seller_type'].astype(int)
        res_df['bathroom'] = res_df['seller_type'].astype(int)
        res_df['0'] = res_df['0'].astype(float)
        res_df['1'] = res_df['1'].astype(float)
        res_df['2'] = res_df['2'].astype(float)
        res_df['3'] = res_df['3'].astype(float)

        res = loaded_model.predict(res_df)

        st.success(f'The expected price is between {res[0]-1000} - {res[0]+1000}')

        st.image(image, caption='Ahmedabad Rent Price Analysis by CHIRAG CHAUHAN')

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