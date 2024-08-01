import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from scikit-learn import datasets
from scikit-learn.ensemble import RandomForestRegressor

# file=input("enter file name:-")
# fil=file+".xlsx"
# df = pd.read_excel(fil)
# def load_data(path: str):
#     data=pd.read_excel(path)
#     return data
# df=load_data("./data.xlsx")
# #print(df)
df=pd.read_excel("data.xlsx")
#st.dataframe(df)
st.write("""
# EXEED PROJECT
""")
st.header('ML in MINING')
st.write('---')
st.write('Inputs are PH,Mg,CaCo3,Depth and area')
st.write('Edit the inputs values to ger results on left slider')
st.header("Collection of data")
st.write("")
with st.expander("data obtained"):
    st.header('data frame obtained')
    st.dataframe(df)


st.write('---')
st.write('---')

import pandas as pd
import numpy as np

# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# target = raw_df.values[1::2, 2]


tra = pd.DataFrame(df, columns=['depth','PH','CaCO3','Mg','amount of mineral found'])
#print(tra)
X=df.iloc[:,1:6]
Y=df.iloc[:,6:7]

depth=df.iloc[:,1]
depth= [float(x) for x in depth]
PH=df.iloc[:,2]
PH= [float(x) for x in PH]
CaCO3=df.iloc[:,3]
CaCO3= [float(x) for x in CaCO3]
Mg=df.iloc[:,4]
Mg= [float(x) for x in Mg]
area=df.iloc[:,5]
area= [float(x) for x in area]

# st.dataframe(df)
# print(df)
st.header("camparing data attributes with amount of mineral found")
graph=st.line_chart(df,x="PH",y=Y)
graph=st.bar_chart(df,x="Mg",y=Y)
graph=st.bar_chart(df,x="depth",y=Y)
graph=st.bar_chart(df,x="CaCO3",y=Y)
st.write("we can conclude that we cannot determine amount of mineral obtained therefore we need to use ML model")
#graph=st.line_chart(df,x="PH",y="area")
# depth=df.iloc[:,1]
# PH=df.iloc[:,2]
# CaCO3=df.iloc[:,3]
# Mg=df.iloc[:,4]
# amount=df.iloc[:,5]
# print(X)
# print(Y)
def user_input_features():
    depth1 = st.sidebar.slider('depth',(np.min(depth)),(np.max(depth)), (np.mean(depth)))
    PH1 = st.sidebar.slider('PH',(np.min(PH)),(np.max(PH)),(np.mean(PH)))
    CaCO31 = st.sidebar.slider('CaCO3',(np.min(CaCO3)),(np.max(CaCO3)), (np.mean(CaCO3)))
    Mg1 = st.sidebar.slider('Mg',(np.min(Mg)),(np.max(Mg)), (np.mean(Mg)))
    area1 = st.sidebar.slider('area',(np.min(area)),(np.max(area)), (np.mean(area)))
    data = {'depth': depth1
            ,
            'PH': PH1,
            'CaCO3': CaCO31,
            'Mg': Mg1,
            'area':area1
           }
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of amount of mineral')

amo=st.write(prediction)
st.header("graph of mineral")
graph=st.bar_chart(df,x="PH",y=amo)
# labels = 'depth','PH','CaCO3','Mg'

# #explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

# fig1, ax1 = plt.subplots()
# ax1.pie(X, labels=labels)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# st.pyplot(fig1)
st.write('---')


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(bbox_inches='tight')
st.write('---')
st.set_option('deprecation.showPyplotGlobalUse', False)

plt.title('Feature importance based on SHAP values (Bar)')
st.set_option('deprecation.showPyplotGlobalUse', False)
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')

#d=X.depth.min()
print(graph)
