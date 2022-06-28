import streamlit as st
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns



#######################Dashboard######################

st.title('Novibet customer segmentation app')
st.text('\n\n\n')



uploaded_file = st.file_uploader('Please upload an excel file')

if uploaded_file is not None:
    clustering_r = pd.read_excel(uploaded_file,sheet_name='clustering_r')
    c_nor_melt= pd.read_excel(uploaded_file,sheet_name='c_nor_melt')
    df_nor_melt = pd.read_excel(uploaded_file,sheet_name='df_nor_melt')
    clusters = pd.read_excel(uploaded_file,sheet_name='clusters')

    st.text('Summary of clusters')
    st.write(clustering_r)
    
    st.text('Bellow you can find a graphical analysis of the clusters')
    fig = plt.figure(figsize=(10, 4))
    sns.lineplot('Attribute', 'Value', hue='Cluster', data=df_nor_melt,palette="pastel")
    st.pyplot(fig)
    st.text('Bellow you can find a graphical radial representation of the variables for each cluster')
    fig2 = px.line_polar(c_nor_melt, r='Value', theta='Attribute',color='Cluster' ,line_close=True)
    fig2.update_traces(fill='toself')
    st.plotly_chart(fig2)
    
    st.text('A more detailed analysis that visualizes the range of the variables for each cluster')
    figParCo= go.Figure(data=
                        go.Parcoords(
                            line = dict(color = clusters['Cluster'],
                                       colorscale = 'HSV'),
                            dimensions = list([
                                dict(
                                    label = 'Cluster', values = clusters['Cluster']),
                                dict(
                                    label = 'loyalty', values = clusters['loyalty']),
                                dict(
                                    label = 'age', values = clusters['age']),
                                dict(
                                    label = 'platforms', values = clusters['platforms']),
                                dict(
                                    label = 'F', values = clusters['F']),
                                dict(
                                    label = 'diff', values = clusters['diff']),
                                dict(
                                    label = 'IsFreeSpinID', values = clusters['IsFreeSpinID']),
                                dict(
                                    label = 'IsLiveID', values = clusters['IsLiveID']),
                                dict(
                                    label = 'valpergame', values = clusters['valpergame']),
                                dict(range = [0,-4000],
                                    label = 'avgdraw', values = clusters['avgdraw']),
                                dict(
                                    label = 'totaltrans', values = clusters['totaltrans']),
                                dict(
                                    label = 'dd_ratio', values = clusters['dd_ratio']),
                                dict(
                                    label = 'Pt%', values = clusters['Pt%']),
                                dict(
                                    label = 'CountryName_Greece', values = clusters['CountryName_Greece']),
                            ])
                        )
                    )
    
    
    st.plotly_chart(figParCo)






        
        






