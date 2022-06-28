#####################libraries######################
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
import matplotlib.pyplot as plt
from functools import reduce
from art import text2art
from datetime import datetime
from datetime import date
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer

'''Set plotting parameters'''
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

#####################functions#######################
def import_csv(path1,path2,path3):
    '''Takes as input filepath and the creates dataframe for games,users,wallet'''
    games = pd.read_csv(path1)
    users = pd.read_csv(path2)
    wallet=pd.read_csv(path3)
    return games,users,wallet

def data_preprocess(games,users,wallet):
    '''Input original novibet dataframes for games(per user/time), bank transactions, user information.
    Variables created:
        'UserProfileId':unique id per customer,
        'firttr':first bank transaction recorded,
        'lasttr':last bank transaction recorded,
        'drawsum': total withdraw money sum,
        'drawcount': number of withdraws, 
        'depsum': total deposit money sum, 
        'depcount': number of deposits,  
        'totaltrans':total number of bank transactions,
        'balance':current balance,
        'avgdraw':the average withdrawal amount,
        'avgdep':the average deposit amount,
        'dd_ratio':the ratio of withdraw/deposition instances
        'loyalty':days since subscription,
        "age":age of player,
        'Sex':Male(1)/Female(0),
        'StatusSysname':Account status- Active(1)/Inactive(0)/Test(0)
        ''platforms':number of games/providiers per player, 
        'F':total number of game records per player, 
        'diff':mean of time difference (in days) between games per player, 
        'Pt%':%of value of profits(novibet) over total sum of monetary ingame transactions(in abs terms), 
        'valpergame':mean monetary value per game
        
        
        '''
    #'''  wallet ''' 
    #''' wallet= transactions 9/21->10/21''' 
    #''' 'created' to datetime''' 
    wallet['Created']= pd.to_datetime(wallet['Created'])
    #''' sort from first to last date''' 
    wallet.sort_values(by=['Created'],inplace=True)
    #''' df for first and last occurence of id->first transaction& last''' 
    start = wallet.groupby(["UserProfileId"], as_index=False)["Created"].first()
    start.rename(columns = {"Created":'start'}, inplace = True)
    end = wallet.groupby(["UserProfileId"], as_index=False)["Created"].last()
    end.rename(columns = {"Created":'end'}, inplace = True)
    #''' amounts and counts per transaction''' 
    drawsum=wallet[(wallet['TypeSysname']=='WITHDRAW')].groupby(["UserProfileId"], as_index=False)["Amount"].sum()
    drawsum.rename(columns = {"Amount":'drawsum'}, inplace = True)
    drawcount=wallet[(wallet['TypeSysname']=='WITHDRAW')].groupby(["UserProfileId"], as_index=False)["Amount"].count()
    drawcount.rename(columns = {"Amount":'drawcount'}, inplace = True)
    depsum=wallet[(wallet['TypeSysname']=='DEPOSIT')].groupby(["UserProfileId"], as_index=False)["Amount"].sum()                 
    depsum.rename(columns = {"Amount":'depsum'}, inplace = True)
    depcount=wallet[(wallet['TypeSysname']=='DEPOSIT')].groupby(["UserProfileId"], as_index=False)["Amount"].count()
    depcount.rename(columns = {"Amount":'depcount'}, inplace = True)    
    #''' define list of DataFrames''' 
    dfs = [start, end, drawsum,drawcount,depsum,depcount]
    #''' merge all DataFrames into one''' 
    transactions = reduce(lambda  left,right: pd.merge(left,right,on=["UserProfileId"],how='left'), dfs)
    #''' rename columns''' 
    transactions.columns = ['UserProfileId', 'firttr', 'lasttr', 'drawsum', 'drawcount', 'depsum', 'depcount']
    #''' feel na in draw and deposits''' 
    transactions.fillna(0,inplace=True)
    #''' total''' 
    transactions['totaltrans']=transactions['drawcount']+transactions['depcount']
    transactions['balance']=transactions['drawsum']+transactions['depsum']
    transactions['avgdraw']=transactions['drawsum']/transactions['drawcount']
    transactions['avgdep']=transactions['depsum']/transactions['depcount']
    transactions['dd_ratio']=transactions['drawcount']/transactions['totaltrans']
    
    #'''  users ''' 
    #''' users= users subscribed up to 2022'''
    #'''check if duplicated and remove them'''
    #pd.concat(g for _, g in users.groupby('UserProfileId') if len(g) > 1)
    users.drop_duplicates(subset=['UserProfileId'], keep='first',inplace=True)
    #''' date to datetime''' 
    users['Registration Date']= pd.to_datetime(users['Registration Date'])
    users['BirthDate']= pd.to_datetime(users['BirthDate'])
    users["age"] = (pd.to_datetime('now') - users['BirthDate']).astype('<m8[Y]')
    d1 = "2021-11-02"
    users['loyalty']=(pd.to_datetime(d1) - users['Registration Date']).dt.days
    #''' replace male female''' 
    users['Sex']=users['Sex'].replace(["M","F"],[1,0],inplace=True)
    #''' replace status''' 
    users['StatusSysname'].replace(['TEST', 'ACTIVE', 'INACTIVE'],[0,1,0],inplace=True)
    #''' replace country''' 
    #''' users.groupby(['CountryName'])['CountryName'].count()''' 
    users['CountryName'].replace(['Finland', 'Hungary','New Zealand', 'India'],'Other',inplace=True)
    #''' keep relevant''' 
    users=users[['UserProfileId','loyalty',"age",'Sex','StatusSysname','CountryName']]
    #''' make country into dummy'''
    user=pd.get_dummies(users, columns=['CountryName'])
    #'''  keep   players who have subscribed within period of other data (sept-nov)'''
    user=user[user['loyalty']>0]
    user.reset_index(drop=True,inplace=True)
    user.drop(columns=['CountryName_Ireland', 'CountryName_Other', 'CountryName_United Kingdom'],inplace=True)
    
    #''' games''' 
    #'''games record 09-10/21
    #''' date to daterime''' 
    games['Date']= pd.to_datetime(games['Date'])
    #''' rename id for all df to have same key''' 
    games.rename(columns = {'UserID':'UserProfileId'}, inplace = True)
    #''' providers per player''' 
    pcur=games.groupby(['UserProfileId'], as_index=False)['Casino_Provider'].nunique()
    pcur.rename(columns = {'Casino_Provider':'platforms'}, inplace = True)
    #''' frequency per id''' 
    freq=games.groupby(['UserProfileId'], as_index=False)['Casino_Provider'].count()
    freq.rename(columns = {'Casino_Provider':'F'}, inplace = True)
    #''' if we wanted to see what each player prefers ->examine popular products etc''' 
    #games.groupby(['UserID','Casino_Provider'])['Casino_Provider'].count()
    income=games[(games['Hold']>0)].groupby(['UserProfileId'], as_index=False)['Hold'].sum()
    income.rename(columns = {'Hold':'I'}, inplace = True)
    #''' profit per player (positive=profit, negative= loss, ''' 
    profit=games.groupby(['UserProfileId'], as_index=False)['Hold'].sum()
    profit.rename(columns = {'Hold':'P'}, inplace = True)
    #''' sum of ingame transactions per player (total monetary value)''' 
    games['absvalue']=games['Hold'].abs()
    absvalue=games.groupby(['UserProfileId'], as_index=False)['absvalue'].sum()
    #'''difference between games measured in days
    games.sort_values(['UserProfileId','Date'],inplace=True)
    games['diff'] = games.groupby('UserProfileId')['Date'].diff()
    games['diff'] = pd.to_numeric(games['diff'].dt.days, downcast='integer')
    #''' fill na difference for single game players
    games['diff'].fillna(0,inplace=True)
    games_occur=games.groupby(['UserProfileId'])['diff'].mean().to_frame()
    #''' average use of free bonus per player'''
    bonus=games.groupby(['UserProfileId'], as_index=False)['IsFreeSpinID'].mean()
    #''' average use of live games per player'''
    live=games.groupby(['UserProfileId'], as_index=False)['IsLiveID'].mean()    
    #''' merge all DataFrames into one'''
    dfss = [pcur, freq,absvalue,profit,games_occur,bonus,live]
    gameid = reduce(lambda  left,right: pd.merge(left,right,on=["UserProfileId"],how='outer'), dfss)
    #'''novibet profit over all ingame transactions per player
    gameid['Pt%']=gameid['P']/gameid['absvalue']
    gameid.drop(axis=1,columns='P',inplace=True)
    #''' fill na Pt where player with single instance and 0 monetary outcome of game
    gameid['Pt%'].fillna(0,inplace=True)
    gameid['valpergame']=gameid['absvalue']/gameid['F']
    gameid.drop(axis=1,columns='absvalue',inplace=True)
 
    return transactions,user,gameid

def clustering(dataset,n):
    model_kmeans = KMeans(n_clusters=n, random_state=0)
    model_kmeans.fit(dataset)
    return model_kmeans,model_kmeans.labels_


def get_app_int(prompt):
    while True:
        try:
            value = int(input(prompt))
        except ValueError:
            print("Sorry, I didn't understand that.")
            continue

        if value not in [1,2]:
            print("Sorry, your response must be 1 or 2.")
            continue
        else:
            break
    return value
def give_input():
    ''''Input directory for each file used'''
    software=int(input("Press 1 for Mac , 2 for windows        "))

    if software is 1:
        pathgames=str(input("Enter path of games csv file      "))
        pathwallet=str(input("Enter path of wallet csv file     "))
        pathusers=str(input("Enter path of users csv file    "))
    elif software is 2:
        b='\\'
        f='/'
        pathgames=str(input("Enter path of games csv file (xlsx)     ")).replace(b,f)
        pathwallet=str(input("Enter path of wallet csv file (xlsx)     ")).replace(b,f)
        pathusers=str(input("Enter path of users csv file (xlsx)     ")).replace(b,f)
    else:
        print("Something went wrong. Please try running script again")
    
    return pathgames,pathwallet,pathusers

#####################App#############################
Art=text2art("NOVIBET",font="block" )
print(Art)
print("\n\n    ....This is a script, used in clustering players.\n\n     --> Please follow the instructsions\n\n author:Christos Logaras \n")





pathgames,pathwallet,pathusers=give_input()
games,users,wallet=import_csv(pathgames,pathusers,pathwallet)

transactions,user,gameid=data_preprocess(games,users,wallet)
dfsl = [user,gameid]
df = reduce(lambda  left,right: pd.merge(left,right,on=["UserProfileId"],how='inner'), dfsl)
dfsl2 = [df,transactions]
df1 = reduce(lambda  left,right: pd.merge(left,right,on=["UserProfileId"],how='left'), dfsl2)
df1.fillna(0,inplace=True)
df1.reset_index(drop=True,inplace=True)
#'''  7(1-test) inactive / 1495 active '''
#df1.groupby(['StatusSysname'])['UserProfileId'].nunique()

#'''cluster the players'''
clusters=df1[['UserProfileId', 'loyalty', 'age','CountryName_Greece', 'platforms', 'F', 'diff', 'IsFreeSpinID', 'IsLiveID', 'Pt%', 'valpergame','totaltrans','avgdraw','dd_ratio']]
X_users_prescaled =  clusters.drop(columns=['UserProfileId'])

customers_fix = pd.DataFrame()
customers_fix[['CountryName_Greece','IsFreeSpinID','IsLiveID','Pt%','dd_ratio']] = clusters[['CountryName_Greece','IsFreeSpinID','IsLiveID','Pt%','dd_ratio']]

customers_fix['age'] = stats.boxcox(clusters['age'])[0]
customers_fix['loyalty'] = stats.boxcox(clusters['loyalty'])[0]
customers_fix['F'] = stats.boxcox(clusters['F'])[0]
customers_fix['platforms'] = stats.boxcox(clusters['platforms'])[0]
customers_fix['valpergame'] = stats.boxcox(clusters['valpergame'])[0]

customers_fix['diff']= clusters['diff'].apply(lambda x: x+0.01)
customers_fix['totaltrans']= clusters['totaltrans'].apply(lambda x: x+0.01)
customers_fix['avgdraw']= clusters['avgdraw'].apply(lambda x: x+1)
customers_fix['avgdraw']=customers_fix['avgdraw'].abs()

customers_fix['diff'] = stats.boxcox(customers_fix['diff'])[0]
customers_fix['totaltrans'] = stats.boxcox(customers_fix['totaltrans'])[0]
customers_fix['avgdraw'] = stats.boxcox(customers_fix['avgdraw'])[0]

scaler = StandardScaler()
scaler.fit(customers_fix)
customers_normalized = scaler.transform(customers_fix)
#'''Checking mean and std'''
#print(customers_normalized.mean(axis = 0).round(3)) # [-0.  0.  0.  0.  0. -0. -0. -0.  0. -0.  0. -0.  0.]
#print(customers_normalized.std(axis = 0).round(3)) # [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]

visualizer = KElbowVisualizer(KMeans(),k=(2,15)).fit(customers_normalized)

model,a=clustering(customers_normalized,7) 
clusters = clusters.assign(Cluster=a)
df_normalized = pd.DataFrame(customers_normalized, columns=['CountryName_Greece', 'IsFreeSpinID', 'IsLiveID', 'Pt%', 'dd_ratio', 'totaltrans', 'avgdraw', 'diff', 'age', 'loyalty', 'F', 'platforms', 'valpergame'])
df_normalized['UserProfileId'] = clusters['UserProfileId']
df_normalized['Cluster'] = a           
df_nor_melt = pd.melt(df_normalized.reset_index(),id_vars=['UserProfileId', 'Cluster'],value_vars=['CountryName_Greece', 'IsFreeSpinID', 'IsLiveID', 'Pt%', 'dd_ratio', 'totaltrans', 'avgdraw', 'diff', 'age', 'loyalty', 'F', 'platforms', 'valpergame'],var_name='Attribute',value_name='Value')
c=df_normalized.groupby('Cluster').agg({
                    'CountryName_Greece':'mean', 
                    'IsFreeSpinID':'mean', 
                    'IsLiveID':'mean', 
                    'Pt%':'mean', 
                    'dd_ratio':'mean', 
                    'totaltrans':'mean', 
                    'avgdraw':'mean', 
                    'diff':'mean', 
                    'age':'mean', 
                    'loyalty':'mean', 
                    'F':'mean', 
                    'platforms':'mean', 
                    'valpergame':'mean'}).round(1)
c_nor_melt = pd.melt(c.reset_index(),id_vars=['Cluster'],value_vars=['CountryName_Greece', 'IsFreeSpinID', 'IsLiveID', 'Pt%', 'dd_ratio', 'totaltrans', 'avgdraw', 'diff', 'age', 'loyalty', 'F', 'platforms', 'valpergame'],var_name='Attribute',value_name='Value')

c0=clusters[clusters['Cluster']==0].describe()
c1=clusters[clusters['Cluster']==1].describe()
c2=clusters[clusters['Cluster']==2].describe()
c3=clusters[clusters['Cluster']==3].describe()
c4=clusters[clusters['Cluster']==4].describe()
c5=clusters[clusters['Cluster']==5].describe()
c6=clusters[clusters['Cluster']==6].describe()
frames = [c0,c1,c2,c3,c4,c5,c6]
clustering_results= pd.concat(frames).reset_index()
clustering_results= clustering_results[clustering_results['index']=='mean']
clustering_results=clustering_results[['Cluster','loyalty', 'age', 'CountryName_Greece', 'platforms', 'F', 'diff', 'IsFreeSpinID', 'IsLiveID', 'Pt%', 'valpergame', 'totaltrans', 'avgdraw', 'dd_ratio']].reset_index(drop=True)

#'''export excel used in dashboard'''
with pd.ExcelWriter("Clusters_Christos_Logaras.xlsx") as writer:
    clustering_results.to_excel(writer, sheet_name='clustering_r', index=False)
    c_nor_melt.to_excel(writer, sheet_name='c_nor_melt', index=False)
    df_nor_melt.to_excel(writer, sheet_name='df_nor_melt', index=False)
    clusters.to_excel(writer, sheet_name='clusters', index=False)
print('\n\n\n')
print('Excell  "Clusters_Christos_Logaras.xlsx" has been created ')
print('\n\n\n')
print('please use command "streamlit run dashboard.py"  to access dashboard')
print('\n\n\n')

print('Using KStest to determine if time difference between games ~ exponential distribution')
print('\n\n\n')
print('Please be patient .....\n\n\n')
print('This might take .....\n\n\n')

games['Date']= pd.to_datetime(games['Date'])
games.sort_values(['UserProfileId','Date'],inplace=True)
games.rename(columns = {'UserID':'UserProfileId'}, inplace = True)
games['diff'] = games.groupby('UserProfileId')['Date'].diff()
games['diff'] = pd.to_numeric(games['diff'].dt.days, downcast='integer')
games['diff'].fillna(0,inplace=True)
print('.......... some .....\n\n\n')
dct={}
for id in gameid['UserProfileId'].unique():
    ks=games[games['UserProfileId']==id]['diff'].to_list()
    p=stats.kstest(ks, 'expon').pvalue
    dct[id] = p
#'''p<.05, reject the null hypothesis:data comes from an exponential'''
expo=pd.DataFrame.from_dict(dct, orient='index')
expo.reset_index(inplace=True)
expo.columns = ['UserProfileId','expo_pval']
expo['null'] = np.where(expo['expo_pval']>0.05, True, False)
print('.......... time .....')
expo.to_csv('customer churn identification.csv')
print('csv file per customer id and p value for KS test, exported !')




