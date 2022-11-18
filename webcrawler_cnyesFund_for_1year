import time
from bs4 import BeautifulSoup as BS
import requests
import pandas as pd

def getFundinfo(strUrl):
  strUrl=strUrl.replace("/report/","/document")
  fundRes = requests.get(strUrl).text 
  soup = BS(fundRes,'lxml') 
  i = soup.select("td")
  time.sleep(0.1)
  return i[0].text,i[3].text,i[4].text,i[7].text,i[12].text

for t in range(1,4):
  fundGroup_str="G"+str(t)

  k = 1
  li1,anueRate1 = [],[]
  while True:
    anueRes = requests.get('https://fund.cnyes.com/search/?focusTab=1&fundGroup='+fundGroup_str+'&order=return1Year&page='+str(k)).text
    soup = BS(anueRes,'lxml')
    li2 = soup.select('._2rwNH a')
    anueRate2 = soup.select('#target_tbody ._1TfYq')
    li1 += li2
    anueRate1 += anueRate2
    k += 1
    if len(li2) < 20 : break
    while(len(li1)%200==0):
      time.sleep(0.5)
      break
 
  myColumns = [ '基金類型', '基金績效', '基金Name',  '操作人',
        '基金規模(美金)', '基金規模(原)', '計價幣別', '成立時間' ]

  if fundGroup_str=="G1": myFData = pd.DataFrame( [], columns=myColumns ) 
  for i in li1:
    for j in anueRate1[li1.index(i)]:  
      f_name,f_operator,f_size,f_currency,f_buildtm=getFundinfo("https://fund.cnyes.com"+i.attrs['href'])  
      ExRate = {'USD': 1,'TWD':0.034,'EUR':1,'ZAR':0.06,'AUD':0.8,'GBP':1.15,'JPY':0.007,'CHF':1,'SEK':0.1,'HKD':0.13,'SGD':0.71,'CNH':0.15}
      arrf_size = f_size.split(' ')
      if len(arrf_size)==3 & len(arrf_size[0])>0:
        f_size_US=int(arrf_size[1].replace(",","").strip())    
        f_size_US=ExRate[arrf_size[0]]*f_size_US
      elif len(arrf_size)==3 and len(arrf_size[0])==0 and f_currency!="":
        ExRate = {'美元': 1,'新台幣':0.034,'歐元':1,'澳元':0.8,'英鎊':1.15,'南非幣':0.06,'日圓':0.007,'瑞士法郎':1,'瑞典克朗':0.1,'港元':0.13,'新加坡元':0.71,'人民幣':0.034}
        f_size_US=int(arrf_size[1].replace(",","").strip())
        f_size_US=int(ExRate[f_currency]*f_size_US)
      else:f_size_US=f_size
      
      if fundGroup_str=="G1":f_group="股票型"
      elif fundGroup_str=="G2":f_group="債券型"
      else:f_group="平衡型"
      df32 = pd.DataFrame( [
          [f_group, j, f_name, f_operator, f_size_US, f_size, f_currency, f_buildtm]
           ]   , columns=myColumns ) 
      myFData = myFData.append(df32,ignore_index=True)
  print("t:"+ fundGroup_str +":累計"+str(len(myFData))+"筆,")
print(len(myFData))
myFData.to_csv('myFG123Data_1118.csv')


# 資料整理 & 清洗
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib import font_manager

df = pd.read_csv('myFG123Data_1118.csv')

filter = (df['成立時間'] < "9999/99/99") & (df['成立時間'] > "1800/99/99")
df = df[filter]

filter2= (df['基金績效'] != "--")
df = df[filter2]

df = df[~df['基金規模(美金)'].str.contains(' ', na=False)]
df = df[~df['基金規模(美金)'].str.contains(',', na=False)]
df = df[~df['基金規模(美金)'].str.contains('-', na=False)]

perf = df['基金績效'].map( lambda x : float(x.strip()) )

df['year'] = pd.DatetimeIndex(df['成立時間']).year
year = df['成立時間'].map( lambda x : int(str(x)[:4]) )

f_operator= df['操作人'].map( lambda x : str(x)[:5])
f_type= df['基金類型'].map( lambda x : str(x)[:8])
f_name= df['基金Name'].map( lambda x : str(x)[:2])
df['公司'] = df['基金Name'].map( lambda x : str(x)[:2])
df['績效Number'] = df['基金績效'].map( lambda x : float(x))
df['績效Level'] = df['基金績效'].map( lambda x : int(float(x)/10) if float(x)>0 else int(float(x)/10)-1  )
df['績效Level_Name'] = df['績效Level'].map( lambda x : f"{x*10} ~ {x*10+10}"  )

gg=df[['公司','績效Number']].sort_values(by=['績效Number'],ascending = [False]).groupby(by= ['公司']).head(1)
index1 = gg.index
df_max = df.filter(items = index1, axis=0)
f_max=df_max["基金Name"].map( lambda x : str(x))
per_max=df_max['基金績效'].map( lambda x : float(x)  )
per_max=df_max['績效Number']

oo=df[['操作人','績效Number']].sort_values(by=['績效Number'],ascending = [False]).groupby(by= ['操作人']).head(1)
index2=oo.index
df_oo = df.filter(items = index2, axis=0)
f_oo=df_oo["操作人"].map( lambda x : str(x))
per_oo=df_oo['基金績效'].map( lambda x : float(x)  )
per_oo=df_oo['績效Number']

f_size = df['基金規模(美金)'].map( lambda x : float(x))
f_size2 = df['基金規模(美金)'].map( lambda x : float(x[:-1]))
f_size2 = f_size2.map( lambda x : x/1000000000)

filterg1_1= (df['基金類型'] == "股票型")
filterg2_1= (df['基金類型'] == "債券型")
filterg3_1= (df['基金類型'] == "平衡型")

g1 = filterg1_1 & perf & f_size2
g2 = filterg2_1 & perf & f_size2
g3 = filterg3_1 & perf & f_size2

levels = list(np.sort(df['績效Level'].unique()))
glv_1 = df[g1][['績效Level','績效Level_Name']].sort_values(by=['績效Level']).groupby(['績效Level','績效Level_Name']).size()
glv_2 = df[g2][['績效Level','績效Level_Name']].sort_values(by=['績效Level']).groupby(['績效Level','績效Level_Name']).size()
glv_3 = df[g3][['績效Level','績效Level_Name']].sort_values(by=['績效Level']).groupby(['績效Level','績效Level_Name']).size()

gY1 = [0] * len(levels)
gY2 = [0] * len(levels)
gY3 = [0] * len(levels)

perfg1 = df[g1]['基金績效'].map( lambda x : float(x)  )
perfg2 = df[g2]['基金績效'].map( lambda x : float(x)  )
perfg3 = df[g3]['基金績效'].map( lambda x : float(x)  )

scaleg1 = df[g1]['基金規模(美金)'].map( lambda x : float( x.split('(')[0] .replace(',','')))
scaleg2 = df[g2]['基金規模(美金)'].map( lambda x : float( x.split('(')[0] .replace(',','')))
scaleg3 = df[g3]['基金規模(美金)'].map( lambda x : float( x.split('(')[0] .replace(',','')))

scale1 = scaleg1.map( lambda x: x/100000000)
scale2 = scaleg2.map( lambda x: x/100000000)
scale3 = scaleg3.map( lambda x: x/100000000)


# 基金績效 & 數量長條圖

a,b,c = -100, -90, 10 
amountli,rate_range = [],[]

for i in range(len(perf)): 
  count = 0
  if a >= max(perf):break
  for j in perf:
    if j >= a and j < b:
      count += 1
  while(count != 0):  
    amountli.append(count)
    rate_range.append(f'{a}~{b}')
    break
  a = b
  b += c
plt.figure(figsize=(12,8))
plt.bar(rate_range, amountli,color = 'm', width = 0.8)
plt.grid()
plt.xlabel("Fund Performance(%)",fontsize=10)
plt.ylabel("Quantity",fontsize=10)
plt.show()

# 各類型基金績效 & 數量長條圖

for idx in glv_1.index:
    if idx[0] in levels :
        i = levels.index(idx[0])
        gY1[i] = glv_1.loc[idx]

for idx in glv_2.index:
    if idx[0] in levels :
        i = levels.index(idx[0])
        gY2[i] = glv_2.loc[idx]

for idx in glv_3.index:
    if idx[0] in levels :
        i = levels.index(idx[0])
        gY3[i] = glv_3.loc[idx]

gY1_and_2  = [x + y for x, y in zip(gY1, gY2)]
gY1_and_23 = [x + y for x, y in zip(gY3, gY1_and_2)]
level_names = [f"{x*10} ~ {x*10+10}" for x in levels]

plt.figure(figsize=(12,8))
zhfont1 = mpl.font_manager.FontProperties(fname='kaiu.ttf',size = 24)
plt.bar(level_names, gY1, color = 'lightsteelblue', width = 0.8, label='股票型')
plt.bar(level_names, gY2, color = 'orange', width = 0.8, bottom = gY1, label='債券型')
plt.bar(level_names, gY3, color = 'limegreen', width = 0.8, bottom = gY1_and_2, label='平衡型')
plt.grid()
plt.xlabel("Fund Performance(%)",fontsize=10)
plt.ylabel("Quantity",fontsize=10)
plt.xticks(rotation= 60 )

for idx,lv_name in enumerate(level_names):
      plt.text(lv_name , gY1_and_23[idx] + 50,gY1_and_23[idx] ,
            fontsize=16 ,horizontalalignment='center' , weight='bold')

      if gY3[idx] >= 100 :
          plt.text(lv_name, (gY1_and_23[idx] + gY1_and_2[idx] -50) /2,
              gY3[idx], fontsize=14, horizontalalignment='center')
          
      if gY2[idx] >= 100 :
          plt.text(lv_name, (gY1[idx] + gY1_and_2[idx]) /2,
              gY2[idx], fontsize=14, horizontalalignment='center')

      if gY1[idx] >= 100 :
          plt.text(lv_name, gY1[idx]/2 ,gY1[idx],
              fontsize=14, horizontalalignment='center')

plt.legend(loc="upper right", prop=zhfont1)
plt.show()

# 成立時間 & 績效散佈圖

data={'year':year,'performance':perf}
df2=pd.DataFrame(data) 
plt.figure(figsize=(15, 8), dpi=80) 
h=sns.jointplot(data=df2,x='year',y='performance')
plt.scatter(df['year'], perf ,marker='.',s=0.01)
plt.grid()
plt.show()

plt.figure(figsize=(15, 6), dpi=150)
plt.scatter(year, perf ,marker='.')
plt.grid()
plt.show()


# 各類型基金規模 & 績效散佈圖

plt.figure(figsize=(8, 6), dpi=150)

for color in ['Stock']:
    plt.scatter(scale1, perfg1 ,c = 'r',label=color ,alpha = 0.5,marker='.',linewidths=0.3)
for color in ['Bond']:
    plt.scatter(scale2, perfg2 ,c = 'g',label=color ,alpha = 0.5,marker='x',linewidths=0.9)
for color in ['Balanced']:
    plt.scatter(scale3, perfg3 ,c = 'b',label=color ,alpha = 0.5,marker='+',linewidths=1.2)

myfont = mpl.font_manager.FontProperties(fname="kaiu.ttf")

plt.title("各類型規模績效", fontproperties=myfont, fontsize=20)
plt.xlabel('基金規模(單位:億美金)', fontproperties=myfont, fontsize=15)
plt.ylabel('績效', fontproperties=myfont, fontsize=15)
plt.xlim(0, 400)
plt.grid()
plt.legend(["Stock", "Bond", "Balanced"], loc='upper right')
plt.show()


#各公司最佳績效圖

plt.figure(figsize=(15, 6), dpi=150)
myfont = font_manager.FontProperties(fname="kaiu.ttf")
plt.rcParams["axes.unicode_minus"]=False 
plt.bar(df_max['公司'] ,per_max, width = 0.5)
plt.title("各公司最佳績效", fontproperties=myfont, fontsize=20)
plt.xlabel('基金公司', fontproperties=myfont, fontsize=5)
plt.ylabel('績效', fontproperties=myfont, fontsize=15)
plt.xticks(df_max['公司'], fontproperties=myfont, fontsize=9)
plt.tick_params(axis='x', which='major', labelsize=6)
plt.xticks(rotation=70)
plt.rcParams['font.sans-serif'] = ['SimHei']


#操作人績效圖

plt.figure(figsize=(15, 6), dpi=200)
myfont = font_manager.FontProperties(fname="kaiu.ttf")
plt.rcParams["axes.unicode_minus"]=False 
plt.bar(df_oo['操作人'] ,per_oo, width = 0.5)
plt.title("各操作人最佳績效", fontproperties=myfont, fontsize=20)
plt.xlabel('操作人', fontproperties=myfont, fontsize=5)
plt.ylabel('績效', fontproperties=myfont, fontsize=15)
plt.xticks(df_oo['操作人'], fontproperties=myfont, fontsize=9)
plt.tick_params(axis='x', which='major', labelsize=5)
plt.xticks(rotation=70)
plt.rcParams['font.sans-serif'] = ['SimHei']
