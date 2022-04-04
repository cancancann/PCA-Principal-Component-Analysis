import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler        #Bağımlı,bağımsız değişkenler 
from sklearn.decomposition import PCA 

#Her bir örnek için 4 özellik bulunur bunlar:
#Taç yaprak uzunluğu--sepal lenght
#Taç yaprak genişliği--sepal width
#Çanak yaprak genişliği--petal lenght
#Çanak yaprak uzunluğu--petal widht

url = 'pca_iris.data'

df = pd.read_csv(url,names=['sepal lenght','sepal width','petal lenght', 'petal widht','target'])
print(df)

#Ayırma işleme

features = ['sepal lenght','sepal width','petal lenght', 'petal widht']
x = df[features]

target = ['target']
y = df[target]


#Scale Etmemiz gerekiyor;

x =StandardScaler().fit_transform(x)
print(x)#scale durumuna getirdik.

#PCA Projection 4 boyuttan 2 boyuta indirgeme:

pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data= principalComponents , columns= ['Principal Component 1','Principal Component 2'])
print(principalDf) #veriyi istediğimiz şekile 2 boyutluya indirgedik.

#Target to PCA:

final_dataframe = pd.concat([principalDf,df['target']],axis=1)
final_dataframe.head()
print(final_dataframe) #Verimizi son haline getirdik ve targeti ekledik.

#Görselleştirme:

targets = ['Iris-setosa','Iris-versicolor', 'Iris-virginica']
colors = ['g','b','r']

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

for target, col in zip(targets,colors):
    dftemp = final_dataframe[df.target == target]
    plt.scatter(dftemp['Principal Component 1'], dftemp['Principal Component 2'], color = col)
plt.savefig('pca.png', dpi=300)  
plt.show()


#Varyans Koruma 

result = pca.explained_variance_ratio_
print(result)

total = pca.explained_variance_ratio_.sum()
print(total) # %96 veri seti korumak

