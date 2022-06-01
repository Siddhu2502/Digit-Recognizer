import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.image as img


# load the test_csv and train_csv
test_csv = pd.read_csv("test.csv")
train_csv = pd.read_csv("train.csv")
sample_csv = pd.read_csv("sample_submission.csv")

# display information of test_csv
print(train_csv.info())

df = pd.concat((train_csv, test_csv))

features = df.columns.drop('label')
print(features.shape)


dum = pd.get_dummies(df, columns=['label'])
df = pd.concat([df['label'], dum]) 

#image = img.imread(dtrain.loc[0])
# display 10 plots in one single plot 
plt.figure(figsize=(20,20))
for i in range (10):
    arr = np.asarray(np.array(train_csv.loc[i][1:]).reshape(-28, 28))
    print(arr.shape)
    plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
    plt.show()