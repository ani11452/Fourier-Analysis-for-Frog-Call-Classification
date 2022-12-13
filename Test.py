# Import necessary libraries
import numpy as np
import librosa
from glob2 import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load audio data and extract features
x = glob("Recordings/*/*.wav")

mfccs = []
for y in x:
    data, sampling_rate = librosa.load(y)
    mfccs.append(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40))

labels = []
for y in ["Spea_multiplicata", "Euphlyctis_cyanophlyctis", "Adenomera_marmorata", "Aplastodiscus_leucopygius", "Elachistocleis_ovalis", "Kassina_senegalensis", "Ameerega_flavopicta", "Rana_draytonii"]:
    for x in range(15):
        labels.append(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mfccs, labels, test_size=0.2, random_state=42)

for i, x in enumerate(X_train):
    x = x.flatten('F')
    lent = len(x)

    if lent < 1760:
        q = np.zeros(1760 - lent)
        con = np.hstack((x, q))
        X_train[i] = con
    elif lent > 1760:
        q = x[0:1761]
        X_train[i] = q

    X_train[i] = x
    for j, y in enumerate(X_train[i]):
        X_train[i][j] = float(y)

for i, x in enumerate(X_test):
    x = x.flatten('F')
    lent = len(x)

    if lent < 1760:
        q = np.zeros(1760 - lent)
        con = np.hstack((x, q))
        X_test[i] = con
    elif lent > 1760:
        q = x[0:1761]
        X_test[i] = q

    X_test[i] = x
    for j, y in enumerate(X_test[i]):
        X_test[i][j] = float(y)

# Train and evaluate the model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
