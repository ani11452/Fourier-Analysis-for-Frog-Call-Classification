# Import necessary libraries
import numpy as np
import librosa
import sklearn
import librosa.display
import matplotlib.pylab as plt
import seaborn as sns
import os
from glob2 import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors as n
from sklearn import model_selection as ms

# Initial Variables:
Data = []
Labels = []

# Process Audio Files
frog_files = os.listdir("Recordings")
del frog_files[1]

# Go through each frog's recordings
for file in frog_files:
    path = "Recordings/" + file + "/*.wav"
    recordings = glob(path)
    samples = []
    labels = []

    # Load sample data for each recording
    for rec in recordings:
        data, sampling_rate = librosa.load(rec)

        # Pad or Splice if not correct number of samples (22050)
        num_samples = len(data)
        if num_samples < sampling_rate:
            pad = np.zeros(sampling_rate - num_samples)
            data = np.hstack((data, pad))
        elif num_samples > sampling_rate:
            data = data[0:sampling_rate + 1]

        # Normalize Samples
        data = data.reshape(-1, 1)
        data = normalize(data, axis=0)
        data = data.flatten()

        # Compute Short-Time Fourier Transform values
        stft = librosa.stft(data, n_fft=512)

        # Convert to Log Power Spectral Density
        S_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

        # Add to samples and labels
        Data.append(S_db.flatten('F'))
        Labels.append(file)

# Randomize Data-Label pairs within sample
temp = list(zip(Data, Labels))
np.random.seed(42)
np.random.shuffle(temp)
res1, res2 = zip(*temp)
Data, Labels = list(res1), list(res2)

# Initialize LOO
loo = ms.LeaveOneOut()
x = loo.get_n_splits(Data)
trainingSet = []
predictions = []
validations = []

# Develop Sets
for train, test in loo.split(Data):
    trainingSet.append(train)
    validations.append(test)

# Initialize Variables for Validation
Truth = []
Preds = []

# Train
for n, train in enumerate(trainingSet):
    # Collect Train and Test data for each validation
    Data_train = [Data[i] for i in train]
    Label_train = [Labels[i] for i in train]
    Data_test = [Data[n]]
    Truth.append(Labels[n])

    # Scale and Normalize data
    scaler = StandardScaler()
    Data_train = scaler.fit_transform(Data_train)
    Data_test = scaler.transform(Data_test)

    # Train model
    model = sklearn.neighbors.KNeighborsClassifier()
    model.fit(Data_train, Label_train)

    # Save model output
    Label_pred = model.predict(Data_test)
    Preds.append((Label_pred))

# Evaluate Model
score = sklearn.metrics.accuracy_score(Truth, Preds)
print('Accuracy Score: ', score)

# Graph Confusion Matrix
confusion = sklearn.metrics.confusion_matrix(Truth, Preds, labels=frog_files)
ax = plt.subplot()
sns.heatmap(confusion, annot=True, ax=ax, cmap="Blues")
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(frog_files, size=5)
ax.yaxis.set_ticklabels(frog_files, size=5)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.autoscale()
plt.tight_layout()
plt.show()
print("Confusion Matrix:\n", confusion)
