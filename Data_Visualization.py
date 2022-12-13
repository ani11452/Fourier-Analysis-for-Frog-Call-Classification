import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import librosa
import librosa.display
from sklearn.preprocessing import normalize


from glob2 import glob

sns.set_theme(style="white", palette=None)

x = glob("Recordings/Ameerega_flavopicta/*.wav")

y, sr = librosa.load(x[0])
y = y.reshape(-1, 1)
y = normalize(y, axis=0)
y = y.flatten()
pd.Series(y).plot(figsize=(10, 5))
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()

D = librosa.stft(y, n_fft=512)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
fig, ax = plt.subplots()
img = librosa.display.specshow(S_db,
                               x_axis='time',
                               y_axis='log',
                               ax=ax)
plt.show()