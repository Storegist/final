import pickle
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from utils.dataloader import DataLoader
from settings.constants import TRAIN_CSV

with open('settings/specifications.json') as f:
    specifications = json.load(f)

raw_train = pd.read_csv(TRAIN_CSV)
x_columns = specifications['description']['X']
y_column = specifications['description']['y']

X_raw = raw_train[x_columns]

loader = DataLoader()
loader.fit(X_raw)
X = loader.load_data()
y = raw_train.Vertical_Segment.iloc[X.index]

model = RandomForestClassifier()
model.fit(X, y)
with open('model/RFC.pickle', 'wb') as f:
    pickle.dump(model, f)
