import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier

URL = "https://raw.githubusercontent.com/marcopeix/MachineLearningModelDeploymentwithStreamlit/master/18_caching_capstone/data/mushrooms.csv"
COLS = ['class', 'odor', 'gill-size', 'gill-color', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'ring-type', 'spore-print-color']


# Read data
df  = pd.read_csv(URL)
df = df[COLS]

# Create pipeline
pipe = Pipeline(steps=[('ordinal_encoder', OrdinalEncoder()),
                ('gradient_boosting_classifier', GradientBoostingClassifier())])
# Fit the pipeline
x = df.drop(['class'], axis=1)
y = df['class']
pipe.fit(x, y)

# Save the pipeline
dump(pipe, 'model.joblib')