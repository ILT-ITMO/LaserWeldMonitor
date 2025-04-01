from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from utils import prepare_dataset


df, is_defect = prepare_dataset('thermograms_analysis/metrics/metrics_40.json')

print(df.head())
print(df.shape)

lr = make_pipeline(PolynomialFeatures(2), StandardScaler(), LogisticRegression())
boosting = CatBoostClassifier(logging_level='Silent')
rf = RandomForestClassifier()

model = rf
model.fit(df, is_defect)


from sklearn.metrics import classification_report, precision_recall_curve
import matplotlib.pyplot as plt

df, is_defect = prepare_dataset('thermograms_analysis/metrics/metrics_new_40.json')

print(df.head())
print(df.shape)

print(classification_report(is_defect, model.predict(df)))

# precision, recall, _ = precision_recall_curve(is_defect, model.predict_proba(df)[:, 1])
# plt.plot(precision, recall, lw=2, color='black')
# plt.xlabel('Precision')
# plt.ylabel('Recall')
# plt.show()