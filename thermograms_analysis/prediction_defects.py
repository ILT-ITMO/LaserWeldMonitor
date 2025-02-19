from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from catboost import CatBoostClassifier
from sklearn.svm import SVC


from utils import *
from nn_classifiers import NNClassifier


df, is_defect = prepare_dataset('metrics_40.json')

print(df.head())
print(df.shape)


### K-FOLD VALIDATION
lr = make_pipeline(PolynomialFeatures(2), StandardScaler(), LogisticRegression())
boosting = CatBoostClassifier(logging_level='Silent')
rf = RandomForestClassifier()
svr = make_pipeline(StandardScaler(), SVC(probability=True))

#validate_model_plot(boosting, df, is_defect)
data = [f"metrics_{i}.json" for i in range(10, 51, 5)]  # list of jsons
models = {'LogisticRegression': lr, 
          'SVC': svr, 
          'RandomForest': rf, 
          'CatBoostClassifier': boosting, 
          'FullyConnectedNet': NNClassifier()}
res = validate_list_models(models, data)
print(res)
res.to_pickle("result.pkl")