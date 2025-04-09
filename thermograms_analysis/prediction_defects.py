from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from catboost import CatBoostClassifier
from sklearn.svm import SVC
import pandas as pd


from utils import validate_list_models, prepare_dataset, validate_model_plot
from modules import NNClassifier


df1, is_defect1 = prepare_dataset('thermograms_analysis/metrics/metrics_40.json')
df2, is_defect2 = prepare_dataset('thermograms_analysis/metrics/metrics_new_40.json')

df = pd.concat((df1, df2), axis=0)
is_defect = pd.concat((is_defect1, is_defect2), axis=0)



### K-FOLD VALIDATION
lr = make_pipeline(PolynomialFeatures(2), StandardScaler(), LogisticRegression())
boosting = CatBoostClassifier(logging_level='Silent')
rf = RandomForestClassifier()
svr = make_pipeline(StandardScaler(), SVC(probability=True))

validate_model_plot(boosting, df, is_defect)

# data = [f"thermograms_analysis/metrics/metrics_{i}.json" for i in range(10, 51, 5)]  # list of jsons
# models = {'LogisticRegression': lr, 
#           'SVC': svr, 
#           'RandomForest': rf, 
#           'CatBoostClassifier': boosting, 
#           'FullyConnectedNet': NNClassifier()}
# res = validate_list_models(models, data)
# print(res)
# res.to_pickle("result.pkl")