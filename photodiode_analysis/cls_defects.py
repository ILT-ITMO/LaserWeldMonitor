from sklearn.ensemble import RandomForestClassifier
from utils import validate_model_plot, generate_ds_mean_std, generate_ds_split_mean_std


train_feat, train_gt, val_feat, val_gt = generate_ds_split_mean_std('photodiode_analysis/data', 'photodiode_analysis/ground_truth.json',
                                512, 32, 0.3)

clf = RandomForestClassifier()

print(train_gt)

validate_model_plot(clf, train_feat, train_gt, val_feat, val_gt, 'he')
