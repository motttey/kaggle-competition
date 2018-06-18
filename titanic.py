import pandas as pd
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

train_df = pd.read_csv("train.csv", header=0)

train_df = pd.read_csv("train.csv").replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)
test_df = pd.read_csv("test.csv").replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)

train_df['Salutation'] = train_df.Name.str.extract(' ([A-Za-z]+).', expand=False)
train_df['Salutation'] = train_df['Salutation'].replace(['Lady', 'Countess','Billiard', 'Carlo', 'Gordon', 'Impe', 'Melkebeke', 'Messemaeker', 'Mulder', 'Pelsmaeker', 'Planke', 'Shawah', 'Steen', 'Velde',
 'Walle','Capt', 'Col', 'Cruyssen', 'y', 'der', 'the','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_df['Salutation'] = train_df['Salutation'].replace('Mlle', 'Miss')
train_df['Salutation'] = train_df['Salutation'].replace('Ms', 'Miss')
train_df['Salutation'] = train_df['Salutation'].replace('Mme', 'Mrs')

train_df['Median_Age'] = train_df["Age"].dropna().median()
train_df[train_df['Salutation'] == 'Master']["Median_Age"] = train_df[train_df['Salutation'] == 'Master']["Age"].median()
#train_df[train_df['Salutation'] == 'Mrs']["Median_Age"] = train_df[train_df['Salutation'] == 'Mrs']["Age"].median()
#train_df[train_df['Salutation'] == 'Miss']["Median_Age"] = train_df[train_df['Salutation'] == 'Miss']["Age"].median()
train_df[train_df['Salutation'] == 'Rare']["Median_Age"] = train_df[train_df['Salutation'] == 'Rare']["Age"].median()

median_age = train_df["Age"].dropna().median()
if len(train_df.Age[train_df.Age.isnull()]) > 0:
  train_df.loc[(train_df.Age.isnull()), "Age"] = train_df.loc[(train_df.Age.isnull()), "Median_Age"]

median_embarked = train_df["Embarked"].dropna().median()
if len(train_df.Embarked[train_df.Embarked.isnull()]) > 0:
  train_df.loc[(train_df.Embarked.isnull()), "Embarked"] = median_embarked

median_fare = train_df["Fare"].dropna().median()
if len(train_df.Fare[train_df.Fare.isnull()]) > 0:
  train_df.loc[(train_df.Fare.isnull()), "Fare"] = median_fare

train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
train_df['IsAlone'] = 0
train_df.loc[train_df['FamilySize'] == 1, 'IsAlone'] = 1

print(train_df.head(3))
grouped = train_df.groupby('Salutation')
print(grouped.Age.median())
print(grouped.size())


train_df = train_df.drop(["Name", "Ticket", "SibSp", "Parch", "Cabin", "PassengerId", "Salutation"], axis=1)
print(train_df.head(3))

test_df['Salutation'] = test_df.Name.str.extract(' ([A-Za-z]+).', expand=False)
test_df['Salutation'] = test_df['Salutation'].replace(['Lady', 'Countess','Billiard', 'Carlo', 'Gordon', 'Impe', 'Melkebeke', 'Messemaeker', 'Mulder', 'Pelsmaeker', 'Planke', 'Shawah', 'Steen', 'Velde',
 'Walle','Capt', 'Col', 'Cruyssen', 'y', 'der', 'the','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test_df['Salutation'] = test_df['Salutation'].replace('Mlle', 'Miss')
test_df['Salutation'] = test_df['Salutation'].replace('Ms', 'Miss')
test_df['Salutation'] = test_df['Salutation'].replace('Mme', 'Mrs')

test_df['Median_Age'] = test_df["Age"].dropna().median()
test_df[test_df['Salutation'] == 'Master']["Median_Age"] = test_df[test_df['Salutation'] == 'Master']["Age"].median()
#test_df[test_df['Salutation'] == 'Mrs']["Median_Age"] = test_df[test_df['Salutation'] == 'Mrs']["Age"].median()
#test_df[test_df['Salutation'] == 'Miss']["Median_Age"] = test_df[test_df['Salutation'] == 'Miss']["Age"].median()
test_df[test_df['Salutation'] == 'Rare']["Median_Age"] = test_df[test_df['Salutation'] == 'Rare']["Age"].median()

if len(test_df.Age[test_df.Age.isnull()]) > 0:
  test_df.loc[(test_df.Age.isnull()), "Age"] = test_df.loc[(test_df.Age.isnull()), "Median_Age"]

median_fare = test_df["Fare"].dropna().median()
if len(test_df.Fare[test_df.Fare.isnull()]) > 0:
  test_df.loc[(test_df.Fare.isnull()), "Fare"] = median_fare

test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1
test_df['IsAlone'] = 0
test_df.loc[test_df['FamilySize'] == 1, 'IsAlone'] = 1

ids = test_df["PassengerId"].values
test_df = test_df.drop(["Name", "Ticket", "SibSp", "Parch", "Cabin", "PassengerId", "Salutation"], axis=1)

print(test_df.head(3))

train_data = train_df.values
test_data = test_df.values
r_state = 5
model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=25, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=15,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=4,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

output = model.fit(train_data[0::, 1::], train_data[0::, 0]).predict(test_data).astype(int)

# export result to be "titanic_submit.csv"
submit_file = open("titanic_submit.csv", "w")
file_object = csv.writer(submit_file)
file_object.writerow(["PassengerId", "Survived"])
file_object.writerows(zip(ids, output))
submit_file.close()
