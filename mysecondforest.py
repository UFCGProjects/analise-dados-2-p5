import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

def handlerData(filepath):
    new_pd = pd.read_csv(filepath, header=0)

    # female = 0, Male = 1
    new_pd['Gender'] = new_pd['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # Embarked from 'C', 'Q', 'S'
    # Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

    # All missing Embarked -> just make them embark from most common place
    # if len(new_pd.Embarked[ new_pd.Embarked.isnull() ]) > 0:
    #     new_pd.Embarked[ new_pd.Embarked.isnull() ] = new_pd.Embarked.dropna().mode().values

    # Ports = list(enumerate(np.unique(new_pd['Embarked'])))    # determine all values of Embarked,
    # Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
    # new_pd.Embarked = new_pd.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

    # All the ages with no data -> make the median of all Ages
    median_age = new_pd['Age'].dropna().median()
    if len(new_pd.Age[ new_pd.Age.isnull() ]) > 0:
        new_pd.loc[ (new_pd.Age.isnull()), 'Age'] = median_age

    # if len(new_pd.Fare[ new_pd.Fare.isnull() ]) > 0:
    #     median_fare = np.zeros(3)
    #     for f in range(0,3):                                              # loop 0 to 2
    #         median_fare[f] = new_pd[ new_pd.Pclass == f+1 ]['Fare'].dropna().median()
    #     for f in range(0,3):                                              # loop 0 to 2
    #         new_pd.loc[ (new_pd.Fare.isnull()) & (new_pd.Pclass == f+1 ), 'Fare'] = median_fare[f]

    # Collect the test data's PassengerIds before dropping it
    ids = new_pd['PassengerId'].values

    # Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
    new_pd = new_pd.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Fare', 'Parch', 'Embarked', 'SibSp'], axis=1)

    return ids, new_pd

def predict(train_ids, train_data, test_ids, test_data):
    print 'Training...'
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

    print 'Predicting...'
    output = forest.predict(test_data).astype(int)

    predictions_file = open("mysecondforest.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(test_ids, output))
    predictions_file.close()
    print 'Done.'


train_ids, train_data = handlerData('train.csv')
test_ids, test_data = handlerData('test.csv')

for t in [train_data, test_data]:
    print t.describe()
    print '---'

predict(train_ids, train_data.values, test_ids, test_data.values)