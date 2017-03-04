import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'expenses', 'exercised_stock_options','poi_from_rate','poi_to_rate'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Besides TOTAL, other data points should be kept in the dataset since there is no evidence that those values are errors.
data_dict.pop('TOTAL')
my_dataset = data_dict

### Task 3: Create new feature(s)
### Two new features of email might be helpful: the proportion of poi_messages to total messages (to and from respectively)
for person in my_dataset:
    if (my_dataset[person]['to_messages']!='NaN')& (my_dataset[person]['to_messages']>0):
        my_dataset[person]['poi_to_rate']=float(my_dataset[person]['from_poi_to_this_person'])/my_dataset[person]['to_messages']
    else:
        my_dataset[person]['poi_to_rate']='NaN'
    if (my_dataset[person]['from_messages']!='NaN')& (my_dataset[person]['from_messages']>0):    
        my_dataset[person]['poi_from_rate']=float(my_dataset[person]['from_this_person_to_poi'])/my_dataset[person]['from_messages']
    else:
        my_dataset[person]['poi_from_rate']='NaN'

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers

### Multiple classifiers were tried in file 'algorithm_exploration_tyler_jin.ipynb', detail process can be reviewed there.
### Here I only adopt the selected classifier.
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=8,
            min_samples_split=20, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')


### Task 5: Tune your classifier to achieve better than .3 precision and recall 

### Detail tuning and testing process can be found in file 'algorithm_exploration_tyler_jin.ipynb'


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)