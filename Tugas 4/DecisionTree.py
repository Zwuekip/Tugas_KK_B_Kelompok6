import pandas as pd
import numpy as np

df = pd.read_csv('trainingdataset.csv', sep=';')
print(df.head())

# Define the calculate entropy function
def calculate_entropy(df_label):
    classes,class_counts = np.unique(df_label,return_counts = True)
    entropy_value = np.sum([(-class_counts[i]/np.sum(class_counts))*np.log2(class_counts[i]/np.sum(class_counts)) 
                        for i in range(len(classes))])
    return entropy_value

# Define the calculate information gain function
def calculate_information_gain(dataset,feature,label): 
    # Calculate the dataset entropy
    dataset_entropy = calculate_entropy(dataset[label])   
    values,feat_counts= np.unique(dataset[feature],return_counts=True)
    
    # Calculate the weighted feature entropy                                # Call the calculate_entropy function
    weighted_feature_entropy = np.sum([(feat_counts[i]/np.sum(feat_counts))*calculate_entropy(dataset.where(dataset[feature]
                              ==values[i]).dropna()[label]) for i in range(len(values))])    
    feature_info_gain = dataset_entropy - weighted_feature_entropy
    return feature_info_gain

# Define the create decision tree function
def create_decision_tree(dataset, df, features, label, parent):
    
    datum = np.unique(df[label], return_counts = True)
    unique_data = np.unique(dataset[label])
    
    if len(unique_data) <= 1:
        return unique_data[0]
    
    elif len(dataset) == 0:
        return unique_data[np.argmax(datum[1])]
    
    elif len(features) == 0:
        return parent
    
    else:
        parent = unique_data[np.argmax(datum[1])]
        
        item_values = [calculate_information_gain(dataset, feature, label) for feature in features]
        
        optimum_feature_index = np.argmax(item_values)
        optimum_feature = features[optimum_feature_index]
        decision_tree = {optimum_feature:{}}
        features = [i for i in features if i != optimum_feature]
        
        for value in np.unique(dataset[optimum_feature]):
            min_data = dataset.where(dataset[optimum_feature] == value).dropna()
            
            min_tree = create_decision_tree(min_data, df, features, label, parent)
            
            decision_tree[optimum_feature][value] = min_tree
        
        return (decision_tree)
    
# Define the predict function
def predict(test_data, decision_tree):
    for nodes in decision_tree.keys():
        value = test_data[nodes]
        decision_tree = decision_tree[nodes][value]
        
        prediction = 0
        if type(decision_tree) is dict:
            prediction = predict(test_data, decision_tree)
        else:
            prediction = decision_tree
            break
            
    return prediction

# Set the features and label
features = df.columns[:-1]
label = 'MSRP'
parent=None
#print(features)

# Train the decision tree model
decision_tree = create_decision_tree(df,df,features,label,parent)

sample_data = {'Year':'Old', 'Transmission Type':'AUTOMATIC', 'Number of Doors':4}
test_data = pd.Series(sample_data)
prediction = predict(test_data, decision_tree)
print(prediction)

sample_data = {'Year':'Old', 'Transmission Type':'MANUAL', 'Number of Doors':2}
test_data = pd.Series(sample_data)
prediction = predict(test_data, decision_tree)
print(prediction)

sample_data = {'Year':'New', 'Transmission Type':'AUTOMATIC', 'Number of Doors':2}
test_data = pd.Series(sample_data)
prediction = predict(test_data, decision_tree)
print(prediction)

sample_data = {'Year':'New', 'Transmission Type':'MANUAL', 'Number of Doors':2}
test_data = pd.Series(sample_data)
prediction = predict(test_data, decision_tree)
print(prediction)

sample_data = {'Year':'Old', 'Transmission Type':'MANUAL', 'Number of Doors':4}
test_data = pd.Series(sample_data)
prediction = predict(test_data, decision_tree)
print(prediction)

#print(decision_tree)