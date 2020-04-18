import numpy as np
import pandas as pd

# Reading Data from CSV File
data = pd.read_csv('Iris.csv')
print(data.head(5))
data.describe()
data_normalized = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
print(data_normalized.head(5))

data_normalized.describe()
# Replacing data to class values
target = data[['Species']].replace(['Iris-virginica', 'Iris-versicolor', 'Iris-setosa'], [0, 1, 2])
print(target.head(5))
# Adding Normalized data's column with target class numeric
data_final = pd.concat([data_normalized, target], axis=1)
print(data_final.head(5))

train_test_per = 70/100.0
# Adding a boolean value to the data for distributing into train and test
data_final['train'] = np.random.rand(len(data_final)) < train_test_per
print(data_final.head(5))

# Checking the randomly alloted train variable and then removing it
train = data_final[data_final.train == 1]
train = train.drop('train', axis=1).sample(frac=1)
print(train.head(5))


test = data_final[data_final.train == 0]
test = test.drop('train', axis=1)
test.sample(n=5)

# Selecting first 4 columns of the training dataset
X = train.values[:, :4]
# Printing first 5 values of Training X
print(X[:5])

# Multiclass division of targets and storing in Y according to 5th column of training data
targets = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
y = np.array([targets[int(x)] for x in train.values[:, 4:5]])
print(y[:5])

num_inputs = len(X[0])
hidden_layer_neurons = 2
np.random.seed(4)
# Initializing weights from Input to Hidden layer
w1 = 2*np.random.random((num_inputs, hidden_layer_neurons)) - 1
print(w1)

# Initializing weights from Hidden Layers to Output
num_outputs = len(y[0])
w2 = 2*np.random.random((hidden_layer_neurons, num_outputs)) - 1
print(w2)

learning_rate = 0.1 # slowly update the network
# Training the Weights
for epoch in range(50000):
    l1 = 1/(1 + np.exp(-(np.dot(X, w1)))) # sigmoid function
    l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))
    er = (abs(y - l2)).mean()
    l2_delta = (y - l2)*(l2 * (1-l2))
    l1_delta = l2_delta.dot(w2.T) * (l1 * (1-l1))
    w2 += l1.T.dot(l2_delta) * learning_rate
    w1 += X.T.dot(l1_delta) * learning_rate
print('Error:', er)

# Using the weights on Test Value
X = test.values[:,:4]
y = np.array([targets[int(x)] for x in test.values[:, 4:5]])

l1 = 1/(1 + np.exp(-(np.dot(X, w1))))
l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))

np.round(l2, 3)

yp = np.argmax(l2, axis=1) # prediction
res = yp == np.argmax(y, axis=1)
correct = np.sum(res)/len(res)

testresult = test[['Species']].replace([0, 1, 2], ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa'])

testresult['Prediction'] = yp
testresult['Prediction'] = testresult['Prediction'].replace([0, 1, 2], ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa'])

print(testresult)
print('Correct:', sum(res), '/', len(res), ':', (correct*100), '%')


def guess_flower(SepalLength, SepalWidth, PetalLength, PetalWidth) :
    Z = [[SepalLength,SepalWidth,PetalLength,PetalWidth]]
    l1 = 1 / (1 + np.exp(-(np.dot(Z, w1))))
    l2 = 1 / (1 + np.exp(-(np.dot(l1, w2))))
    np.round(l2, 3)
    yp = np.argmax(l2, axis=1)
    # yp = yp.replace([0, 1, 2], ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa'])
    if yp == 0:
        flowertype = 'Iris-virginica'
    elif yp == 1:
        flowertype = 'Iris-versicolor'
    elif yp == 2:
        flowertype = 'Iris-setosa'
    print(flowertype)


guess_flower(6.6, 3,  4.4,  1.4) # Virginica

guess_flower(5.,  3.4,  1.5,  0.2) # Setosa

guess_flower(6.5,  2.8,  4.6,  1.5) # Virginica
