The neural net model can be used for classification problems: the neural net classifies a large amount of incoming data into a finite number of possible classes. 

As with other machine learning techniques, the procedure to develop the model is to train the neural net on a training set of data, verify the training on a cross-validation set, and test the model on a test set. 

The neural net model consists of layers. The first layer is the input training data. The last layer is the predicted outcome layer. In between are intermediate feature layers. The number of layers and the number of features in each layer are aspects of the model. 

Neural networks are "trained" to "learn" to predict outcomes correctly by refining mathematical relationships between layers. In this model, the assumption is that the mathematical relationship from layer to layer consists of an affine transformation (similar to linear regression) composed with a sigmoid transformation (similar to logistic regression). In other words, to get from one layer to the next layer, multiply the coordinates X of the first layer by a matrix Theta of coefficients, and then obtain the coordinates of the next layer as sigmoid(Theta*X). 

The model uses mathematical optimization routines to compute the parameters of each layer of coefficients Theta. 

When outcomes consist of only two classes (yes/no, sick/healthy, 0/1) these neural net models produce probabilities for each outcome, and typically a probability of 0.5 or greater predicts an outcome of 1, and less than 0.5 predicts an outcome of 0. 

When more three or more classes are to be predicted, this neural net model uses One Vs. All classification. 

In the scripts, a simple two-layer neural network is assumed. This can easily be extended to more layers. The scripts accept variable dimensions of features and output classes. 

