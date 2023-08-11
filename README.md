# Pytorch-Data-Science-Using-Breast-Cancer-Example-Data-in-Python
Using sklearn's open source breast cancer dataset for datascience with Pytorch in Python:

    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()

and splitting the training and testing numpy inputs matrix and target output labers:

    x = data.data
    y = data.target
    import numpy as np
    def split_data(x, y):
        _split_ones = np.round(np.where(y == 1.0)[0].shape[0]*0.5, decimals=0).astype(int)
        _split_zeros = np.round(np.where(y == 0.0)[0].shape[0]*0.5, decimals=0).astype(int)
        training_indexes = np.concatenate([np.where(y == 1.0)[0][:_split_ones], np.where(y == 0.0)[0][:_split_zeros]], axis=0)
        testing_indexes = np.concatenate([np.where(y == 1.0)[0][_split_ones:], np.where(y == 0.0)[0][_split_zeros:]], axis=0)
        return x[training_indexes, :], y[training_indexes], x[testing_indexes, :], y[testing_indexes]
    
    _x_train, _y_train, _x_test, _y_test = split_data(x, y)


Pytorch provides an optimal environment for constructing neural network weights matrixes and conducting machine learning datascience; we start with constructing a class for the torch module:

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable
    
    class Model(nn.Module):
        def __init__(self, input_dimensions, hidden_dimensions, output_labels, feature_names, input_layer=None, output_layer=None):
            super().__init__()
            self.hidden_dimensions = hidden_dimensions
            self.output_labels = output_labels
            self.feature_names = feature_names
            self.input_layer = nn.Linear(input_dimensions, self.hidden_dimensions) if input_layer is None else input_layer
            self.output_layer = nn.Linear(self.hidden_dimensions, np.unique(self.output_labels).shape[0]) if output_layer is None else output_layer
    
        def forward(self, x):
            return F.softmax(self.output_layer(F.relu(self.input_layer(x))), dim=1)
    
        def get_weights(self):
            return self.input_layer.weight.detach().numpy(), self.output_layer.weight.detach().numpy()
    
        def find_significant_features(self, top_num_features=1):
            w, _ = self.get_weights()
            max_contribution_per_feature = np.reshape(np.max(w, axis=1), [w.shape[0]])
            return np.argsort(max_contribution_per_feature)[-top_num_features:]





Two tensors which we'll call input_layer and output_layer, both of which also compose the hidden units as their second dimension which in this example we'll use size 30:

![image](https://github.com/OriYarden/Pytorch-Data-Science-Using-Breast-Cancer-Example-Data-in-Python/assets/137197657/cc384bcb-d298-4df3-9653-53f4c499bd7b)


![image](https://github.com/OriYarden/Pytorch-Data-Science-Using-Breast-Cancer-Example-Data-in-Python/assets/137197657/57664a04-1859-4334-bf4b-b8b683023363)

After 1,000 training iterations:

![image](https://github.com/OriYarden/Pytorch-Data-Science-Using-Breast-Cancer-Example-Data-in-Python/assets/137197657/4d0c6f9d-041c-47e9-96ed-7bc0418394d9)


![image](https://github.com/OriYarden/Pytorch-Data-Science-Using-Breast-Cancer-Example-Data-in-Python/assets/137197657/c0527328-f0ba-442e-bd96-728d2c794280)

with the most contributing feature being "mean concave points":

![image](https://github.com/OriYarden/Pytorch-Data-Science-Using-Breast-Cancer-Example-Data-in-Python/assets/137197657/ba1c9e83-487c-4a80-b3cb-bf76537d9e11)

Now this isn't a very large dataset and we can't draw direct conclusions just because the weights representing the input features that had the highest values in the input_layer tensor; further research must be conducted to uncover potential links between features such as "mean concave points" and breast cancer diagnonsis.

Though if we plot the patients diagnosed with breast cancer (Red) and patients without breast cancer (Blue) for the 30 input features:

![image](https://github.com/OriYarden/Pytorch-Data-Science-Using-Breast-Cancer-Example-Data-in-Python/assets/137197657/21196a74-d640-4fec-921c-e3472b7657f8)

We can see that "mean concave points" feature shows significant differences for diagnoses.


[to be continued]






