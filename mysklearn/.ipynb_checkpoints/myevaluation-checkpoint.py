import mysklearn.myutils as myutils
import numpy as np
import math

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    
    if random_state is not None:
        # TODO: seed your random number generator
        # you can use the math module or use numpy for your generator
        # choose one and consistently use that generator throughout your code
        np.random.seed(random_state)
    
    if shuffle: 
        # TODO: shuffle the rows in X and y before splitting
        # be sure to maintain the parallel order of X and y!!
        # note: the unit test for train_test_split() does not test
        # your use of random_state or shuffle, but you should still 
        # implement this and check your work yourself
        for i in range(len(X)):
            rand_index = np.random.randint(len(X))
            tmp = X[rand_index]
            X[rand_index] = X[i]
            X[i] = tmp
            tmp = y[rand_index]
            y[rand_index] = y[i]
            y[i] = tmp
    if isinstance(test_size, float):
        test_size = math.ceil(len(X)*test_size)
    split_index = len(X)-test_size
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """

    fold_indicies = [i for i in range(len(X))]
    #fold_indicies = [[i for j in range(n_splits) if i%n_splits == j] for i in range(len(X))]
    train_indicies = [fold_indicies for i in range(n_splits)]
    """
    for i in range(len(test_indicies)):
        train_indicies.append([])
        for j in range(len(fold_indicies)):
            if j != i:
                train_indicies[i].append(j)
    """
    folds = []


    n = 0
    fold_length = math.ceil(len(fold_indicies)/n_splits)

    for i in range(n_splits):
        folds.append([])
        for j in range(fold_length):
            if n < len(fold_indicies):
                folds[i].append(fold_indicies[n])
                n += 1

    train_folds = [[index for index in fold_indicies if index not in folds[i]] for i in range(len(folds))]
        
    """
    n = 0
    i = 0
    while i < len(fold_indicies):
        folds[i//n_splits].append(fold_indicies[n])
        i += 1
        n += 1
    """
    return train_folds, folds

    """
    print(folds)
    print(train_indicies, test_indicies)
    print([[fold_indicies[j] for j in range(len(fold_indicies)) if i != j] for i in train_indicies], [[fold_indicies[i]] for i in test_indicies])
    # TODO: fix this
    return [[fold_indicies[j] for j in range(len(fold_indicies)) if i != j] for i in train_indicies], [[fold_indicies[i]] for i in test_indicies] # TODO: fix this
    """
def stratified_kfold_cross_validation(X, y, n_splits=10):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    
    """
    fold_indicies = [[i for j in range(n_splits) if i%n_splits == j] for i in range(len(X))]
    test_indicies = [i for i in range(len(fold_indicies))]
    train_indicies = [[]]
    for i in range(len(test_indicies)):
        for j in range(len(fold_indicies)):
            if j != i:
                train_indicies[i].append(j)
    """ 
    grouped_values, grouped_indicies = myutils.group_by(X, y)
    flattened_groups = [val for group in grouped_values for val in group]
    flattened_indicies = [index for group in grouped_indicies for index in group]
    folds = [[] for i in range(n_splits)]


    for i in range(len(flattened_indicies)):
            folds[i%n_splits].append(flattened_indicies[i])
    
    train_folds = [[index for index in flattened_indicies if index not in folds[i]] for i in range(len(folds))]
    return train_folds, folds
                
        
    """
    fold_indicies = [[i for j in range(n_splits) if i%n_splits == j] for i in range(len(flattened_groups))]
    test_indicies = [i for i in range(len(fold_indicies))]
    train_indicies = [[]]
    for i in range(len(test_indicies)):
        for j in range(len(fold_indicies)):
            if j != i:
                train_indicies[i].append(j)
    
    return [[fold_indicies[j] for j in range(len(fold_indicies)) if i != j] for i in train_indicies], [[fold_indicies[i]] for i in test_indicies] # TODO: fix this
    folds = []
    for i in range(n_splits):
        folds.append([])
    
    
    n = 0
    for indices in index_folds:
        i = 0
        while i < len(indices):
            folds[i % n_splits].append(index_folds[n][i])
            i += 1
        n += 1
    #return [[fold_indicies[j] for j in range(len(fold_indicies) if i != j] for i in train_indicies], [[fold_indicies[i]] for i in test_indicies] # TODO: fix this
    return [], [] # TODO: fix this
    """
def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    num_labels = len(labels)
    matrix = [[sum([1 for k in range(len(y_pred)) if y_true[k] == labels[i] and y_pred[k] == labels[j]]) for j in range(num_labels)] for i in range(num_labels)]
    return matrix
    """
    for i in range(num_labels):
        for j in range(num_labels):
            sum([1 for k in range(len(y_pred)) if y_true[k] == i and y_pred[k] == j])
            for k in range(len(y_pred)):
                if y_true[k] == i and y_pred[k] == j



    return [] # TODO: fix this
    """
