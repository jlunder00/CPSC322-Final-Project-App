"""
Programmer: Jason Lunder
Class: CptS 322-01, Spring 2021
Programming Assignment #6
4/15/21
A collection of classifiers to be used in EDA
"""
import mysklearn.myutils as myutils
import math
import random

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope 
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        xmean = sum([sum(x)/len(x) for x in X_train])/len([sum(x)/len(x) for x in X_train])
        ymean = sum(y_train)/len(y_train)
        self.slope = sum([(((sum(X_train[i])/len(X_train[i]))-xmean)*(y_train[i]-ymean)) for i in range(len(X_train))])/sum([((sum(x)/len(x))-xmean)**2 for x in X_train])
        
        self.intercept = ymean - self.slope*xmean

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        return [(sum(x)/len(x))*self.slope+self.intercept for x in X_test]


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test, categorical=False):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = [[0 if test_sample[j] == train_sample[j] else 1 for j in range(len(test_sample))] for train_sample in self.X_train for test_sample in X_test] if categorical == True else [[math.sqrt(sum([(test_sample[j]-train_sample[j])**2 for j in range(len(test_sample))])) for train_sample in self.X_train] for test_sample in X_test]
        neighbor_indices = []
        for i in range(len(distances)):
            neighbor_indices.append([])
            max_exclude = [item for item in distances[i]]
            for j in range(self.n_neighbors):
                neighbor_indices[i].append(max_exclude.index(max(max_exclude)))
                max_exclude = [max_exclude[k] if k != neighbor_indices[i][-1] else min(max_exclude)-1 for k in range(len(max_exclude))]
        return distances, neighbor_indices
    
    def select_class_label(self, indices):
        possible_classes = []
        for item in self.y_train:
            if item not in possible_classes:
                possible_classes.append(item)
        classification_counts = [sum([1 for j in self.y_train if i == j]) for i in possible_classes]
        return possible_classes[classification_counts.index(max(classification_counts))]



    def predict(self, X_test, categorical=False):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        distances, neighbor_indices = self.kneighbors(X_test, categorical)
        print("!!!")
        
        return [self.select_class_label(indices) for indices in neighbor_indices] # TODO: fix this
    

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.y_train = y_train
        self.X_train = X_train

        classes = list(set(self.y_train))
        self.priors = {}
        for item in classes:
            self.priors[item] = sum([1 for c in self.y_train if c == item])/len(self.y_train)
        
        self.posteriors = {}
        n_features = len(self.X_train[-1])
        attrs = [[x for x in item] for item in self.X_train]
        attrs = [[item[j] for item in attrs] for j in range(n_features)]
        possible_xs = [list(set(item)) for item in attrs]
        for c in classes:
            self.posteriors[c] = {}
            for i in range(len(attrs)):
                for x in possible_xs[i]:
                    text = "att"+str(i)+"="+str(x)
                    self.posteriors[c][text] = (sum([1 for j in range(len(attrs[i])) if attrs[i][j] == x and c == y_train[j]])/len(attrs[i]))/self.priors[c]


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        classes = [y for y in self.priors]
        y_predicted = []
        
        for item in X_test:
            
            for y in classes:
                print(y)
            for i in range(len(item)):
                print(i, item[i])
            
            probabilities = [math.prod([self.posteriors[y]["att"+str(i)+"="+str(item[i])] for i in range(len(item))])*self.priors[y] for y in classes]
            
            y_predicted.append(classes[probabilities.index(max(probabilities))])
        return y_predicted

class MyZeroRClassifier:

    def __init__(self):
        self.X_train = None 
        self.y_train = None
        self.y_predicted = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        possible_ys = list(set(y_train))

        y_counts = [sum([1 for y in y_train if y == c]) for c in possible_ys]
        self.y_predicted = possible_ys[y_counts.index(max(y_counts))]
    
    def predict(self, X_test):
        return self.y_predicted

class MyRandomClassifer:

    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.frequencies = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        possible_ys = list(set(self.y_train))
        self.frequencies = [sum([1 for y in self.y_train if y == c])/len(self.y_train) for c in possible_ys]

    def predict(self, X_test):
        possible_ys = list(set(self.y_train))
        return random.choices(possible_ys, weights=tuple(self.frequencies), k=len(X_test))

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        header = myutils.build_head([self.X_train[i] + [self.y_train[i]] for i in range(len(self.X_train))])
        train = [self.X_train[i] + [self.y_train[i]] for i in range(len(self.X_train))]

        attrs = header.copy()
        attr_doms = {}
        for attr in attrs:
            attr_doms[attr] = sorted(myutils.classes(myutils.get_col(attr, train, header)))
        self.tree = myutils.mytdidt(train, attrs[:-1], attr_doms)
       

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        header = myutils.build_head(self.X_train)
        y_predicted = [myutils.tdidt_predict(header, self.tree, instance) for instance in X_test]
        return y_predicted # TODO: fix this

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        myutils.tdidt_rule_gen(attribute_names, self.tree, "IF ", class_name)

        pass # TODO: fix this

    # BONUS METHOD
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).

        Notes: 
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this


class MyRandomForestGenerator:
    """Represents a decision tree classifier.
    Attributes:
    X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
                y_train(list of obj): The target y values (parallel to X_train).
        The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None
        self.y_train = None
        self.best_trees = []

    def fit(self, X_train, y_train, N, M, F):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
            The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
                Build a decision tree using the nested list representation described in class.
                Store the tree in the tree attribute.
                Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train

        trees = []
        for _ in range(N):
            tree = []
            r, v = myutils.bootstrap(X_train, y_train)
            
            
            header = myutils.build_header(self.X_train)
            att_doms = myutils.att_domain(self.X_train, header)
            
            train = [X_train[i] + [y_train[i]] for i in range(0,len(y_train))]
            
            attrs = header.copy()
            # initial tdidt() call
            new_tree = myutils.random_forest_tdidt(train, attrs,header,att_doms, F)
            #add tree to trees
            tree.append(new_tree)
            tree_sols = []
            for x in v[0]:
                tree_sols.append(myutils.classify_tdidt(new_tree, x))
                
            acc = 0
            for x in range(len(tree_sols)):
                if tree_sols[x] == v[1][x]:
                    acc+=1
            acc = acc/len(tree_sols)
            #add acc to the tree array
            tree.append(acc)
            trees.append(tree)
        #step 3
        for tree in trees:
            #compute acc based of of the
            if len(self.best_trees) < M:
                #add it to best_trees array
                self.best_trees.append(tree)
            else:
                for item in self.best_trees:
                    if item[1] < tree[1]:
                        index = self.best_trees.index(item)
                        self.best_trees[index] = tree
                        break
                        
                        

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
            Args:
                X_test(list of list of obj): The list of testing samples
                    The shape of X_test is (n_test_samples, n_features)

            Returns:
                y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        tree_sols = []
        for x in self.best_trees:
            new_tree = x[0]
            answers = []
            for x in X_test:
                answers.append(myutils.classify_tdidt(new_tree, x))
            tree_sols.append(answers)

        answers = [] 
        for idx in range(len(tree_sols[0])):
            values = []
            for row in tree_sols:
                values.append([row[idx]])
            result = myutils.majority_rule(values)
            answers.append(result)
        return answers
