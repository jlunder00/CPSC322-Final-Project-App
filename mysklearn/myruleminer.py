"""
Programmer: Jason Lunder
Class: CPCS 322-01, Spring 2021
Programming Assignment #7
4/28/21
This implements an association rule miner for generating association rules
"""
import mysklearn.myutils as myutils

class MyAssociationRuleMiner:
    """Represents an association rule miner.

    Attributes:
        minsup(float): The minimum support value to use when computing supported itemsets
        minconf(float): The minimum confidence value to use when generating rules
        X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
        rules(list of dict): The generated rules

    Notes:
        Implements the apriori algorithm
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, minsup=0.25, minconf=0.8):
        """Initializer for MyAssociationRuleMiner.

        Args:
            minsup(float): The minimum support value to use when computing supported itemsets
                (0.25 if a value is not provided and the default minsup should be used)
            minconf(float): The minimum confidence value to use when generating rules
                (0.8 if a value is not provided and the default minconf should be used)
        """
        self.minsup = minsup
        self.minconf = minconf
        self.X_train = None 
        self.rules = None

    def fit(self, X_train):
        """Fits an association rule miner to X_train using the Apriori algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)

        Notes:
            Store the list of generated association rules in the rules attribute
            If X_train represents a non-market basket analysis dataset, then: 
                Attribute labels should be prepended to attribute values in X_train
                    before fit() is called (e.g. "att=val", ...).
                Make sure a rule does not include the same attribute more than once
        """
        self.rules = myutils.apriori(X_train, self.minsup, self.minconf) 

    def print_association_rules(self):
        """Prints the association rules in the format "IF val AND ... THEN val AND...", one rule on each line.

        Notes: 
            Each rule's output should include an identifying number, the rule, the rule's support, the rule's confidence, and the rule's lift 
            Consider using the tabulate library to help with this: https://pypi.org/project/tabulate/
        """
        ce = 0
        for ruleset in self.rules:
            for rule in ruleset:
                print(str(ce), "IF", end=" ")
                ce += 1
                for i in range(len(rule[0])):
                    print(str(rule[0][i]), end=" ")
                    if i < len(rule[0])-1:
                        print("AND", end=" ")
                    else:
                        print("THEN", end=" ")
                for i in range(len(rule[1])):
                    print(str(rule[1][i]), end=" ")
                    if i < len(rule[1])-1:
                        print("AND", end=" ")
                print(rule[2]["support"], rule[2]["confidence"], rule[2]["lift"])

                
