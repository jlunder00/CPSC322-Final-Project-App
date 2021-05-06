"""
Programmer: Jason Lunder
Class: CPCS 322-01, Spring 2021
Programming Assignment #7
4/28/21
This is a set of utility functions to be used in the EDA
"""
import mysklearn.mypytable
from mysklearn.mypytable import MyPyTable
import random
import math

def group_by(X, y):
    """
    Groups attributes together
    
    Args: X, y
    
    Returns: group_list, group_ind
    """
    groups = list(set(y))
    grouping_list = [[] for g in groups]
    grouping_indicies = [[] for g in groups]
    for i in range(len(X)):
        grouping_list[groups.index(y[i])].append(X[i])
        grouping_indicies[groups.index(y[i])].append(i)
    return grouping_list, grouping_indicies

def get_column(fname, col_name):
    return MyPyTable().load_from_file(fname).get_column(col_name, False)

def categorize_given_boxes(values, bounds):
    #clever, but unusable, as it unintentionally sorts the list passed to it
    categorized_vals = []
    
    for item in values:
        for bound in bounds:
            if bound[1] == None:
                if bound[2] == None or item < bound[2]:
                    categorized_vals.append(bound[0])
                    break;
            elif bound[2] == None:
                if bound[1] <= item:
                    categorized_vals.append(bound[0])
                    break
            elif bound[1] <= item and item < bound[2]:
                categorized_vals.append(bound[0])
                break
    return categorized_vals

def weights_to_categorical(weights):
    """discretizes the weights attribute
        Args:
            weights: the weights column
        Returns:
            the discretized weights
    """ 
    cats = []
    for weight in weights:
        if weight < 2000:
            cats.append(1)
        elif weight < 2500:
            cats.append(2)
        elif weight < 3000:
            cats.append(3)
        elif weight < 3500:
            cats.append(4)
        else:
            cats.append(5)
    
    return cats

def mpg_to_rating(column):
    """discretizes the mpg attribute
        Args:
            column: the mpg column
        Returns:
            the discretized mpg attribute
    """
    ratings = []
    for it in column:
        if it >= 45:
            ratings.append(10)
        elif it >= 37:
            ratings.append(9)
        elif it >= 31:
            ratings.append(8)
        elif it >= 27:
            ratings.append(7)
        elif it >= 24:
            ratings.append(6)
        elif it >= 20:
            ratings.append(5)
        elif it >= 17:
            ratings.append(4)
        elif it >= 15:
            ratings.append(3)
        elif it >= 14:
            ratings.append(2)
        else:
            ratings.append(1)
    
    return ratings

def get_instances(table, col_names):
    return MyPyTable(col_names, table)


def mytdidt(instances, attrs, attr_doms, prev=None):
    """
    This function recursively builds a decision tree
        Args: instances, attributes, attribute_domains, prev

        Returns:
            tree
    """
    split_attr = calculate_entropy_attr(instances, attrs) 
    
    if prev == None:
        prev = len(instances)
    tree = ["Attribute", split_attr]
    

    partitions = partition_instances(instances, split_attr, attrs, attr_doms)
    attrs.remove(split_attr)

    for attr_val, partition in partitions.items():
        value_subtree = ["Value", attr_val]

        if len(partition) > 0 and all_same_classes(partition):
            leaf = ["Leaf", partition[-1][-1], len(partition), len(instances)]
            value_subtree.append(leaf)
        elif len(partition) > 0 and len(attrs) == 0:
            leaf = ["Leaf", clash(partition), len(partition), len(instances)]
            value_subtree.append(leaf)
        elif len(partition) == 0:
            tree = ["Leaf", clash(instances), len(instances), prev]
            return tree
        else:
            subtree = mytdidt(partition, attrs.copy(), attr_doms, len(instances))
            value_subtree.append(subtree)
        tree.append(value_subtree)
    return tree

   
def clash(instances):
    vals, counts = get_freq(sorted(get_col_idx(-1, instances)))
    return vals[counts.index(max(counts))]

def get_freq(col):
    values = []
    counts = []

    for value in col:
        if value not in values:
            # first time we have seen this value
            values.append(value)
            counts.append(1)
        else:
            # we have seen this value before 
            counts[-1] += 1 # ok because the list is sorted

    return values, counts 

    
def get_col_idx(idx, instances):
    return [instances[i][idx] for i in range(len(instances))]

def get_col(attr, instances, head):
    attr_idx = head.index(attr)
    return [instances[i][attr_idx] for i in range(len(instances))]

def build_head(xs):
    return ["att"+str(i) for i in range(len(xs[0]))]

def all_same_classes(partition):
    return all(x[-1] == partition[0][-1] for x in partition)

def classes(instances):
    """
    Build a list of the possible class values
        Args: instances

        Returns:
            possible_classes
    """
    possible_classes = []
    for item in instances:
        if item not in possible_classes:
            possible_classes.append(item)
    return possible_classes

def partition_instances(instances, split_attr, head, attr_doms=None):
    """
    This function partitions a tree based upon the values at the root node

    args: header, tree, instance

    returns: the partitions
    """
    attr_dom = classes(get_col(split_attr, instances, head)) if attr_doms == None else attr_doms[split_attr]
    attr_idx = head.index(split_attr)
    partitions = {}
    
    for attr_val in attr_dom:
        partitions[attr_val] = []
        for inst in instances:
            if inst[attr_idx] == attr_val:
                partitions[attr_val].append(inst[:attr_idx] + inst[attr_idx+1:])
    return partitions

def calc_ent(attr, instances, attr_idx):
    """
    calculate the entropy of the current partition

    args: instances, attribues, attribute_index

    returns: the entropy of the current partition
    """
    pos_vals = []

    val_counts = []

    for inst in instances:
        if inst[attr_idx] not in pos_vals:
            pos_vals.append(inst[attr_idx])
            val_counts.append(1)
        else:
            val_counts[pos_vals.index(inst[attr_idx])] += 1

    ent = 0.0
    if sum(val_counts) > 0:
        for val_idx, val in enumerate(pos_vals):
            classes = []
            class_counts = []
            for inst in instances:
                if inst[attr_idx] == val:
                    if inst[-1] not in classes:
                        classes.append(inst[-1])
                        class_counts.append(1)
                    else:
                        class_counts[classes.index(inst[-1])] += 1
            ne = 0.0
            for ct in class_counts:
                if ct > 0:
                    val = ct/val_counts[val_idx]
                    ne += (-val) * math.log(val, 2)
            ent += val_counts[val_idx] / sum(val_counts) * ne
    return ent


def tdidt_predict(header, tree, instance):
    """
    recursively traces the decision tree to find the predicted value given an instance

    args: header, tree, instance

    returns: the predicted value
    """
    # returns "True" or "False" if a leaf node is hit
    # None otherwise 
    info_type = tree[0]
    if info_type == "Attribute":
        # get the value of this attribute for the instance
        attribute_index = header.index(tree[1])
        instance_value = instance[attribute_index]
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                # recurse, we have a match!!
                return tdidt_predict(header, value_list[2], instance)
    else: # Leaf
        return tree[1]

def calculate_entropy_attr(instances, attrs):
    """
    calculate the entropy of a single attribute

    args: instances, attrs

    returns: the entropy of a single attribute
    """
    entropies = [calc_ent(attr, instances, attrs.index(attr)) for attr in attrs]
    return attrs[entropies.index(min(entropies))]

def tdidt_rule_gen(header, tree, rule, class_name):
    """
    Recursively generates rule strings representing the rules denoted by the decision tree

    args: header, tree, rule, class_name

    returns: print statements
    """
    info_type = tree[0]
    if info_type == "Attribute":
        rule += header[int(tree[1][3:])] if header != None else tree[1]
        
        for i in range(2, len(tree)):
            value_list = tree[i]
            tdidt_rule_gen(header, value_list[2], rule+" == "+str(value_list[1])+" AND ", class_name)
    else: # Leaf
        print(rule[:-4] + "THEN " + class_name + " == " + str(tree[1]) + " ")
        return

def print_mat(label, matrix):
    h = ""
    for i in range(len(label)):
        h += "\t"+str(label[i])
    h += "\tTotal"
    print(h)
    print("-------------------------------------------------------------")
    for i in range(len(matrix)):
        row = str(label[i])+"|"
        for item in matrix[i]:
            row += "\t"+str(item)
        row += "\t"+str(sum(matrix[i]))
        print(row)

def build_header(instances):
    header = []
    for col in range(len(instances[0])):
        header.append("att"+str(col))
    return header

def att_domain(instances, header):
    dic = {}
    for i in range(len(header)):
        vals = []
        for row in instances:
            if not row[i] in vals:
                vals.append(row[i])
        dic[header[i]] = vals
    return dic

def select_attribute(instances, attrs, header, attribute_domain):
    #need to do entropy based selection
    entropy_scores = []
    for att in attrs:
        dic = attribute_domain[att]
        row = []
        for _ in range(len(dic)):
            row.append(0)
        values = []
        classes = []
        att_index = header.index(att)
        for item in instances:
            if not item[-1] in classes:
               classes.append(item[-1])
               values.append(row.copy())
            class_idx = classes.index(item[-1])
            val_idx = dic.index(item[att_index])
            values[class_idx][val_idx] += 1
        
        ae = []
        counts = []
        for col in range(len(values[0])):
            total = 0
            vals = []
            ent = 0
            for row in range(len(values)):
                value = values[row][col]
                total += value
                vals.append(value)
            for i in vals:
                if i > 0:
                    ent += -(i/total)*(math.log2(i/total))
            ae.append(ent)
            counts.append(total)  

        ne = 0
        for idx in range(len(ae)):
            ne += (counts[idx]/len(instances))*ae[idx]
        entropy_scores.append(ne)

    index = entropy_scores.index(min(entropy_scores))
    return attrs[index]

def all_same_class(instances):
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False 
    return True

def partition_instances(instances, split_attribute, header, attribute_domains):
    # comments refer to split_attribute "level"
    attr_dom = attribute_domains[split_attribute] 
    attr_idx = header.index(split_attribute) 

    partitions = {} # key (attribute value): value (partition)
    # task: try to finish this
    for attribute_value in attr_dom:
        partitions[attribute_value] = []
        for instance in instances:
            if instance[attr_idx] == attribute_value:
                partitions[attribute_value].append(instance)
    return partitions 

def majority_leaf(partition):
    classes = []
    num_classes = []
    for element in partition:
        if not element[-1] in classes:
            classes.append(element[-1])
            num_classes.append(1)
        else:
            index = classes.index(element[-1])
            num_classes[index] += 1
    instances = max(num_classes)
    index = num_classes.index(instances)
    return classes[index], instances


def tdidt(current_instances, attrs, header, attribute_domain):
    split_attribute = select_attribute(current_instances, attrs,header,attribute_domain)
    
    attrs.remove(split_attribute) # Python is pass by object reference!!
    tree = ["Attribute", split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, header, attribute_domain)
    #print("partitions:", partitions)

    # for each partition, repeat unless one of the following occurs (base case)
    for attribute_value, partition in partitions.items():
        values_subtree = ["Value", attribute_value]
        # TODO: append your leaf nodes to this list appropriately
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and all_same_class(partition):
            #print("CASE 1")
            leaf = ["Leaf", partition[0][-1], len(partition), len(current_instances)]
            values_subtree.append(leaf)
            tree.append(values_subtree)  

        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(attrs) == 0:
            #print("CASE 2")
            val, instances = majority_leaf(partition)
            #get number of values in the partition. use insted of len(partition)
            leaf = ["Leaf", val, instances, len(current_instances)]
            values_subtree.append(leaf)
            tree.append(values_subtree)  

        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            #print("CASE 3")
            tree = []
            #get all the instances of the higher up attribute and use as partition
            #for the majority leaf and return tree
            val, instances = majority_leaf(current_instances)
            leaf = ["Leaf", val, instances, len(current_instances)]
            return leaf

        else: # all base cases are false, recurse!!
            #print("Recurse")
            #print(attrs)
            subtree = tdidt(partition, attrs.copy(),header, attribute_domain)
            values_subtree.append(subtree)
            tree.append(values_subtree)         
    return tree

def random_forest_tdidt(current_instances, attrs, header, attribute_domain, F):
    attrs = random_attribute_subset(attrs, F)
    split_attr = select_attribute(current_instances, attrs,header,attribute_domain)

    attrs.remove(split_attr) # Python is pass by object reference!!
    tree = ["Attribute", split_attr]

    partitions = partition_instances(current_instances, split_attr, header, attribute_domain)

    for attribute_value, partition in partitions.items():
        values_subtree = ["Value", attribute_value]
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and all_same_class(partition):
            leaf = ["Leaf", partition[0][-1], len(partition), len(current_instances)]
            values_subtree.append(leaf)
            tree.append(values_subtree)  

        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(attrs) == 0:
            val, instances = majority_leaf(partition)
            #get number of values in the partition. use insted of len(partition)
            leaf = ["Leaf", val, instances, len(current_instances)]
            values_subtree.append(leaf)
            tree.append(values_subtree)  

        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            tree = []
            val, instances = majority_leaf(current_instances)
            leaf = ["Leaf", val, instances, len(current_instances)]
            return leaf

        else: # all base cases are false, recurse!!
            subtree = tdidt(partition, attrs.copy(),header, attribute_domain)
            values_subtree.append(subtree)
            tree.append(values_subtree)  
         
    return tree


def classify_tdidt(tree, inst):
    pred = ""
    if tree[0] == "Attribute":
        attr = tree[1]
        index = int(attr[-1])
        val = inst[index]
        for i in range(2,len(tree)):
            if tree[i][0] == "Value":
                if tree[i][1] == val:
                    return classify_tdidt(tree[i][2], inst)
    if tree[0] == "Leaf":
        pred = tree[1]
    return pred

def print_rules(tree, attr_n, class_name, rules):
    if tree[0] == "Attribute":
        attribute = tree[1]
        index = int(attribute[-1])
        att_name = attr_n[index]
        if rules == "":
            rules+="IF "
        else:
            rules+=" AND "
        rules += str(att_name) + " == "

        for i in range(2,len(tree)):
            if tree[i][0] == "Value":
                temp = str(tree[i][1])
                print_rules(tree[i][2], attr_n, class_name, rules+temp)
    if tree[0] == "Leaf":
        rules += " THEN "+ str(class_name) +" = "+str(tree[1])
        print(rules)

def bootstrap(X, y, test=.37, train=.63):
    n = len(X)
    train = math.ceil(n*train)
    test = math.floor(n*test)
    train_set_x = []
    train_set_y = []
    test_set_x = []
    test_set_y = []
    for _ in range(train):
        rand_index = random.randrange(train)
        train_set_x.append(X[rand_index])
        train_set_y.append(y[rand_index])
    for _ in range(test):
        rand_index = random.randrange(test)
        test_set_x.append(X[rand_index])
        test_set_y.append(y[rand_index])
    return [train_set_x,train_set_y], [test_set_x,test_set_y]
    
def random_attribute_subset(attributes, F):
    # shuffle and pick first F
    shuffled = attributes[:] # make a copy
    random.shuffle(shuffled)
    return shuffled[:F]

def majority_rule(data):
    classes = []
    num_classes = []
    for element in data:
        if not element == [""]:
            if  not element[-1] in classes:
                classes.append(element[-1])
                num_classes.append(1)
            else:
                index = classes.index(element[-1])
                num_classes[index] += 1
    instances = max(num_classes)
    index = num_classes.index(instances)
    return classes[index]


def prepend_attribute_label(table, header):
    '''
    Add header names to beginning of attibute values

    args: table, header

    returns: table values modified
    '''
    for row in table:
        for i in range(len(row)):
            row[i] = header[i] + "=" + str(row[i])

def check_row_match(inst, row):
    return 1 if 0 not in [0 if item not in row else 1 for item in inst] else 0
        
def compute_unique_values(table):
    '''
    find all unique values of a table

    args: table

    returns: a sorted list of the unique values in the table
    '''
    unique = set()
    for row in table:
        for value in row: 
            unique.add(value)
    return sorted(list(unique))

def compute_k_minus_1_subsets(itemset):
    '''
    Generates all subsets of length k-1

    args: itemset

    returns: the subsets of length k-1 have been built up
    '''
    subsets = []
    for i in range(len(itemset)):
        subsets.append(itemset[:i] + itemset[i + 1:])
    return subsets

def apriori(table, minsup, minconf):
    '''
    generate a set of association rules given a table and confidence/support values

    args: table, minsup, minconf

    returns: the association rules have been determined for the dataset within the constraints of the confidence and support
    '''
    supported_itemsets = []
    I = compute_unique_values(table)
    Lk  = supported_singletons(minsup, I, table)
    k = 2
    Lkminus1 = Lk
    lks = [Lk]
    while len(Lkminus1) > 0:
        Ck = combinations(Lkminus1, k)
        Lk = prune_unsupported(Ck, minsup, table)
        Lkminus1 = Lk
        lks.append(Lk)
        k += 1
    for i in range(1, k-2):
        supported_itemsets.extend(lks[i])

    rules = generate_apriori_rules(supported_itemsets, table, minconf, k)
    return rules

def all_subsets(l):
    '''
    Generate all possible subsets of a list

    args: l

    returns: a list of subsets of the original list
    '''
    lists = [[]]
    for i in range(len(l)):
        o = lists[:]
        new = l[i]
        for j in range(len(lists)):
            lists[j] = lists[j]+[new]
        lists = o+lists
    return lists[1:]

def generate_apriori_rules(sup_itemsets, table, minconf, k):
    '''
    generate rules at the end of the apriori algorithm

    args: sup_itemsets, table, minconf, k

    returns: a list of rules to be used to print out the rules
    '''
    rules = []
    for i in range(len(sup_itemsets)):
        rules.append([])
        rhses = all_subsets(sup_itemsets[i])
        for rhs in rhses:
            lhs = [it for it in sup_itemsets[i] if it not in rhs]

            nleft, nright, nboth, ntotal = compute_rule_counts([lhs, rhs], table)
            conf = nboth/nleft
            if conf >= minconf:
                rules[i].append([lhs, rhs, compute_rule_interestingness([lhs, rhs], table)])
    return [rule for rule in rules if len(rule) > 0]

def check_row_match(terms, row):
    # returns 1 if terms is a subset of row, 0 otherwise
    for term in terms:
        if term not in row:
            return 0
    return 1

def compute_rule_counts(rule, table):
    Nleft = Nright = Nboth = 0
    Ntotal = len(table)
    for row in table:
        # add 1 to Nleft if rule["lhs"] is a subset of row
        Nleft += check_row_match(rule[0], row)
        Nright += check_row_match(rule[1], row)
        Nboth += check_row_match(rule[0] + rule[1], row)

    return Nleft, Nright, Nboth, Ntotal

def compute_rule_interestingness(rule, table):
    Nleft, Nright, Nboth, Ntotal = compute_rule_counts(rule, table)
    return {"confidence" : Nboth / Nleft,
            "support" : Nboth / Ntotal,
            "completeness" : Nboth / Nright,
            "lift" : (Nboth/Ntotal)/((Nleft/Ntotal)*(Nright/Ntotal))}

def compute_k_minus_1_subsets(itemset):
    subsets = []
    for i in range(len(itemset)):
        subsets.append(itemset[:i] + itemset[i + 1:])
    return subsets

def combinations(items, k):
    '''
    Create the candidate set Ck

    args: items, k

    returns: the candidate set of items
    '''
    combos = [items[i]+[it for it in items[j] if it not in items[i]] for i in range(len(items)-1) for j in range(i, len(items)) if k-2 <= 0 or set(items[i][:k-2]) == set(items[j][:k-2])]
    to_check = [compute_k_minus_1_subsets(c) for c in combos]
    return [combos[i] for i in range(len(combos)) if False not in [check in items for check in to_check[i]]]

def prune_unsupported(Ck, minsup, table):
    '''
    Prune the candidate set Ck based on support

    args: Ck, minsup, table

    returns: the set of supported itemsets
    '''
    return [item for item in Ck if (sum([1 if False not in [it in row for it in item] else 0 for row in table])/len(table)) >= minsup]
                


def supported_singletons(minsup, unique_vals, table):
    return [[item] for item in unique_vals if (sum([1 for row in table for val in row if val == item])/len(table)) >= minsup]


