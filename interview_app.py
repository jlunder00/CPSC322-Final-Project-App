# we are going to use a micro web framework called Flask
# to create our web app (for running an API service)
import os 
import pickle 
from flask import Flask, jsonify, request 

# by default flask runs on port 5000

app = Flask(__name__)

# we need to define "routes", functions that 
# handle requests
# let's add a route for the "homepage"
@app.route("/", methods=["GET"])
def index():
    # return content and status code
    return "Welcome to my App!!", 200

# add a route for "/predict" API endpoint
@app.route("/predict", methods=["GET"])
def predict():
    # goal is to parse the query string to the args
    # query args are in the request object
    coin = request.args.get("coin", "")
    print(coin)
    print(request.args.get("att0", ""))
    level = int(request.args.get("att0", ""))
    lang = request.args.get("att1", "")
    tweets = int(request.args.get("att2", ""))
    phd = int(request.args.get("att3", ""))
    print("level:", level, lang, tweets, phd)
    # task: extract the other three parameters
    # level, lang, tweets, phd
    # make a prediction with the tree
    # respond to the client with the prediction in a JSON object
    print(type(level))
    prediction = predict_interviews_well([level, lang, tweets, phd], coin)
    if prediction is not None:
        result = {"prediction": prediction} 
        return jsonify(result), 200 
    else:
        return "Error making prediction", 400
    
def classify_tdidt(tree, instance):
    prediction = ""
    print(tree[0][0])
    if tree[0][0] == "Attribute":
        attribute = tree[0][1]
        print(attribute, tree[0])
        index = int(attribute[-1])
        print(index, instance)
        value = instance[index]
        print(value, "pls")
        for x in range(2,len(tree[0])):
            print(tree[0][x], tree[0][x][0])
            if tree[0][x][0] == "Value":
                print(tree[0][x][1], value, tree[0][x][1] == value, type(tree[0][x][1]), type(value))
                if tree[0][x][1] == value:
                    return classify_tdidt(tree[0][x][2], instance)
    if tree[0] == "Leaf":
        prediction = tree[1]
    return prediction

def majority_rule(data):
    classes = []
    num_classes = []
    for item in data:
        if not item == [""]:
            if  not item[-1] in classes:
                classes.append(item[-1])
                num_classes.append(1)
            else:
                index = classes.index(item[-1])
                num_classes[index] += 1
    instances = max(num_classes)
    index = num_classes.index(instances)
    return classes[index]

def predict(header, trees, instance):
    """Makes predictions for test instances in X_test.
            Args:
                X_test(list of list of obj): The list of testing samples
                    The shape of X_test is (n_test_samples, n_features)

            Returns:
                y_predicted(list of obj): The predicted target y values (parallel to X_test)
    """
    tree_solutions = []
    print(tree_solutions)
    for x in trees:
        new_tree = x[0]
        print(x[0])
        answers = []
        answers.append(classify_tdidt(new_tree, instance))
        tree_solutions.append(answers)
    print(tree_solutions, "Tree solutions")

    answers = [] 
    for idx in range(len(tree_solutions[0])):
        print(idx, "IN HERE")
        values = []
        for row in tree_solutions:
            print(row)
            values.append([row[idx]])
        result = majority_rule(values)
        answers.append(result)
    return answers
    

def predict_interviews_well(instance, coin):
    # I need the tree and its header in order to make a prediction for instance
    # import pickle.... depickle tree.p
    infile = open("tree.p", "rb")
    coins, header, tree = pickle.load(infile)
    infile.close()

    print("coins:", coins)
    print("header:", header)
   # print("tree:", tree)
    print(coins.index(coin))
    # traverse the tree to make a prediction
    # write a recursive algorithm to do this (predict() for PA6)
    try:
        return predict(header, [tree], instance)
    except:
        # something went wrong
        return None 


if __name__ == "__main__":
    # deployment notes
    # two main categories of deployment
    # host your own server OR use a cloud provider
    # quite a few cloud provider options... AWS, Heroku, Azure, DigitalOcean,...
    # we are going to use Heroku (backend as a service BaaS)
    # there are quite a few ways to deploy a Flask app on Heroku
    # 1. deploy the app directly on an ubuntu "stack" (e.g. Procfile and requirements.txt)
    # 2. deploy the app as a Docker container on a container "stack" (e.g. Dockerfile)
    # 2.A. build a Docker image locally and push the image to a container registry 
    # (e.g. Heroku's registry)
    # **2.B.** define a heroku.yml and push our source code to Heroku's git
    # and Heroku is going to build the Docker image (and register it)
    # 2.C. define main.yml and push our source code to Github and Github
    # (via a Github Action) builds the image and pushes the image to Heroku

    # we need to change app settings for deployment
    port = os.environ.get("PORT", 5000)
    app.run(debug=False, host="0.0.0.0", port=port)
