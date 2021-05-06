# TODO: your reusable plotting functions here
import matplotlib.pyplot as plt
import importlib
import utils
import math

def bar_chart(name, column):
    plt.figure(figsize=(15, 4))
    vals = []
    x = []
    y = []
    for item in column:
        try:
            index = vals.index(item)
            y[index] = y[index] + 1
        except:
            vals.append(item)
            x.append(len(x))
            y.append(1)
    
    plt.bar(x,y)
    
    plt.title("{} Frequencies".format(name))
    plt.xlabel("Categories")
    plt.ylabel("Frequency")
     
    plt.xticks(x, vals, rotation=45, horizontalalignment="center")
    plt.show()
    
def count_bar_chart(name, table, cols):
    plt.figure()
    y = [sum(table.get_column(column)) for column in cols]
    plt.bar(cols, y)
    
    plt.title("{} Counts".format(name))
    plt.xlabel("Catagories")
    plt.ylabel("Counts")
    plt.show()
    
def pie_chart(table, cols):
    plt.figure()
    sales_totals = []
    sales_totals = [sum(table.get_column(item)) for item in cols]
    plt.pie(sales_totals, labels=cols, autopct="%1.1f%%")
    plt.show()
    
    
def histogram(table, column):
    plt.figure(figsize=(20, 4))
    plt.title("{} Distribution".format(column))
    plt.hist(sorted(utils.cast_to_float(table.get_column(column))), alpha=0.75, color="b")
    plt.show()
    
def scatter(table, x, y):
    plt.figure()
    plt.title("{} vs {}".format(x, y))
    x_data = utils.cast_to_float(table.get_column(x))
    y_data = utils.cast_to_float(table.get_column(y))
    print(len(x_data))
    print(len(y_data))
    plt.scatter(x_data, y_data)
    m, b, r, cov = compute_slope_intercept(x_data, y_data)
    #plt.plot([min(x_data), max(x_data)], [m * min(x_data) + b, m * max(x_data) + b], c="r", lw=5)
    #plt.text(0,0,"r: "+str(r)+" cov: "+str(cov), bbox={})
    plt.grid(True)
    plt.show()
    
def compute_slope_intercept(x, y):
    mean_x = sum(x)/len(x)
    mean_y = sum(y)/len(y)
    sumXYMeans = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))])
    sumXSquaredMeans = sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
    sumYSquaredMeans = sum([(y[i] - mean_y) ** 2 for i in range(len(y))])
    m = sumXYMeans/sumXSquaredMeans
    b = mean_y - m * mean_x
    r = sumXYMeans/math.sqrt(sumXSquaredMeans*sumYSquaredMeans)
    cov = sumXYMeans/len(x)
    return m, b, r, cov

def box_plot(table, x, y):

    plt.figure(figsize=(25, 4))
    genreCol = table.get_column("Genres")
    genres = x
    
    data = []
    data = [[y[i] for i in range(len(genreCol)) if genre in genreCol[i]] for genre in genres]
    plt.boxplot(data)
    labels = ["$"+genre+"$" for genre in genres]
    plt.xticks(list(range(1, len(labels)+1)), labels, visible=True)
    plt.show()