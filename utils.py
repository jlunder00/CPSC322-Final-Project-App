# TODO: your reusable general-purpose functions here
from mysklearn.mypytable import MyPyTable 
import matplotlib.pyplot as plt
import importlib
import os
import plot_utils
import utils
import numpy as np

def categorize_given_boxes(table, column, bounds):
    col_vals = table.get_column(column)
    col = table.column_names.index(column)
    categorized_vals = []
    for item in col_vals:
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
    table.data = [[table.data[i][j] if j != col else categorized_vals[i] for j in range(len(table.data[i]))] for i in range(len(table.data))]
    
def cast_to_float(data):
    values = []
    for item in data:
        try:
            values.append(float(item))
        except:
            item = ''.join(c for c in item if c.isdigit())
            if len(item) > 0:
                values.append(float(item))
    return values


def two_attr_difference_table(header, table, const_name, attr1, attr2, absolute_val):
    const_col = table.get_column(const_name)
    attr1_col = table.get_column(attr1)
    attr2_col = table.get_column(attr2)
    
    inverted_table = create_inverted_table(attr1_col, attr2_col, const_col, absolute_val)
    return MyPyTable(header, [[inverted_table[j][i] for j in range(len(inverted_table))] for i in range(len(table.data))])

def create_inverted_table(col1, col2, col3, absolute_val):
    return [col3, [col2[i] - col1[i] if not absolute_val else abs(col2[i] - col1[i]) for i in range(len(col1))]]

    
def sma(i, column, n):
    return [i-n, sum(column[i-n:i])/n]
    
def sma_table(header, table, col_name, num_datapoints):
    return MyPyTable(header, [sma(i, table.get_column(col_name), num_datapoints) for i in range(num_datapoints, len(table.data))])

def ema(column, prev_ema, i, m, n):
    # multilpier: 2 ÷ (number of observations + 1)]
    # EMA = Closing price x multiplier + EMA (previous day) x (1-multiplier)
    return [i-n, column[i]*m + prev_ema[-1]*(1-m)]

def ema_table(header, table, col_name, sma_size):
    prev_ema = sma(sma_size, table.get_column(col_name), sma_size)
    print(prev_ema)
    multiplier = 2/(sma_size+1)
    emas = [prev_ema]
    col = table.get_column(col_name)
    for i in range(sma_size+1, len(table.data)):
        prev_ema = ema(col, prev_ema, i, multiplier, sma_size)
        emas.append(prev_ema)
    return MyPyTable(header, emas)

def percent_change(instance1, instance2):
    return (instance2/instance1)-1


def percent_change_column(header, table, col_name1, col_name2):
    col_1 = table.get_column(col_name1)
    col_2 = table.get_column(col_name2)
    return [percent_change(col_1[i], col_2[i]) for i in range(len(table.data))]
    
def calcMedian(columnList):
    columnList.sort()
    return (columnList[len(columnList)//2] if len(columnList)%2 == 1 else (columnList[len(columnList)//2]+columnList[(len(columnList)-1)//2])/2) if len(columnList) > 0 else 0, len(columnList)//2 if len(columnList)%2 == 1 else (len(columnList)-1)//2

def calc_quartiles(column):
    column.sort()
    median, medianidx = calcMedian(column)
    first_volume, idx = calcMedian(column[:medianidx])
    third_volume, idx = calcMedian(column[medianidx:])
    return [column[0], first_volume, median, third_volume, column[len(column)-1]]
    
def percents_to_categorical(percents):
    """discretizes the percents attribute
        Args:
            percents: the percents column
        Returns:
            the discretized percents
    """ 
    cats = []
    for percent in percents:
        if percent < -0.50:
            cats.append(-10)
        elif percent < -0.45:
            cats.append(-9)
        elif percent < -0.40:
            cats.append(-8)
        elif percent < -0.35:
            cats.append(-7)
        elif percent < -0.30:
            cats.append(-6)
        elif percent < -0.25:
            cats.append(-5)
        elif percent < -0.20:
            cats.append(-4)
        elif percent < -0.15:
            cats.append(-3)
        elif percent < -0.10:
            cats.append(-2)
        elif percent < -0.05:
            cats.append(-1)
        elif percent < 0.05:
            cats.append(1)
        elif percent < 0.10:
            cats.append(2)
        elif percent < 0.15:
            cats.append(3)
        elif percent < 0.20:
            cats.append(4)
        elif percent < 0.25:
            cats.append(5)
        elif percent < 0.30:
            cats.append(6)
        elif percent < 0.35:
            cats.append(7)
        elif percent < 0.40:
            cats.append(8)
        elif percent < 0.45:
            cats.append(9)
        elif percent < 0.50:
            cats.append(10)
        else:
            cats.append(0)

    return cats

def volumes_to_categorical(volumes, list_in):
    """discretizes the volumes attribute
        Args:
            volumes: the volumes column
        Returns:
            the discretized volumes
    """ 
    cats = []
    
    for volume in volumes:
        if (list_in[0] <= volume and volume <= list_in[1]):
            cats.append(0)
        elif (list_in[1] <= volume and volume <= list_in[2]):
            cats.append(1)
        elif (list_in[2] <= volume and volume <= list_in[3]):
            cats.append(2)
        elif (list_in[3] <= volume and volume <= list_in[4]):
            cats.append(3)
        else:
            print("What have you Done???")

    return cats 



def split_on_quantiles(col, quantiles):
    lst = []
    for j in range(len(col)):
        for i in range(len(quantiles)-1):
            if quantiles[i] <= col[j] and col[j] <= quantiles[i+1]:
                lst.append(i)
    return lst
    #return [i for i in range(len(quantiles)-1) for j in range(len(col)) if quantiles[i] <= col[j] and col[j] <= quantiles[i+1]]
    
    