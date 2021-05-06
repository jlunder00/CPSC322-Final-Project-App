# TODO: copy your mypytable.py solution from PA2-PA5 here
"""
Programmer: Jason Lunder
Class: CptS 322-01, Spring 2021
Programming Assignment #2
2/17/21
This is the MyPyTable class, which provides functionality for data storage and various data science related operations
"""
import copy
import csv 
import math
from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """
        Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col = -1;
        if type(col_identifier) == int:
            col = col_identifier
        elif type(col_identifier) == str:
            col = self.column_names.index(col_identifier)
        else:
            raise ValueError
        return [item[col] for item in self.data]

    def isNumeric(self, value):
        try:
            float(value)
            return True
        except:
            return False
    
    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        self.data = [[float(value) if self.isNumeric(value) else value for value in item] for item in self.data]

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.

        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        for row in rows_to_drop:
            self.data = [item for item in self.data if item != row]

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename, newline='') as fin:
            reader = csv.reader(fin)
            self.column_names = reader.__next__()
            self.data = [row for row in reader]
        self.convert_to_numeric()
        return self 

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, 'w', newline='') as fout:
            writer = csv.writer(fout)
            writer.writerow(self.column_names)
            writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely baed on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """
        duplicates = []
        for i in range(len(self.data)-1):
            for j in range(i+1, len(self.data)):
                match = True
                for col_name in key_column_names:
                    if self.data[i][self.column_names.index(col_name)] != self.data[j][self.column_names.index(col_name)]:
                        match = False
                if match:
                    duplicates.append(self.data[j])
                    break
        return duplicates

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        self.data = [row for row in self.data if "NA" not in row and "N/A" not in row]

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous self.data, fill missing values in a column by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col = self.column_names.index(col_name)
        nonEmptyVals = [row[col] for row in self.data if row[col] != "NA" and row[col] != "N/A"]
        avg = sum(nonEmptyVals)/len(nonEmptyVals) if len(nonEmptyVals) > 0 else 0
        self.data = [[row[i] if i != col or row[i] != "NA" else avg for i in range(len(row))] for row in self.data]

    def col_is_numeric(self, col_name):
        return not(False in [(self.isNumeric(item) or item == "NA" or item == "N/A") for item in self.get_column(col_name)])
    
    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        table = []
        summary_col_names = ["attribute", "min", "max", "mid", "avg", "median"]
        k = 0
        for i in range(len(col_names)):
            col_name = col_names[i]
            if not(self.col_is_numeric(col_name)):
                continue
            self.replace_missing_values_with_column_average(col_name)
            col = self.column_names.index(col_name) 
            columnList = [row[col] for row in self.data]
            if len(columnList) > 0:
                table.append([])
            for j in range(len(summary_col_names)):
                if len(columnList) > 0:
                    if j == 0:
                        table[k].append(col_names[i])
                    else:
                        table[k].append(self.calculate_summary_stat(columnList, summary_col_names[j]))
            k += 1
        return MyPyTable(summary_col_names, table)
    
    def calcMin(self, columnList):
        return min(columnList) if len(columnList) > 0 else 0

    def calcMax(self, columnList):
        return max(columnList) if len(columnList) > 0 else 0

    def calcMid(self, columnList):
        return (max(columnList)+min(columnList))/2 if len(columnList) > 0 else 0

    def calcAvg(self, columnList):
        return sum(columnList)/len(columnList) if len(columnList) > 0 else 0

    def calcMedian(self, columnList):
        columnList.sort()
        return (columnList[len(columnList)//2] if len(columnList)%2 == 1 else (columnList[len(columnList)//2]+columnList[(len(columnList)-1)//2])/2) if len(columnList) > 0 else 0

    def calculate_summary_stat(self, columnList, operation):
        switcher = {
            "min" : self.calcMin,
            "max" : self.calcMax,
            "mid" : self.calcMid,
            "avg" : self.calcAvg,
            "median" : self.calcMedian}
        func = switcher[operation]
        return func(columnList)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        table = []
        key_col_table_self = [[row[i] for i in range(len(self.column_names)) if self.column_names[i] in key_column_names] for row in self.data]
        key_col_table_other = [[row[i] for i in range(len(other_table.column_names)) if other_table.column_names[i] in key_column_names] for row in other_table.data]
        
        other_cols = [col for col in other_table.column_names if col not in key_column_names]
        overall_cols = self.column_names+other_cols
        
        for i in range(len(self.data)):
            for j in range(len(other_table.data)):
                if key_col_table_self[i] == key_col_table_other[j]:
                    table.append(self.data[i]+ [other_table.data[j][k] for k in range(len(other_table.column_names)) if other_table.column_names[k] not in key_column_names])

        return MyPyTable(overall_cols, table) # TODO: fix this

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        other_table.reorder_key_cols(key_column_names)
        key_col_table_self = [[row[i] for i in range(len(self.column_names)) if self.column_names[i] in key_column_names] for row in self.data]
        key_col_table_other = [[row[i] for i in range(len(other_table.column_names)) if other_table.column_names[i] in key_column_names] for row in other_table.data]
        
        other_cols = [col for col in other_table.column_names if col not in key_column_names]
        overall_cols = self.column_names+other_cols

        table = self.data
        for i in range(len(self.data)):
            for j in range(len(other_table.data)):
                if key_col_table_self[i] == key_col_table_other[j]:
                    table[i] = self.data[i]+[other_table.data[j][k] for k in range(len(other_table.column_names)) if other_table.column_names[k] not in key_column_names]
        
        for j in range(len(self.data)):
            if key_col_table_self[j] not in key_col_table_other:
                    allowed = [item in self.column_names for item in overall_cols]
                    replacement = []
                    k = 0
                    for i in range(len(allowed)):
                        if allowed[i]:
                            replacement += [self.data[j][k]]
                            k += 1
                        else:
                            replacement += ["NA"]
                    table[j] = replacement
        
        for j in range(len(other_table.data)):
            if key_col_table_other[j] not in key_col_table_self:
                allowedItems = [item in other_table.column_names for item in overall_cols]
                appendableData = []
                k = 0
                for i in range(len(allowedItems)):
                    if allowedItems[i]:
                        appendableData.append(other_table.data[j][k])
                        k += 1
                    else:
                        appendableData.append("NA")

                table.append(appendableData)
        table = [row for row in table if row != []]
        table = [[float(item) if self.isNumeric(item) else item for item in row] for row in table]
        return MyPyTable(overall_cols, table)

    def reorder_key_cols(self, key_cols):
        cur_col_indicies = [self.column_names.index(item) for item in key_cols]
        for i in range(len(cur_col_indicies)):
            for j in range(len(key_cols)):
                if self.column_names[i] == key_cols[j]:
                    self.swap_cols(self.column_names[i], self.column_names[j], i, j)

    def swap_cols(self, start_col_name, end_col_name, start_col_index, end_col_index):
        end_col = self.get_column(end_col_name)
        self.data = [[row[i] if i != end_col_index else row[start_col_index] for i in range(len(row))] for row in self.data]
        self.column_names = [self.column_names[i] if i != end_col_index else start_col_name for i in range(len(self.column_names))]
        self.data = [[self.data[i][j] if j != start_col_index else end_col[i] for j in range(len(self.data[i]))] for i in range(len(self.data))]
        self.column_names = [self.column_names[i] if i != start_col_index else end_col_name for i in range(len(self.column_names))]

    def drop_col(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data and removes it to make a new table.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            a new table missing the specified column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        column = -1
        if type(col_identifier) == int:
            column = col_identifier
        elif type(col_identifier) == str:
            column = self.column_names.index(col_identifier)
        else:
            raise ValueError
        del self.column_names[column]
        new_table = copy.deepcopy(self.data)
        for i,row in enumerate(self.data):
            new_table[i].pop(column)
        return new_table # TODO: fix this
