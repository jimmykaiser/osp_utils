import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sas7bdat import SAS7BDAT
pd.options.mode.chained_assignment = None 
import seaborn as sns

pd.options.display.max_rows = 100
pd.options.display.max_columns = 100



def try_int(value):
    try:
        return int(value)
    except:
        return value
    
def read_from_sas(filename, data_directory):
    """Return a dataframe from given SAS table."""
    with SAS7BDAT(data_directory + filename + ".sas7bdat") as f:
        df = f.to_data_frame()
    return df

def round_correct(value, decimal = 0):
    """
    Round a pandas series the same way that SAS and R do
    Use with pandas: some_series.apply(round_correct, args =(number_of_decimals_to_round_to,))
    """
    reg_round = round(value, decimal)
    rounding_error = .5 * (1/10**decimal)
    val_to_add = 1 * (1/10**decimal)
    if value >= 0:
        if round(value - reg_round, decimal + 3) == rounding_error:
            return reg_round + val_to_add
        else:
            return reg_round
    else:
        if round(value - reg_round, decimal + 3) == -rounding_error:
            return reg_round - val_to_add
        else:
            return reg_round

def round_all_columns(df, n_decimals):
    for col in df.columns:
        try:
            df[col]= df[col].apply(round_correct, args = (n_decimals,))
        except:
            pass    
    return df


def get_district(dbn):
	"""Get leading two numbers of DBN as district"""
	return dbn[0:2]

def get_bn(dbn):
	"""Get trailing 4 numbers of DBN as bn"""
	return dbn[2:]

def get_borough(dbn):
	"""Get borough from dbn"""
	return dbn[2]

def create_z_score(list_of_vals):
    """Calculate z-scores for a given list"""
    group_mean = np.nanmean(list_of_vals)
    group_std = np.nanstd(list_of_vals)
    zs = [abs((x-group_mean)/group_std) for x in list_of_vals]
    return zs

def fix_ascii_col_names(df):
    """Columns from SQL sometimes start with weird characters - this fixes them"""
    df.columns = [x.encode('ascii', 'ignore').decode("UTF-8") for x in df.columns]
    
def create_bn_col(df, dbn_col = "dbn"):
    """Create bn column from given dbn column"""
    df['bn'] = df[dbn_col].apply(lambda x: x[2:])


def percentile_rank_groups_10000(df, col_to_groupby, col_to_rank, method_to_rank = "average"):
    """
    Convert numerical column to percentile rank in the same method that SAS uses
    See http://support.sas.com/documentation/cdl/en/proc/61895/HTML/default/viewer.htm#a000146840.htm
    """
    df['raw_rank'] = df[[col_to_rank, col_to_groupby]].groupby(col_to_groupby).rank(method = method_to_rank)
    df['groupby_size'] = df.groupby(col_to_groupby)[col_to_rank].transform('size')
    df['per_rank'] = ((df.raw_rank*100)/(df.groupby_size + 1)).round(1)


def autolabel(rects, ax):
    """Use to label bar plots"""
    for rect in rects:
        height = rect.get_height()
        height_val = round_correct(rect.get_height()*100,1)
        ax.text(rect.get_x() + (rect.get_width()/2), height,
                s = str(height_val) + "%",
                ha='center', va='bottom', fontsize = 10)

def return_dups(dataframe, variable):
    return dataframe[dataframe[variable].isin(dataframe[dataframe[variable].duplicated()][variable])]

def get_first_non_null(x):
    """
    Get first non null value from row
    Usage: df[cols].apply(get_first_non_null, axis = 1)
    """
    if x.first_valid_index() is None:
        return None
    else:
        return x[x.first_valid_index()]
    
def get_last_non_null(x):
    """
    Get last non null value from row
    Usage: df[cols].apply(get_last_non_null, axis = 1)
    """
    if x.last_valid_index() is None:
        return None
    else:
        return x[x.last_valid_index()]

# Use to display multiple output lines in one cell
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

def print_sprint_ids(ids):
    for id in ids:
        print("'" + str(id) + "',")

#RPSG public spreadsheet columns names
city_columns = ['grade', 'year', 'category', 'n_tested', 'scale_score', 'level_1_n', 'level_1_per', 'level_2_n', 'level_2_per',
                       'level_3_n', 'level_3_per','level_4_n', 'level_4_per', 'level_34_n', 'level_34_per']

def grouped_weighted_avg(values, weights, by):
    "Usage: grouped_weighted_avg(values=df[values_col], weights=df[weight_col], by=df[grouped_col])"
    return (values * weights).groupby(by).sum() / weights.groupby(by).sum()

def kappa(cm):
    """
    Compares your classifier with a random classifier that predicts the classes as
    often as your classifier does.
    Its values range from -1 to 1.
    If its value is positive, your classifier is doing better than chance.
    If its value is negative, then your classifier is doing worse than chance.
    """
    num_classes = len(cm)
    sum_all = 0
    sum_diag = 0
    sum_rands = 0
    for i in range(0, num_classes):
        sum_diag = sum_diag + cm[i, i]
        sum_col = 0
        sum_row = 0
        for j in range(0, num_classes):
            sum_col = sum_col + cm[j, i]
            sum_row = sum_row + cm[i, j]
            sum_all = sum_all + cm[i, j]
        sum_rands = sum_rands + sum_row * sum_col
    acc = sum_diag * 1.0 / sum_all
    rand = sum_rands * 1.0 / (sum_all * sum_all)
    return (acc - rand) / (1 - rand)