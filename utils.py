from collections import defaultdict,OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re

pd.set_option('display.max_columns', None)

import datetime
import numpy as np
from joblib import Parallel, delayed
from time import sleep
from utils import *
from sklearn.mixture import BayesianGaussianMixture
import scipy.stats as stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import importlib
# import utils
import warnings

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
plt.rcParams.update({'font.size': 14})
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.core.common.SettingWithCopyWarning)

def clean_jobs(df_jobs):
    # converte le colonne scelte da testo a Timestamp
    df_jobs['started_on'] = pd.to_datetime(df_jobs['started_on'], errors = 'coerce')
    df_jobs['ended_on']   = pd.to_datetime(df_jobs['ended_on'], errors = 'coerce')
    
    # separa il ruolo 'founder'
    df_jobs.loc[
        df_jobs['title'].str.contains('founder', case = False, na = False),
        'job_type'
    ] = 'founder'


def clean_organizations(df_organizations):
    # converte le colonne scelte da testo a Timestamp
    df_organizations['founded_on'] = pd.to_datetime(df_organizations['founded_on'], errors = 'coerce')
    df_organizations['closed_on']  = pd.to_datetime(df_organizations['closed_on'], errors = 'coerce')
    
    # crea la colonna 'area' con l'indicazione dell'area geografica ('Europe' o 'USA')
    dict_country_to_area = cl.defaultdict(lambda: None)
    dict_country_to_area.update(
        {c: 'USA' for c in ['USA', 'ASM', 'GUM', 'MNP', 'PRI', 'UMI', 'VIR']} |
        {c: 'Europe' for c in [
            'ALB', 'AND', 'ARM', 'AUT', 'AZE', 'BLR', 'BEL', 'BIH', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA', 'GEO', 'DEU', 'GRC',
            'HUN', 'ISL', 'IRL', 'ITA', 'KAZ', 'LVA', 'LIE', 'LTU', 'LUX', 'MLT', 'MDA', 'MCO', 'MNE', 'NLD', 'MKD', 'NOR', 'POL', 'PRT', 'ROM',
            'RUS', 'SMR', 'SRB', 'SVK', 'SVN', 'ESP', 'SWE', 'CHE', 'TUR', 'UKR', 'GBR', 'VAT', 'ALA', 'GGY', 'JEY', 'FRO', 'GIB', 'GRL', 'IMN', 'SJM']}
    )
    df_organizations['area'] = df_organizations['country_code'].map(dict_country_to_area)

# def clean_text(s):
#     s = str(s).lower()
#     l_of_words = ['and ','&',',','//','\\','the ','of ','co-']
#     for w in l_of_words:
#         s = s.replace(w,'')
#     if 'chief executive officer' in s:
#         s=s.replace('chief executive officer','ceo')
#     if 'founder ceo' in s:
#         s=s.replace('founder ceo','ceo founder') 
#     l = s.split(' ')
#     try:
#         l.remove('')
#     except:
#         dummy=1
#     # if 'ceo' in l and any(s.endswith('founder') or s.startswith('founder') for s in l):
#     #     return ' '.join(l)
#     # else:
#     #     return ''
#     return ' '.join(l)
    

def clean_text(s):
    s = str(s).lower()
    l_of_words = ['and ','&',',','//','\\','the ','of ',
                  'co-','vice','vice-','(cro)','(',')',
                  '#','+','•','►',"'",'—','-']
    for w in l_of_words:
        s=s.replace(w,'')
    if 'chief executive officer' in s:
        s=s.replace('chief executive officer','ceo')
    if 'founder ceo' in s:
        s=s.replace('founder ceo','ceo founder') 
    l = s.split(' ')
    try:
        l.remove('')
    except:
        dummy=1
    # if 'ceo' in l and any(s.endswith('founder') or s.startswith('founder') for s in l):
    #     return ' '.join(l)
    # else:
    #     return ''
    return ' '.join(l)

def group_roles(df, col_name, groups):
    """
    Group roles in a DataFrame column based on a dictionary of groups.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
        col_name (str): The name of the column containing the roles.
        groups (dict): A dictionary where keys are the group names and values are lists of roles to be included in that group.
        
    Returns:
        pandas.DataFrame: A new DataFrame with a new column 'group' containing the grouped roles as a list.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Create an empty 'group' column with a list of empty lists
    df_copy['group'] = [[] for _ in range(len(df_copy))]
    
    # Convert the roles in the dictionary to lowercase and create regex patterns
    lowercase_groups = {group_name: [re.escape(role.lower()) for role in roles] for group_name, roles in groups.items()}
    
    # Iterate over each group in the dictionary
    for group_name, roles in lowercase_groups.items():
        # Create a regex pattern to match any of the roles in the group
        pattern = '|'.join(r'\b{}\b'.format(role) for role in roles)
        
        # Find rows where the role matches any role in the group
        group_mask = df_copy[col_name].str.lower().str.contains(pattern, regex=True)
        
        # Append the group name to the 'group' column for matching rows
        df_copy.loc[group_mask, 'group'] = df_copy.loc[group_mask, 'group'].apply(lambda x: x + [group_name])
    
    # Remove duplicates from the 'group' column
    df_copy['group'] = df_copy['group'].apply(lambda x: list(set(x)))
    
    # Set the 'group' column to ['Other'] for rows where it's still empty
    df_copy.loc[df_copy['group'].str.len() == 0, 'group'] = [['Other']]
    
    return df_copy

def find_job_word(df,field,word):
    return df[df[field].str.contains(word, case=False)]

def abbreviate_title(title):
    words = title.split()
    t = False
    other_jobs = ['president','founder','director']
    if title == 'editor-in-chief' or title == 'editor in chief':
        return 'employee'

    if len(words) <= 3 :
        abbreviated = ''.join([word[0] for word in words])
        t=True
    
    if len(words) == 4 and any(word in words for word in other_jobs):
        job = list(set(words) & set(other_jobs))[0]
        words.remove(job)
        abbreviated = ''.join([word[0] for word in words])
        abbreviated += ' '+job
        t=True

    if t:
        return abbreviated
    else:
        return title
    
def abbreviate_title_round_2(title):
    words = title.split()
    t = False
    other_jobs = ['president','founder','director','executive']

    if len(words) > 2 :
        abbreviated = ''.join([word[0] for word in words])
        t=True
        
    
    if len(words) > 2 and any(word in words for word in other_jobs):
        job = list(set(words) & set(other_jobs))
        for j in job:
            words.remove(j)
        abbreviated = ''.join([word[0] for word in words])+' '
        abbreviated += ' '.join(job)
        
        t=True

    if t:
        return abbreviated
    else:
        return title

def remove_executives(s):
    s = str(s)
    l = s.split(' ')
    try:
        l.remove('executive')
    except:
        dummy=1
    
    return ' '.join(l)



def move_founder_to_end(title):
    tokens = title.split()
    if 'founder' in tokens:
        tokens.remove('founder')
        tokens.append('founder')
    return ' '.join(tokens)

# def get_final_title(row):
#     cleaned_title = row['cleaned_title']
#     if cleaned_title in chief_df.set_index('cleaned_title')['abbreviated_cleaned_title_round3'].dropna().to_dict():
#         return chief_df.set_index('cleaned_title').at[cleaned_title, 'abbreviated_cleaned_title_round3']
#     else:
#         return cleaned_title

def filter_by_year_range(df, date_column, start_year, end_year):
    """
    Filter a DataFrame to keep only rows where the date in the specified column
    falls within the given year range (inclusive).

    Args:
        df (pandas.DataFrame): The input DataFrame.
        date_column (str): The name of the column containing the dates.
        start_year (int): The start year of the range (inclusive).
        end_year (int): The end year of the range (inclusive).

    Returns:
        pandas.DataFrame: A new DataFrame containing only the rows that fall within the specified year range.
    """
    # Convert the date column to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])

    # Create the start and end date ranges
    start_date = pd.Timestamp(year=start_year, month=1, day=1)
    end_date = pd.Timestamp(year=end_year, month=12, day=31)

    # Filter the DataFrame based on the date range
    filtered_df = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]

    return filtered_df



def show_clusters(df,
                  dimension_labels,
                  colors,
                  Z,
                  normalized=True, 
                  difference=False,
                  clusters = 7):

    cluster_labels = fcluster(Z, t=clusters, criterion='maxclust')

    # Extract the fingerprint of each cluster
    fingerprints = []
    for cluster in range(1, clusters+1):
        cluster_members = df.iloc[cluster_labels == cluster]
        fingerprint = cluster_members.mean().values
        fingerprints.append(fingerprint)


    # Number of arrays and dimensions
    if normalized:
        normalized_arrays = [array / np.sum(array) for array in fingerprints]
        if difference:
            rep_fing = normalized_arrays[0]
            normalized_arrays = [rep_fing - fing for fing in normalized_arrays]
    else:
        normalized_arrays = fingerprints

    n_arrays = len(normalized_arrays)
    n_dimensions = len(normalized_arrays[0])

    # X locations for the groups
    indices = np.arange(n_arrays)

    # Width of a bar
    bar_width = 0.1

    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(20, 6))

    # Generate bars for each dimension and add labels
    for j in range(n_dimensions):
        values = [array[j] for array in normalized_arrays]
        ax.bar(indices + j * bar_width, values, bar_width, color=colors[j], label=dimension_labels[j])

    # Add some text for labels, title, and axes ticks
    ax.set_xlabel('Index')
    ax.set_ylabel('Values')
    ax.set_title('Bar Chart of Arrays')
    ax.set_xticks(indices + bar_width * (n_dimensions - 1) / 2)
    ax.set_xticklabels([f'Cluster {i+1}' for i in range(n_arrays)])

    # Create custom legend
    legend_handles = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(n_dimensions)]
    bbox_coords = (1.02, 0.5)  # x, y coordinates for the legend box
    ax.legend(legend_handles, dimension_labels, title="Dimensions", bbox_to_anchor=bbox_coords, loc='center left', borderaxespad=0.)
    # Display the plot
    plt.tight_layout()
    plt.show()
