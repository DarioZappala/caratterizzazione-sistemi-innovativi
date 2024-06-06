from collections import defaultdict,OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
import ast
from scipy.stats import entropy
from sklearn.linear_model import LinearRegression


pd.set_option('display.max_columns', None)
import random
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
import networkx as nx
from itertools import combinations
import igraph as ig

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
plt.rcParams.update({'font.size': 14})
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.core.common.SettingWithCopyWarning)

# ------------------------------

european_countries = ['ALB', 'AND', 'ARM', 'AUT', 'AZE', 'BLR', 'BEL', 'BIH', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA', 'GEO', 'DEU', 'GRC',
 'HUN', 'ISL', 'IRL', 'ITA', 'KAZ', 'LVA', 'LIE', 'LTU', 'LUX', 'MLT', 'MDA', 'MCO', 'MNE', 'NLD', 'MKD', 'NOR', 'POL', 'PRT', 'ROM',
 'RUS', 'SMR', 'SRB', 'SVK', 'SVN', 'ESP', 'SWE', 'CHE', 'TUR', 'UKR', 'GBR', 'VAT', 'ALA', 'GGY', 'JEY', 'FRO', 'GIB', 'GRL', 'IMN', 'SJM']

us_countries = ['USA', 'ASM', 'GUM', 'MNP', 'PRI', 'UMI', 'VIR']

groups = {
    'ExCo': ['ceo', 'cto', 'cfo', 'coo', 'cmo', 'cio', 'cso', 'cpo', 'cco', 'cro', 'svp', 'evp', 'cdo', 'cbo', 'cxo'],
    'Founder': ['founder'],
    'Engineering': ['software engineer', 'data scientist'],
    'Leadership': ['president', 'managing director', 'director', 'vp', 'chairman', 'executive director', 'general manager', 'vp engineering'],
    'Board': ['board member', 'member board director', 'board directors', 'chairman board', 'advisory board member', 'board observer'],
    'Ownership': ['owner', 'partner', 'managing partner', 'founding partner'],
    'Investor':['investor'],
    'Advisory': ['advisor', 'consultant'],
    'Other': ['member', 'team member', 'associate', 'product manager', 'principal']
}

dimension_labels = list(groups.keys())
colors = sns.color_palette("tab10", n_colors=len(dimension_labels))

# -------------------------------

def updateFitness(X,Q):

    return X @ Q

# -------------------------------

def updateComplexity(X,F):

    F_1 = 1/F
    return 1/(X.T @ F_1)

# -------------------------------

def distance(A,B):

    return np.sum(np.abs(A - B))

# -------------------------------

def fitnessComplexity(X,n_rounds=100,toll = 1e-4):

    F = np.ones(X.shape[0])
    Q = np.ones(X.shape[1])
    
    for i in range(n_rounds):
        F_tilde = updateFitness(X,Q)
        Q_tilde = updateComplexity(X,F)
        F = F_tilde / np.mean(F_tilde)
        Q = Q_tilde / np.mean(Q_tilde)

        if distance(Q,Q_tilde) < toll:
            print(f"\nConverged in {i} steps")
            break

    return F,Q

# -------------------------------

def entropyTrendPlot(l_coeff):
    plt.figure(figsize=(10,8))
    plt.plot(sorted(l_coeff,reverse=True))
    plt.title('Entropy trends coefficients')
    plt.axhline(y=0)
    plt.ylabel('Angular coefficient alpha')
    plt.xlabel('Cities')
    
# -------------------------------

def createDendogram(df,Z,threshold = 0.23):
    """
    Creates and displays a hierarchical clustering dendrogram for the given DataFrame and linkage matrix.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to be clustered. The index of the DataFrame should contain the labels (e.g., city names) to be displayed on the dendrogram.
    Z : numpy.ndarray
        The linkage matrix obtained from hierarchical clustering. This matrix is used to plot the dendrogram.
    threshold : float, optional
        The color threshold for the dendrogram. Clusters will be colored differently if their distance is above this threshold. Default value is 0.23.

    Returns:
    --------
    None
        This function does not return any value. It generates and displays a dendrogram plot.

    Notes:
    ------
    The function uses the `dendrogram` function from `scipy.cluster.hierarchy` to generate the plot. The labels for the dendrogram are taken from the index of the input DataFrame `df`.
    The x-axis labels are rotated by 80 degrees to improve readability.
    """
    
    plt.figure(figsize=(20, 8))

    # Generate the dendrogram
    dendro = dendrogram(Z, 
                        labels=df.index,
                        show_leaf_counts=True,
                        color_threshold=threshold)

    # Customize the tick label font size
    plt.setp(plt.gca().get_xticklabels(), fontsize=10)
    plt.setp(plt.gca().get_yticklabels(), fontsize=14)

    plt.title("Hierarchical Clustering Dendrogram", fontsize=16)
    plt.xlabel("Cities", fontsize=14)
    plt.ylabel("Distance", fontsize=14)
    plt.xticks(rotation=80)  # Rotate x-axis labels if necessary

    plt.tight_layout()
    plt.show()

def clean_jobs(df_jobs):
    # converte le colonne scelte da testo a Timestamp
    df_jobs['started_on'] = pd.to_datetime(df_jobs['started_on'], errors = 'coerce')
    df_jobs['ended_on']   = pd.to_datetime(df_jobs['ended_on'], errors = 'coerce')
    
    # separa il ruolo 'founder'
    df_jobs.loc[
        df_jobs['title'].str.contains('founder', case = False, na = False),
        'job_type'
    ] = 'founder'

# -------------

def plotClusters(df,
                 dimension_labels,
                 colors,
                 Z,
                 normalized=True, 
                 difference=False,
                 clusters=7,
                 bar_width = 0.1,
                 return_cities=False):
    """
    Plots clusters using hierarchical clustering and bar charts for each dimension. Optionally, returns the cities included in each cluster.

    Parameters:
    --------
    df (pd.DataFrame): 
        The DataFrame containing the data to be clustered, with cities as the index.
    dimension_labels (list): 
        List of labels for the dimensions being plotted.
    colors (list): 
        List of colors for the bars in the bar chart, corresponding to each dimension.
    Z (ndarray):  
        The linkage matrix from hierarchical clustering.
    normalized (bool, optional): 
        Whether to normalize the fingerprints of each cluster. Defaults to True.
    difference (bool, optional): 
        If True, plots the difference between the first cluster and others. Only used if normalized is True. Defaults to False.
    clusters (int, optional): 
        The number of clusters to form. Defaults to 7.
    bar_width (float): 
        Width of the bar charts.
    return_cities (bool, optional): 
        If True, returns a dictionary with clusters as keys and lists of cities as values. Defaults to False.

    Returns:
    --------
        dict (optional): A dictionary with clusters as keys and lists of cities as values, if return_cities is True.
    """
    
    # Generate cluster labels
    cluster_labels = fcluster(Z, t=clusters, criterion='maxclust')

    # Extract the fingerprint of each cluster
    fingerprints = []
    for cluster in range(1, clusters+1):
        cluster_members = df.iloc[cluster_labels == cluster]
        fingerprint = cluster_members.mean().values
        fingerprints.append(fingerprint)

    # Normalize the fingerprints if required
    if normalized:
        normalized_arrays = [array / np.sum(array) for array in fingerprints]
        if difference:
            rep_fing = normalized_arrays[0]
            normalized_arrays = [rep_fing - fing for fing in normalized_arrays]
    else:
        normalized_arrays = fingerprints

    # Determine the number of arrays and dimensions
    n_arrays = len(normalized_arrays)
    n_dimensions = len(normalized_arrays[0])

    # X locations for the groups
    indices = np.arange(n_arrays)

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

    # Optionally return the cities in each cluster
    if return_cities:
        cluster_cities = {cluster: df.index[cluster_labels == cluster].tolist() for cluster in range(1, clusters+1)}
        return cluster_cities
    
# -------------

def heatmapPlot(df,city):
    # Plot the heatmap with axes swapped
    plt.figure(figsize=(20, 6))
    sns.heatmap(df, cmap='RdBu_r', annot=False, linewidths=.002)
    plt.title(city+' Trend')
    plt.xlabel('Year')
    plt.ylabel('Groups')
    plt.xticks(rotation=45)
    plt.show()

# -------------

def entropyPlot(entropies,city):# Plot the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(range(2000, 2024), entropies)
    plt.title('Entropy over the years of city '+city)
    plt.xlabel('Year')
    plt.ylabel('Entropy')
    plt.show()

# -------------

    
def networkFingerprint(df_org_foundation, 
                        df_jobs_cleaned, 
                        year, 
                        minimum_nr_of_companies=100,
                        city_to_consider='all'):
    """
    Generates a network fingerprint of organizational relationships within cities based on shared personnel and group affiliations up to a specified year.

    Parameters:
    --------
    
    df_org_foundation (pd.DataFrame): 
        DataFrame containing organizational data with columns 'uuid', 'founded_on', 'country_code', 'city', and 'total_funding_usd'.
    df_jobs_cleaned (pd.DataFrame): 
        DataFrame containing job data with columns 'uuid', 'started_on', 'org_uuid', 'person_uuid', and 'group'.
    year (int): 
        The cutoff year up to which the data should be considered.
    city_to_consider (str, optional): 
        Specific city to consider for the analysis. Defaults to 'all' for considering all cities.

    Returns:
        pd.DataFrame: A pivot table normalized by row sums, representing the network fingerprint of organizational relationships within cities based on shared personnel and group affiliations.
    """
    
    # Filter data up to the specified year and rename columns for merging
    org_uty = filter_up_to_year(df_org_foundation, 'founded_on', year).rename(columns={'uuid': 'org_uuid'})
    jobs_uty = filter_up_to_year(df_jobs_cleaned, 'started_on', year).rename(columns={'uuid': 'person'})

    # Merge organizational and job data, and drop unnecessary columns
    df = pd.merge(
            org_uty,
            jobs_uty,
            on='org_uuid'
        ).drop(
            columns=['started_on', 'founded_on', 'total_funding_usd', 'country_code']
        ).rename(
            columns={'org_uuid': 'organization'}
        )
    
    df.explode('group').reset_index(drop=True)

    # Group the data by city
    grouped = df.groupby('city')

    results = []
    for city, group in grouped:
        if city_to_consider != 'all' and city != city_to_consider:
            continue
        if len(group) < minimum_nr_of_companies:
            continue
        org_person_dict = group.groupby('person_uuid')['organization'].apply(list).to_dict()
        org_group_dict = group.set_index('organization')['group'].to_dict()
        
        edges = set()  # Use a set to avoid duplicate edges
        for orgs in org_person_dict.values():
            if len(orgs) > 1:
                for org_pair in combinations(orgs, 2):
                    edges.add(tuple(sorted(org_pair)))  # Sort to ensure uniqueness and convert to tuple

        for org1, org2 in edges:
            group1 = org_group_dict[org1]
            group2 = org_group_dict[org2]
            for g1 in group1:
                for g2 in group2:
                    group_pair = '-'.join(sorted([g1, g2]))  # Ensure each element is a string
                    results.append((city, group_pair)) 

    # Count the group pairs
    group_pair_counts = pd.DataFrame(results, columns=['city', 'group_pair'])
    group_pair_counts['count'] = 1
    group_pair_counts = group_pair_counts.groupby(['city', 'group_pair']).sum().reset_index()

    # Pivot the table to get the desired format
    pivot_table = group_pair_counts.pivot_table(index='city', columns='group_pair', values='count', fill_value=0)

    # Sort rows by the sum of counts in descending order
    pivot_table['row_sum'] = pivot_table.sum(axis=1)
    pivot_table = pivot_table.sort_values(by='row_sum', ascending=False).drop(columns=['row_sum'])

    # Sort columns to form a triangular matrix
    sorted_columns = pivot_table.sum(axis=0).sort_values(ascending=False).index
    pivot_table    = pivot_table[sorted_columns]

    # Normalize the pivot table by row sums
    pivot_table_normalized = pivot_table.div(pivot_table.sum(axis=1), axis=0)
    
    return pivot_table_normalized

# ----------

def calculate_angular_coeff(values):
    X = np.array([i for i in range(len(values))]).reshape(-1,1)
    # Create a range of values for the independent variable (e.g., time)
    y = values

    # Create and fit the linear regression model
    model = LinearRegression().fit(X, y)

    # Get the angular coefficient (slope)
    return model.coef_[0]

# ------------

def clean_text(s):
    s = str(s).lower()
    l_of_words = ['and ','&',',','//','\\','the ','of ',
                  'co-','vice','vice-','(cro)','(',')',
                  '#','+','•','►',"'",'—','-','/','\t']
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
    return ' '.join(l)

# -----------

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

# -----------

def entropyTrend(df, s_year, e_year, df_job, df_org):
    """
    Calculate the entropy trend of organizational groups over a range of years for each city in the dataset.

    Parameters:
    --------
        
    df (pd.DataFrame): 
        DataFrame with cities as index and organizational data.
    s_year (int): 
        Start year for the range.
    e_year (int):  
        End year for the range.
    df_job (pd.DataFrame): 
        DataFrame containing job data.
    df_org (pd.DataFrame): 
        DataFrame containing organizational data.

    Returns:
    --------
    
        list: List of angular coefficients representing the entropy trend for each city.
        dict: Dictionary mapping cities to their angular coefficients.
    """
    
    l_coeff = list()  # List to store angular coefficients
    city_coeff = defaultdict(float)  # Dictionary to store city-specific angular coefficients
    
    # Loop through each city in the DataFrame index
    for city in list(df.index):
        # Skip specific cities
        try:
            if city in ['Campbell', 'Johannesburg', 'Rockville', 'Waltham']:
                continue

            # Initialize matrix to store normalized counts for each year
            matrix = np.zeros((e_year - s_year, len(dimension_labels)))
            unique_groups = dimension_labels  # List of unique groups (dimension labels)
            
            # Loop through each year in the specified range
            for year_index, start_year in enumerate(range(s_year, e_year)):
                end_year = start_year  # Set end year to the start year (single year filtering)
                
                # Filter organizational data by the year range
                df_org_filtered = filter_by_year_range(df_org, 'founded_on', start_year, end_year)
                
                # Filter job data by the year range
                df_jobs_filtered = filter_by_year_range(df_job, 'started_on', start_year, end_year)
                
                # Merge filtered organization and job data on organization UUIDs
                df_merge = pd.merge(df_org_filtered[df_org_filtered.city == city], df_jobs_filtered, left_on='uuid', right_on='org_uuid')
                
                # Explode the 'groups' column into separate rows
                exploded = df_merge.explode('group').reset_index(drop=True)
                
                # Count occurrences of unique elements grouped by 'city' and 'group'
                group_counts = exploded.groupby(['city', 'group']).size().reset_index(name='count')
                
                # Normalize the group counts by the total counts within the city
                if not group_counts.empty:
                    group_counts['normalized_count'] = group_counts['count'] / group_counts['count'].sum()
                    
                    # Assign counts to the appropriate position in the matrix
                    for i, group in enumerate(unique_groups):
                        if group in group_counts['group'].values:
                            matrix[year_index, i] = group_counts[group_counts['group'] == group]['normalized_count'].values[0]
            
            # Transpose the matrix for better processing
            transposed_matrix = matrix.T
            
            # Calculate entropy for each group of counts in the matrix
            entropies = [entropy(group_counts, base=2) for group_counts in matrix]
            
            # Calculate and store the angular coefficient of the entropy trend
            angular_coeff = calculate_angular_coeff(entropies)
            l_coeff.append(angular_coeff)
            city_coeff[city] = angular_coeff
        except:
            continue

    return l_coeff, city_coeff  # Return the list and dictionary of angular coefficients

# -----------

def heatmapPlot(df,city):
    # Plot the heatmap with axes swapped
    plt.figure(figsize=(20, 6))
    sns.heatmap(df, cmap='RdBu_r', annot=False, linewidths=.002)
    plt.title(city+' Trend')
    plt.xlabel('Year')
    plt.ylabel('Groups')
    plt.xticks(rotation=45)
    plt.show()

# -----------

def entropyPlot(entropies,city):# Plot the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(range(2000, 2024), entropies)
    plt.title('Entropy over the years of city '+city)
    plt.xlabel('Year')
    plt.ylabel('Entropy')
    plt.show()
    
# -----------

def find_job_word(df,field,word):
    return df[df[field].str.contains(word, case=False)]

# -----------

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
    
    return title

# -----------
    
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
    return title

# -----------

def remove_executives(s):
    s = str(s)
    l = s.split(' ')
    try:
        l.remove('executive')
    except:
        dummy=1
    
    return ' '.join(l)

# -----------

def move_founder_to_end(title):
    tokens = title.split()
    if 'founder' in tokens:
        tokens.remove('founder')
        tokens.append('founder')
    return ' '.join(tokens)

# -----------

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

def calculate_entropy(row):
    # Count the frequency of unique elements in the row
    unique_values, counts = np.unique(row, return_counts=True)
    
    # Calculate the probability of each unique value
    probabilities = counts / len(row)
    
    # Calculate the entropy using the Shannon entropy formula
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy

def create_category_graph(df):
    G = nx.Graph()
    for row in df.itertuples(index=False):
        categories = row.category_list.split(',')
        for cat1, cat2 in zip(categories, categories[1:]):
            cat1 = cat1.strip()
            cat2 = cat2.strip()
            if G.has_edge(cat1, cat2):
                G[cat1][cat2]['weight'] += 1
            else:
                G.add_edge(cat1, cat2, weight=1)
    return G

def networkx_to_igraph(nx_graph):
    ig_graph = ig.Graph(directed=nx_graph.is_directed())
    mapping = {name: idx for idx, name in enumerate(nx_graph.nodes())}
    
    ig_graph.add_vertices(len(mapping))
    for name, idx in mapping.items():
        ig_graph.vs[idx]['name'] = name

    edges = [(mapping[u], mapping[v]) for u, v in nx_graph.edges()]
    ig_graph.add_edges(edges)
    weights = [nx_graph[u][v]['weight'] for u, v in nx_graph.edges()]
    ig_graph.es['weight'] = weights
    
    return ig_graph

def print_random_node_properties(graph, num_nodes=10):
    # Ensure we don't try to sample more nodes than exist in the graph
    num_nodes = min(num_nodes, len(graph.nodes))

    # Sample random nodes
    random_nodes = random.sample(graph.nodes, num_nodes)

    # Print the properties of the sampled nodes
    for node in random_nodes:
        print(f"Node: {node}")
        for key, value in graph.nodes[node].items():
            print(f"  {key}: {value}")
        print()  # Print a blank line for better readability


def merge_graphs(g1, g2):
    """
    Merge two NetworkX graphs and update the weights of overlapping edges.
    
    Args:
        g1 (networkx.Graph): The first input graph.
        g2 (networkx.Graph): The second input graph.
    
    Returns:
        networkx.Graph: A new graph that is the result of merging g1 and g2.
    """
    # Create a new graph
    merged_graph = nx.Graph()
    
    # Add edges and weights from the first graph
    for u, v, data in g1.edges(data=True):
        weight = data.get('weight', 1.0)
        if merged_graph.has_edge(u, v):
            merged_graph[u][v]['weight'] += weight
        else:
            merged_graph.add_edge(u, v, weight=weight)
    
    # Add edges and weights from the second graph
    for u, v, data in g2.edges(data=True):
        weight = data.get('weight', 1.0)
        if merged_graph.has_edge(u, v):
            merged_graph[u][v]['weight'] += weight
        else:
            merged_graph.add_edge(u, v, weight=weight)
    
    return merged_graph

def filter_up_to_year(df, date_column, year_s):
    # Convert the date column to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])

    # Create the start and end date ranges
    year = pd.Timestamp(year=year_s, month=12, day=31)

    # Filter the DataFrame based on the date range
    filtered_df = df[df[date_column] <= year]

    return filtered_df

# pivot_table_normalized

# Define a custom function to convert a string representation of a list to an actual list
def str_to_list(s):
    try:
        # Use ast.literal_eval to safely evaluate the string to a list
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # If the string cannot be parsed, return it as is or handle it accordingly
        return s