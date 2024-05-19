import pandas as pd
import collections as cl

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
    df_organizations['closed_on'] = pd.to_datetime(df_organizations['closed_on'], errors = 'coerce')
    
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

def clean_text(s):
    s = str(s).lower()
    l_of_words = ['and ','&',',','//','\\','the ','of ','co-']
    for w in l_of_words:
        s = s.replace(w,'')
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
    