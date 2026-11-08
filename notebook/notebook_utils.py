import os
import pandas as pd 
from pathlib import Path
import numpy as np
import pickle
from tqdm.auto import tqdm
import seaborn as sns
from seaborn._statistics import EstimateAggregator
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import ipywidgets as ipw
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import random 
import time
from plotly.subplots import make_subplots
import pyarrow.parquet as pypq
import textwrap 
import networkx as nx 
import gc 
from datetime import datetime
import json
from box import Box
import pyarrow.dataset as ds
import pyarrow as pa
# from rich import print


plt.rcParams.update({'font.size': 22})
sns.set(style="ticks", context="talk")

pd.options.plotting.backend = 'plotly'

if pd.__version__ >= '2.1':
  print('Using PyArrow strings!')
  pd.options.future.infer_string = True
# pio.templates.default = 'seaborn+presentation'

tqdm.pandas(unit_scale=True)

#unique concepts
works_concepts_unique = pd.read_csv(os.environ['OPENALEX_CSV'] + '/concepts.csv.gz')
#dictionaries
dict_concept_id_name = works_concepts_unique[['concept_id','concept_name']].set_index('concept_id').to_dict()['concept_name']
dict_concept_name_id = works_concepts_unique[['concept_id','concept_name']].set_index('concept_name').to_dict()['concept_id']
dict_concept_id_level = works_concepts_unique[['concept_id','level']].set_index('concept_id').to_dict()['level']

# works topics
works_topics = pd.read_csv(os.environ['OPENALEX_CSV'] + '/topics.csv.gz')
dict_topic_id_name = works_topics[['topic_id', 'topic_name']].set_index('topic_id').to_dict()['topic_name']


def parse_authorships(work_id, publication_year, authorships, author_skip_ids, inst_skip_ids, inst_info_d):
    authorship_rows = []
    num_authors = 0
    has_complete_institution_info = True and (len(authorships) > 0)  # to make sure empty authorships dont trigger True

    for authorship in authorships:
        if author_id := authorship.get('author', {}).get('id'):
            author_id = convert_openalex_id_to_int(author_id)

            if author_id in author_skip_ids:  # skip if author has been deleted
                continue

            num_authors += 1  # increase the count of authors

            author_name = authorship.get('author', {}).get('display_name', pd.NA)
            raw_author_name = authorship.get('raw_author_name', pd.NA)

            institutions = []  # <- NEW way (Jan 26)
            for i in authorship.get('institutions'):
                inst_id = convert_openalex_id_to_int(i.get('id'))
                if inst_id not in inst_skip_ids:  # TODO: needs testing
                    institutions.append(i)

            institution_ids = [convert_openalex_id_to_int(i.get('id')) for i in institutions]

            institution_ids = [i for i in institution_ids if i]
            institution_ids = institution_ids or [None]
            has_complete_institution_info = has_complete_institution_info and (institution_ids != [None])

            institution_names = [i.get('display_name') for i in institutions]
            institution_names = [i for i in institution_names if i]
            institution_names = institution_names or [None]

            country_codes = [i.get('country_code') for i in institutions]
            country_codes = [i for i in country_codes if i]
            country_codes = country_codes or [None]

            lineages = [[convert_openalex_id_to_int(l) for l in i.get('lineage', [None])] for i in institutions]
            lineages = [i for i in lineages if i]
            lineages = lineages or [[None]]

            for institution_id, institution_name, country_code, lineage_inst in zip(
                    institution_ids, institution_names, country_codes, lineages,
            ):
                # level = 0
                for level, lin_inst_id in enumerate(lineage_inst):
                    inst_id = lin_inst_id

                    if inst_id in inst_skip_ids:  # old inst id has been deleted
                        continue

                    assigned_institution = (institution_id == lin_inst_id)
                    inst_name = inst_info_d['institution_name'].get(inst_id, pd.NA)
                    country_code = inst_info_d['country_code'].get(inst_id, pd.NA)

                    # raw affil string is forcibly set to pd.NA
                    raw_affil_string = authorship.get('raw_affiliation_string', pd.NA)
                    raw_affil_string = raw_affil_string if raw_affil_string != '' else pd.NA

                    authorship_rows.append({
                        'work_id': work_id,
                        'publication_year': publication_year,
                        'author_position': authorship.get('author_position', pd.NA),
                        'author_id': author_id,
                        'author_name': author_name,
                        'raw_author_name': raw_author_name,
                        'is_corresponding': authorship.get('is_corresponding', pd.NA),
                        'institution_lineage_level': level,
                        'assigned_institution': assigned_institution,
                        'institution_id': inst_id,
                        'institution_name': inst_name,
                        'raw_affiliation_string': raw_affil_string,
                        'country_code': country_code,
                    })
                    # level += 1

    return authorship_rows


def write_parquet(pandas_df, path, num_partitions, preserve_index=False, schema=None, compression='snappy', **args):
    """
    Write a pandas parquet into a directory with multiple chunks using pyarrow.dataset
    """ 
    print(f'Writing table ({len(pandas_df):,} rows) to {str(path)!r}')
    
    tic = time.time()
    max_rows_per_file = len(pandas_df)//num_partitions
    max_rows_per_group = min(500_000, max_rows_per_file)
    
    table = pa.Table.from_pandas(pandas_df, preserve_index=preserve_index)
    file_options = ds.ParquetFileFormat().make_write_options(compression='brotli')

    with tqdm(total=num_partitions, desc='Writing parquets....') as pbar:
        def update_pbar(written_file):
            pbar.set_postfix_str(Path(written_file.path).stem)
            pbar.update(1)
            return 
          
        ds.write_dataset(
            table, path,
            format='parquet', max_rows_per_file=max_rows_per_file,
            max_rows_per_group=max_rows_per_group,
            schema=schema, 
            file_options=file_options,
            file_visitor=update_pbar,
            # compression=compression,
            existing_data_behavior='delete_matching',
            # coerce_timestamps='ms', allow_truncated_timestamps=True,
        )
    toc = time.time()
    
    return 
    

def read_manifest(kind: str, snapshot_dir) -> Box:
    manifest_path = snapshot_dir / kind / 'manifest'
    create_date = datetime.fromtimestamp(manifest_path.stat().st_ctime).strftime("%a, %b %d %Y")

    raw_data = Box(json.load(open(manifest_path)))
    print(f'Reading {kind!r} manifest created on {create_date}. {len(raw_data.entries):,} files, '
          f'{raw_data.meta.record_count:,} records.')
    data = Box({'len': raw_data.meta.record_count})

    entries = []
    for raw_entry in raw_data.entries:
        filename = snapshot_dir / raw_entry.url.replace('s3://openalex/data/', '')
        entry = Box({'filename': filename, 'kind': kind,
                     'count': raw_entry.meta.record_count,
                     'updated_date': '_'.join(filename.parts[-2:]).replace('.gz', '')})
        if entry.count > 0:
            entries.append(entry)

    data['entries'] = sorted(entries, key=lambda x: x.count)

    return data


def line(error_y_mode=None, **kwargs):
    """Extension of `plotly.express.line` to use error bands."""
    ERROR_MODES = {'bar','band','bars','bands',None}
    if error_y_mode not in ERROR_MODES:
        raise ValueError(f"'error_y_mode' must be one of {ERROR_MODES}, received {repr(error_y_mode)}.")
    if error_y_mode in {'bar','bars',None}:
        fig = px.line(**kwargs)
    elif error_y_mode in {'band','bands'}:
        if 'error_y' not in kwargs:
            raise ValueError(f"If you provide argument 'error_y_mode' you must also provide 'error_y'.")
        figure_with_error_bars = px.line(**kwargs)
        fig = px.line(**{arg: val for arg,val in kwargs.items() if arg != 'error_y'})
        for data in figure_with_error_bars.data:
            x = list(data['x'])
            y_upper = list(data['y'] + data['error_y']['array'])
            y_lower = list(data['y'] - data['error_y']['array'] if data['error_y']['arrayminus'] is None else data['y'] - data['error_y']['arrayminus'])
            color = f"rgba({tuple(int(data['line']['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))},.3)".replace('((','(').replace('),',',').replace(' ','')
            fig.add_trace(
                go.Scatter(
                    x = x+x[::-1],
                    y = y_upper+y_lower[::-1],
                    fill = 'toself',
                    fillcolor = color,
                    line = dict(
                        color = 'rgba(255,255,255,0)'
                    ),
                    hoverinfo = "skip",
                    showlegend = False,
                    legendgroup = data['legendgroup'],
                    xaxis = data['xaxis'],
                    yaxis = data['yaxis'],
                )
            )
        # Reorder data as said here: https://stackoverflow.com/a/66854398/8849755
        reordered_data = []
        for i in range(int(len(fig.data)/2)):
            reordered_data.append(fig.data[i+int(len(fig.data)/2)])
            reordered_data.append(fig.data[i])
        fig.data = tuple(reordered_data)
    return fig
    

def conf_interval(data, aggfunc='mean', errorfunc=('ci', 95), 
                  return_errors=False):
    """
    data: set of values, can be a pandas series or numpy array like  
    aggfunc: function to aggregate: mean/median 
    errorfunc: ('ci', 95), 'sd' (standard dev), 'se' (standard error)
    return_errors: return difference between the means if True, else return the absolute boundaries 
    """
    data = np.array(data)
    data = data[~np.isnan(data)]   # remove NaNs
    
    if len(data) == 0:
        res = namedtuple('result', 'y ymin ymax')
        res.y = np.nan
        res.ymin = np.nan
        res.ymax = np.nan
    else:
        agg = EstimateAggregator(aggfunc, errorfunc)
        df = pd.DataFrame({'y': data})
        res = agg(df, 'y')
    
    result = res.y
    errorfunc_name = aggfunc + '_' + ''.join(map(str, errorfunc))
    
    if return_errors:
        y_error_min, y_error_max =  result - res.ymin, res.ymax - result

        ser = pd.Series({
            'count': len(data),
            aggfunc: result, 
            f'{errorfunc_name}_error_min': y_error_min,
            f'{errorfunc_name}_error_max': y_error_max})
    else:
        ser = pd.Series({aggfunc: result, 'count': len(data),
                         f'{errorfunc_name}_min': res.ymin,
                         f'{errorfunc_name}_max': res.ymax})
    return ser 



def value_counts(df, col):
    """
    better value counts. do both counts and percentags at the same time
    """
    counts_ser = (
        df
        [col]
        .value_counts()
        .sort_index()
        .rename('counts')
    )
    perc_ser = (
        df
        [col]
        .value_counts(normalize=True)
        .rename('perc')
    )
    combined_df = (
        pd.concat([counts_ser, perc_ser], axis=1)
        .sort_values(by='counts', ascending=False)
        .assign(
            perc=lambda df_: df_.perc.map('{:.2%}'.format)
        )
    )
    return combined_df


def dump_pickle(obj, path):
    if isinstance(path, str):
        path = Path(path)
    with open(path, 'wb') as fp:
        pickle.dump(obj, fp)
    print(f'Pickle saved at {str(path)!r}')
    return

def load_pickle(path):
    if isinstance(path, str):
        path = Path(path)
    assert path.exists()
    with open(path, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def read_temporal_parquet_slices(basepath, outpath, start_year, end_year, replace=False):
    """
    Read only rows within the year range [start_year, end_year]
    revamped Jul 2026, kept the min_score threshold the same as before
    """
    min_score = 0.3
    columns_d = dict(
        works=['work_id', 'doi', 'type', 'publication_date', 'publication_year', 'num_authors', 'language'],
        works_authorships=['work_id', 'author_id', 'institution_id'],
        works_concepts=['work_id', 'concept_id', 'score'],
    )
    # work_types = {'article', 'journal-article', 'proceedings-article', 'posted-content', 'book-chapter'}
    ## changed Jul 2026
    work_types = {'article', 'journal-article', 'proceedings-article', 'posted-content', 'book-chapter', 'preprint', 'review', 'conference-paper'}
    works_filters = [[
        ('publication_year', '>=', start_year), 
        ('publication_year', '<=', end_year), 
        ('num_authors', '>=', 2),    # added Jul 2026
        ('num_authors', '<=', 30),   # added Jul 2026
        ('type', 'in', work_types),  # added Jul 2026
        ('is_retracted', '=', False), # added Jul 2026
        ('is_paratext', '=', False),  # added Jul 2026
    ]]

    works_path = outpath / f'works_{start_year}_{end_year}.parquet'
    works_authors_path = outpath / f'works_authors_{start_year}_{end_year}.parquet'
    works_concepts_path = outpath / f'works_concepts_{start_year}_{end_year}.parquet'

    if works_path.exists() and not replace:
        print('Reading existing works table!')
        works_df = read_parquet(works_path)
    else:
        # read the works table first 
        works_df = read_parquet(basepath / 'works', columns=columns_d['works'], engine='pyarrow', filters=works_filters)
    
        ## filter out works without DOIs 
        works_df = (
            works_df
            .query('~doi.isna()', engine='python')
        )
        works_df = shuffle_jan1_works(works_df)
        works_df.to_parquet(works_path)
    work_ids = set(works_df.work_id)

    if works_authors_path.exists() and not replace:
        print('Reading existing works_authors table!')
        works_authors_df = read_parquet(works_authors_path)
        
    else:
        authorship_filters = [[
            ('publication_year', '>=', start_year), 
            ('publication_year', '<=', end_year), 
            ('assigned_institution', '=', True),
            ('work_id', 'in', work_ids),
        ]]
        works_authors_df = read_parquet(
            basepath / 'works_authorships', 
            columns=columns_d['works_authorships'], 
            engine='pyarrow', 
            filters=authorship_filters,
        )
        works_authors_df = (
            works_authors_df
            .merge(
                works_df[['work_id', 'publication_date', 'publication_year']],
                on='work_id',
            )
        )
        works_authors_df.to_parquet(works_authors_path)

    if works_concepts_path.exists() and not replace:
        print('Reading existing works_concepts table!')
        works_concepts_df = read_parquet(works_concepts_path)
    else:
        works_concepts_filters = [[
            ('publication_year', '>=', start_year), 
            ('publication_year', '<=', end_year), 
            ('score', '>', min_score),
            ('work_id', 'in', work_ids),
        ]]
        works_concepts_df = read_parquet(
            basepath / 'works_concepts', 
            columns=columns_d['works_concepts'], 
            engine='pyarrow', 
            filters=works_concepts_filters,
        )
        works_concepts_df = (
            works_concepts_df
            .astype({'score': 'float'})
            .merge(
                works_df[['work_id', 'publication_year']],
                on='work_id',
            )
        )
        works_concepts_df.to_parquet(works_concepts_path)
    
    return works_df, works_authors_df, works_concepts_df

def get_concept_name(concept_id):
    return dict_concept_id_name[concept_id]

def str_time_prop(start, end, time_format, prop):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formatted in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, time_format))
    etime = time.mktime(time.strptime(end, time_format))

    ptime = stime + prop * (etime - stime)

    return np.datetime64(time.strftime(time_format, time.localtime(ptime)))


def random_date(start, end, prop):
    #random_date("2020-1-1", "2020-12-31", random.random())
    return str_time_prop(start, end, '%Y-%m-%d', prop)

def shuffle_jan1_works(works, seed=0):
    """
    Shuffle the works on Jan 1 randomly to the remaining days of the year

    seed: int (default 0) -> reproducible. Same input frame always yields the same dates.
          Pass seed=None for the old non-deterministic behaviour.

    NOTE: this function was unseeded before Sep 2026, so any parquet written by an earlier
    run cannot be reproduced -- those dates are gone. A seed only makes runs from here on
    repeatable; it does not recover an older shuffle.
    """
    rng = np.random.default_rng(seed)
    years_list = list(range(works.publication_year.min(), works.publication_year.max()+1))
    works.loc[:, 'shuffled_date'] = False

    for year in tqdm(years_list):
        date_1jan = np.datetime64(str(year)+"-01-01")
        ix = works['publication_date'] == date_1jan
        num_1jan = int(ix.sum())
        if num_1jan == 0:
            continue
        # Vectorized equivalent of the old per-row
        #   [random_date(f'{year}-01-01', f'{year}-12-31', random.random()) for _ in range(n)]
        # random.random() is [0, 1), so the old code could reach Jan 1 .. Dec 30 but never
        # Dec 31; span_days below reproduces exactly that range. ~10x faster -- the old
        # version also built a full `.query()` copy just to take its len().
        span_days = (np.datetime64(str(year)+"-12-31") - date_1jan).astype('timedelta64[D]').astype(int)
        offsets = rng.integers(0, span_days, size=num_1jan).astype('timedelta64[D]')
        works.loc[ix, 'shuffled_date'] = True
        works.loc[ix, 'publication_date'] = date_1jan + offsets

    works = (
        works
        .assign(
            publication_year=lambda df_: df_.publication_date.dt.year,   # for some publication_year doesnt match with the year in the publication date
            publication_month=lambda df_: df_.publication_date.dt.month, 
        )
    )
    
    return works

def get_monthly_works_authors_stats(
    topic, month_idx, date, 
    topic_works_authors, topic_newcomers_month, 
    topic_neophytes_month, topic_bellwethers_month,
    topic_experts_month, topic_repeat_authors_month, trial=0,
):
    """
    get work stats for topics work authors 
    """
    monthly_works_author_stats = (
        topic_works_authors
        .groupby('work_id', sort=False, as_index=False)
        .agg({
            'work_id': 'first',
            'author_id': [
                'count', 
                lambda x: len(set(x) & topic_newcomers_month), 
                lambda x: len(set(x) & topic_neophytes_month), 
                lambda x: len(set(x) & topic_experts_month),
                lambda x: len(set(x) & topic_repeat_authors_month),
                lambda x: len(set(x) & topic_bellwethers_month),
            ]
        })
        .droplevel(0, axis=1)
        .rename(columns={'first': 'work_id', 'count': 'num_authors', 
                         '<lambda_0>': 'num_newcomers', '<lambda_1>': 'num_neophytes', 
                         '<lambda_2>': 'num_experts', '<lambda_3>': 'num_repeats', '<lambda_4>': 'num_bellwethers'})
        .assign(month_idx=month_idx, topic=topic, date=date)
    )
    return monthly_works_author_stats


def shuffle_works_authors(topic, topic_works_authors, trials=5, max_tries=100):
    """
    Try to shuffle works authors table keeping the following same
    * number of authors per work
    * number of works per author
    """
    topic_works_authors = (
        topic_works_authors
        .assign(
            orig_num_authors=lambda df_: df_.groupby('work_id', sort=False).author_id.transform('count'),  # orig number of authors per paper
        )
    )
    shuffled_dfs = []
    # max_tries = len(topic_works_authors) // 2
    
    mismatches = []
    with tqdm(range(trials), desc=f'Shuffling {len(topic_works_authors):,} authorships', leave=False, position=0) as pbar:    
        for trial in range(trials):
            rng = np.random.default_rng()
            tries = 0
            properly_shuffled = False
            best_shuffle, best_score = None, float('inf')
            with tqdm(total=max_tries, desc='Tries', leave=False, position=1) as pbar2:
                for tries in range(max_tries):
                    shuffled_works_authors = (    
                        topic_works_authors
                        .assign(
                            # shuffled_author_id=lambda df_: np.random.permutation(df_.author_id),  # shuffle the author ID column
                            shuffled_author_id=lambda df_: rng.permutation(df_.author_id),  # shuffle the author IDs        
                            shuffled_num_authors=lambda df_: df_.groupby('work_id', sort=False).author_id.transform('nunique'))
                    )
    
                    duplicate_authorships = (
                        shuffled_works_authors.loc[:, ['work_id', 'shuffled_author_id']].duplicated().sum()  # find duplicate rows with same work_id, shuffled_author_id
                    )
    
                    author_works_count = (
                        shuffled_works_authors
                        .groupby('author_id', sort=False)
                        .work_id
                        .count()
                        .rename('works_count')
                        .to_frame()
                        .join(
                            shuffled_works_authors
                            .groupby('shuffled_author_id', sort=False)
                            .work_id
                            .nunique()
                            .rename('shuffled_works_count')
                        )
                    )
                    authors_work_count_mismatch = (
                        author_works_count
                        .query('works_count!=shuffled_works_count')  # make sure the number of works per author stays the same
                    )
                    mismatches.append((duplicate_authorships, len(authors_work_count_mismatch), duplicate_authorships+len(authors_work_count_mismatch)))
                    properly_shuffled = (duplicate_authorships == 0) and (len(authors_work_count_mismatch) == 0)
                    total_error = duplicate_authorships + len(authors_work_count_mismatch)

                    if total_error < best_score:
                        best_score = total_error
                        best_shuffle = shuffled_works_authors.copy()
                    
                    pbar2.set_postfix_str(f'{tries=:,} dupes {duplicate_authorships:,} count misses {len(authors_work_count_mismatch):,}')
                    pbar2.update(1)
                    
                    if properly_shuffled:
                        break

            if tries > 1:
                pbar.set_postfix_str(f'Latest error: {best_score:,}')
                # print(f'Shuffling {len(topic_works_authors):,} authorships took {tries:,} tries. Error: {best_score:,}')
        
            assert best_shuffle is not None
            
            shuffled_dfs.append((
                best_shuffle
                [['work_id', 'shuffled_author_id']]
                .rename(columns={'shuffled_author_id': 'author_id'})
                .drop_duplicates(subset=['work_id', 'author_id'])  # drop duplicated authorship rows 
                .assign(topic=topic, trial=trial+1)
            ))
            
            if tries == max_tries:
                print(f'Exceeded {max_tries:,} tries to shuffle')
                print(f'mismatches: {sorted(mismatches, key=lambda x: x[-1])[: 5]}')
            pbar.update(1)

    return shuffled_dfs
    

def shuffle_works_authors_old(topic,  
                          topic_works_authors,
                          trials=5):    
    # degree sequences bipartite graph: authors productivity and team sizes
    deg_seq_authors_M = (
        topic_works_authors
        .groupby('author_id', sort=False)
        .work_id
        .count()
        .to_frame()
    )
    deg_seq_teams_M = (
        topic_works_authors
        .groupby('work_id', sort=False)
        .author_id
        .count()
        .to_frame()
    )
    
    #corrispondence source and author_id
    author_id_list_M = list(deg_seq_authors_M.index)
    author_id_dict_M = dict(zip(range(len(author_id_list_M)),author_id_list_M))
    
    work_id_list_M = list(deg_seq_teams_M.index)
    work_id_dict_M = dict(zip(range(len(author_id_list_M), len(author_id_list_M)+len(deg_seq_teams_M)), 
                              work_id_list_M))
    
    
    
    #for each month
    deg_seq_authors_M = list(deg_seq_authors_M['work_id'])
    deg_seq_teams_M = list(deg_seq_teams_M['author_id'])

    #sample baseline graphs
    shuffled_dfs = []
    for seed_ in tqdm(range(trials), desc=f'randomizing {len(topic_works_authors):,} authorships', leave=False): #number of samples random baseline
        shuffled_works_authors = (
            random_baseline(deg_seq_authors_M, deg_seq_teams_M, author_id_dict_M, work_id_dict_M, seed_)
            .assign(topic=topic, trial=seed_+1)
        )
        
        # monthly_author_count = get_monthly_works_authors_stats(
        #         topic=topic, month_idx=month_idx, date=month_end_date, 
        #         topic_works_authors=shuffled_works_authors, 
        #         topic_newcomers_month=newcomers_covid_month,
        #         topic_neophytes_month=neophytes_month, 
        #         topic_experts_month=covid_expert_month,
        #         topic_repeat_authors_month=covid_repeat_authors_month,
        #     )
        shuffled_dfs.append(shuffled_works_authors)
    
    return shuffled_dfs


def random_baseline(deg_seq_authors_M, deg_seq_teams_M, 
                    author_id_dict_M, work_id_dict_M,
                    seed_):
    """
    deg_seq_authors_M: degree sequence for authors in month M [number of papers]
    deg_seq_teams_M: degree sequence for works in month M [number of authors per work]
    author_id_dict_M: mapping between author_ids -> int starting at 0 
    seed_: seed for the random number 
    """
    #2 possibilities: repetitions or not exact same distribution
    G_conf = nx.bipartite.configuration_model(aseq=deg_seq_authors_M, bseq=deg_seq_teams_M, seed=seed_)
    
    df_conf = nx.to_pandas_edgelist(G_conf) #list edges
    source_set = set(df_conf['source'])
    target_set = set(df_conf['target'])
    
    df_conf_dupl = df_conf[df_conf.duplicated()]#.reset_index(drop=True) #duplicated edges
    df_conf_dupl_indexes = list(df_conf_dupl.index)
    df_conf_dupl_len = len(df_conf_dupl)

    r = 0
    while df_conf_dupl_len!=0:
        index_i = df_conf_dupl_indexes[0] #first row duplicated 
        row = df_conf_dupl.loc[[index_i]]
        author_id = row['source'].values[0]
        work_id = row['target'].values[0]
        work_id_neig = set(G_conf.neighbors(work_id))
        cond = True

        while cond: #while find work such that set of authors intersection empty 
            random.seed(r)
            work_id_random = random.sample(list(target_set), 1)[0] #chose other work_id that month randomly
            work_id_random_neig = set(G_conf.neighbors(work_id_random))
            if len(work_id_neig.intersection(work_id_random_neig)) != 0: 
                #if intersection authors not empty, select randomly another work
                cond = True
                r += 1
            else: #swap the duplicated edge with a random one of this work 
                author_id_random = random.sample(list(work_id_random_neig), 1)[0]
                index_random = df_conf.query('source==@author_id_random & target==@work_id_random').index[0]   
                # delate duplicate edge #delate random edge work random 
                df_conf = df_conf.drop([index_i,index_random]) #
                df_conf = df_conf.reset_index(drop=True)
                #add new edges
                df_conf.loc[len(df_conf)] = [author_id_random,work_id]
                df_conf.loc[len(df_conf)] = [author_id,work_id_random]
                #modify graph
                G_conf.remove_edges_from([(author_id,work_id),(author_id_random,work_id_random)])
                G_conf.add_edges_from([(author_id_random,work_id),(author_id,work_id_random)])
                cond = False
                r += 1

        df_conf = df_conf.reset_index(drop=True)
        df_conf_dupl = df_conf[df_conf.duplicated()]#.reset_index(drop=True) #duplicated edges
        df_conf_dupl_indexes = list(df_conf_dupl.index)
        df_conf_dupl_len = len(df_conf_dupl)
        df_conf_dupl_len = 0

    # #check no duplicates
    # df_conf_dupl = df_conf[df_conf.duplicated()].reset_index(drop=True) #duplicated edges
    # display(df_conf_dupl)
    
    #check same degree distribution
    deg_seq_authors_check = list((
        df_conf
        .groupby('source') #.groupby('author_id')
        .target
        .count()
        .to_frame()
    )['target'])
    deg_seq_teams_check = list((
        df_conf
        .groupby('target')
        .source #.author_id
        .count()
        .to_frame()
    )['source']) #['author_id'])

    assert np.array_equal(np.sort(np.array(deg_seq_authors_M)),np.sort(np.array(deg_seq_authors_check)))
    assert np.array_equal(np.sort(np.array(deg_seq_teams_M)),np.sort(np.array(deg_seq_teams_check)))

    #correspondent author_id
    df_conf['author_id'] = df_conf['source'].map(author_id_dict_M)
    df_conf['work_id'] = df_conf['target'].map(work_id_dict_M)

    shuffled_df = (
        df_conf
        .drop(columns=['source', 'target'])
        [['work_id', 'author_id']]
    )
    
    return shuffled_df


def combine_and_assign_phases_author_classes(df, start_date_ow, end_date_ow):
    """
    assign phase and author classes to dfs
    """
    if isinstance(df, list): 
        df = pd.concat(df)
        
    df = (
        df
        .assign(
            date=lambda df_: pd.to_datetime(df_.date, infer_datetime_format=True),
            # phase=lambda df_: pd.cut(
            #     df_.date,
            #     pd.period_range(start=str(start_date_ow), end=str(end_date_ow), freq='6M').to_timestamp(),
            #     labels=['Phase I', 'Phase II', 'Phase III', 'Phase IV', 'Phase V', 'Phase VI'],
            #     include_lowest=True,
            # ),
            
        )
    )
    
    if 'work_id' in df.columns and 'num_experts' in df.columns:
        df = (
            df
            .assign(
                frac_expert=lambda df_: df_.num_experts.div(df_.num_authors),
                class_=lambda df_ : pd.cut(
                    df_.frac_expert,
                    bins=[0, 0.000001, 0.25, 0.5, 0.75, 0.999, 1],
                    include_lowest=True,
                    labels=['0%', '1-25%', '26-50%', '51-75%', '76-99%', '100%'],
                ),
            )
        )
        
    return df
    

def read_topic_stat_csvs(topic, csv_path, shuffles=20):
    """
    read monthly stats, works authors summary, and shuffled authorships 
    """
    print(f'{topic=} from {csv_path.stem!r}')
    
    monthly_stats = pd.read_csv(csv_path / f'{topic}.csv', parse_dates=['date'])
    display(monthly_stats.head(2))

    works_authors_stats = pd.read_csv(csv_path / f'{topic}_author_classes.csv', parse_dates=['date'])
    display(works_authors_stats.head(2))
    
    suffix = '.gz' if shuffles != 10 else ''
    shuffled_path = csv_path / f'{topic}_randomized_author_classes-{shuffles}.csv{suffix}'
    
    try:
        shuffled_works_authors_stats = pd.read_csv(shuffled_path, engine='c', parse_dates=['date'])
    
        display(shuffled_works_authors_stats.head(2))
    except pd.errors.EmptyDataError:
        shuffled_works_authors_stats = pd.DataFrame()
    
    return monthly_stats, works_authors_stats, shuffled_works_authors_stats


def get_topic_info(topic:str, ow_duration: int = 3, tw_duration: int = 5, verbose=True):
    """
    get topic_ids, prior experience topic ids, 
    tw_duration: training window duration in years
    ow_duration: observation window duration in years
    """
    topic_ = topic.replace('-noJan1', '')
    if topic_ == 'COVID':
        topic_ids = {3008058167,3006700255,3007834351}
        topic_ids_exp = {107130276, 89623803, 116675565, 524204448}
        start_date_ow = pd.to_datetime('2020-01-01')

    elif topic_ == 'COVID_524204448':
        topic_ids = {3008058167,3006700255,3007834351}
        topic_ids_exp = {107130276, 89623803, 116675565}
        start_date_ow = pd.to_datetime('2020-01-01')

    elif topic_ == 'COVID_116675565':
        topic_ids = {3008058167,3006700255,3007834351}
        topic_ids_exp = {107130276, 89623803, 524204448}
        start_date_ow = pd.to_datetime('2020-01-01')

    elif topic_ == 'COVID_89623803':
        topic_ids = {3008058167,3006700255,3007834351}
        topic_ids_exp = {107130276, 116675565, 524204448}
        start_date_ow = pd.to_datetime('2020-01-01')
    
    elif topic_ == 'COVID_107130276':
        topic_ids = {3008058167,3006700255,3007834351}
        topic_ids_exp = {89623803, 116675565, 524204448}
        start_date_ow = pd.to_datetime('2020-01-01')

    elif topic_ == 'COVID_PH':
        topic_ids = {3008058167,3006700255,3007834351}
        topic_ids_exp = {107130276, 89623803, 116675565, 524204448, 138816342}  # public health 
        start_date_ow = pd.to_datetime('2020-01-01')

    elif topic_ == 'COVID_MED_PH':
        topic_ids = {3008058167,3006700255,3007834351}
        topic_ids_exp = {71924100, 138816342}  # medicine and public health only
        start_date_ow = pd.to_datetime('2020-01-01')
          
    elif topic_ == 'DL':  # deep learning
        topic_ids = {108583219, 147168706, 81363708, 2988773926, 66322947}    # transformers: 66322947
        topic_ids_exp = {50644808,179717631,136389625,8038995}
        start_date_ow = pd.to_datetime('2017-01-01') 
        
    elif topic_ == 'DL_2020':  # deep learning but starts in 2020
        topic_ids = {108583219, 147168706, 81363708, 2988773926, 66322947}  
        topic_ids_exp = {50644808,179717631,136389625,8038995}
        start_date_ow = pd.to_datetime('2020-01-01')
        
    elif topic_ == 'DL_ML':  # deep learning with ML 
        topic_ids = {108583219, 147168706, 81363708, 2988773926, 66322947}  
        topic_ids_exp = {119857082}
        # topic_ids_exp = {50644808,179717631,136389625,8038995}
        start_date_ow = pd.to_datetime('2017-01-01') 
    
    elif topic_ == 'StatPhy':
        topic_ids = {121864883}
        topic_ids_exp = {149288129, 51329190, 202213908, 29912722, 99874945, 151342819}
        start_date_ow = pd.to_datetime('2017-01-01') 

    elif topic_ == 'StatMech':
        topic_ids = {99874945}
        topic_ids_exp = {121864883}
        start_date_ow = pd.to_datetime('2017-01-01') 
    
    elif topic_ == 'QED':  # quantum electrodynamics
        topic_ids = {3079626}       # quantum electrodynamics
        topic_ids_exp = {62520636}  # quantum mechanics
        start_date_ow = pd.to_datetime('2017-01-01') 
        
    elif topic_ == 'Cryptography':
        topic_ids = {203062551}     # cryptography
        topic_ids_exp = {38652104}  # computer security
        start_date_ow = pd.to_datetime('2017-01-01') 
     
    elif topic_ == 'H1N1':  
        topic_ids = {2910793863, 3017774372, 2777546802}  # C2777546802 is influenza type A 
        # topic_ids = {2910793863, 3017774372}
        topic_ids_exp = {107130276, 89623803, 116675565, 524204448} #same COVID 
        start_date_ow = pd.to_datetime('2009-01-01')
        
    elif topic_ == 'HIV':  
        topic_ids = {2780195530, 3013748606}  
        topic_ids_exp = {107130276, 89623803, 116675565, 524204448} #same COVID 
        start_date_ow = pd.to_datetime('1981-01-01')
        
    elif topic_ == 'Ebola':  
        topic_ids = {2777469322,2909623084,2910344980,2778933410}   # 22 is Ebola virus, 84 is Ebolavirus, 980: ebola hemorrhagic, 10: ebola vaccine
        topic_ids_exp = {107130276, 89623803, 116675565, 524204448} #same COVID 
        start_date_ow = pd.to_datetime('2014-01-01')
        
    elif topic_ == 'Zika':  
        topic_ids = {2777053367}  
        topic_ids_exp = {107130276, 89623803, 116675565, 524204448} #same COVID 
        start_date_ow = pd.to_datetime('2015-01-01')
        
    elif topic_ == 'MERS': 
        topic_ids = {2776525042,2777691041, 2777648638, 2778137277}   # 38: Coronavirus, 77: Betacoronavirus
        topic_ids_exp = {107130276, 89623803, 116675565, 524204448} #same COVID 
        start_date_ow = pd.to_datetime('2012-01-01')
        
    elif topic_ == 'SARS':  
        topic_ids = {3020799909, 2778859668, 3007834351, 2777648638, 2778137277}  # C2778859668: SARS 
        # topic_ids = {3020799909, 2778859668}  # C2778859668: SARS 
        topic_ids_exp = {107130276, 89623803, 116675565, 524204448} #same COVID 
        start_date_ow = pd.to_datetime('2002-01-01')
        
    else:
        raise NotImplementedError(f'{topic}')

    # end_date_ow = start_date_ow + pd.DateOffset(years=ow_duration)
    end_date_ow = pd.to_datetime(f'{start_date_ow.year + 2}-12-31')
    observation_window = pd.date_range(start_date_ow, end_date_ow, freq='ME')

    end_date_tw = start_date_ow  # end date of exposure window is the start date of OW
    start_date_tw = start_date_ow - pd.DateOffset(years=tw_duration)
    end_date_tw = end_date_tw - pd.DateOffset(days=1) # end one day earlier 
    training_window = pd.date_range(start_date_tw, end_date_tw, freq='ME')

    if verbose:
        print(f'{topic!r} Core topics')
        for id in topic_ids:
            print(f'\t{id}, {dict_concept_id_name[id]} (level {dict_concept_id_level[id]})')
    
        print(f'\n{topic!r} Prior experience topics:')
        for id in topic_ids_exp:
            print(f'\t{id}, {dict_concept_id_name[id]} (level {dict_concept_id_level[id]})')
        
        print(f'Training window: {start_date_tw.to_period("D")} to {end_date_tw.to_period("D")}')
        print(f'Observation window: {start_date_ow.to_period("D")} to {end_date_ow.to_period("D")}')
    
    return topic_ids, topic_ids_exp, start_date_tw, end_date_tw, start_date_ow, end_date_ow


def convert_openalex_id_to_int(openalex_id):
    if not openalex_id:
        return None
    try:
        openalex_id = openalex_id.strip().replace('https://openalex.org/', '')
        id_ = int(openalex_id[1:])
    except Exception:
        id_ = None
    return id_


def read_parquet_v2(path, engine='pyarrow', convert_dtypes=True, quiet=False, **args):
    """
    Read a parquet file
    """
    if isinstance(path, str):
        path = Path(path)

    if 'filters' in args:
        engine = 'pyarrow'
    
    if not path.name.endswith('.parquet'):
        ## check if a file exists without the extension 
        dir_exists = path.exists()
        if not dir_exists:  # try adding the parquet extension
            if path.with_suffix('.parquet').exists():
                path = path.with_suffix('.parquet')
                
    name = path.stem
    
    if not quiet:
        print(f'\n(v2) Reading {name!r} from {str(path)!r} using {engine=}')
    
    tic = time.time()
    ## read using pyarrow 
    # df = pd.read_parquet(path, engine=engine, **args)
    # filters = args.get('filters')
    # columns = args.get('columns')
    
    table = pypq.read_table(
        path, memory_map=True, **args,
    )

    dtype_mapping = {
        pa.int8(): pd.Int8Dtype(),
        pa.int16(): pd.Int16Dtype(),
        pa.int32(): pd.Int32Dtype(),
        pa.int64(): pd.Int64Dtype(),
        pa.uint8(): pd.UInt8Dtype(),
        pa.uint16(): pd.UInt16Dtype(),
        pa.uint32(): pd.UInt32Dtype(),
        pa.uint64(): pd.UInt64Dtype(),
        pa.bool_(): pd.BooleanDtype(),
        pa.float32(): pd.Float32Dtype(),
        pa.float64(): pd.Float64Dtype(),
        pa.string(): pd.StringDtype(),
    }

    df = table.to_pandas(split_blocks=True, self_destruct=True, types_mapper=dtype_mapping.get)
    del table
    toc = time.time()
    print(f'Read {len(df):,} rows from {path.stem!r} in {toc-tic:.2f} sec.')
    
    if convert_dtypes:
        tic = time.time()
        size_before = df.memory_usage(deep=True).sum() / 1024 / 1024 / 1024

        string_cols_d = {}
        # string_type = 'string[python]'
        string_type = 'string[pyarrow]'
        for col, dtype in df.dtypes.to_dict().items():
            if pd.__version__ <= '2.1':
              if dtype == 'object' or dtype == 'string':  # convert object columns to string
                  string_cols_d[col] = string_type
            if col == 'type' or col == 'concept_name':
                if dtype != 'category':
                    string_cols_d[col] = 'category'
            if col == 'publication_month':
                if dtype != 'uint8':
                    string_cols_d[col] = 'uint8'
            if col == 'score':
                if dtype != 'float16':
                    string_cols_d[col] = 'float16'
                
        df = df.astype(string_cols_d) 
        
        size_after = df.memory_usage(deep=True).sum() / 1024 / 1024 / 1024
        toc = time.time()
        print(f'Converting dtypes took {toc-tic:.2f} sec. Size before: {size_before:.2f}GB, after: {size_after:.2f}GB')

    if not quiet:
        display(df.head(3))
    return df



def read_parquet(path, engine='pyarrow', convert_dtypes=True, quiet=False, **args):
    """
    Read a parquet file
    """
    if isinstance(path, str):
        path = Path(path)

    if 'filters' in args:
        engine = 'pyarrow'
    
    if not path.name.endswith('.parquet'):
        ## check if a file exists without the extension 
        dir_exists = path.exists()
        if not dir_exists:  # try adding the parquet extension
            if path.with_suffix('.parquet').exists():
                path = path.with_suffix('.parquet')
                
    name = path.stem
    
    if not quiet:
        print(f'\nReading {name!r} from {str(path)!r} using {engine=}')
    
    tic = time.time()
    df = pd.read_parquet(path, engine=engine, **args)
    toc = time.time()
    print(f'Read {len(df):,} rows from {path.stem!r} in {toc-tic:.2f} sec.')
    
    if convert_dtypes:
        tic = time.time()
        size_before = df.memory_usage(deep=True).sum() / 1024 / 1024 / 1024

        string_cols_d = {}
        # string_type = 'string[python]'
        string_type = 'string[pyarrow]'
        for col, dtype in df.dtypes.to_dict().items():
            if pd.__version__ <= '2.1':
              if dtype == 'object' or dtype == 'string':  # convert object columns to string
                  string_cols_d[col] = string_type
            if col == 'type' or col == 'concept_name':
                if dtype != 'category':
                    string_cols_d[col] = 'category'
            if col == 'publication_month':
                if dtype != 'uint8':
                    string_cols_d[col] = 'uint8'
            if col == 'score':
                if dtype != 'float16':
                    string_cols_d[col] = 'float16'
                
        df = df.astype(string_cols_d) 
        
        size_after = df.memory_usage(deep=True).sum() / 1024 / 1024 / 1024
        toc = time.time()
        print(f'Converting dtypes took {toc-tic:.2f} sec. Size before: {size_before:.2f}GB, after: {size_after:.2f}GB')

    if not quiet:
        display(df.head(3))
    return df
    

def peek_parquet(path):
    """
    peeks at a parquet file (or a directory containing parquet files) and prints the following:
    * Path
    * schema
    * number of pieces (fragments)
    * number of rows 
    """
    if isinstance(path, str):
        path = Path(path)
        
    parq_file = pypq.ParquetDataset(path)
    piece_count = len(parq_file.fragments)
    schema = textwrap.indent(parq_file.schema.to_string(), ' '*4)
    row_count = sum(frag.count_rows() for frag in parq_file.fragments)
    
    st = [
        f'Name: {path.stem!r}',  
        f'Path: {str(path)!r}',
        f'Files: {piece_count:,}',
        f'Rows: {row_count:,}',
        f'Schema:\n{schema}',
        f'5 random rows:',
    ]
    print('\n'.join(st))
    sample_df = parq_file.fragments[0].head(5).to_pandas()  # read 5 rows from the first fragment
    display(sample_df)

    return


def plot_bar_pd(df,column_x1,column_x2,column_y,title):
    plt.figure(figsize=(20,10))
    plt.bar(df[column_x1].astype(str) + "/" + df[column_x2].astype(int).astype(str),df[column_y]) 
    plt.xticks(rotation=90,fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(title,fontsize=25)
    plt.show()

    
def plot_author_class_counts(topic_works_authors, phase=True, normalize=True, title=None):
    """
    Plot author class counts.
    If phase: plot phase wise counts, else plot monthly counts 
    """
    topic = topic_works_authors.topic.values[0]
    
    if phase:
        x, y = 'works_count', 'phase'
        groupby_col = 'phase'
        plot_kind = 'barh'
        plot_args = dict(text='works_count')
    else:
        x, y = 'date', 'works_count'
        groupby_col = 'date'
        plot_kind = 'area'
        plot_args = dict()
    
    df = (
        topic_works_authors
        .groupby(['topic', groupby_col], sort=False)
        .class_
        .value_counts(normalize=normalize)
        .rename('works_count')
        .to_frame()
        .reset_index() 
    )
    
    if normalize:
        df = (
            df
            # .assign(works_count=lambda df_: df_.works_count.mul(100))  # multiply by 100
        )
        text_auto = '.0%'
        s = 'percentage'
    else:
        text_auto = '.0s'
        s = 'counts'
    if phase:
        plot_args.update(dict(text_auto=text_auto))
    
    if title is None:
        title = f'{topic} phase works {s} by author class'
        height = 500
    else:
        height = 400
    
    fig = (
        df
        .plot
        (
            x=x, y=y, color='class_', 
            category_orders=dict(class_=['0%', '1-25%', '26-50%', '51-75%', '76-99%', '100%']),
            height=height, title=title,
            color_discrete_sequence=px.colors.qualitative.Safe,
            labels={'phase': '', 'works_count': '', 'class_': ''},
            kind=plot_kind, **plot_args,
        )
    )
    
    if phase:
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title='Counts')
        
        if normalize:
            fig.update_xaxes(tickformat=',.0%')
            fig.update_xaxes(title='Percentage')
            
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=0.7,
    ))
    if 'COVID' in topic or '2020' in topic:
        xrange = ['2019-12-15', '2023-01-15']
    else:
        xrange = ['2016-12-15', '2020-01-15']
    fig.update_xaxes(range=xrange)
    return fig 


def make_combined_area_plot(topic_monthly_stats, kind='author_classes', aggregate=True, phase=True, v2=True):
    if v2:
        area_plot_fn = make_area_plot_v2
    else:
        area_plot_fn = make_area_plot
    
    if kind == 'author_classes':
        fig1 = go.FigureWidget(area_plot_fn(topic_monthly_stats, normalize=False, aggregate=aggregate, phase=phase))
        fig2 = go.FigureWidget(area_plot_fn(topic_monthly_stats, normalize=True, aggregate=aggregate, phase=phase, title='', showlegend=False))

    elif kind == 'work_classes':
        fig1 = go.FigureWidget(plot_author_class_counts(topic_works_authors=topic_monthly_stats, phase=phase, normalize=False))
        fig2 = go.FigureWidget(plot_author_class_counts(topic_works_authors=topic_monthly_stats, phase=phase, normalize=True, title=''))
    
    fig2.layout.update(margin=dict(t=0))
    subplots = ipw.VBox([fig1, fig2])
    return subplots


def make_paired_area_plot(topic, stats_d, stats_d_comb, kind='author_classes', phase=True, aggregate=True):
    """
    Make a paired plot to compare stats for bellwethers and bellwethers combined with experts
    """
    title1, title2 = f'{topic} separate bellwethers', f'{topic} bellwethers combined w/ experts'
    
    if kind == 'author_classes':
        fig1 = go.FigureWidget(make_area_plot(stats_d[topic][0], normalize=True, aggregate=aggregate, phase=phase, title=title1, showlegend=True))
        
        fig2 = go.FigureWidget(make_area_plot(stats_d_comb[topic][0], normalize=True, aggregate=aggregate, phase=phase, title=title2, showlegend=False))
    
    elif kind == 'work_classes':
        fig1 = go.FigureWidget(plot_author_class_counts(topic_works_authors=stats_d[topic][1], phase=phase, normalize=True, title=title1))
        fig2 = go.FigureWidget(plot_author_class_counts(topic_works_authors=stats_d_comb[topic][1], phase=phase, normalize=True, title=title2))
    
        fig2.layout.update(margin=dict(t=90))
    subplots = ipw.VBox([fig1, fig2])
    return subplots
    

def make_area_plot(monthly_stats_df, phase=True, aggregate=False, normalize=False, title=None, showlegend=True):
    """
    make area plots from monthly stats
    if aggregate: don't separate first-timers and repeat authors
    if counts: plot counts instead of percentages
    """    
    # counts = True -> overall counts, False for percentage
    topic = monthly_stats_df.topic.unique().tolist()[0]
    bell = False
    if 'topic_bellwethers' in  monthly_stats_df.columns:
        bell = True 

    if aggregate:
        agg_st = 'aggregated' 
        y = ['experts', 'neophytes', 'newcomers']
        color_discrete_sequence = ['dodgerblue', 'coral', 'oliveDrab']
        if bell:
            y = ['experts', 'bellwethers', 'neophytes', 'newcomers']
            color_discrete_sequence = ['dodgerblue', 'orchid', 'coral', 'oliveDrab']
    else: 
        agg_st = 'divided'
        y = ['experts_first', 'experts_repeat', 'neophytes_first', 'neophytes_repeat', 
             'newcomers_first', 'newcomers_repeat'] 
        color_discrete_sequence = ['mediumturquoise', 'dodgerblue', 'orange', 'coral', 'palegreen', 'olivedrab']

        if bell:
            y = ['experts_first', 'experts_repeat', 'bellwethers_first', 'bellwethers_repeat',
                 'neophytes_first', 'neophytes_repeat', 'newcomers_first', 'newcomers_repeat'] 
            color_discrete_sequence = ['mediumturquoise', 'dodgerblue', 'pink', 'orchid', 'orange', 'coral', 'palegreen', 'olivedrab']
    

    # cols = ['date', 'topic', 'month_idx', 'phase', 'works', 'authors'] + [c for c in y if 'repeat' not in c]
    
    df = (
        monthly_stats_df
        .rename(columns=lambda f: f.replace('concept_', ''))
        .rename(columns=lambda f: f.replace('topic_', ''))
        .assign(
            experts_repeat=lambda df_: df_.experts - df_.experts_first,
            neophytes_repeat=lambda df_: df_.neophytes - df_.neophytes_first,
            newcomers_repeat=lambda df_: df_.newcomers - df_.newcomers_first,
        )
    )

    if bell:
        df = (
            df
            .assign(
                bellwethers_repeat=lambda df_: df_.bellwethers - df_.bellwethers_first,
            )
        )
    

    if phase:
        x, y =  y, 'phase'
        ymin = -10
        kind = 'barh'
        plot_args = dict(text_auto='.0s', text='value')
    else:
        kind = 'area'
        x = 'date'
        plot_args = dict(markers=False, text='value')
        if 'COVID' in topic:
            xrange = ['2020-02-01', '2023-04-01']
        else:
            xrange = ['2016-12-01', '2020-03-01']
    
    
    if not phase:
        if normalize:
            count_st = 'percentage'
            df = (
                df
                .set_index(['topic', 'date', 'month_idx', 'phase'])
                .pipe(lambda df_: df_.div(df_.authors, axis=0))
                .reset_index()
            )
            val = 'fraction'
            texttemplate = '%{y:.0%}'
        else:
            count_st = 'counts'
            texttemplate = '%{y:.0s}'
            val = 'Counts'
    
    else:
        val = 'Counts'
        count_st = 'counts'
        df = (
            df
            .groupby('phase', as_index=False)
            .sum(numeric_only=True)
        )
        if normalize:
            val = 'Authors %'
            df = (
                df
                .set_index(['month_idx', 'phase'])
                .pipe(lambda df_: df_.div(df_.authors, axis=0))
                .reset_index()
            )
            plot_args['text_auto'] = '.0%'
    
    # display(f'{x=} {plot_args=}',  df.head(2))
    if title is None:
        title = f'{topic} {agg_st} author {count_st}'
        height = 500
    else:
        height = 400
        
    fig = (
        df
        .plot
        (
            x=x, 
            y=y,
            log_y=False,
            labels={'value': val, 'phase': '', 'variable': '', 'Counts': ''},
            color_discrete_sequence=color_discrete_sequence,
            title=title, height=height, #width=1500,
            kind=kind, **plot_args
        )
    )

    if phase:
        fig.update_yaxes(autorange="reversed")
        if normalize:
            fig.update_xaxes(tickformat=',.0%')
            
    else:
        fig.for_each_trace(lambda t: t.update(textfont_color=t.fill, textposition='bottom center', textfont={'size': 13}))
        fig.update_xaxes(
            dtick="M2",
            ticklabelmode="period",
            range=xrange,
            tickformat="%b\n%Y"
        )
        
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig.update_traces(texttemplate=texttemplate)
    
    if showlegend:
        if aggregate:
            t = 100
        else:
            t = 150
    else:
        t = 50
    # if aggregate:
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=0.95,),
        showlegend=showlegend,
        margin=dict(t=t),
    )

    return fig


def make_area_plot_v2(monthly_stats_df, phase=True, aggregate=False, normalize=False, title=None, showlegend=True):
    """
    make area plots from monthly stats
    if aggregate: don't separate first-timers and repeat authors
    if counts: plot counts instead of percentages
    """    
    # counts = True -> overall counts, False for percentage
    topic = monthly_stats_df.topic.unique().tolist()[0]
    bell = False
    
    monthly_stats_df = (
        monthly_stats_df
        .drop(columns=['topic_bellwethers', 'topic_bellwethers_first'])
        .rename(columns=lambda f: f.replace('neophytes', 'bellwethers'))
    )
    # display(monthly_stats_df.head(3))

    if aggregate:
        agg_st = 'aggregated' 
        y = ['experts', 'bellwethers', 'newcomers']

    else: 
        agg_st = 'divided'
        y = ['experts_first', 'experts_repeat', 'bellwethers_first', 'bellwethers_repeat', 
             'newcomers_first', 'newcomers_repeat'] 

    # cols = ['date', 'topic', 'month_idx', 'phase', 'works', 'authors'] + [c for c in y if 'repeat' not in c]
    
    df = (
        monthly_stats_df
        .rename(columns=lambda f: f.replace('concept_', ''))
        .rename(columns=lambda f: f.replace('topic_', ''))
        .assign(
            experts_repeat=lambda df_: df_.experts - df_.experts_first,
            bellwethers_repeat=lambda df_: df_.bellwethers - df_.bellwethers_first,
            newcomers_repeat=lambda df_: df_.newcomers - df_.newcomers_first,
        )
    )

    if phase:
        x, y =  y, 'phase'
        ymin = -10
        kind = 'barh'
        plot_args = dict(text_auto='.0s', text='value')
    else:
        kind = 'area'
        x = 'date'
        # plot_args = dict(markers=False, text='value')
        plot_args = dict()
        if 'COVID' in topic or '2020' in topic:
            xrange = ['2019-12-15', '2023-01-15']
        else:
            xrange = ['2016-12-15', '2020-01-15']
    
    if not phase:
        if normalize:
            count_st = 'percentage'
            df = (
                df
                .set_index(['topic', 'date', 'month_idx'])
                .pipe(lambda df_: df_.div(df_.authors, axis=0))
                .reset_index()
            )
            val = 'fraction'
            texttemplate = '%{y:.0%}'
        else:
            count_st = 'counts'
            texttemplate = '%{y:.0s}'
            val = 'Counts'
    
    else:
        val = 'Counts'
        count_st = 'counts'
        df = (
            df
            .groupby('phase', as_index=False)
            .sum(numeric_only=True)
        )
        if normalize:
            val = 'Authors %'
            df = (
                df
                .set_index(['month_idx', 'phase'])
                .pipe(lambda df_: df_.div(df_.authors, axis=0))
                .reset_index()
            )
            plot_args['text_auto'] = '.0%'
    
    # display(f'{x=} {plot_args=}',  df.head(2))
    if title is None:
        title = f'{topic} {agg_st} author {count_st}'
        height = 500
    else:
        height = 400
        
    fig = (
        df
        .plot
        (
            x=x, 
            y=y,
            log_y=False,
            labels={'value': val, 'date': '', 'phase': '', 'variable': '', 'Counts': ''},
            # color_discrete_sequence=color_discrete_sequence,
            title=title, height=height, #width=1500,
            kind=kind, **plot_args
        )
    )

    if normalize:
        fig.update_xaxes(tickformat=',.0%')
    
    if phase:
        fig.update_yaxes(autorange="reversed") 
    else:
        # fig.for_each_trace(lambda t: t.update(textfont_color=t.fill, textposition='bottom center', textfont={'size': 13}))
        fig.update_xaxes(
            dtick="M2",
            ticklabelmode="period",
            range=xrange,
            tickformat="%b\n%Y"
        )
        
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig.update_traces(texttemplate=texttemplate)
    
    if showlegend:
        if aggregate:
            t = 100
        else:
            t = 150
    else:
        t = 50
    # if aggregate:
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=0.95,),
        showlegend=showlegend,
        margin=dict(t=t),
    )

    return fig



def make_random_baseline_plot(
    topic_works_authors_counts, shuffled_works_authors_counts, num_samples=20,
    phase=True,
    kind='bar', title=None):
    
    if phase:
        groupby_cols = ['topic', 'phase', 'class_']
        plot_args = dict(facet_col='phase', facet_col_wrap=3, height=1000)
    else:
        groupby_cols = ['topic', 'class_']
        plot_args = dict(height=300)
    
    actual_phase_works_counts = (
        topic_works_authors_counts
        .groupby(groupby_cols)
        .work_id
        .count()
        .rename('actual_works_count')
        # .to_frame()
    )

    frac_counts_df = []

    for split_ix in range(num_samples):
        split_df = (
            shuffled_works_authors_counts
            .iloc[split_ix: : num_samples]
        )
        shuffled_phase_works_counts = (
            split_df
            .groupby(groupby_cols)
            .work_id
            .count() 
            # .add(0.0000001)  # to prevent div by 0
            .rename('shuffled_works_count')
            # .to_frame()
        )
        # display(shuffled_phase_works_counts)
        frac_phase_works_counts = (
            actual_phase_works_counts
            .div(shuffled_phase_works_counts)
        )
        frac_counts_df.append(frac_phase_works_counts)
        # display(frac_phase_works_counts)

    frac_counts_combined = (
        pd.concat(frac_counts_df, axis=1)
        .replace([np.inf, -np.inf], np.nan)
    )
    # display(frac_counts_combined.round(2).head(3))
    
    topic = frac_counts_combined.reset_index().topic.values[0]
    
    df = (
        frac_counts_combined
        .assign(
            mean=lambda df_: df_.iloc[:, :num_samples].mean(axis=1),
            ci=lambda df_: df_.apply(lambda row: sns.utils.ci(row[: num_samples]), axis=1),
            ci_minus=lambda df_: df_.apply(lambda row: row['mean'] - row['ci'][0], axis=1),
            ci_plus=lambda df_: df_.apply(lambda row: row['ci'][1] - row['mean'], axis=1),
        )
        .join(
            actual_phase_works_counts
        )
        .reset_index()
    )
    
    if title is None:
        title = f'{topic!r} wrt {num_samples} random baselines'
    
    if kind == 'bar':
        fig = (
            df
            .plot
            .bar(
                x='class_', y='mean', color='class_', text='actual_works_count',
                error_y='ci_plus', error_y_minus='ci_minus',
                height=600, text_auto='.2f',
                category_orders=dict(class_=['0%', '1-25%', '26-50%', '51-75%', '76-99%', '100%']),
                color_discrete_sequence=px.colors.qualitative.Safe, 
                labels={'phase': '', 'works_count': '', 'class_': ''},
                title=title, log_y=True, **plot_args,
            )
        )
        fig.update_traces(width=0.75, texttemplate='%{text:.0s}', marker=dict(line=dict(width=2, color='gray')))
        fig.update_yaxes(matches=None)
        fig.add_hline(y=1, line_color='red', line_width=2, line_dash='dash')
        
    elif kind == 'scatter':
        fig = (
            df
            .plot
            .scatter(
                y='mean', x='class_', error_y='ci_plus', error_y_minus='ci_minus',
                text='actual_works_count', color='class_',
                size='actual_works_count', size_max=25,
                category_orders={'class_': ['0%', '1-25%', '26-50%', '51-75%', '76-99%', '100%']},
                color_discrete_sequence=px.colors.qualitative.Safe, log_y=True,
                labels={'class_': ''}, **plot_args,
                title=title,
            )
        )
        fig.update_traces(
            texttemplate='%{text:.0s}', 
            textposition='top center', 
            marker_line_width=1.5, marker_line_color='white', marker_sizemin=10, 
            # marker=dict(size=15), 
            cliponaxis=False,
        )
        fig.add_hline(y=1, line_color='red', line_width=2, line_dash='dash')
        fig.update_yaxes(matches=None, dtick=1, tickformat='.0f')
        
    fig.update_xaxes(type='category')
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    # fig.update_xaxes(showgrid=True, ticks="outside")
    
    
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.2,
            xanchor="right",
            x=1,
        ),
        margin=dict(t=120),
    )
    
    return fig


def make_monthly_counts_plot(topic, works, works_concepts, works_authors):
    counts_df = get_combined_TW_OW_monthly_counts(topic=topic, works=works,
                                                  works_concepts=works_concepts, 
                                                  works_authors=works_authors)
    fig = go.Figure()

    start_date_ow, end_date_ow = (
            str(counts_df.query('month_idx==1').index.values[0]), 
            str(counts_df.query('month_idx==36').index.values[0])
    )

    fig.add_trace(
        go.Scatter(x=counts_df.index, y=counts_df.works, name='works',
                   line=dict(color='red')), 
    )

    fig.add_trace(
        go.Scatter(x=counts_df.index, y=counts_df.exp_works, name='exp_works',
                  line=dict(color='green')), 
    )

    fig.add_trace(
        go.Scatter(x=counts_df.index, y=counts_df.authors, name='authors',
                   line=dict(color='coral', dash='dashdot')), 
    )

    fig.add_trace(
        go.Scatter(x=counts_df.index, y=counts_df.exp_authors, name='exp_authors',
                   line=dict(color='limegreen', dash='dashdot')), 
    )

    fig.update_layout(
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        title=f'{topic} counts',
    )
    fig.add_vrect(
            x0=str(start_date_ow), x1=str(end_date_ow), line_width=0, fillcolor='cyan', opacity=0.10,
    )
    fig.update_yaxes(type='log') #, range=[2, 4])

    return fig
    


def plot_monthly_counts_v2(topic, monthly_counts=None, works=None, works_concepts=None, works_authors=None, title=None):
    if title is None:
        title = f'{topic!r} monthly counts'
    
    if monthly_counts is None:
        monthly_counts = (
            get_combined_TW_OW_monthly_counts(topic=topic, works=works, works_concepts=works_concepts, works_authors=works_authors)
        )
    monthly_counts = (
        monthly_counts
        .reset_index()
            .rename(columns={'index': 'date'})
    )
    start_date_ow, end_date_ow = (
        str(monthly_counts.query('month_idx==1').date.values[0]), 
        str(monthly_counts.query('month_idx==36').date.values[0])
    )

    fig = (
        monthly_counts
        .rename(columns={
            'works': f'{topic} works', 'exp_works': f'{topic} expert works',
            'authors': f'{topic} authors', 'exp_authors': f'{topic} expert authors',
        })
        .plot
        .line(
            x='date',
            y=[f'{topic} works', f'{topic} expert works', f'{topic} authors', f'{topic} expert authors'], height=450,
            log_y=True,
            labels={'variable': '', 'date': '', 'Topic works': topic, 'exp_works': 'Expert works'},
            title=title,
        )
    )

    fig.for_each_trace(lambda t: t.update(textfont_color=t.fill, textposition='bottom center', textfont={'size': 8}))
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_traces(texttemplate='%{y:.0s}')
    fig.update_xaxes(
        dtick="M6",
        ticklabelmode="period",
        tickformat="%b\n%Y")
    # fig.update_yaxes(range=yrange)
    fig.add_vrect(
        x0=str(start_date_ow), x1=str(end_date_ow), line_width=0, fillcolor='tomato', opacity=0.10,
    )
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    # fig.add_vline(x=start_date_ow, line_width=3, line_dash="dash", line_color="green")
    return fig




def plot_monthly_works_counts(topic, monthly_counts=None, works=None, works_concepts=None, works_authors=None):
    if monthly_counts is None:
        monthly_counts = (
            get_combined_TW_OW_monthly_counts(topic=topic, works=works, works_concepts=works_concepts, works_authors=works_authors)
        )
    monthly_counts = (
        monthly_counts
        .reset_index()
            .rename(columns={'index': 'date'})
    )
    start_date_ow, end_date_ow = (
        str(monthly_counts.query('month_idx==1').date.values[0]), 
        str(monthly_counts.query('month_idx==36').date.values[0])
    )

    fig = (
        monthly_counts
        .rename(columns={'works': f'{topic} works', 'exp_works': f'{topic} expert works'})
        .plot
        .line(
            x='date',
            y=[f'{topic} works', f'{topic} expert works'], height=450,
            labels={'variable': '', 'date': '', 'Topic works': topic, 'exp_works': 'Expert works'},
            title=f'{topic} monthly works count',
        )
    )

    fig.for_each_trace(lambda t: t.update(textfont_color=t.fill, textposition='bottom center', textfont={'size': 8}))
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_traces(texttemplate='%{y:.0s}')
    fig.update_xaxes(
        dtick="M6",
        ticklabelmode="period",
        tickformat="%b\n%Y")
    # fig.update_yaxes(range=yrange)
    fig.add_vrect(
        x0=str(start_date_ow), x1=str(end_date_ow), line_width=0, fillcolor="red", opacity=0.10,
    )
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    # fig.add_vline(x=start_date_ow, line_width=3, line_dash="dash", line_color="green")
    return fig


def get_monthly_concept_works_count(concepts: set, topic: str, works, works_concepts, works_authors):
    """
    Monthly work counts for concepts under this topic
    """
    concept_works = (
        works
        [works.index.isin(
            set(works_concepts
            [
                (works_concepts
                .concept_id.isin(concepts)) 
            ].work_id)
        )]
        .query('num_authors>1')
    )
    
    concept_works_authors = (
        works_authors
        [works_authors.work_id.isin(set(concept_works.index))]
    )
    
    monthly_concept_works = (
        concept_works
        .reset_index()
        .set_index('publication_date')
        .resample('M')
        .work_id
        .count()
        .rename('works')
        .to_frame()
        .assign(topic=topic)
        .rename(columns={'publication_date': 'date'})
    )
    
    monthly_concept_authors = (
        concept_works_authors
        .set_index('publication_date')
        .resample('M')
        .author_id
        .nunique()
        .rename('authors')
        .to_frame()
        .assign(topic=topic)
        .rename(columns={'publication_date': 'date'})
    )
    
    df = (
        monthly_concept_works
        .join(
            monthly_concept_authors
            .authors
        )
    )
    
    return df


def filter_dataframes_timerange(whole_works, whole_works_authors, whole_works_concepts, start_date, end_date, exclude_solo_authors=True):
    # print(f'{start_date.date()} to {end_date.date()}')
    works_ow = (
        whole_works
        [whole_works.publication_date.between(start_date, end_date)]
    )
    if exclude_solo_authors:
        works_ow = (
            works_ow
            [works_ow.num_authors>1]
        )
    # print(f'{len(works_ow):,} works, discarding solo works')
    work_ids = set(works_ow.index)
    
    works_authors_ow = (
        whole_works_authors
        [whole_works_authors.work_id.isin(work_ids)]
    )
    # print(f'{works_authors_ow.author_id.nunique():,} authors')

    works_concepts_ow = (
        whole_works_concepts
        [whole_works_concepts.work_id.isin(work_ids)]
    )
    # print(f'{len(works_concepts_ow):,} works concept tags')

    return works_ow, works_authors_ow, works_concepts_ow


def get_monthly_stats_old(key_concepts, expert_concepts, topic, start_date_ow, ow_duration=2, tw_duration=5,
                      works_ew=None, works_ow=None):
    print(f'{topic} Key concepts:')
    for id in key_concepts:
        print(id, dict_concept_id_name[id]) 
    print() 
    
    print(f'{topic} Expert concepts:')
    for id in expert_concepts:
        print(id, dict_concept_id_name[id]) 
    print()
    
    start_date_ow = pd.to_datetime(start_date_ow)
    end_date_ow = start_date_ow + pd.DateOffset(years=ow_duration)
    observation_window = pd.date_range(start_date_ow, end_date_ow, freq='M')

    end_date_ew = start_date_ow  # end date of exposure window is the start date of OW
    start_date_ew = start_date_ow - pd.DateOffset(years=ew_duration)
    end_date_ew = end_date_ew - pd.DateOffset(days=1) # end one day earlier 
    exposure_window = pd.date_range(start_date_ew, end_date_ew, freq='M')
    print(f'{topic}')
    print(f'Exposure window: {start_date_ew} to {end_date_ew}')
    print(f'Observation window: {start_date_ow} to {end_date_ow}')
    
    if works_ew is None:
        works_ew, works_authors_ew, works_concepts_ew = filter_dataframes_timerange(works, works_authors, 
                                                                                    works_concepts, start_date_ew, end_date_ew)
    
    if works_ow is None:
        works_ow, works_authors_ow, works_concepts_ow = filter_dataframes_timerange(works, works_authors, 
                                                                                    works_concepts, start_date_ow, end_date_ow)

    # initialize these sets at the start of the Observation Window. 
    existing_authors = set(   # all authors active in the last 5 years
        works_authors_ew
        .author_id
    )
    print(f'{len(existing_authors)=:,}')

    experts = set(         # wrote a paper on the expert concepts in the EW 
        works_authors_ew
        [
            works_authors_ew
            .work_id
            .isin(
                set(
                    works_concepts_ew
                    [(works_concepts_ew.score>min_score) & (works_concepts_ew.concept_id.isin(expert_concepts))]
                    .work_id
                )
            )
        ]
        .author_id
    )
    print(f'{len(experts)=:,}')
    
    existing_covid_authors = set(         # wrote a paper on the expert concepts in the EW 
        works_authors_ew
        [
            works_authors_ew
            .work_id
            .isin(
                set(
                    works_concepts_ew
                    [(works_concepts_ew.score>min_score) & (works_concepts_ew.concept_id.isin(key_concepts))]
                    .work_id
                )
            )
        ]
        .author_id
    )
    print(f'{len(existing_covid_authors)=:,}')
    
    # existing_covid_or_expert_authors = experts   # set of authors who have written something with COVID or some expert concept. 
                                                 # starts the same as experts 
    # print(f'{len(existing_covid_or_expert_authors)=:,}')    
    
    rows = []
    covid_monthly_author_dfs = []
    ### separate experts from repeat authors 
    for i, month_end_date in enumerate(tqdm(observation_window)):       
        month_start_date = month_end_date - pd.offsets.MonthBegin()
        assert month_start_date.day==1 and month_start_date.month==month_end_date.month

        works_month, works_authors_month, works_concepts_month = filter_dataframes_timerange(works_ow, works_authors_ow, 
                                                                                             works_concepts_ow, month_start_date, month_end_date)
        
        all_authors_month = set(works_authors_month.author_id)  # set of all authors for that month

        covid_work_ids_month = set(     # set of works tagged with COVID
            works_concepts_month
            [(works_concepts_month.score>min_score) & (works_concepts_month.concept_id.isin(key_concepts))]
            .work_id
        )
        # print(f'{len(covid_work_ids_month)=:,}')

        covid_works_authors_month = (    # works authorships for covid papers 
            works_authors_month
            [works_authors_month.work_id.isin(covid_work_ids_month)]
        )
        
        covid_authors_month = set(       # set of authors who wrote something tagged with COVID
            covid_works_authors_month
            .author_id
        )
        # print(f'{len(covid_authors_month)=:,}')

        expert_work_ids_month = set(    # set of works tagged with COVID expert concepts 
            works_concepts_month
            [(works_concepts_month.score>min_score) & (works_concepts_month.concept_id.isin(expert_concepts))]
            .work_id
        )
        # print(f'{len(expert_work_ids_month)=:,}')

        expert_authors_month = set(       # set of authors who wrote something tagged with COVID expert concepts
            works_authors_month
            [works_authors_month.work_id.isin(expert_work_ids_month)]
            .author_id
        )
        # print(f'{len(expert_authors_month)=:,}')

        brand_new_authors_month = all_authors_month - existing_authors   # brand new authors overall 
        # print(f'{len(brand_new_authors_month)=:,}')

        brand_new_authors_covid_month =  covid_authors_month - existing_authors  # brand new authors who wrote a COVID paper 
        # print(f'{len(brand_new_authors_covid_month)=:,}')

        # neophytes -> existing authors who wrote a COVID paper for the first time 
        # how to separate them from existing authors who've written multiple COVID papers? keep track of people who have written COVID papers 
        neophytes_month = {author for author in (covid_authors_month & existing_authors) if author not in existing_covid_authors}
        # print(f'{len(neophytes_month)=:,}')
        
        # authors who wrote a paper in that month who are experts
        covid_expert_month = covid_authors_month & experts 
        covid_repeat_authors = covid_authors_month & existing_covid_authors
        # covid_expert_or_repeat_authors_month = covid_authors_month & existing_covid_or_expert_authors
        
        ### create works rows by calculating the number of brand new authors, neophytes, experts 
        monthly_author_count = (
            covid_works_authors_month
            .groupby('work_id', sort=False, as_index=False)
            .agg({
                'work_id': 'first',
                'publication_date': 'last',  # otherwise both work_id and date end up with the same column name after dropping level 0 column
                'author_id': ['count', 
                              lambda x: len(set(x) & brand_new_authors_covid_month), 
                              lambda x: len(set(x) & neophytes_month), 
                              lambda x: len(set(x) & covid_expert_month),
                              lambda x: len(set(x) & covid_repeat_authors)]
            })
            .droplevel(0, axis=1)
            .rename(columns={'first': 'work_id', 'last': 'publication_date', 'count': 'num_authors', 
                             '<lambda_0>': 'num_brand_new', '<lambda_1>': 'num_neophytes', 
                             '<lambda_2>': 'num_experts', '<lambda_3>': 'num_repeats'})
            .assign(month_idx=i+1, topic=topic)
        )
        covid_monthly_author_dfs.append(monthly_author_count)
        
        ### update all the sets 
        existing_authors.update(all_authors_month)  # add all authors for that month
        # print(f'{len(existing_authors)=:,}')

        # existing_covid_or_expert_authors.update(covid_authors_month | expert_authors_month)  # add covid authors and expert authors
        existing_covid_authors.update(covid_authors_month)
        
        # print(f'{len(existing_covid_or_expert_authors)=:,}')

        rows.append(dict(topic=topic, date=month_end_date.date(), month_idx=i+1, 
                         all_works=len(works_month), all_authors=len(all_authors_month),
                         concept_works=len(covid_work_ids_month), concept_active_authors=len(covid_authors_month), 
                         brand_new_authors_concept=len(brand_new_authors_covid_month), concept_neophytes=len(neophytes_month), 
                         concept_expert=len(covid_expert_month), concept_repeat_authors=len(covid_repeat_authors),
                         expert_works=len(expert_work_ids_month), expert_active_authors=len(expert_authors_month),
                         existing_authors_overall=len(existing_authors), existing_concept_authors_overall=len(existing_covid_authors),
                         brand_new_authors_overall=len(brand_new_authors_month), 
                         ))
    df = pd.DataFrame(rows)

    return df, pd.concat(covid_monthly_author_dfs)


def get_combined_TW_OW_monthly_counts(topic, works, works_concepts, works_authors, ow_duration=3, tw_duration=5):
    key_concepts, expert_concepts, start_date_tw, \
    end_date_tw, start_date_ow, end_date_ow = get_topic_info(topic=topic)
    
    observation_window = pd.date_range(start_date_ow, end_date_ow, freq='M')
    training_window = pd.date_range(start_date_tw, end_date_tw, freq='M')

    works_count = get_monthly_concept_works_count(concepts=key_concepts, topic=topic, works=works, works_concepts=works_concepts, works_authors=works_authors)
    exp_works_count = get_monthly_concept_works_count(concepts=expert_concepts, topic=topic, works=works, works_concepts=works_concepts, works_authors=works_authors)

    works_count = works_count.reindex(pd.date_range(start_date_tw, end_date_ow, freq='M'), fill_value=0)

    exp_works_count = exp_works_count.reindex(pd.date_range(start_date_tw, end_date_ow, freq='M'), fill_value=0)
    
    exp_works_count = (
        pd.concat([
        exp_works_count
        .loc[training_window]
        .assign(
            month_idx=lambda df_: range(-len(df_), 0, 1),
        ),
        exp_works_count
        .loc[observation_window]
        .assign(
            month_idx=lambda df_: range(1, len(df_)+1)
        )],
        axis=0,
        )
        .rename(columns={'works': 'exp_works', 'authors': 'exp_authors'})
    )
    
    works_count = (
        pd.concat([
        works_count
        .loc[training_window]
        .assign(
            month_idx=lambda df_: range(-len(df_), 0, 1),
        ),
        works_count
        .loc[observation_window]
        .assign(
            month_idx=lambda df_: range(1, len(df_)+1)
        )],
        axis=0,
        )
    )

    # combine exp and concept
    combined_count = pd.concat([
            works_count,
            exp_works_count
            [['exp_works', 'exp_authors']]
        ],
        axis=1)
    return combined_count


def clean_csvs_class_date(topic, csv_path):
    """
    renames field to topic, assign author classes and phases
    """
    topic_ids, topic_ids_exp, start_date_tw, end_date_tw, start_date_ow, end_date_ow = get_topic_info(topic)
    monthly_stats_df = pd.read_csv(csv_path / f'{topic}.csv', parse_dates=['date'])
    display(monthly_stats_df.head(2))

    topic_works_authors = pd.read_csv(csv_path / f'{topic}_author_classes.csv', parse_dates=['publication_date'])
    display(topic_works_authors.head(2))
    monthly_stats_df = (
        monthly_stats_df
        .rename(columns={'field': 'topic'})
        .assign(
            trial=0,
            date=lambda df_: pd.to_datetime(df_.date, infer_datetime_format=True),
            # phase=lambda df_: pd.cut(
            #     df_.date,
            #     pd.period_range(start=str(start_date_ow), end=str(end_date_ow), freq='6M').to_timestamp(),
            #     labels=['Phase I', 'Phase II', 'Phase III', 'Phase IV', 'Phase V', 'Phase VI'],
            #     include_lowest=True,
            # ),
        )
    )
    
    topic_works_authors = (
        topic_works_authors
        .rename(columns={'field': 'topic', 'publication_date': 'date'})
        .assign(
            trial=0,
            date=lambda df_: pd.to_datetime(df_.date, infer_datetime_format=True),
            frac_expert=lambda df_: df_.num_experts.div(df_.num_authors),
            phase=lambda df_: pd.cut(
                df_.date,
                pd.period_range(start=str(start_date_ow), end=str(end_date_ow), freq='6M').to_timestamp(),
                labels=['Phase I', 'Phase II', 'Phase III', 'Phase IV', 'Phase V', 'Phase VI'],
                include_lowest=True,
            ),
            class_=lambda df_ : pd.cut(
                df_.frac_expert,
                bins=[0, 0.000001, 0.25, 0.5, 0.75, 0.999, 1],
                include_lowest=True,
                labels=['0%', '1-25%', '26-50%', '51-75%', '76-99%', '100%'],
            ),
        )
    )
    monthly_stats_df.to_csv(csv_path / 'cleaned' / f'{topic}.csv', index=False)
    topic_works_authors.to_csv(csv_path / 'cleaned' / f'{topic}_author_classes.csv', index=False)
    return 


def make_histogram_pair(x, field, title, figsize=(16, 10), bins=25):
    # histogram on linear scale
    x = x[x>0]
    fig, axs = plt.subplots(nrows=2, figsize=figsize)
    hist, bins, _ = axs[0].hist(x, bins=bins)

    # histogram on log scale. 
    # Use non-equal bin sizes, such that they look equal on log scale.
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]), len(bins))

    axs[1].hist(x, bins=logbins)
    axs[1].set_xscale('log')
    fig.suptitle(f'{title} {field} papers')
    axs[0].set_title('linear bins')
    axs[1].set_title('log bins');
    
    for ax in axs:
        ax.set_ylabel('Counts')
        ax.grid(which='major', axis='y')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return fig


def make_histogram_pair(x, title='', figsize=(16, 10), bins=25):
    # histogram on linear scale
    x = x[x>0]
    fig, axs = plt.subplots(nrows=2, figsize=figsize)
    hist, bins, _ = axs[0].hist(x, bins=bins)

    # histogram on log scale. 
    # Use non-equal bin sizes, such that they look equal on log scale.
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]), len(bins))

    axs[1].hist(x, bins=logbins)
    axs[1].set_xscale('log')
    fig.suptitle(f'{title}')
    axs[0].set_title('linear bins')
    axs[1].set_title('log bins');
    
    for ax in axs:
        ax.set_ylabel('Counts')
        ax.grid(which='major', axis='y')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return fig

def log_log_hist(x, title='', figsize=(12, 5), ax=None):
    x = x[x>0]
    if ax is None:
        fig, axs = plt.subplots(nrows=1, figsize=figsize)
    else:
        axs = ax
    hist, bins, _ = axs[0].hist(x, bins=bins)

    # histogram on log scale. 
    # Use non-equal bin sizes, such that they look equal on log scale.
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]), len(bins))

    axs[1].hist(x, bins=logbins)
    axs[1].set_xscale('log')
    fig.suptitle(f'{title}')
    axs[1].set_title('log bins');
    
    for ax in axs:
        ax.set_ylabel('Counts')
        ax.grid(which='major', axis='y')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return fig


UPSET_FACE, UPSET_EDGE, UPSET_SHADE = '#9ac1f2', '#408ae6', '#d2f1fa'

def plot_upset(counts, set_sizes, ax_title='# Outbreak Scientists', figsize=(15, 7.5),
               min_subset_size=0, ascending=False):
  """UpSet plot of DISJOINT intersections.

  counts: Series indexed by a boolean MultiIndex (one level per set), values = number of
          members in exactly that combination of sets.
  set_sizes: Series mapping set name -> the TOTAL cardinality of that set, i.e. every member of
          the set including those it shares with other sets. This is what the UpSet design
          specifies for the left-hand panel, and it is NOT a sum over `counts`: summing is only
          equivalent when `counts` holds disjoint cells, and is meaningless otherwise.
  ascending: order the columns smallest bar first; the default is largest first, the UpSet
          convention of sorting by cardinality.
  """
  sets = list(counts.index.names)
  counts = counts[counts >= min_subset_size].sort_values(ascending=ascending)
  n_cols = len(counts)

  fmt = lambda v: f'{v*1e-6:.0f}M' if v >= 1e6 else (f'{v*1e-3:.0f}K' if v >= 1e3 else f'{v:.0f}')

  fig = plt.figure(figsize=figsize)
  gs = GridSpec(2, 2, width_ratios=[1, 3.6], height_ratios=[2.1, 1],
                hspace=0.06, wspace=0.22, figure=fig)
  ax_bars = fig.add_subplot(gs[0, 1])
  ax_mat = fig.add_subplot(gs[1, 1], sharex=ax_bars)
  ax_sets = fig.add_subplot(gs[1, 0], sharey=ax_mat)
  fig.add_subplot(gs[0, 0]).axis('off')

  # top: intersection sizes
  x = range(n_cols)
  ax_bars.bar(x, counts.values, width=0.62, facecolor=UPSET_FACE, edgecolor=UPSET_EDGE, linewidth=1.2)
  for xi, v in zip(x, counts.values):
    ax_bars.text(xi, v, f'{v:,}', ha='center', va='bottom', fontsize=9, rotation=90)
  ax_bars.set_ylabel(ax_title)
  ax_bars.set_ylim(0, counts.max() * 1.30)
  ax_bars.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: fmt(v)))
  ax_bars.spines[['top', 'right']].set_visible(False)
  ax_bars.tick_params(labelbottom=False, bottom=False)
  ax_bars.grid(axis='y', color='0.9', linewidth=0.8)
  ax_bars.set_axisbelow(True)

  # bottom: membership matrix, rows ordered by set size so the big sets sit on top
  order = list(set_sizes.sort_values(ascending=False).index)
  y_of = {s: len(order) - 1 - i for i, s in enumerate(order)}
  for i, s in enumerate(order):
    if i % 2 == 0:
      ax_mat.axhspan(y_of[s] - 0.5, y_of[s] + 0.5, color=UPSET_SHADE, zorder=0)
  for xi, idx in enumerate(counts.index):
    present = [s for s, flag in zip(sets, idx) if flag]
    ys = [y_of[s] for s in present]
    ax_mat.scatter([xi] * len(order), [y_of[s] for s in order], s=70, color='#dfe8f5', zorder=1)
    ax_mat.scatter([xi] * len(ys), ys, s=70, color=UPSET_EDGE, zorder=3)
    if len(ys) > 1:
      ax_mat.plot([xi, xi], [min(ys), max(ys)], color=UPSET_EDGE, linewidth=1.8, zorder=2)
  # set names live in the gap between the two panels, not on top of the size bars
  for s_ in order:
    ax_mat.text(-0.95, y_of[s_], s_, ha='right', va='center', fontsize=12)
  ax_mat.set_yticks([])
  ax_mat.set_ylim(-0.6, len(order) - 0.4)
  ax_mat.set_xlim(-0.7, n_cols - 0.3)
  ax_mat.set_clip_on(False)
  ax_mat.tick_params(bottom=False, labelbottom=False, left=False)
  ax_mat.spines[['top', 'right', 'bottom', 'left']].set_visible(False)

  # left: total size of each set
  ax_sets.barh([y_of[s] for s in order], [set_sizes[s] for s in order],
               height=0.55, facecolor=UPSET_FACE, edgecolor=UPSET_EDGE, linewidth=1.2)
  for s in order:
    ax_sets.text(set_sizes[s], y_of[s], f'{set_sizes[s]:,}  ', ha='right', va='center', fontsize=9)
  ax_sets.invert_xaxis()
  ax_sets.set_xlabel('Set size')
  ax_sets.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: fmt(v)))
  ax_sets.set_yticks([])
  ax_sets.tick_params(labelleft=False, left=False)
  ax_sets.spines[['top', 'right', 'left']].set_visible(False)
  return fig


def upset_to_latex(counts, set_sizes, ylabel='\\# Outbreak Scientists',
                   caption='UpSet plot', label='fig:upset', ascending=False,
                   bar_width='13cm', bar_height='5.4cm', matrix_height='2.6cm'):
  """Emit a TikZ/pgfplots UpSet figure with the data inlined.

  counts: Series indexed by a boolean MultiIndex, one level per set. Values are whatever
          the caller wants plotted -- disjoint cells, or unions.
  set_sizes: Series set name -> TOTAL cardinality of that set (see plot_upset); the UpSet
          design specifies this panel as the full set size, never a sum over the bars.
  Preamble: \\usepackage{pgfplots} and \\pgfplotsset{compat=1.18}.
  """
  counts = counts.sort_values(ascending=ascending)
  sets = list(counts.index.names)
  rows = list(set_sizes.sort_values(ascending=False).index)   # biggest set on top
  y_of = {s: len(rows) - i for i, s in enumerate(rows)}
  n, vmax = len(counts), counts.max()

  # \pgfmathparse overflows above 16384, so let pgfplots' own fpu do the /1000
  thousands = ('scaled y ticks={real:1000}, ytick scale label code/.code={}, '
               'yticklabel={\\pgfmathprintnumber[precision=0]{\\tick}K}')

  bars = ' '.join(f'({i+1},{v})' for i, v in enumerate(counts.values))
  bar_labels = '\n'.join(
    f'      \\node[rotate=90, anchor=west, inner sep=2pt, font=\\sffamily\\scriptsize] '
    f'at (axis cs:{i+1},{v}) {{{v:,}}};'
    for i, v in enumerate(counts.values))

  on, off, lines = [], [], []
  for i, idx in enumerate(counts.index):
    present = [s for s, flag in zip(sets, idx) if flag]
    for s in rows:
      (on if s in present else off).append(f'({i+1},{y_of[s]})')
    if len(present) > 1:
      ys = [y_of[s] for s in present]
      # emitted as \addplot, not \draw: a raw \draw inside a pgfplots axis lands on a lower
      # layer than the plots, which put the grey dots on top of the connectors
      lines.append(f'      \\addplot[upsetedge, line width=1pt, mark=none, forget plot] '
                   f'coordinates {{({i+1},{min(ys)}) ({i+1},{max(ys)})}};')

  shading = '\n'.join(
    f'      \\fill[upsetshade] (axis cs:0.4,{y_of[s]-0.5}) rectangle (axis cs:{n+0.6},{y_of[s]+0.5});'
    for i, s in enumerate(rows) if i % 2 == 0)
  size_bars = ' '.join(f'({set_sizes[s]},{y_of[s]})' for s in rows)
  size_labels = '\n'.join(
    f'      \\node[anchor=east, inner sep=3pt, font=\\sffamily\\scriptsize] '
    f'at (axis cs:{set_sizes[s]},{y_of[s]}) {{{set_sizes[s]:,}}};' for s in rows)

  return f"""% Generated from first-principles-jul2026.ipynb -- do not hand-edit.
% Preamble: \\usepackage{{pgfplots}} \\pgfplotsset{{compat=1.18}}
\\definecolor{{upsetface}}{{HTML}}{{9AC1F2}}
\\definecolor{{upsetedge}}{{HTML}}{{408AE6}}
\\definecolor{{upsetshade}}{{HTML}}{{D2F1FA}}

\\begin{{figure*}}[t]
  \\centering
  \\begin{{tikzpicture}}[font=\\sffamily\\small]
    %% ---- membership matrix (anchors the other two panels) -------------------
    \\begin{{axis}}[
      name=matrix, anchor=south west, at={{(0cm,0cm)}},
      width={bar_width}, height={matrix_height}, scale only axis,
      % pgfplots defers ALL marks past ALL plot paths so they escape clipping, which paints the
      % grey dots over the connectors no matter the source order. This draws marks in order.
      clip marker paths=true,
      xmin=0.4, xmax={n+0.6}, ymin=0.4, ymax={len(rows)+0.6},
      axis line style={{draw=none}}, tick style={{draw=none}},
      xtick=\\empty, ytick={{{','.join(str(y_of[s]) for s in rows)}}},
      yticklabels={{{','.join(f'{{{s}}}' for s in rows)}}},
      yticklabel style={{anchor=east, xshift=-2pt}},
    ]
{shading}
      \\addplot[only marks, mark=*, mark size=2.1pt, black!12] coordinates {{{' '.join(off)}}};
{chr(10).join(lines)}
      \\addplot[only marks, mark=*, mark size=2.6pt, upsetedge] coordinates {{{' '.join(on)}}};
    \\end{{axis}}

    %% ---- intersection sizes -------------------------------------------------
    \\begin{{axis}}[
      name=bars, at={{(matrix.north west)}}, anchor=south west, yshift=0.5cm,
      width={bar_width}, height={bar_height}, scale only axis,
      ybar, bar width=7pt,
      xmin=0.4, xmax={n+0.6}, ymin=0, ymax={vmax*1.22:.0f},   % headroom for the rotated value label on the tallest bar
      axis x line=none, axis y line*=left,
      ylabel={{{ylabel}}}, ylabel style={{align=center}},
      scaled y ticks=false, {thousands},
      ymajorgrids, grid style={{black!12}},
    ]
      \\addplot[fill=upsetface, draw=upsetedge, line width=0.7pt] coordinates {{{bars}}};
{bar_labels}
    \\end{{axis}}

    %% ---- set sizes: TOTAL cardinality of each set, never a sum over the bars -
    \\begin{{axis}}[
      name=sizes, at={{(matrix.south west)}}, anchor=south east, xshift=-1.95cm,
      width=3.0cm, height={matrix_height}, scale only axis,
      xbar, bar width=7pt, x dir=reverse,
      xmin=0, xmax={set_sizes.max()*1.5:.0f}, ymin=0.4, ymax={len(rows)+0.6},
      ytick=\\empty, axis y line=none, axis x line*=bottom,
      xlabel={{Set size}}, xtick={{0,40000,80000}},
      xmajorgrids, grid style={{black!12}},
      scaled x ticks={{real:1000}}, xtick scale label code/.code={{}},
      xticklabel={{\\pgfmathprintnumber[precision=0]{{\\tick}}K}},
      xticklabel style={{font=\\sffamily\\scriptsize}},
    ]
      \\addplot[fill=upsetface, draw=upsetedge, line width=0.7pt] coordinates {{{size_bars}}};
{size_labels}
    \\end{{axis}}
  \\end{{tikzpicture}}
  \\caption{{{caption}}}
  \\label{{{label}}}
\\end{{figure*}}
"""
