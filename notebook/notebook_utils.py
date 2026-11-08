"""Helper functions for first-principles.ipynb.

The notebook star-imports this module, so the imports below are also the notebook's namespace (pd, np, px,
plt, Path, tqdm, ...). Importing reads concepts.csv.gz from $OPENALEX_CSV for the concept lookups.
"""
import os
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
import seaborn as sns
from seaborn._statistics import EstimateAggregator
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
import time
import textwrap
import gc
import json

plt.rcParams.update({'font.size': 22})
sns.set(style='ticks', context='talk')
pd.options.plotting.backend = 'plotly'
tqdm.pandas(unit_scale=True)

# OpenAlex concept lookups
_concepts = pd.read_csv(os.environ['OPENALEX_CSV'] + '/concepts.csv.gz')
dict_concept_id_name = _concepts.set_index('concept_id')['concept_name'].to_dict()
dict_concept_id_level = _concepts.set_index('concept_id')['level'].to_dict()

UPSET_FACE, UPSET_EDGE, UPSET_SHADE = '#9ac1f2', '#408ae6', '#d2f1fa'


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
        # random day offset in [0, days to Dec 31), i.e. Jan 1 to Dec 30
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
        topic_ids_exp = {107130276, 89623803, 116675565, 524204448} #same COVID 
        start_date_ow = pd.to_datetime('2002-01-01')
        
    else:
        raise NotImplementedError(f'{topic}')

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


def read_parquet(path, engine='pyarrow', convert_dtypes=True, quiet=False, **args):
    """
    Read a parquet file
    """
    if isinstance(path, str):
        path = Path(path)

    if 'filters' in args:
        engine = 'pyarrow'
    
    if not path.name.endswith('.parquet'):
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
