# python -m streamlit run app_nms.py

abslex = r"abuseLexicon.xlsx"

vac_red = r"PolarIs1_VaccRed_up_ext.xlsx"   #r"PolarIs1_VaccRed_up.xlsx"
vac_tw = r"Polaris2_2_up.xlsx"  # r"Polaris2_2.xlsx"
cch_red = r"Polaris3_2_up.xlsx"  #  r"Polaris3_2.xlsx"
cch_tw =  r"Polaris4_up.xlsx"  # r"Polaris4.xlsx"
us16 = r"app_US2016_up.xlsx"  # app_US2016_up


# imports
import streamlit as st
from PIL import Image
from collections import Counter
import pandas as pd
pd.set_option("max_colwidth", 400)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
#plt.style.use("seaborn-talk")

from scipy.stats import pearsonr, pointbiserialr, spearmanr

import spacy
nlp = spacy.load('en_core_web_sm')

pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


import networkx as nx
import pylab as pyl
from networkx.drawing.nx_agraph import graphviz_layout
try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
    print('nx_agraph')
except ImportError:
    try:
        import pydotplus
        from networkx.drawing.nx_pydot import graphviz_layout
        print('nx_pydot')
    except ImportError:
        raise ImportError("This example needs Graphviz and either "
                              "PyGraphviz or PyDotPlus")

import plotly.express as px
import plotly
import plotly.graph_objects as go
import wordcloud
from wordcloud import WordCloud, STOPWORDS

import nltk
nltk.download('stopwords')
from nltk.text import Text


# functions

ethos_mapping = {0: 'neutral', 1: 'support', 2: 'attack'}
valence_mapping = {0: 'neutral', 1: 'positive', 2: 'negative'}


def clean_text(df, text_column, text_column_name = "content"):
  import re
  new_texts = []
  for text in df[text_column]:
    text_list = str(text).lower().split(" ")
    new_string_list = []
    for word in text_list:
      if 'http' in word:
        word = "url"
      elif ('@' in word) and (len(word) > 1):
        word = "@"
      if (len(word) > 1) and not word == 'amp' and not (word.isnumeric()):
        new_string_list.append(word)
    new_string = " ".join(new_string_list)
    new_string = re.sub("\d+", " ", new_string)
    new_string = new_string.replace('\n', ' ')
    new_string = new_string.replace('  ', ' ')
    new_string = new_string.strip()
    new_texts.append(new_string)
  df[text_column_name] = new_texts
  return df

def make_word_cloud(comment_words, width = 1100, height = 650, colour = "black", colormap = "brg"):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(collocations=False, max_words=100, colormap=colormap, width = width, height = height,
                background_color ='black',
                min_font_size = 16, stopwords = stopwords).generate(comment_words) # , stopwords = stopwords

    fig, ax = plt.subplots(figsize = (width/ 100, height/100), facecolor = colour)
    ax.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()
    return fig, wordcloud.words_.keys()


def prepare_cloud_lexeme_data(data_neutral, data_support, data_attack):

  # neutral df
  neu_text = " ".join(data_neutral['sentence_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
  count_dict_df_neu_text = Counter(neu_text.split(" "))
  df_neu_text = pd.DataFrame( {"word": list(count_dict_df_neu_text.keys()),
                              'neutral #': list(count_dict_df_neu_text.values())} )
  df_neu_text.sort_values(by = 'neutral #', inplace=True, ascending=False)
  df_neu_text.reset_index(inplace=True, drop=True)
  #df_neu_text = df_neu_text[~(df_neu_text.word.isin(stops))]

  # support df
  supp_text = " ".join(data_support['sentence_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
  count_dict_df_supp_text = Counter(supp_text.split(" "))
  df_supp_text = pd.DataFrame( {"word": list(count_dict_df_supp_text.keys()),
                              'support #': list(count_dict_df_supp_text.values())} )

  df_supp_text.sort_values(by = 'support #', inplace=True, ascending=False)
  df_supp_text.reset_index(inplace=True, drop=True)
  #df_supp_text = df_supp_text[~(df_supp_text.word.isin(stops))]

  merg = pd.merge(df_supp_text, df_neu_text, on = 'word', how = 'outer')

  #attack df
  att_text = " ".join(data_attack['sentence_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
  count_dict_df_att_text = Counter(att_text.split(" "))
  df_att_text = pd.DataFrame( {"word": list(count_dict_df_att_text.keys()),
                              'attack #': list(count_dict_df_att_text.values())} )

  df_att_text.sort_values(by = 'attack #', inplace=True, ascending=False)
  df_att_text.reset_index(inplace=True, drop=True)
  #df_att_text = df_att_text[~(df_att_text.word.isin(stops))]

  df2 = pd.merge(merg, df_att_text, on = 'word', how = 'outer')
  df2.fillna(0, inplace=True)
  df2['general #'] = df2['support #'] + df2['attack #'] + df2['neutral #']
  df2['word'] = df2['word'].str.replace("'", "_").replace("”", "_").replace("’", "_")
  return df2


import random
def wordcloud_lexeme(dataframe, lexeme_threshold = 90, analysis_for = 'support', cmap_wordcloud = 'Greens'):
  '''
  analysis_for:
  'support',
  'attack',
  'both' (both support and attack)

  cmap_wordcloud: best to choose from:
  gist_heat, flare_r, crest, viridis

  '''
  if analysis_for == 'attack':
    #print(f'Analysis for: {analysis_for} ')
    cmap_wordcloud = 'Reds' #gist_heat
    dataframe['precis'] = (round(dataframe['attack #'] / dataframe['general #'], 3) * 100).apply(float) # att
  elif analysis_for == 'both':
    #print(f'Analysis for: {analysis_for} ')
    cmap_wordcloud = 'autumn' #viridis
    dataframe['precis'] = (round((dataframe['support #'] + dataframe['attack #']) / dataframe['general #'], 3) * 100).apply(float) # both supp & att
  else:
    #print(f'Analysis for: {analysis_for} ')
    dataframe['precis'] = (round(dataframe['support #'] / dataframe['general #'], 3) * 100).apply(float) # supp

  dfcloud = dataframe[(dataframe['precis'] >= int(lexeme_threshold)) & (dataframe['general #'] > 3) & (dataframe.word.map(len)>3)]
  #print(f'There are {len(dfcloud)} words for the analysis of language {analysis_for} with precis threshold equal to {lexeme_threshold}.')
  n_words = dfcloud['word'].nunique()
  text = []
  for i in dfcloud.index:
    w = dfcloud.loc[i, 'word']
    w = str(w).strip()
    if analysis_for == 'both':
      n = int(dfcloud.loc[i, 'support #'] + dfcloud.loc[i, 'attack #'])
    else:
      n = int(dfcloud.loc[i, str(analysis_for)+' #']) #  + dfcloud.loc[i, 'attack #']   dfcloud.loc[i, 'support #']+  general
    l = np.repeat(w, n)
    text.extend(l)

  import random
  random.shuffle(text)
  st.write(f"There are {n_words} words.")
  if n_words < 1:
      st.error('No words with a specified threshold. \n Choose lower value of threshold.')
      st.stop()
  figure_cloud, figure_cloud_words = make_word_cloud(" ".join(text), 1000, 620, '#1E1E1E', str(cmap_wordcloud)) #gist_heat / flare_r crest viridis
  return figure_cloud, dfcloud, figure_cloud_words



def transform_text(dataframe, text_column):
  data = dataframe.copy()
  pos_column = []

  for doc in nlp.pipe(data[text_column].apply(str)):
    pos_column.append(" ".join(list( token.pos_ for token in doc)))
  data["POS_tags"] = pos_column
  return data



def UserRhetStrategy(data_list):
    st.write("## Rhetoric Strategy")
    add_spacelines()
    rhetoric_dims = ['ethos', 'pathos']
    df = data_list[0].copy()
    user_stats_df = user_stats_app(data = df)
    user_stats_df.fillna(0, inplace=True)
    cc = ['size',
          'ethos_n', 'ethos_support_n', 'ethos_attack_n',
          'pathos_n', 'pathos_negative_n', 'pathos_positive_n',
          ]
    user_stats_df[cc] = user_stats_df[cc].astype('int')
    user_stats_df_desc = user_stats_df.describe().round(3)
    cols_strat_zip = [
                    ('ethos_support_percent', 'pathos_positive_percent'),
                    ('ethos_attack_percent', 'pathos_negative_percent'),
                        ]
    cols_strat = ['ethos_support_percent', 'pathos_positive_percent',
                'ethos_attack_percent', 'pathos_negative_percent'
                  ]
    user_stats_df[cols_strat] = user_stats_df[cols_strat].round(-1)
    user_stats_df[cols_strat] = user_stats_df[cols_strat].astype('int')
    range_list = []
    number_users = []
    rhetoric_list = []
    bin_low = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
    bin_high = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    dimensions = ['ethos_support_percent', 'pathos_positive_percent']
    for dim in dimensions:
        for val in zip(bin_low, bin_high):
            rhetoric_list.append(dim)
            range_list.append(str(val))
            count_users = len(user_stats_df[ (user_stats_df[dim] >= int(val[0])) & (user_stats_df[dim] <= int(val[1]))])
            number_users.append(count_users)
    heat_df = pd.DataFrame({'range': range_list, 'values': number_users, 'dimension':rhetoric_list})
    heat_df['dimension'] = heat_df['dimension'].str.replace("_percent", "")

    heat_grouped = heat_df.pivot(index='range', columns='dimension', values='values')
    range_list_at = []
    number_users_at = []
    rhetoric_list_at = []
    dimensions_at = ['ethos_attack_percent', 'pathos_negative_percent']
    for dim in dimensions_at:
        for val in zip(bin_low, bin_high):
            rhetoric_list_at.append(dim)
            range_list_at.append(str(val))
            count_users = len(user_stats_df[ (user_stats_df[dim] >= int(val[0])) & (user_stats_df[dim] <= int(val[1]))])
            number_users_at.append(count_users)
    heat_df_at = pd.DataFrame({'range': range_list_at, 'values': number_users_at, 'dimension':rhetoric_list_at})
    heat_df_at['dimension'] = heat_df_at['dimension'].str.replace("_percent", "")
    heat_grouped_at = heat_df_at.pivot(index='range', columns='dimension', values='values')

    sns.set(style = 'whitegrid', font_scale=1.4)
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    sns.heatmap(heat_grouped_at, ax=axes[1], cmap='Reds', linewidths=0.1, annot=True)
    sns.heatmap(heat_grouped, ax=axes[0], cmap='Greens', linewidths=0.1, annot=True)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("range - percentage of texts %\n")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    plt.tight_layout(pad = 3)
    plt.show()
    st.write("#### Strategies - Overview")
    _, hsm, _ = st.columns([1, 15, 1])
    with hsm:
        st.pyplot(fig)
    add_spacelines(2)

    sns.set(style = 'whitegrid', font_scale=1.06)
    fig, axes = plt.subplots(2, 1, figsize=(10, 14))
    axes = axes.flatten()
    nz = 0
    for cz in cols_strat_zip[:1]:
        data_crosstab = pd.crosstab(user_stats_df[ user_stats_df[[cz[0], cz[1]]].all(axis=1) ][cz[0]],
                                    user_stats_df[ user_stats_df[[cz[0], cz[1]]].all(axis=1) ][cz[1]],
                                    margins = False)
        #data_crosstab = data_crosstab[data_crosstab[data_crosstab.columns].any(axis=1)]
        #data_crosstab = data_crosstab[data_crosstab[data_crosstab.columns].any(axis=0)]
        htmp = sns.heatmap(data_crosstab, ax=axes[nz], cmap='Greens', linewidths=0.1, annot=True)
        nz += 1

    for cz in cols_strat_zip[1:]:
        data_crosstab = pd.crosstab(user_stats_df[ user_stats_df[[cz[0], cz[1]]].all(axis=1) ][cz[0]],
                                    user_stats_df[ user_stats_df[[cz[0], cz[1]]].all(axis=1) ][cz[1]],
                                    margins = False)
        #data_crosstab = data_crosstab[data_crosstab[data_crosstab.columns].any(axis=1)]
        #data_crosstab = data_crosstab[data_crosstab[data_crosstab.columns].any(axis=0)]
        htmn = sns.heatmap(data_crosstab, ax=axes[nz], cmap='Reds', linewidths=0.1, annot=True)
        nz += 1
    htmp.set_yticklabels(htmp.get_yticklabels(), rotation=0)
    htmn.set_yticklabels(htmn.get_yticklabels(), rotation=0)
    plt.tight_layout(pad = 2.3)
    #plt.xticks(rotation = 0)
    #plt.yticks(rotation = 0)
    plt.show()

    # need to adjust it
    #st.write("#### Strategies - Cross View")
    #_, hsm2, _ = st.columns([1, 3, 1])
    #with hsm2:
        #st.pyplot(fig)
    #st.write('***********************************************************************')





def UserRhetStrategy1(data_list):
    st.write(f" ### Rhetoric Strategies")
    df = data_list[0].copy()
    if len(data_list) > 1:
        for i, d in enumerate(data_list):
            df = pd.concat( [df, data_list[i+1]], axis=0, ignore_index=True )
    add_spacelines(1)
    plot_type_strategy = st.radio("Type of the plot", ('heatmap', 'histogram'))
    add_spacelines(1)

    rhetoric_dims = ['ethos', 'pathos']
    pathos_cols = ['pathos_label']

    user_stats_df = user_stats_app(df)
    user_stats_df.fillna(0, inplace=True)
    for c in ['text_n', 'ethos_n', 'ethos_support_n', 'ethos_attack_n', 'pathos_n', 'pathos_negative_n', 'pathos_positive_n']:
           user_stats_df[c] = user_stats_df[c].apply(int)

    user_stats_df_desc = user_stats_df.describe().round(3)
    cols_strat = ['ethos_support_percent', 'ethos_attack_percent',
                  'pathos_positive_percent', 'pathos_negative_percent']
    if plot_type_strategy == 'histogram':
        def plot_strategies(data):
            i = 0
            for c in range(2):
                sns.set(font_scale=1, style='whitegrid')
                print(cols_strat[c+i], cols_strat[c+i+1])
                fig_stats, axs = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
                axs[0].hist(data[cols_strat[c+i]], color='#009C6F')
                title_str0 = " ".join(cols_strat[c+i].split("_")[:-1]).capitalize()
                axs[0].set_title(title_str0)
                axs[0].set_ylabel('number of users\n')
                axs[0].set_xlabel('\npercentage of texts %')
                axs[0].set_xticks(np.arange(0, 101, 10))

                axs[1].hist(data[cols_strat[c+i+1]], color='#9F0155')
                title_str1 = " ".join(cols_strat[c+i+1].split("_")[:-1]).capitalize()
                axs[1].set_xlabel('\npercentage of texts %')
                axs[1].yaxis.set_tick_params(labelbottom=True)
                axs[1].set_title(title_str1)
                axs[1].set_xticks(np.arange(0, 101, 10))
                plt.show()
                i+=1
                st.pyplot(fig_stats)
                add_spacelines(2)
        plot_strategies(data = user_stats_df)

    elif plot_type_strategy == 'heatmap':
        range_list = []
        number_users = []
        rhetoric_list = []
        bin_low = [0, 11, 21, 31, 41, 51, 61, 71, 81, 91]
        bin_high = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        dimensions = ['ethos_support_percent', 'pathos_positive_percent']
        for dim in dimensions:
            for val in zip(bin_low, bin_high):
                rhetoric_list.append(dim)
                range_list.append(str(val))
                count_users = len(user_stats_df[ (user_stats_df[dim] >= int(val[0])) & (user_stats_df[dim] <= int(val[1]))])
                number_users.append(count_users)
        heat_df = pd.DataFrame({'range': range_list, 'values': number_users, 'dimension':rhetoric_list})
        heat_df['dimension'] = heat_df['dimension'].str.replace("_percent", "")
        heat_grouped = heat_df.pivot(index='range', columns='dimension', values='values')

        range_list_at = []
        number_users_at = []
        rhetoric_list_at = []
        dimensions_at = ['ethos_attack_percent', 'pathos_negative_percent']
        for dim in dimensions_at:
            for val in zip(bin_low, bin_high):
                rhetoric_list_at.append(dim)
                range_list_at.append(str(val))
                count_users = len(user_stats_df[ (user_stats_df[dim] >= int(val[0])) & (user_stats_df[dim] <= int(val[1]))])
                number_users_at.append(count_users)
        heat_df_at = pd.DataFrame({'range': range_list_at, 'values': number_users_at, 'dimension':rhetoric_list_at})
        heat_df_at['dimension'] = heat_df_at['dimension'].str.replace("_percent", "")
        heat_grouped_at = heat_df_at.pivot(index='range', columns='dimension', values='values')

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        sns.heatmap(heat_grouped_at, ax=axes[1], cmap='Reds', linewidths=0.1, annot=True)
        sns.heatmap(heat_grouped, ax=axes[0], cmap='Greens', linewidths=0.1, annot=True)
        axes[0].set_xlabel("")
        axes[0].set_ylabel("range - percentage of texts %\n")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("")
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)
        plt.show()
        st.pyplot(fig)
        add_spacelines(2)

    with st.container():
        ethos_strat = user_stats_df[(user_stats_df.ethos_percent > user_stats_df.ethos_percent.std()+user_stats_df.ethos_percent.mean()) & \
                (user_stats_df.pathos_percent < user_stats_df.pathos_percent.std()+user_stats_df.pathos_percent.mean())]

        pathos_strat = user_stats_df[(user_stats_df.ethos_percent < user_stats_df.ethos_percent.std()+user_stats_df.ethos_percent.mean()) & \
                (user_stats_df.pathos_percent > user_stats_df.pathos_percent.std()+user_stats_df.pathos_percent.mean())]

        col1, col2, col3 = st.columns([1, 3, 3])
        with col1:
            st.write('')
        with col2:
            st.write(f"Dominant **ethos** strategy ")
            col2.metric(str(ethos_strat.shape[0]) + " users", str(round(ethos_strat.shape[0] / len(user_stats_df) * 100, 1)) + "%")

        with col3:
            st.write(f"Dominant **pathos** strategy ")
            col3.metric(str(pathos_strat.shape[0]) + " users", str(round(pathos_strat.shape[0] / len(user_stats_df) * 100, 1)) + "%")
        #add_spacelines(2)
        #dominant_percent_strategy = round(pathos_strat.shape[0] / len(user_stats_df) * 100, 1) + round(ethos_strat.shape[0] / len(user_stats_df) * 100, 1)
        #col2.write(f"##### **{round(dominant_percent_strategy, 1)}%** of users have one dominant rhetoric strategy.")
        add_spacelines(2)



emosn = ['sadness', 'anger', 'fear', 'disgust']
emosp = ['joy'] # 'surprise'
emos_map = {'joy':'emotion_positive', 'surprise':2, 'sadness':'emotion_negative', 'anger':'emotion_negative',
            'fear':'emotion_negative', 'disgust':'emotion_negative', 'neutral':'emotion_neutral'}

# app version
#@st.cache
def user_stats_app(data, source_column = 'source', ethos_column = 'ethos_label', emotion_column = 'pathos_label'):
  dataframe = data.copy() # data_list[0].copy()
  dataframe[source_column] = dataframe[source_column].astype('str')

  if not 'neutral' in dataframe[ethos_column]:
      dataframe[ethos_column] = dataframe[ethos_column].map(ethos_mapping)
  if not 'neutral' in dataframe[emotion_column]:
      dataframe[emotion_column] = dataframe[emotion_column].map(valence_mapping)

  sources_list = dataframe[dataframe[source_column] != 'nan'][source_column].unique()
  dataframe = dataframe[dataframe[source_column].isin(sources_list)]
  dataframe = dataframe.rename(columns = {'sentence':'text'})
  df = pd.DataFrame(columns = ['user', 'text_n',
                               'ethos_n', 'ethos_support_n', 'ethos_attack_n',
                               'pathos_n', 'pathos_negative_n', 'pathos_positive_n',
                             'ethos_percent', 'ethos_support_percent', 'ethos_attack_percent',
                             'pathos_percent', 'pathos_negative_percent', 'pathos_positive_percent',
                             ])
  users_list = []
  d1 = dataframe.groupby(source_column, as_index=False).size()
  d1 = d1[ d1['size'] > 1]
  sources_list = d1[source_column].unique()
  dataframe = dataframe[dataframe[source_column].isin(sources_list)]

  d2 = dataframe.groupby([source_column, ethos_column], as_index=False)['text'].size()
  d2 = d2.pivot(index = source_column, columns = ethos_column, values = 'size')
  d2 = d2.fillna(0).reset_index()
  d2.columns = ['ethos_'+c+"_n" if i >= 1 else c for i, c in enumerate(d2.columns) ]

  d22 = pd.DataFrame(dataframe.groupby(source_column)[ethos_column].value_counts(normalize=True).round(3)*100)
  d22.columns = ['percent']
  d22 = d22.reset_index()
  d22 = d22.pivot(index = source_column, columns = ethos_column, values = 'percent')
  d22 = d22.fillna(0).reset_index()
  d22.columns = ['ethos_'+c+"_percent" if i >= 1 else c for i, c in enumerate(d22.columns) ]
  #st.dataframe(d2)
  #st.dataframe(d22)
  d3 = dataframe.groupby([source_column, emotion_column], as_index=False)['text'].size()
  d3 = d3.pivot(index = source_column, columns = emotion_column, values = 'size')
  d3 = d3.fillna(0).reset_index()
  d3 = d3[[source_column,  'negative', 'positive']]
  d3.columns = ['pathos_'+c+"_n" if i >= 1 else c for i, c in enumerate(d3.columns) ]
  #st.dataframe(d3)
  d32 = pd.DataFrame(dataframe.groupby(source_column)[emotion_column].value_counts(normalize=True).round(3)*100)
  d32.columns = ['percent']
  d32 = d32.reset_index()
  d32 = d32.pivot(index = source_column, columns = emotion_column, values = 'percent')
  d32 = d32.fillna(0).reset_index()
  d32 = d32[[source_column,  'negative', 'positive']]
  d32.columns = ['pathos_'+c+"_percent" if i >= 1 else c for i, c in enumerate(d32.columns) ]

  df = d1.merge(d2, on = source_column, how = 'left')
  df = df.merge(d22, on = source_column, how = 'left')
  df = df.merge(d3, on = source_column, how = 'left')
  df = df.merge(d32, on = source_column, how = 'left')
  #df = df.fillna(0)
  #st.dataframe(df)
  df['pathos_n'] = df.pathos_negative_n + df.pathos_positive_n
  df['ethos_n'] = df.ethos_attack_n + df.ethos_support_n
  return df


def standardize(data):
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  data0 = data.copy()
  scaled_values = scaler.fit_transform(data0)
  data0.loc[:, :] = scaled_values
  return data0


def user_rhetoric_v2(data, source_column = 'source', ethos_col = 'ethos_label',
                  pathos_col = 'pathos_label'):

  import warnings
  dataframe = data.copy()

  dataframe[source_column] = dataframe[source_column].apply(str)
  sources_list = dataframe[ ~(dataframe[source_column].isin(['nan', ''])) ][source_column].unique()
  metric_value = []
  users_list = []
  if not 'neutral' in dataframe[ethos_col]:
      dataframe[ethos_col] = dataframe[ethos_col].map(ethos_mapping)
  if not 'neutral' in dataframe[pathos_col]:
      dataframe[pathos_col] = dataframe[pathos_col].map(valence_mapping)

  map_ethos_weight = {'attack':-1, 'neutral':0, 'support':1}
  map_pathos_weight = {'negative':-1, 'neutral':0, 'positive':1}
  for u in sources_list:
    users_list.append(str(u))
    df_user = dataframe[dataframe[source_column] == u]
    ethos_pathos_user = 0
    df_user_rhetoric = df_user.groupby([str(pathos_col), str(ethos_col)], as_index=False).size()
    # map weights
    df_user_rhetoric[pathos_col] = df_user_rhetoric[pathos_col].map(map_pathos_weight)
    df_user_rhetoric[ethos_col] = df_user_rhetoric[ethos_col].map(map_ethos_weight)

    ethos_pathos_sum_ids = []

    for id in df_user_rhetoric.index:
      ethos_pathos_val = np.sum(df_user_rhetoric.loc[id, str(pathos_col):str(ethos_col)].to_numpy())
      ethos_pathos_val = ethos_pathos_val * df_user_rhetoric.loc[id, 'size']
      ethos_pathos_sum_ids.append(ethos_pathos_val)

    ethos_pathos_user = np.sum(ethos_pathos_sum_ids)
    try:
        metric_value.append(int(ethos_pathos_user))
    except:
        metric_value.append(0)
  df = pd.DataFrame({'user': users_list, 'rhetoric_metric': metric_value})
  return df




def add_spacelines(number_sp=2):
    for xx in range(number_sp):
        st.write("\n")


@st.cache_data#(allow_output_mutation=True)
def load_data(file_path, indx = True, indx_col = 0):
  '''Parameters:
  file_path: path to your excel or csv file with data,

  indx: boolean - whether there is index column in your file (usually it is the first column) --> default is True

  indx_col: int - if your file has index column, specify column number here --> default is 0 (first column)
  '''
  if indx == True and file_path.endswith(".xlsx"):
    data = pd.read_excel(file_path, index_col = indx_col)
  elif indx == False and file_path.endswith(".xlsx"):
    data = pd.read_excel(file_path)

  elif indx == True and file_path.endswith(".csv"):
    data = pd.read_csv(file_path, index_col = indx_col)
  elif indx == False and file_path.endswith(".csv"):
    data = pd.read_csv(file_path)
  return data


@st.cache_data
def lemmatization(dataframe, text_column = 'sentence', name_column = False):
  '''Parameters:
  dataframe: dataframe with your data,

  text_column: name of a column in your dataframe where text is located
  '''
  df = dataframe.copy()
  lemmas = []
  for doc in nlp.pipe(df[text_column].astype('str')):
    lemmas.append(" ".join([token.lemma_ for token in doc if (not token.is_punct and not token.is_stop and not token.like_num and len(token) > 1) ]))

  if name_column:
      df[text_column] = lemmas
  else:
      df[text_column+"_lemmatized"] = lemmas
  return df



def ttr_lr(t, n, definition = 'TTR'):
    '''
    https://core.ac.uk/download/pdf/82620241.pdf
    Torruella, J., & Capsada, R. (2013). Lexical statistics and tipological structures: a measure of lexical richness. Procedia-Social and Behavioral Sciences, 95, 447-454.

    definition:
    TTR (type-token ratio) (1957, Templin),
    RTTR (root type-token ratio) (1960, Giraud),
    CTTR (corrected type-token ratio) (1964, Carrol),
    H (1960, Herdan),
    M (1966, Mass),
    '''
    definition = str(definition).upper()
    if definition == 'TTR':
        coeff = round(t / n, 2)
    elif definition == 'RTTR':
        coeff = round(t / np.sqrt(n), 2)
    elif definition == 'CTTR':
        coeff = round(t / np.sqrt(n*2), 2)
    elif definition == 'H':
        coeff = round(np.log(t) / np.log(n), 2)
    elif definition == 'M':
        coeff = round((np.log(n) - np.log(t)) / np.log2(n), 2)
    return coeff



def compnwords(dataframe, column_name = 'sentence'):
    data = dataframe.copy()
    data['nwords'] = data[column_name].astype('str').str.split().map(len)
    return data





def StatsLog(df_list, an_type = 'ADU-based'):
    #st.write("#### Sentence Length Analysis")
    add_spacelines(2)

    st.write("#### ADU-based analytics")
    conn_list = [' Logos Attack', ' Logos Support']
    map_naming = {'attack':'Ethos Attack', 'neutral':'Neutral', 'support':'Ethos Support',
            'Default Conflict': ' Logos Attack',
            'Default Rephrase' : ' Neutral',
            'Default Inference' : ' Logos Support'}
    rhetoric_dims = ['ethos', 'logos']
    df_list_et = df_list[0]
    df_list_et['nwords'] = df_list_et['sentence'].str.split().map(len)
    if not 'neutral' in df_list_et['ethos_label'].unique():
        df_list_et['ethos_label'] = df_list_et['ethos_label'].map(ethos_mapping).map(map_naming)
    df_list_log = df_list[1]
    import re
    df_list_log['locution_conclusion'] = df_list_log.locution_conclusion.apply(lambda x: " ".join( str(x).split(':')[1:]) )
    df_list_log['locution_premise'] = df_list_log.locution_premise.apply(lambda x: " ".join( str(x).split(':')[1:]) )
    df_list_log['sentence'] = df_list_log.locution_premise.astype('str')# + " " + df_list_log.locution_conclusion.astype('str')
    df_list_log['nwords_conclusion'] = df_list_log['locution_conclusion'].str.split().map(len)
    df_list_log['nwords_premise'] = df_list_log['locution_premise'].str.split().map(len)
    df_list_log['nwords'] = df_list_log[['nwords_conclusion', 'nwords_premise']].mean(axis=1).round(2)

    df_list_log.connection = df_list_log.connection.map(map_naming)
    df_list_log_stats = df_list_log.groupby(['connection'], as_index=False)['nwords'].mean().round(2)
    log_all = df_list_log_stats[df_list_log_stats.connection.isin([' Logos Attack', ' Logos Support'])].nwords.mean().round(2)
    df_list_log_stats.loc[len(df_list_log_stats)] = [' Logos All', log_all]

    df_list_et_stats = df_list_et.groupby(['ethos_label'], as_index=False)['nwords'].mean().round(2)
    et_all = df_list_et_stats[df_list_et_stats.ethos_label.isin(['Ethos Support','Ethos Attack'])].nwords.mean().round(2)
    df_list_et_stats.loc[len(df_list_et_stats)] = ['Ethos All', et_all]
    #st.stop()

    if an_type == 'Relation-based':
            df_list_log_stats = df_list_log.groupby(['id_connection', 'connection'], as_index=False)[['nwords_premise', 'nwords_conclusion']].sum().round(2)
            df_list_log_stats = df_list_log_stats.groupby(['connection'], as_index=False)[['nwords_premise', 'nwords_conclusion']].mean().round(2)
            df_list_log_stats['nwords'] = df_list_log_stats[['nwords_conclusion', 'nwords_premise']].mean(axis=1).round(2)
            log_all = df_list_log_stats[df_list_log_stats.connection.isin([' Logos Attack', ' Logos Support'])].nwords.mean().round(2)
            df_list_log_stats.loc[len(df_list_log_stats)] = [' Logos All', log_all]

            #st.write(df_list_log_stats)

    #df_list_et = compnwords(df_list_et, column_name = 'sentence')
    #df_list_log_stats = compnwords(df_list_log_stats, column_name = 'sentence')

    cet_desc, c_log_stats_desc = st.columns(2)
    #df_list_et_desc = pd.DataFrame(df_list_et[df_list_et.ethos_label.isin(['Ethos Support','Ethos Attack'])].groupby('ethos_label').nwords.describe().round(2).iloc[:, 1:])
    df_list_et_desc = pd.DataFrame(df_list_et.groupby('ethos_label').nwords.describe().round(2).iloc[:, 1:])
    df_list_et_desc = df_list_et_desc.T
    with cet_desc:
        st.write("ADU Length for **Ethos**: ")
        st.write(df_list_et_desc)
    #df_list_log_stats_desc = pd.DataFrame(df_list_log_stats[df_list_log_stats.connection.isin(conn_list)].groupby('connection').nwords.describe().round(2).iloc[:, 1:])
    conn_list = [' Logos Attack', ' Logos Support', ' Neutral']
    df_list_log_stats_desc = pd.DataFrame(df_list_log[df_list_log.connection.isin(conn_list)].groupby('connection').nwords.describe().round(2).iloc[:, 1:])
    if an_type == 'Relation-based':
        df_list_log_stats = df_list_log.groupby(['id_connection', 'connection'], as_index=False)[['nwords_premise', 'nwords_conclusion']].sum().round(2)
        df_list_log_stats_desc = pd.DataFrame(df_list_log_stats[df_list_log_stats.connection.isin(conn_list)].groupby('connection').nwords.describe().round(2).iloc[:, 1:])

    df_list_log_stats_desc = df_list_log_stats_desc.T
    with c_log_stats_desc:
        st.write("ADU Length for **Logos**: ")
        st.write(df_list_log_stats_desc)

    add_spacelines(1)
    #cstat1, cstat2, cstat3, cstat4 = st.columns(4)
    cstat1, cstat2, _, _ = st.columns(4)
    with cstat1:
        le = df_list_et_desc.loc['mean', 'Ethos Attack']
        ll = df_list_log_stats_desc.loc['mean', ' Logos Attack']
        lrel = round((le *100 / ll)- 100, 2)
        #st.write(le, ll, lrel)
        st.metric('Ethos Attack vs. Logos Attack', f" {le} vs. {ll} ", str(lrel)+'%')

    with cstat2:
        le = df_list_et_desc.loc['mean', 'Ethos Support']
        ll = df_list_log_stats_desc.loc['mean', ' Logos Support']
        lrel = round((le *100 / ll)- 100, 2)
        st.metric('Ethos Support vs. Logos Support', f" {le} vs. {ll} ", str(lrel)+'%')

    #with cstat3:
        #le = df_list_et_desc.loc['mean', 'Ethos Attack']
        #ll = df_list_log_stats_desc.loc['mean', ' Logos Attack']
        #lrel = round((ll *100 / le)- 100, 2)
        #st.metric('Logos Attack vs. Ethos Attack', f" {ll} vs. {le} ", str(lrel)+'%')

    #with cstat4:
        #le = df_list_et_desc.loc['mean', 'Ethos Support']
        #ll = df_list_log_stats_desc.loc['mean', ' Logos Support']
        #lrel = round((ll *100 / le)- 100, 2)
        #st.metric('Logos Support vs. Ethos Support', f" {ll} vs. {le} ", str(lrel)+'%')

    #st.write(df_list_log_stats)
    #st.write(df_list_et_stats)
    df_list_et_stats.columns = ['connection', 'nwords']
    df_list_desc = pd.concat( [df_list_log_stats,
                                df_list_et_stats], axis = 0, ignore_index=True )

    #df_list_desc = df_list_desc.reset_index()
    #st.write(df_list_desc)
    #st.stop()
    df_list_desc.columns = ['category', 'mean']
    df_list_desc.loc[:4, 'dimension'] = 'Logos'
    df_list_desc.loc[4:, 'dimension'] = 'Ethos'
    df_list_desc = df_list_desc.sort_values(by = ['dimension', 'category'])
    #df_list_desc['category'] = df_list_desc['category'].str.replace(' Ethos Neutral', 'Neutral').str.replace(' Logos  Neutral', ' Neutral')

    sns.set(font_scale = 1.4, style = 'whitegrid')
    f_desc = sns.catplot(data = df_list_desc, x = 'category', y = 'mean', col = 'dimension',
                kind = 'bar', palette = {'Ethos Attack':'#BB0000', 'Neutral':'#3B3591', 'Ethos Support':'#026F00', 'Ethos All':'#6C6C6E',
                        ' Logos Attack':'#BB0000', ' Neutral':'#3B3591', ' Logos Support':'#026F00', ' Logos All':'#6C6C6E'},
                        height = 4, aspect = 1.4, sharex=False)
    f_desc.set(xlabel = '', ylabel = 'mean ADU length', ylim = (0, np.max(df_list_desc['mean']+2)))
    f_desc.set_xticklabels(fontsize = 13)
    for ax in f_desc.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
            ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    st.pyplot(f_desc)
    st.write("************************************************************************")

    st.write("#### Word-based analytics")
    df_list_et['nwords'] = df_list_et['sentence'].astype('str').apply(lambda x: np.mean( [ len(w)  for w in x.split()] ) )
    #df_list_log_stats['nwords'] = df_list_log_stats['sentence'].astype('str').apply(lambda x: np.mean( [ len(w)  for w in x.split()] ) )
    df_list_log['nwords_conclusion'] = df_list_log['locution_conclusion'].astype('str').apply(lambda x: np.mean( [ len(w)  for w in x.split()] ) )
    df_list_log['nwords_premise'] = df_list_log['locution_premise'].astype('str').apply(lambda x: np.mean( [ len(w)  for w in x.split()] ) )
    df_list_log['nwords'] = df_list_log[['nwords_conclusion', 'nwords_premise']].mean(axis=1).round(2)


    df_list_log_stats = df_list_log.groupby(['connection'], as_index=False)['nwords'].mean().round(2)
    log_all = df_list_log_stats[df_list_log_stats.connection.isin([' Logos Attack', ' Logos Support'])].nwords.mean().round(2)
    df_list_log_stats.loc[len(df_list_log_stats)] = [' Logos All', log_all]

    df_list_et_stats = df_list_et.groupby(['ethos_label'], as_index=False)['nwords'].mean().round(2)
    et_all = df_list_et_stats[df_list_et_stats.ethos_label.isin(['Ethos Support','Ethos Attack'])].nwords.mean().round(2)
    df_list_et_stats.loc[len(df_list_et_stats)] = ['Ethos All', et_all]
    #st.stop()

    if an_type == 'Relation-based':
            df_list_log_stats = df_list_log.groupby(['id_connection', 'connection'], as_index=False)[['nwords_premise', 'nwords_conclusion']].sum().round(2)
            df_list_log_stats = df_list_log_stats.groupby(['connection'], as_index=False)[['nwords_premise', 'nwords_conclusion']].mean().round(2)
            df_list_log_stats['nwords'] = df_list_log_stats[['nwords_conclusion', 'nwords_premise']].mean(axis=1).round(2)
            log_all = df_list_log_stats[df_list_log_stats.connection.isin([' Logos Attack', ' Logos Support'])].nwords.mean().round(2)
            df_list_log_stats.loc[len(df_list_log_stats)] = [' Logos All', log_all]


    #cet_desc, c_log_stats_desc = st.columns(2)
    #df_list_et_desc = pd.DataFrame(df_list_et.groupby('ethos_label').nwords.describe().round(2).iloc[:, 1:])
    #df_list_et_desc = df_list_et_desc.T
    #with cet_desc:
        #st.write("Word Length for **Ethos**: ")
        #st.write(df_list_et_desc)
    #df_list_log_stats_desc = pd.DataFrame(df_list_log_stats.groupby('connection').nwords.describe().round(2).iloc[:, 1:])
    #df_list_log_stats_desc = df_list_log_stats_desc.T
    #with c_log_stats_desc:
        #st.write("Word Length for **Logos**: ")
        #st.write(df_list_log_stats_desc)

    df_list_et_stats.columns = ['connection', 'nwords']
    add_spacelines(1)
    cstat12, cstat22 ,_, _= st.columns(4)
    with cstat12:
        le = df_list_et_stats[df_list_et_stats.connection == 'Ethos Attack'].nwords.iloc[0]
        ll =  df_list_log_stats[df_list_log_stats.connection == ' Logos Attack'].nwords.iloc[0]
        lrel = round((le *100 / ll)- 100, 2)
        st.metric('Ethos Attack vs. Logos Attack', f" {le} vs. {ll} ", str(lrel)+'%')

    with cstat22:
        le = df_list_et_stats[df_list_et_stats.connection == 'Ethos Support'].nwords.iloc[0]
        ll =  df_list_log_stats[df_list_log_stats.connection == ' Logos Support'].nwords.iloc[0]
        lrel = round((le *100 / ll)- 100, 2)
        st.metric('Ethos Support vs. Logos Support', f" {le} vs. {ll} ", str(lrel)+'%')


    df_list_desc = pd.concat( [df_list_log_stats,
                                df_list_et_stats], axis = 0, ignore_index=True )

    df_list_desc.columns = ['category', 'mean']
    df_list_desc.loc[:4, 'dimension'] = 'Logos'
    df_list_desc.loc[4:, 'dimension'] = 'Ethos'
    df_list_desc = df_list_desc.sort_values(by = ['dimension', 'category'])
    #df_list_desc['category'] = df_list_desc['category'].str.replace(' Ethos Neutral', 'Neutral').str.replace(' Logos  Neutral', ' Neutral')


    sns.set(font_scale = 1.4, style = 'whitegrid')
    f_desc2 = sns.catplot(data = df_list_desc, x = 'category', y = 'mean', col = 'dimension',
                kind = 'bar', palette = {'Ethos Attack':'#BB0000', 'Neutral':'#3B3591', 'Ethos Support':'#026F00', 'Ethos All':'#6C6C6E',
                        ' Logos Attack':'#BB0000', ' Neutral':'#3B3591', ' Logos Support':'#026F00', ' Logos All':'#6C6C6E'},
                        height = 4, aspect = 1.4, sharex=False)
    f_desc2.set(xlabel = '', ylabel = 'mean word length', ylim = (0, np.max(df_list_desc['mean']+2)))
    f_desc2.set_xticklabels(fontsize = 13)
    for ax in f_desc2.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
            ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    st.pyplot(f_desc2)

    st.write("************************************************************************")

    lr_coeff = st.selectbox("Choose a measure of lexical richness",
                ['TTR', 'RTTR', 'CTTR', 'H', 'M'])

    with st.expander('Lexical richness measures'):
        st.write('''
        A first class of indices based on the direct relationship between the number of terms and words (type-token):
        **TTR**: (type-token ratio) (1957, Templin);\n


        TTR with corrections:
        **RTTR**:  (root type-token ratio) (1960, Giraud),
        **CTTR**: (corrected type-token ratio) (1964, Carrol);\n

        A second class of indices has been developed using formulae based on logarithmic function:
        **Herdan H**:  (1960, Herdan),
        **Mass M**:  (1966, Mass).
        ''')
    add_spacelines(1)
    dims_ttr = ['Ethos Attack', 'Ethos Support', ' Logos Attack', ' Logos Support']
    vals_ttr = []

    df_list_log['sentence'] = df_list_log['locution_premise'].astype('str') + " " + df_list_log['locution_conclusion'].astype('str')
    df_list_log_stats = lemmatization(df_list_log, 'sentence') # premise_lemmatized
    #st.write(df_list_log_stats)

    lr_def = str(lr_coeff)

    colttr1, colttr2, colttr3, colttr4 = st.columns(4)
    ttr_ea1 = " ".join( df_list_et[df_list_et['ethos_label'] == 'Ethos Attack']['sentence_lemmatized'].astype('str').str.lower().values )
    ttr_ea1 = ttr_ea1.split()
    df_list_et_targets = df_list_et.Target.dropna().str.lower().values
    ttr_ea1 = list(w for w in ttr_ea1 if not w in df_list_et_targets)
    ttr_ea1_token = len(ttr_ea1)
    ttr_ea1_type = len( set(ttr_ea1) )
    ttr_ea = ttr_lr(t = ttr_ea1_type, n = ttr_ea1_token, definition = lr_def)
    #ttr_ea = round(ttr_ea1_type / ttr_ea1_token, 2) # np.sqrt()  np.sqrt(ttr_ea1_token*2)
    vals_ttr.append(ttr_ea)
    with colttr1:
        st.metric('Lexical Richness of Ethos Attack', ttr_ea)


    ttr_es1 = " ".join( df_list_et[df_list_et['ethos_label'] == 'Ethos Support']['sentence_lemmatized'].astype('str').str.lower().values )
    ttr_es1 = ttr_es1.split()
    ttr_es1 = list(w for w in ttr_es1 if not w in df_list_et_targets)
    ttr_es1_token = len(ttr_es1)
    ttr_es1_type = len( set(ttr_es1) )
    ttr_es = ttr_lr(t = ttr_es1_type, n = ttr_es1_token, definition = lr_def)
    #ttr_es = round(ttr_es1_type / ttr_es1_token, 2)
    vals_ttr.append(ttr_es)
    with colttr2:
        st.metric('Lexical Richness of Ethos Support', ttr_es)

    ttr_ea1 = " ".join( df_list_log_stats[df_list_log_stats['connection'] == ' Logos Attack']['sentence_lemmatized'].astype('str').str.lower().values )
    ttr_ea1 = ttr_ea1.split()
    ttr_ea1 = list(w for w in ttr_ea1 if not w in df_list_et_targets)
    ttr_ea1_token = len(ttr_ea1)
    ttr_ea1_type = len( set(ttr_ea1) )
    ttr_ea = ttr_lr(t = ttr_ea1_type, n = ttr_ea1_token, definition = lr_def)
    #ttr_ea = round(ttr_ea1_type / ttr_ea1_token, 2)
    vals_ttr.append(ttr_ea)
    with colttr3:
        st.metric('Lexical Richness of Logos Attack', ttr_ea)


    ttr_es1 = " ".join( df_list_log_stats[df_list_log_stats['connection'] == ' Logos Support']['sentence_lemmatized'].astype('str').str.lower().values )
    ttr_es1 = ttr_es1.split()
    ttr_es1 = list(w for w in ttr_es1 if not w in df_list_et_targets)
    ttr_es1_token = len(ttr_es1)
    ttr_es1_type = len( set(ttr_es1) )
    ttr_es = ttr_lr(t = ttr_es1_type, n = ttr_es1_token, definition = lr_def)
    #ttr_es = round(ttr_es1_type / ttr_es1_token, 2)
    vals_ttr.append(ttr_es)
    with colttr4:
        st.metric('Lexical Richness of Logos Support', ttr_es)

    df_ttr_stats = pd.DataFrame({'category':dims_ttr, 'ttr ratio':vals_ttr})

    df_ttr_stats.loc[:1, 'dimension'] = 'Ethos'
    df_ttr_stats.loc[2:, 'dimension'] = 'Logos'
    df_ttr_stats = df_ttr_stats.sort_values(by = ['dimension', 'category'])
    val = round(np.max(df_ttr_stats['ttr ratio']) / 5, 2)

    sns.set(font_scale = 1.4, style = 'whitegrid')
    f_desc2 = sns.catplot(data = df_ttr_stats, x = 'category', y = 'ttr ratio', col = 'dimension',
                kind = 'bar', palette = {'Ethos Attack':'#BB0000', ' No Ethos':'#022D96', 'Ethos Support':'#026F00',
                        ' Logos Attack':'#BB0000', ' Logos  Rephrase':'#D7A000', ' Logos Support':'#026F00'},
                        height = 4, aspect = 1.4, sharex=False)
    f_desc2.set(xlabel = '', ylabel = lr_def,
                ylim = (0, np.max(df_ttr_stats['ttr ratio'])+val ) ) # 'TTR (type-token ratio)'  'CTTR \n(corrected type-token ratio)'
    for ax in f_desc2.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
            ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    st.pyplot(f_desc2)
    st.stop()



@st.cache_data
def PolarizingNetworksSub_new(df3):
    pos_n = []
    neg_n = []
    neu_n = []

    tuples_tree = {
     'pos': [],
     'neg': [],
     'neu': []}
    for i in df3.index:
     if df3.ethos_label.loc[i] == 'support':
       tuples_tree['pos'].append(tuple( [str(df3.source.loc[i]), str(df3.Target.loc[i]).replace("@", '')] ))
       pos_n.append(df3.source.loc[i])

     elif df3.ethos_label.loc[i] == 'attack':
       tuples_tree['neg'].append(tuple( [str(df3.source.loc[i]), str(df3.Target.loc[i]).replace("@", '')] ))
       neg_n.append(df3.source.loc[i])

     elif df3.ethos_label.loc[i] == 'neutral':
       tuples_tree['neu'].append(tuple( [str(df3.source.loc[i]), str(df3.Target.loc[i]).replace("@", '')] ))
       neu_n.append(df3.source.loc[i])

    G = nx.DiGraph()

    default_weight = 0.7
    for nodes in tuples_tree['neu']:
        n0 = nodes[0]
        n1 = nodes[1]
        if n0 != n1:
            if G.has_edge(n0,n1):

                if G[n0][n1]['weight'] < 4:
                    G[n0][n1]['weight'] += default_weight
            else:
                G.add_edge(n0,n1, weight=default_weight, color='blue')

    default_weight = 0.9
    for nodes in tuples_tree['pos']:
        n0 = nodes[0]
        n1 = nodes[1]
        if n0 != n1:

            if G.has_edge(n0,n1):
              ll = list(G.edges([n0], data=True))

              for nn0, ii0 in enumerate(ll):
                if ll[nn0][0] == n0 and ll[nn0][1] == n1:

                  if ll[nn0][-1]['color'] == 'green':

                    if G[n0][n1]['weight'] < 4:
                        G[n0][n1]['weight'] += default_weight
            else:
                G.add_edge(n0,n1, weight=default_weight, color='green')


    default_weight = 0.9
    for nodes in tuples_tree['neg']:
        n0 = nodes[0]
        n1 = nodes[1]
        if n0 != n1:

            if G.has_edge(n0,n1):
              ll = list(G.edges([n0], data=True))

              for nn0, ii0 in enumerate(ll):
                if ll[nn0][0] == n0 and ll[nn0][1] == n1:

                  if ll[nn0][-1]['color'] == 'red':

                    if G[n0][n1]['weight'] < 4:
                        G[n0][n1]['weight'] += default_weight
            else:
                G.add_edge(n0,n1, weight=default_weight, color='red')



    colors_nx_node = {}
    for n0 in G.nodes():
        if not (n0 in neu_n or n0 in neg_n or n0 in pos_n):
            colors_nx_node[n0] = 'grey'
        elif n0 in neu_n and not (n0 in neg_n or n0 in pos_n):
            colors_nx_node[n0] = 'blue'
        elif n0 in pos_n and not (n0 in neg_n or n0 in neu_n):
            colors_nx_node[n0] = 'green'
        elif n0 in neg_n and not (n0 in neu_n or n0 in pos_n):
            colors_nx_node[n0] = 'red'
        else:
            colors_nx_node[n0] = 'gold'
    nx.set_node_attributes(G, colors_nx_node, name="color")
    return G




def FellowsDevils_new(df_list):
    st.write("### Fellows - Devils")
    meth_feldev = 'frequency'#st.radio("Choose a method of calculation", ('frequency', 'log-likelihood ratio') )    selected_rhet_dim = 'ethos'
    #selected_rhet_dim = selected_rhet_dim+"_label"
    add_spacelines(1)
    df = df_list[0]
    #st.write(df)
    df['target'] = df.Target.values
    df['source'] = df.source.astype('str').str.strip()
    df.target = df.target.astype('str').str.strip()
    df = df[df.target != 'nan']
    if not 'neutral' in df['ethos_label'].unique():
        df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        df['ethos'] = df['ethos_label']

    df = df.drop_duplicates(subset = ['source', 'sentence'])
    df.target = df.target.str.replace('humans', 'people')
    df = df[df.ethos != 'neutral']

    src = df.source.unique()
    df['sentence_lemmatized'] = df['target']

    if meth_feldev != 'frequency':
        ddmsc = ['support', 'attack']
        odds_list_of_dicts = []
        effect_list_of_dicts = []
        freq_list_of_dicts = []
        # 1 vs rest
        #num = np.floor( len(df) / 10 )
        for ii, ddmsc1 in enumerate(ddmsc):
            dict_1vsall_percent = {}
            dict_1vsall_effect_size = {}
            dict_1vsall_freq = {}

            ddmsc12 = set(ddmsc).difference([ddmsc1])
            #all100popular = Counter(" ".join( df.lemmatized.values ).split()).most_common(100)
            #all100popular = list(w[0] for w in all100popular)

            ddmsc1w = " ".join( df[df[selected_rhet_dim] == ddmsc1 ].sentence_lemmatized.fillna('').astype('str').values ).split() # sentence_lemmatized
            c = len(ddmsc1w)
            #ddmsc1w = list(w for w in ddmsc1w if not w in all100popular)
            ddmsc1w = Counter(ddmsc1w).most_common() # num
            if ddmsc1 in ['positive', 'support']:
                ddmsc1w = [w for w in ddmsc1w if w[1] >= 3 ]
            else:
                ddmsc1w = [w for w in ddmsc1w if w[1] > 3 ]
            #print('**********')
            #print(len(ddmsc1w), ddmsc1w)
            #print([w for w in ddmsc1w if w[1] > 2 ])
            #print(len([w for w in ddmsc1w if w[1] > 2 ]))
            ddmsc1w_word = dict(ddmsc1w)

            #st.write( list(ddmsc12)[0] )

            ddmsc2w = " ".join( df[ df[selected_rhet_dim] == list(ddmsc12)[0] ].sentence_lemmatized.fillna('').astype('str').values ).split() # sentence_lemmatized
            d = len(ddmsc2w)
            #ddmsc2w = list(w for w in ddmsc2w if not w in all100popular)
            ddmsc2w = Counter(ddmsc2w).most_common()
            ddmsc2w_word = dict(ddmsc2w)


            ddmsc1w_words = list( ddmsc1w_word.keys() )
            for n, dim in enumerate( ddmsc1w_words ):

                a = ddmsc1w_word[dim]
                try:
                    b = ddmsc2w_word[dim]
                except:
                    b = 0.5

                ca = c-a
                bd = d-b

                E1 = c*(a+b) / (c+d)
                E2 = d*(a+b) / (c+d)

                g2 = 2*((a*np.log(a/E1)) + (b* np.log(b/E2)))
                g2 = round(g2, 2)

                odds = round( (a*(d-b)) / (b*(c-a)), 2)

                if odds > 1:

                    if g2 > 10.83:
                        #print(f"{dim, g2, odds} ***p < 0.001 ")
                        dict_1vsall_percent[dim] = odds
                        dict_1vsall_effect_size[dim] = 0.001
                        dict_1vsall_freq[dim] = a
                    elif g2 > 6.63:
                        #print(f"{dim, g2, odds} **p < 0.01 ")
                        dict_1vsall_percent[dim] = odds
                        dict_1vsall_effect_size[dim] = 0.01
                        dict_1vsall_freq[dim] = a
                    elif g2 > 3.84:
                        #print(f"{dim, g2, odds} *p < 0.05 ")
                        dict_1vsall_percent[dim] = odds
                        dict_1vsall_effect_size[dim] = 0.05
                        dict_1vsall_freq[dim] = a
            #print(dict(sorted(dict_1vsall_percent.items(), key=lambda item: item[1])))
            odds_list_of_dicts.append(dict_1vsall_percent)
            effect_list_of_dicts.append(dict_1vsall_effect_size)
            freq_list_of_dicts.append(dict_1vsall_freq)

        df_odds_pos = pd.DataFrame({
                    'word':odds_list_of_dicts[0].keys(),
                    'odds':odds_list_of_dicts[0].values(),
                    'frequency':freq_list_of_dicts[0].values(),
                    'effect_size_p':effect_list_of_dicts[0].values(),
        })
        df_odds_pos['category'] = 'fellows'
        df_odds_neg = pd.DataFrame({
                    'word':odds_list_of_dicts[1].keys(),
                    'odds':odds_list_of_dicts[1].values(),
                    'frequency':freq_list_of_dicts[1].values(),
                    'effect_size_p':effect_list_of_dicts[1].values(),
        })
        df_odds_neg['category'] = 'devils'

        df_odds_neg = df_odds_neg.sort_values(by = 'odds', ascending = False)
        df_odds_neg = df_odds_neg[df_odds_neg.word != 'user']

        df_odds_pos = df_odds_pos.sort_values(by = 'odds', ascending = False)
        df_odds_pos = df_odds_pos[df_odds_pos.word != 'user']

    else:
        df.target = df.target.str.replace("-", "").str.replace(".", "").str.replace(",", "")

        #df.target = df.target.str.replace("@", "")
        df = df[df.target != 'nan']
        df_odds_pos = pd.DataFrame( df[df.ethos == 'support'].target.value_counts() ).reset_index()
        df_odds_pos.columns = ['word', 'size']
        df_odds_pos = df_odds_pos[df_odds_pos['size'] > 1]
        df_odds_pos['category'] = 'fellows'
        df_odds_pos.index += 1

        df_odds_neg = pd.DataFrame( df[df.ethos == 'attack'].target.value_counts() ).reset_index()
        df_odds_neg.columns = ['word', 'size']
        df_odds_neg = df_odds_neg[df_odds_neg['size'] > 1]
        df_odds_neg['category'] = 'devils'
        df_odds_neg.index += 1


    targets_list = df.target.astype('str').unique()
    df_odds_pos = df_odds_pos[df_odds_pos.word.isin(targets_list)]
    df_odds_neg = df_odds_neg[df_odds_neg.word.isin(targets_list)]

    df_odds_neg = df_odds_neg.reset_index(drop=True)
    df_odds_pos = df_odds_pos.reset_index(drop=True)
    df_odds_pos.index += 1
    df_odds_neg.index += 1


    df_odds_pos_words = set(df_odds_pos.word.values)
    df_odds_neg_words = set(df_odds_neg.word.values)


    tab_odd, tab_fellow = st.tabs(['Tables', 'Analytics'])
    with tab_odd:
        oddpos_c, oddneg_c = st.columns(2, gap = 'large')
        cols_odds = ['source', 'sentence', 'ethos', 'target']


        with oddpos_c:
            st.write(f'Number of entities regarded as **{df_odds_pos.category.iloc[0]}**: {len(df_odds_pos)} ')
            st.dataframe(df_odds_pos)
            add_spacelines(1)
            pos_list_freq = df_odds_pos.word.tolist()
            #freq_word_pos = st.multiselect('Choose entities for network analytics', pos_list_freq, pos_list_freq[2])
            #df_odds_pos_words = set(freq_word_pos)
            #df0p = df[df.target.isin(df_odds_pos_words)]


        with oddneg_c:
            st.write(f'Number of entities regarded as **{df_odds_neg.category.iloc[0]}**: {len(df_odds_neg)} ')
            st.dataframe(df_odds_neg)
            add_spacelines(1)
            neg_list_freq = df_odds_neg.word.tolist()

        pos_list_freq.extend(neg_list_freq)
        list_freq = list( set(pos_list_freq) )

        #entity_list = list( set(pos_list_freq).union() )
        options1 = st.multiselect('First group of entities', list_freq, list( set(['Trump']).intersection(set(list_freq)) )[:1] )
        names_w_cops_grp2 = set(list_freq) - set(options1)
        names_w_cops_grp2 = list(names_w_cops_grp2)
        try:
            options2 = st.multiselect('Second group of entities', names_w_cops_grp2, names_w_cops_grp2[0])
        except:
            st.info("No connections between chosen entities. Choose different entity from the list.")
        add_spacelines(1)

        #freq_word_neg = st.multiselect('Choose entities for network analytics', pos_list_freq, list( set(['vonderleyen', 'GretaThunberg', 'JoeBiden' 'LeoDiCaprio', 'MikeHudema']).intersection(set(pos_list_freq)) )[:])
        pos_tr = list( set(options1).union(set(options2)) )
        #df0n = df[df.target.isin(df_odds_neg_words)]
        pos_sr = df[ (df.Target.isin(pos_tr)) & (df.ethos.isin(['support', 'attack'])) ].source.unique()


        df0p = df[ (df.source.isin(pos_sr)) | (df.Target.isin(pos_sr)) ]
        df0p.Target = df0p.Target.astype('str')
        df0p = df0p[df0p.Target != 'nan']
        df0pss = df0p.groupby(['source', ], ).Target.unique()
        df0pss = pd.DataFrame(df0pss).reset_index()
        df0pss['val'] = df0pss.Target.apply( lambda x: 1 if len( set(pos_tr).intersection(set(x)) ) >= 2 else 0 )
        df0pss = df0pss[df0pss.val == 1]
        df0p = df0p[ (df0p.source.isin(df0pss.source.unique()))  ]

        df0p_src = df0p.source.unique()
        df0p = df0p[ (df0p.Target.isin(df0p_src)) | (df0p.Target.isin(pos_tr)) ]

        if df0p.shape[0] < 1:
            st.info("No connections between chosen entities. Choose different entity from the list.")
            st.stop()


    with tab_fellow:
        st.write("#### Network Analytics")
        add_spacelines(1)
        slidekk = st.slider('Choose the value of parameter **k** in the newtwork plot', 0, 100, 15)
        slidekk = slidekk / 100
        slideiter = st.slider('Choose the value of parameter **iterations** in the newtwork plot', 0, 100, 12)



        df0p_graph = df0p.groupby(['source', 'Target', 'ethos'], as_index=False).size()
        add_spacelines(1)

        #st.write(df0p_graph )
        df0p_graph = df0p_graph.sort_values(by = ['source', 'ethos', 'size'], ascending = [True, False, False])
        df0p_graph = df0p_graph.drop_duplicates( ['source', 'Target'] )

        df0p_graph['g1_bool']  = np.where(df0p_graph.Target.isin(options1), 1, 0)
        df0p_graph['g2_bool']  = np.where(df0p_graph.Target.isin(options2), 1, 0)
        df0p_graph_src = df0p_graph.groupby('source', as_index = False)[['g1_bool', 'g2_bool']].sum()

        #df0p_graph_src = df0p_graph_src[ (df0p_graph_src.g1_bool == 1) & (df0p_graph_src.g2_bool == 1) ]
        df0p_graph_src = df0p_graph.groupby(['source', 'ethos'], as_index = False)[['g1_bool', 'g2_bool']].sum()

        #st.write(df0p_graph )
        #st.write(df0p_graph_src)

        df0p_graph_src_sup_enemy = df0p_graph_src[ (df0p_graph_src[['g1_bool', 'g2_bool']].sum(axis = 1) == 1) & (df0p_graph_src.ethos == 'support') ].source.unique()
        df0p_graph_src_att_enemy = df0p_graph_src[ (df0p_graph_src[['g1_bool', 'g2_bool']].sum(axis = 1) == 1) & (df0p_graph_src.ethos == 'attack') ].source.unique()
        df0p_graph_src_enemy = df0p_graph_src[ (df0p_graph_src.source.isin(df0p_graph_src_sup_enemy)) & (df0p_graph_src.source.isin(df0p_graph_src_att_enemy)) ]
        df0p_graph_src_enemy2 = df0p_graph_src_enemy.groupby('source', as_index = False)[['g1_bool', 'g2_bool']].sum()
        df0p_graph_src_enemy2 = df0p_graph_src_enemy2[ (df0p_graph_src_enemy2.g1_bool >= 1) & (df0p_graph_src_enemy2.g2_bool >= 1)  ]
        #st.write( df0p_graph_src_enemy2 )

        df0p_graph_src_confirm = df0p_graph_src_enemy2.copy()

        #df0p_graph_src_fellows = df0p_graph_src[ (df0p_graph_src.g1_bool > 1) | (df0p_graph_src.g2_bool > 1) ]
        #st.write(df0p_graph_src_fellows)
        #df0p_graph_src_confirm = pd.concat( [df0p_graph_src_enemy2, df0p_graph_src_fellows], axis = 0, ignore_index = True )

        #df0p_graph_nn_corob = df0p_graph_src_enemy2.shape[0] + df0p_graph_src_fellows.shape[0]
        df0p_graph_nn_corob = df0p_graph_src_confirm.shape[0]
        df0p_graph_nn_corob = round(df0p_graph_nn_corob * 100 / df0p_graph_src.source.nunique(), 3)

        node_s = 100
        sns.set(font_scale=1.35, style='whitegrid')

        posu = df0p_graph.groupby('source').ethos.unique()
        #st.write(posu)
        posu = posu.reset_index()
        posu.ethos = posu.ethos.apply(lambda x: " ".join(x))
        posu = posu[posu.ethos.str.split().map(len) == 1]
        negu = posu[posu.ethos == 'attack']
        posu = posu[posu.ethos == 'support']
        fig1, ax1 = plt.subplots(figsize = (12, 10))

        G = PolarizingNetworksSub_new(df0p)
        sns.set(font_scale=1, style='whitegrid')
        widths = list(nx.get_edge_attributes(G,'weight').values())
        widths = [ w - 0.2 if w < 2.5 else 2.5 for w in widths ]
        colors = list(nx.get_edge_attributes(G,'color').values())
        colors_nodes = nx.get_node_attributes(G, "color").copy()
        #st.write(df0p_graph_src_confirm.source.unique())
        for cn in df0p_graph_src_confirm.source.unique():
            colors_nodes[cn] = 'purple'
        for cn in negu.source.unique():
            colors_nodes[cn] = 'red'
        for cn in posu.source.unique():
            colors_nodes[cn] = 'green'
        colors_nodes = list(colors_nodes.values())
        #st.write(colors_nodes)
        #st.write(nx.get_node_attributes(G, "color"))


        pos = nx.drawing.layout.spring_layout(G, k= slidekk, iterations=slideiter, seed=6)

        nx.draw_networkx(G, with_labels=False, pos = pos,
               width=widths, edge_color=colors,
               alpha=0.75, node_color = colors_nodes, node_size = 450)

        font_names = ['Sawasdee', 'Gentium Book Basic', 'FreeMono']
        family_names = ['sans-serif', 'serif', 'fantasy', 'monospace']
        pos_tr = list( x.replace("@", "") for x in pos_tr )

        text = nx.draw_networkx_labels(G, pos, font_size=9,
            labels = { n:n if not (n in pos_tr or n in pos_sr) else '' for n in nx.nodes(G) } )

        for i, nodes in enumerate(pos_tr):
            # extract the subgraph
            g = G.subgraph(pos_tr[i])
            # draw on the labels with different fonts
            nx.draw_networkx_labels(g, pos, font_size=12.5, font_weight='bold', font_color = 'darkgreen')

        for i, nodes in enumerate(pos_sr):
            # extract the subgraph
            g = G.subgraph(pos_sr[i])
            # draw on the labels with different fonts
            nx.draw_networkx_labels(g, pos, font_size=10, )

        import matplotlib.patches as mpatches
        att_users_only = mpatches.Patch(color='red', label='negative')
        polar_users = mpatches.Patch(color='purple', label='polarising')
        sup_users_only = mpatches.Patch(color='green', label='positive')
        mix_users = mpatches.Patch(color='gold', label='mixed')
        #neu_users_only = mpatches.Patch(color='blue', label='neutral')
        #targ_only = mpatches.Patch(color='grey', label='target only')
        plt.legend(handles=[att_users_only, sup_users_only, polar_users, mix_users],
                    loc = 'upper center', bbox_to_anchor = (0.5, 1.045), ncol = 5,) #  title = f''
        plt.draw()
        plt.show()
        st.pyplot(fig1)

        #st.write(df0p_graph_nn_corob, df0p_graph_src.source.nunique())
        add_spacelines(2)

        cor_col1, cor_col2 = st.columns(2)
        with cor_col1:
            st.write("Users that coroborate the hypothesis")
            st.dataframe( df0p_graph_src_confirm[['source']] )
            add_spacelines(1)

        with cor_col2:
            xx = ['coroborate', 'falsify']
            st.metric("Percentage of ties that coroborate the hypothesis", round(df0p_graph_nn_corob))

        st.stop()



        with st.expander("Data cases"):
            add_spacelines(1)

            df0p_graph_src = df0p_graph.source.unique()
            st.write( "Cases of ethotic appeals from the network analytics" )
            st.dataframe(df0p_graph[ df0p_graph.Target.isin( pos_tr )])
            add_spacelines(1)

            st.write( "Content of ethotic appeals " )
            cols = ['created_at', 'text', 'username', 'ethos', 'target']
            st.dataframe( df[ (df.username.isin(df0p_graph_src)) & (df.target.isin(freq_word_neg)) ][cols] )
            add_spacelines(1)




def StatsLog_compare(df_list, an_type = 'ADU-based'):
    #st.write("#### Sentence Length Analysis")
    add_spacelines(2)
    conn_list = [' Logos Attack', ' Logos Support']
    map_naming = {'attack':'Ethos Attack', 'neutral':' No Ethos', 'support':'Ethos Support',
            'Default Conflict': ' Logos Attack',
            'Default Rephrase' : ' Logos  Rephrase',
            'Default Inference' : ' Logos Support'}
    rhetoric_dims = ['ethos', 'logos']
    df_list_et = pd.concat([df_list[0], df_list[-2]], axis=0, ignore_index = True)

    #df_list_et = df_list[0]
    if not 'neutral' in df_list_et['ethos_label'].unique():
        df_list_et['ethos_label'] = df_list_et['ethos_label'].map(ethos_mapping).map(map_naming)
    #df_list_log = df_list[1]
    df_list_log = pd.concat([df_list[1], df_list[-1]], axis=0, ignore_index = True)
    import re
    df_list_log['locution_conclusion'] = df_list_log.locution_conclusion.apply(lambda x: " ".join( str(x).split(':')[1:]) )
    df_list_log['locution_premise'] = df_list_log.locution_premise.apply(lambda x: " ".join( str(x).split(':')[1:]) )
    df_list_log['sentence'] = df_list_log.locution_premise.astype('str')# + " " + df_list_log.conclusion.astype('str')
    df_list_log_stats = df_list_log.groupby(['corpus', 'locution_premise', 'id_connection', 'connection'])['sentence'].apply(lambda x: " ".join(x)).reset_index()

    if an_type == 'Relation-based':
            df_list_log['locution_premise'] = df_list_log['locution_premise'].astype('str')
            df_list_log['locution_conclusion'] = df_list_log['locution_conclusion'].astype('str')

            dfp = df_list_log.groupby(['corpus', 'id_connection', 'connection'])['locution_premise'].apply(lambda x: " ".join(x)).reset_index()
            #dfc = df_list_log.groupby(['id_connection', 'connection'])['locution_conclusion'].apply(lambda x: " ".join(x)).reset_index()
            #dfp = dfp.merge(dfc, on = ['id_connection', 'connection']) #pd.concat([dfp, dfc.iloc[:, -1:]], axis=1) #dfp.merge(dfc, on = ['id_connection', 'connection'])
            dfp = dfp.drop_duplicates()

            dfp['sentence'] = dfp.locution_premise.astype('str')#+ " " + dfp['conclusion'].astype('str')
            import re
            dfp['sentence'] = dfp['sentence'].apply(lambda x: re.sub(r"\W+", " ", str(x)))
            dfp['sentence'] = dfp['sentence'].astype('str').str.lower()
            df_list_log_stats = dfp.copy()

    df_list_log_stats.connection = df_list_log_stats.connection.map(map_naming)


    df_list_et = compnwords(df_list_et, column_name = 'sentence')
    df_list_log_stats = compnwords(df_list_log_stats, column_name = 'sentence')

    cet_desc, c_log_stats_desc = st.columns(2)
    df_list_et_desc = pd.DataFrame(df_list_et[df_list_et.ethos_label.isin(['Ethos Support','Ethos Attack'])].groupby(['corpus', 'ethos_label']).nwords.describe().round(2).iloc[:, 1:])
    #df_list_et_desc = df_list_et_desc.T
    with cet_desc:
        st.write("Sentence Length for **Ethos**: ")
        st.write(df_list_et_desc)
        #st.stop()

    df_list_log_stats_desc = pd.DataFrame(df_list_log_stats[df_list_log_stats.connection.isin(conn_list)].groupby(['corpus', 'connection']).nwords.describe().round(2).iloc[:, 1:])
    #df_list_log_stats_desc = df_list_log_stats_desc.T
    with c_log_stats_desc:
        st.write("Sentence Length for **Logos**: ")
        st.write(df_list_log_stats_desc)


    add_spacelines(1)
    df_list_log_stats_desc = df_list_log_stats_desc.reset_index()
    df_list_et_desc = df_list_et_desc.reset_index()
    #st.write(df_list_log_stats_desc.columns)
    coprs_names1 = df_list_log_stats_desc.corpus.iloc[0]
    coprs_names2 = df_list_log_stats_desc.corpus.iloc[2]
    coprs_names = [coprs_names1, coprs_names2]

    #cstat1, cstat2, cstat3, cstat4 = st.columns(4)
    cstat1, cstat2 = st.columns(2)
    for cname in coprs_names:
        with cstat1:
            st.write(f"**{cname}**")
            le = df_list_et_desc[(df_list_et_desc.corpus == cname) & (df_list_et_desc.ethos_label == 'Ethos Attack')]['mean'].iloc[0]
            ll = df_list_log_stats_desc[(df_list_log_stats_desc.corpus == cname) &\
                    (df_list_log_stats_desc.connection == ' Logos Attack')]['mean'].iloc[0]
            #ll = df_list_log_stats_desc[df_list_log_stats_desc.corpus == cname].loc['mean', ' Logos Attack']
            lrel = round((le *100 / ll)- 100, 2)
            #st.write(le, ll, lrel)
            st.metric('Ethos Attack vs. Logos Attack', f" {le} vs. {ll} ", str(lrel)+'%')

        with cstat2:
            st.write(f"**{cname}**")
            #le = df_list_et_desc.loc['mean', 'Ethos Support']
            #ll = df_list_log_stats_desc.loc['mean', ' Logos Support']
            le = df_list_et_desc[(df_list_et_desc.corpus == cname) & (df_list_et_desc.ethos_label == 'Ethos Support')]['mean'].iloc[0]
            ll = df_list_log_stats_desc[(df_list_log_stats_desc.corpus == cname) &\
                    (df_list_log_stats_desc.connection == ' Logos Support')]['mean'].iloc[0]
            lrel = round((le *100 / ll)- 100, 2)
            st.metric('Ethos Support vs. Logos Support', f" {le} vs. {ll} ", str(lrel)+'%')

    add_spacelines(2)
    df_list_log_stats_desc['dimension'] = 'Logos'
    df_list_log_stats_desc = df_list_log_stats_desc.rename(columns = {'connection':'category'})
    df_list_et_desc = df_list_et_desc.rename(columns = {'ethos_label':'category'})
    df_list_et_desc['dimension'] = 'Ethos'

    df_list_desc = pd.concat( [df_list_log_stats_desc, df_list_et_desc], axis = 0, ignore_index = True )
    #st.write(df_list_desc)
    df_list_desc = df_list_desc.sort_values(by = ['dimension', 'category'])

    sns.set(font_scale = 1.15, style = 'whitegrid')
    f_desc = sns.catplot(data = df_list_desc, x = 'category', y = 'mean', col = 'dimension', row = 'corpus',
                kind = 'bar', palette = {'Ethos Attack':'#BB0000', ' No Ethos':'#022D96', 'Ethos Support':'#026F00',
                        ' Logos Attack':'#BB0000', ' Logos  Rephrase':'#D7A000', ' Logos Support':'#026F00'},
                        height = 4, aspect = 1.4, sharex=False)
    f_desc.set(xlabel = '', ylabel = 'mean sentence length', ylim = (0, np.max(df_list_desc['mean']+2)))
    for ax in f_desc.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
            ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

    plt.tight_layout(pad=2)
    st.pyplot(f_desc)
    st.write("************************************************************************")


    if an_type == 'ADU-based':
        #cet_desc, c_log_stats_desc = st.columns(2)
        #st.write(df_list_et)
        #n_words_all_list = df_list[-1]['premise'].tolist() + df_list[-1]['conclusion'].tolist() + df_list_et.sentence.tolist()
        vals = []
        cats_all = []
        corps_all = []

        for cname in coprs_names:
            n_words_all_list = df_list_log_stats[df_list_log_stats.corpus == cname].sentence.tolist() +\
                        df_list_et[df_list_et.corpus == cname].sentence.tolist()

            n_words_all_series = pd.DataFrame({'text':n_words_all_list})
            n_words_all_series = n_words_all_series.drop_duplicates()
            n_words_all_series['nl'] = n_words_all_series.text.str.split().map(len)
            n_words_all = n_words_all_series['nl'].mean().round(2)

            df_list_et_desc = pd.DataFrame(df_list_et[(df_list_et.ethos_label.isin(['Ethos Support','Ethos Attack'])) & (df_list_et.corpus == cname)].groupby('ethos_label').nwords.describe().round(2).iloc[:, 1:])
            df_list_et_desc = df_list_et_desc.T

            df_list_log_stats_desc = pd.DataFrame(df_list_log_stats[(df_list_log_stats.connection.isin(conn_list)) & (df_list_log_stats.corpus == cname)].groupby('connection').nwords.describe().round(2).iloc[:, 1:])
            df_list_log_stats_desc = df_list_log_stats_desc.T


            add_spacelines(1)
            n_words_all = round(n_words_all, 1)


            le = df_list_et_desc.loc['mean', 'Ethos Attack']
            lrel = round(le -n_words_all, 1)
            vals.append(n_words_all)
            cats_all.append('All')
            corps_all.append(cname)
            vals.append(le)
            cats_all.append('Ethos Attack')
            corps_all.append(cname)
            #st.write(le, ll, lrel)

            le = df_list_et_desc.loc['mean', 'Ethos Support']
            lrel = round(le -n_words_all, 1)
            #lrel = round((le *100 / ll)- 100, 2)
            vals.append(le)
            cats_all.append('Ethos Support')
            corps_all.append(cname)

            le = df_list_log_stats_desc.loc['mean', " Logos Attack"]
            lrel = round(le -n_words_all, 1)
            #st.write(le, ll, lrel)
            vals.append(le)
            cats_all.append(" Logos Attack")
            corps_all.append(cname)

            le = df_list_log_stats_desc.loc['mean', " Logos Support"]
            #ll = df_list_log_stats_desc.loc['mean', ' Logos Support']
            lrel = round(le -n_words_all, 1)
            vals.append(le)
            cats_all.append( " Logos Support")
            corps_all.append(cname)


        df_list_desc = pd.DataFrame([corps_all, cats_all, vals]).T
        df_list_desc.columns = ['corpus', 'category', 'mean']
        #add_spacelines(1)
        #st.write(df_list_desc)
        #st.stop()

        sns.set(font_scale = 1, style = 'whitegrid')
        f_desc = sns.catplot(data = df_list_desc, x = 'category', y = 'mean', col = 'corpus',
                    kind = 'bar', palette = {'Ethos Attack':'#BB0000', 'All':'#022D96', 'Ethos Support':'#026F00',
                            ' Logos Attack':'#BB0000', ' Logos  Rephrase':'#D7A000', ' Logos Support':'#026F00'},
                            aspect = 1.65, sharex=False, height=4)
        f_desc.set(xlabel = '', ylabel = 'mean sentence length', ylim = (0, np.max(df_list_desc['mean']+1)))
        for ax in f_desc.axes.ravel():
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

        plt.tight_layout(pad=2)
        _, c_log_stats_desc_all, _ = st.columns([1,20,1])
        with c_log_stats_desc_all:
            st.pyplot(f_desc)
        st.write("************************************************************************")

    st.stop()






def OddsRatioLog_compare(df_list, selected_rhet_dim, an_type = 'ADU-based'):
    rhetoric_dims = ['ethos', 'logos']

    if selected_rhet_dim == 'ethos_label':
        df = df_list[0]
        if not 'neutral' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str').str.lower().str.replace('ahould', 'should')

    if selected_rhet_dim == 'logos_label':
        df = df_list[1]
        #st.write(df)
        df['logos_label'] = df.connection.map({
                            'Default Conflict': 'attack',
                            'Default Rephrase' : 'Rephrase',
                            'Default Inference' : 'support'
        }).fillna('other')

        #df['sentence'] = df.premise
        df['sentence_lemmatized'] = df['premise'].astype('str') + " " + df['conclusion'].astype('str')
        if an_type != 'Relation-based':
            df = lemmatization(df, 'sentence_lemmatized', name_column = True)
            df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
            df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str').str.lower().str.replace('ahould', 'should')

        elif an_type == 'Relation-based':
            df['premise'] = df['premise'].astype('str')
            df['conclusion'] = df['conclusion'].astype('str')
            dfp = df.groupby(['id_connection', 'logos_label'])['premise'].apply(lambda x: " ".join(x)).reset_index()
            dfc = df.groupby(['id_connection', 'logos_label'])['conclusion'].apply(lambda x: " ".join(x)).reset_index()
            dfp = dfp.merge(dfc, on = ['id_connection', 'logos_label']) #pd.concat([dfp, dfc.iloc[:, -1:]], axis=1) #dfp.merge(dfc, on = ['id_connection', 'connection'])
            dfp = dfp.drop_duplicates()

            dfp['sentence_lemmatized'] = dfp.premise.astype('str')+ " " + dfc['conclusion'].astype('str')
            #st.write(dfp)
            import re
            dfp['sentence_lemmatized'] = dfp['sentence_lemmatized'].apply(lambda x: re.sub(r"\W+", " ", str(x)))
            dfp = lemmatization(dfp, 'sentence_lemmatized', name_column = True)
            df = dfp.copy()
            df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
            df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str').str.lower().str.replace('ahould', 'should')
        #if not 'sentence_lemmatized' in df.columns:
            #df = lemmatization(df, 'sentence')


    ddmsc = ['support', 'attack']
    if selected_rhet_dim == 'pathos_label':
        ddmsc = ['positive', 'negative']

    odds_list_of_dicts = []
    effect_list_of_dicts = []
    count_list_of_dicts = []
    # 1 vs rest
    #num = np.floor( len(df) / 10 )
    for ddmsc1 in ddmsc:
        dict_1vsall_percent = {}
        dict_1vsall_effect_size = {}
        dict_1vsall_count = {}
        #all100popular = Counter(" ".join( df.lemmatized.values ).split()).most_common(100)
        #all100popular = list(w[0] for w in all100popular)

        ddmsc1w = " ".join( df[df[selected_rhet_dim] == ddmsc1].sentence_lemmatized.fillna('').astype('str').values ).split()
        c = len(ddmsc1w)
        #ddmsc1w = list(w for w in ddmsc1w if not w in all100popular)
        ddmsc1w = Counter(ddmsc1w).most_common() # num
        ddmsc1w = [w for w in ddmsc1w if w[1] >= 3 ]

        #if ddmsc1 in ['positive', 'support']:
            #ddmsc1w = [w for w in ddmsc1w if w[1] >= 3 ]
        #else:
            #ddmsc1w = [w for w in ddmsc1w if w[1] > 3 ]

        ddmsc1w_word = dict(ddmsc1w)

        ddmsc2w = " ".join( df[df[selected_rhet_dim] != ddmsc1].sentence_lemmatized.fillna('').astype('str').values ).split()
        d = len(ddmsc2w)
        #ddmsc2w = list(w for w in ddmsc2w if not w in all100popular)
        ddmsc2w = Counter(ddmsc2w).most_common()
        ddmsc2w_word = dict(ddmsc2w)


        ddmsc1w_words = list( ddmsc1w_word.keys() )
        for n, dim in enumerate( ddmsc1w_words ):

            a = ddmsc1w_word[dim]
            try:
                b = ddmsc2w_word[dim]
            except:
                b = 0.5

            ca = c-a
            bd = d-b

            E1 = c*(a+b) / (c+d)
            E2 = d*(a+b) / (c+d)

            g2 = 2*((a*np.log(a/E1)) + (b* np.log(b/E2)))
            g2 = round(g2, 2)

            odds = round( (a*(d-b)) / (b*(c-a)), 2)

            if odds > 1 and len(dim) > 2:
                if g2 > 10.83:
                    #print(f"{dim, g2, odds} ***p < 0.001 ")
                    dict_1vsall_percent[dim] = odds
                    dict_1vsall_effect_size[dim] = 0.001
                    dict_1vsall_count[dim] = a
                elif g2 > 6.63:
                    #print(f"{dim, g2, odds} **p < 0.01 ")
                    dict_1vsall_percent[dim] = odds
                    dict_1vsall_effect_size[dim] = 0.01
                    dict_1vsall_count[dim] = a
                elif g2 > 3.84:
                    #print(f"{dim, g2, odds} *p < 0.05 ")
                    dict_1vsall_percent[dim] = odds
                    dict_1vsall_effect_size[dim] = 0.05
                    dict_1vsall_count[dim] = a
        #print(dict(sorted(dict_1vsall_percent.items(), key=lambda item: item[1])))
        odds_list_of_dicts.append(dict_1vsall_percent)
        effect_list_of_dicts.append(dict_1vsall_effect_size)
        count_list_of_dicts.append(dict_1vsall_count)

    df_odds_pos = pd.DataFrame({
                'word':odds_list_of_dicts[0].keys(),
                'odds':odds_list_of_dicts[0].values(),
                'effect_size_p':effect_list_of_dicts[0].values(),
                'frequency': count_list_of_dicts[0].values(),
    })
    df_odds_pos['category'] = ddmsc[0]
    df_odds_neg = pd.DataFrame({
                'word':odds_list_of_dicts[1].keys(),
                'odds':odds_list_of_dicts[1].values(),
                'effect_size_p':effect_list_of_dicts[1].values(),
                'frequency': count_list_of_dicts[1].values(),

    })
    df_odds_neg['category'] = ddmsc[1]
    df_odds_neg = df_odds_neg[df_odds_neg.word != 'bewp']
    df_odds_neg = df_odds_neg.sort_values(by = ['odds'], ascending = False)
    df_odds_pos = df_odds_pos.sort_values(by = ['odds'], ascending = False)


    df_odds_neg = transform_text(df_odds_neg, 'word')
    df_odds_pos = transform_text(df_odds_pos, 'word')
    pos_list = ['NOUN', 'VERB', 'NUM', 'PROPN', 'ADJ', 'ADV']
    df_odds_neg = df_odds_neg[df_odds_neg.POS_tags.isin(pos_list)]
    df_odds_pos = df_odds_pos[df_odds_pos.POS_tags.isin(pos_list)]
    df_odds_neg = df_odds_neg.reset_index(drop=True)
    df_odds_pos = df_odds_pos.reset_index(drop=True)
    df_odds_pos.index += 1
    df_odds_neg.index += 1

    df_odds_pos_tags_summ = df_odds_pos.POS_tags.value_counts(normalize = True).round(2)*100
    df_odds_neg_tags_summ = df_odds_neg.POS_tags.value_counts(normalize = True).round(2)*100
    df_odds_pos_tags_summ = df_odds_pos_tags_summ.reset_index()
    df_odds_pos_tags_summ.columns = ['POS_tags', 'percentage']

    df_odds_neg_tags_summ = df_odds_neg_tags_summ.reset_index()
    df_odds_neg_tags_summ.columns = ['POS_tags', 'percentage']

    df_odds_pos_tags_summ = df_odds_pos_tags_summ[df_odds_pos_tags_summ.percentage > 1]
    df_odds_neg_tags_summ = df_odds_neg_tags_summ[df_odds_neg_tags_summ.percentage > 1]

    oddpos_c, oddneg_c = st.columns(2)
    dimm = selected_rhet_dim.split("_")[0]
    with oddpos_c:
        st.write(f'Number of {dimm} {df_odds_pos.category.iloc[0]} words: {len(df_odds_pos)} ')
        st.dataframe(df_odds_pos)
        add_spacelines(1)
        st.dataframe(df_odds_pos_tags_summ)
        add_spacelines(1)

    with oddneg_c:
        st.write(f'Number of {dimm} {df_odds_neg.category.iloc[0]} words: {len(df_odds_neg)} ')
        st.dataframe(df_odds_neg)
        add_spacelines(1)
        st.dataframe(df_odds_neg_tags_summ)
        add_spacelines(1)







def OddsRatioLog(df_list, an_type = 'ADU-based'):
    st.write("### Lexical Analysis - Odds Ratio")
    add_spacelines(2)
    rhetoric_dims = ['ethos', 'logos']
    selected_rhet_dim = st.selectbox("Choose a rhetoric strategy for analysis", rhetoric_dims, index=0)
    selected_rhet_dim = selected_rhet_dim.replace('ethos', 'ethos_label').replace('logos', 'logos_label')
    add_spacelines(1)

    if selected_rhet_dim == 'ethos_label':
        df = df_list[0]
        if not 'neutral' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)

        df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str').str.lower().str.replace('ahould', 'should')
    if selected_rhet_dim == 'logos_label':
        df = df_list[1]
        #st.write(df)
        df['logos_label'] = df.connection.map({
                            'Default Conflict': 'attack',
                            'Default Rephrase' : 'Rephrase',
                            'Default Inference' : 'support'
        }).fillna('other')

        #df['sentence'] = df.premise
        df['sentence_lemmatized'] = df['premise'].astype('str') + " " + df['conclusion'].astype('str')
        if an_type != 'Relation-based':
            df = lemmatization(df, 'sentence_lemmatized', name_column = True)
            df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
            df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str').str.lower().str.replace('ahould', 'should')

        elif an_type == 'Relation-based':
            df['premise'] = df['premise'].astype('str')
            df['conclusion'] = df['conclusion'].astype('str')
            dfp = df.groupby(['id_connection', 'logos_label'])['premise'].apply(lambda x: " ".join(x)).reset_index()
            dfc = df.groupby(['id_connection', 'logos_label'])['conclusion'].apply(lambda x: " ".join(x)).reset_index()
            dfp = dfp.merge(dfc, on = ['id_connection', 'logos_label']) #pd.concat([dfp, dfc.iloc[:, -1:]], axis=1) #dfp.merge(dfc, on = ['id_connection', 'connection'])
            dfp = dfp.drop_duplicates()

            dfp['sentence_lemmatized'] = dfp.premise.astype('str')+ " " + dfc['conclusion'].astype('str')
            #st.write(dfp)
            import re
            dfp['sentence_lemmatized'] = dfp['sentence_lemmatized'].apply(lambda x: re.sub(r"\W+", " ", str(x)))
            dfp = lemmatization(dfp, 'sentence_lemmatized', name_column = True)
            df = dfp.copy()
            df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
            df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str').str.lower().str.replace('ahould', 'should')
        #if not 'sentence_lemmatized' in df.columns:
            #df = lemmatization(df, 'sentence')


    ddmsc = ['support', 'attack']
    if selected_rhet_dim == 'pathos_label':
        ddmsc = ['positive', 'negative']

    odds_list_of_dicts = []
    effect_list_of_dicts = []
    count_list_of_dicts = []
    # 1 vs rest
    #num = np.floor( len(df) / 10 )
    for ddmsc1 in ddmsc:
        dict_1vsall_percent = {}
        dict_1vsall_effect_size = {}
        dict_1vsall_count = {}
        #all100popular = Counter(" ".join( df.lemmatized.values ).split()).most_common(100)
        #all100popular = list(w[0] for w in all100popular)

        ddmsc1w = " ".join( df[df[selected_rhet_dim] == ddmsc1].sentence_lemmatized.fillna('').astype('str').values ).split()
        c = len(ddmsc1w)
        #ddmsc1w = list(w for w in ddmsc1w if not w in all100popular)
        ddmsc1w = Counter(ddmsc1w).most_common() # num
        ddmsc1w = [w for w in ddmsc1w if w[1] > 3 ]

        #if ddmsc1 in ['positive', 'support']:
            #ddmsc1w = [w for w in ddmsc1w if w[1] >= 3 ]
        #else:
            #ddmsc1w = [w for w in ddmsc1w if w[1] > 3 ]

        ddmsc1w_word = dict(ddmsc1w)

        ddmsc2w = " ".join( df[df[selected_rhet_dim] != ddmsc1].sentence_lemmatized.fillna('').astype('str').values ).split()
        d = len(ddmsc2w)
        #ddmsc2w = list(w for w in ddmsc2w if not w in all100popular)
        ddmsc2w = Counter(ddmsc2w).most_common()
        ddmsc2w_word = dict(ddmsc2w)


        ddmsc1w_words = list( ddmsc1w_word.keys() )
        for n, dim in enumerate( ddmsc1w_words ):

            a = ddmsc1w_word[dim]
            try:
                b = ddmsc2w_word[dim]
            except:
                b = 0.5

            ca = c-a
            bd = d-b

            E1 = c*(a+b) / (c+d)
            E2 = d*(a+b) / (c+d)

            g2 = 2*((a*np.log(a/E1)) + (b* np.log(b/E2)))
            g2 = round(g2, 2)

            odds = round( (a*(d-b)) / (b*(c-a)), 2)

            if odds > 1 and len(dim) > 2:
                if g2 > 10.83:
                    #print(f"{dim, g2, odds} ***p < 0.001 ")
                    dict_1vsall_percent[dim] = odds
                    dict_1vsall_effect_size[dim] = 0.001
                    dict_1vsall_count[dim] = a
                elif g2 > 6.63:
                    #print(f"{dim, g2, odds} **p < 0.01 ")
                    dict_1vsall_percent[dim] = odds
                    dict_1vsall_effect_size[dim] = 0.01
                    dict_1vsall_count[dim] = a
                elif g2 > 3.84:
                    #print(f"{dim, g2, odds} *p < 0.05 ")
                    dict_1vsall_percent[dim] = odds
                    dict_1vsall_effect_size[dim] = 0.05
                    dict_1vsall_count[dim] = a
        #print(dict(sorted(dict_1vsall_percent.items(), key=lambda item: item[1])))
        odds_list_of_dicts.append(dict_1vsall_percent)
        effect_list_of_dicts.append(dict_1vsall_effect_size)
        count_list_of_dicts.append(dict_1vsall_count)

    df_odds_pos = pd.DataFrame({
                'word':odds_list_of_dicts[0].keys(),
                'odds':odds_list_of_dicts[0].values(),
                'effect_size_p':effect_list_of_dicts[0].values(),
                'frequency': count_list_of_dicts[0].values(),
    })
    df_odds_pos['category'] = ddmsc[0]
    df_odds_neg = pd.DataFrame({
                'word':odds_list_of_dicts[1].keys(),
                'odds':odds_list_of_dicts[1].values(),
                'effect_size_p':effect_list_of_dicts[1].values(),
                'frequency': count_list_of_dicts[1].values(),

    })
    df_odds_neg['category'] = ddmsc[1]
    df_odds_neg = df_odds_neg[df_odds_neg.word != 'bewp']
    df_odds_neg = df_odds_neg.sort_values(by = ['odds'], ascending = False)
    df_odds_pos = df_odds_pos.sort_values(by = ['odds'], ascending = False)


    df_odds_neg = transform_text(df_odds_neg, 'word')
    df_odds_pos = transform_text(df_odds_pos, 'word')
    pos_list = ['NOUN', 'VERB', 'NUM', 'PROPN', 'ADJ', 'ADV']
    df_odds_neg = df_odds_neg[df_odds_neg.POS_tags.isin(pos_list)]
    df_odds_pos = df_odds_pos[df_odds_pos.POS_tags.isin(pos_list)]
    df_odds_neg = df_odds_neg.reset_index(drop=True)
    df_odds_pos = df_odds_pos.reset_index(drop=True)
    df_odds_pos.index += 1
    df_odds_neg.index += 1
    df_odds_neg['abusive'] = df_odds_neg.word.apply(lambda x: " ".join( set(x.lower().split()).intersection(abus_words)  ))
    df_odds_neg['abusive'] = np.where( df_odds_neg['abusive'].fillna('').astype('str').map(len) > 1 , 'abusive', 'non-abusive' )
    df_odds_pos['abusive'] = df_odds_pos.word.apply(lambda x: " ".join( set(x.lower().split()).intersection(abus_words)  ))
    df_odds_pos['abusive'] = np.where( df_odds_pos['abusive'].fillna('').astype('str').map(len) > 1, 'abusive', 'non-abusive' )


    df_odds_pos_tags_summ = df_odds_pos.POS_tags.value_counts(normalize = True).round(2)*100
    df_odds_neg_tags_summ = df_odds_neg.POS_tags.value_counts(normalize = True).round(2)*100
    df_odds_pos_tags_summ = df_odds_pos_tags_summ.reset_index()
    df_odds_pos_tags_summ.columns = ['POS_tags', 'percentage']
    df_odds_neg_tags_summ = df_odds_neg_tags_summ.reset_index()
    df_odds_neg_tags_summ.columns = ['POS_tags', 'percentage']

    df_odds_pos_tags_summ = df_odds_pos_tags_summ[df_odds_pos_tags_summ.percentage > 1]
    df_odds_neg_tags_summ = df_odds_neg_tags_summ[df_odds_neg_tags_summ.percentage > 1]

    df_odds_pos_abs= df_odds_pos.abusive.value_counts(normalize = True).round(3)*100
    df_odds_neg_abs = df_odds_neg.abusive.value_counts(normalize = True).round(3)*100
    df_odds_pos_abs = df_odds_pos_abs.reset_index()
    df_odds_pos_abs.columns = ['abusive', 'percentage']
    df_odds_neg_abs = df_odds_neg_abs.reset_index()
    df_odds_neg_abs.columns = ['abusive', 'percentage']

    df_odds_pos_words = set(df_odds_pos.word.values)
    df_odds_neg_words = set(df_odds_neg.word.values)

    df_odds = pd.concat( [df_odds_pos, df_odds_neg], axis = 0, ignore_index = True )
    df_odds = df_odds.sort_values(by = ['category', 'odds'], ascending = False)
    df['odds_words_'+df_odds_pos.category.iloc[0]] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_pos_words) ))
    df['odds_words_'+df_odds_neg.category.iloc[0]] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_neg_words) ))


    tab_odd, tab_pos, tab_abuse = st.tabs(['Odds', 'POS', 'Abusiveness'])
    with tab_odd:
        oddpos_c, oddneg_c = st.columns(2, gap = 'large')
        if selected_rhet_dim == 'ethos_label':
            cols_odds = ['source', 'sentence', 'ethos_label', 'Target',
                     'odds_words_'+df_odds_pos.category.iloc[0], 'odds_words_'+df_odds_neg.category.iloc[0]]

        elif selected_rhet_dim == 'pathos_label':
            cols_odds = ['source', 'sentence', 'pathos_label', 'Target',
                     'odds_words_'+df_odds_pos.category.iloc[0], 'odds_words_'+df_odds_neg.category.iloc[0]]

        elif selected_rhet_dim == 'logos_label':
            cols_odds = ['conclusion', 'premise', 'logos_label',
                     'odds_words_'+df_odds_pos.category.iloc[0], 'odds_words_'+df_odds_neg.category.iloc[0]]

        dimm = selected_rhet_dim.split("_")[0]
        with oddpos_c:
            st.write(f'Number of {dimm} {df_odds_pos.category.iloc[0]} words: {len(df_odds_pos)} ')
            st.dataframe(df_odds_pos)
            #add_spacelines(1)
            #st.dataframe(df_odds_pos_tags_summ)
            add_spacelines(1)
            st.write(f'Cases with **{df_odds_pos.category.iloc[0]}** words:')
            dfp = df[ df['odds_words_'+df_odds_pos.category.iloc[0]].str.split().map(len) >= 1 ][cols_odds]
            dfp = dfp[dfp[selected_rhet_dim].isin(['support', 'positive'])].reset_index(drop=True)
            st.dataframe(dfp) # .set_index('source')

        with oddneg_c:
            st.write(f'Number of {dimm} {df_odds_neg.category.iloc[0]} words: {len(df_odds_neg)} ')
            st.dataframe(df_odds_neg)
            #add_spacelines(1)
            #st.dataframe(df_odds_neg_tags_summ)
            add_spacelines(1)
            st.write(f'Cases with **{df_odds_neg.category.iloc[0]}** words:')
            dfn = df[ df['odds_words_'+df_odds_neg.category.iloc[0]].str.split().map(len) >= 1 ][cols_odds]
            dfn = dfn[dfn[selected_rhet_dim].isin(['attack', 'negative'])].reset_index(drop=True)
            st.dataframe(dfn) # .set_index('source')

    with tab_pos:
        sns.set(font_scale = 1.25, style = 'whitegrid')
        df_odds_pos_tags_summ['category'] = df_odds_pos.category.iloc[0]
        df_odds_neg_tags_summ['category'] = df_odds_neg.category.iloc[0]
        df_odds_pos = pd.concat([df_odds_pos_tags_summ, df_odds_neg_tags_summ], axis = 0, ignore_index=True)
        ffp = sns.catplot(kind='bar', data = df_odds_pos,
        y = 'POS_tags', x = 'percentage', hue = 'POS_tags', aspect = 1.3, height = 5, dodge=False,
        legend = False, col = 'category')
        ffp.set(ylabel = '')
        plt.tight_layout(w_pad=3)
        st.pyplot(ffp)
        add_spacelines(1)

        oddpos_cpos, oddneg_cpos = st.columns(2, gap = 'large')
        with oddpos_cpos:
            st.write(f'POS analysis of **{dimm} {df_odds_pos.category.iloc[0]}** words')
            add_spacelines(1)
            st.dataframe(df_odds_pos_tags_summ)
            add_spacelines(1)

        with oddneg_cpos:
            st.write(f'POS analysis of **{dimm} {df_odds_neg.category.iloc[0]}** words')
            add_spacelines(1)
            st.dataframe(df_odds_neg_tags_summ)
            add_spacelines(1)


    with tab_abuse:
        sns.set(font_scale = 1, style = 'whitegrid')
        df_odds_pos_abs['category'] = df_odds_pos.category.iloc[0]
        df_odds_neg_abs['category'] = df_odds_neg.category.iloc[0]
        df_odds_abs = pd.concat([df_odds_pos_abs, df_odds_neg_abs], axis = 0, ignore_index=True)
        ffp = sns.catplot(kind='bar', data = df_odds_abs,
        y = 'abusive', x = 'percentage', hue = 'abusive', aspect = 1.3, height = 3,dodge=False,
        palette = {'abusive':'darkred', 'non-abusive':'grey'}, legend = False, col = 'category')
        ffp.set(ylabel = '')
        plt.tight_layout(w_pad=3)
        st.pyplot(ffp)

        oddpos_cab, oddneg_cab = st.columns(2, gap = 'large')
        with oddpos_cab:
            st.write(f'Abusiveness analysis of **{dimm} {df_odds_pos.category.iloc[0]}** words')
            add_spacelines(1)
            st.dataframe(df_odds_pos_abs)
            add_spacelines(1)
            #ffp = sns.catplot(kind='bar', data = df_odds_pos_abs,
            #y = 'abusive', x = 'percentage', hue = 'abusive', aspect = 1.3, height = 3,dodge=False,
            #palette = {'abusive':'darkred', 'non-abusive':'grey'}, legend = False)
            #ffp.set(ylabel = '', title = f'Abusiveness of {dimm} {df_odds_pos.category.iloc[0]} words')
            #st.pyplot(ffp)
            add_spacelines(1)
            if df_odds_pos_abs.shape[0] > 1:
                st.write(df_odds_pos[df_odds_pos['abusive'] == 'abusive'])

        with oddneg_cab:
            st.write(f'Abusiveness analysis of **{dimm} {df_odds_neg.category.iloc[0]}** words')
            add_spacelines(1)
            st.dataframe(df_odds_neg_abs)
            add_spacelines(1)
            #ffn = sns.catplot(kind='bar', data = df_odds_neg_abs,
            #y = 'abusive', x = 'percentage', hue = 'abusive', aspect = 1.3, height = 3, dodge=False,
            #palette = {'abusive':'darkred', 'non-abusive':'grey'}, legend = False)
            #ffn.set(ylabel = '', title = f'Abusiveness of {dimm} {df_odds_neg.category.iloc[0]} words')
            #st.pyplot(ffn)
            add_spacelines(1)
            if df_odds_neg_abs.shape[0] > 1:
                st.write(df_odds_neg[df_odds_neg['abusive'] == 'abusive'])



@st.cache_data
def assignprons(data, col_take = 'sentence'):
    df = data.copy()
    prons = {'he', 'she', 'you', 'his', 'him', 'her', 'hers', 'your', 'yours', 'herself', 'himself', 'yourself'}
    prons3rd = {'they', 'their', 'theirs', 'them', 'themselves'}
    prons_verbs = {'are', 'were', 'have been', "weren't", "aren't", "haven't"}

    import re
    df[col_take] = df[col_take].apply(lambda x: re.sub(r"\W+", " ", str(x)))
    df['pronouns_singular'] = df[col_take].apply(lambda x: " ".join( set(x.split()).intersection(prons) ) )
    df['pronouns_plural'] = df[col_take].apply(lambda x: " ".join( set(x.split()).intersection(prons3rd) ) )
    df['plural_TOBE_verbs'] = df[col_take].apply(lambda x: " ".join( set(x.split()).intersection(prons_verbs) ) )
    return df


@st.cache_data
def count_categories(dataframe, categories_column, spliting = False, prefix_txt = 'pos'):
  if spliting:
    dataframe[categories_column] = dataframe[categories_column].str.split()

  dataframe["merge_indx"] = range(0, len(dataframe))
  from collections import Counter
  dataframe = pd.merge(dataframe, pd.DataFrame([Counter(x) for x in dataframe[categories_column]]).fillna(0).astype(int).add_prefix(str(prefix_txt)), how='left', left_on="merge_indx", right_index=True)
  dataframe.drop(["merge_indx"], axis=1, inplace=True)
  if spliting:
    dataframe[categories_column] = dataframe[categories_column].apply(lambda x: " ".join(x))
  return dataframe


def PronousLoP(df_list):
    st.write("### Language of Polarization Cues")
    add_spacelines(2)

    radio_prons_cat_dict = {
            'singular pronouns':'pronouns_singular',
            'plural pronouns':'pronouns_plural',
            'plural TO BE verbs':'plural_TOBE_verbs'}
    radio_prons_cat = st.multiselect('Choose a method of searching for pronouns',
                    ['singular pronouns',
                    'plural pronouns'], ['plural pronouns'])
    df = df_list[0]

    if not 'neutral' in df['ethos_label'].unique():
        df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
    if not 'pathos_label' in df['pathos_label'].unique():
        df['pathos_label'] = df['pathos_label'].map(valence_mapping)

    df = clean_text(df, 'sentence', text_column_name = "sentence_lemmatized")
    df = assignprons(data = df, col_take = 'sentence_lemmatized')
    cols_odds1 = ['source', 'sentence_lemmatized',
                'pronouns_plural', 'plural_TOBE_verbs', 'pronouns_singular',
                'ethos_label', 'pathos_label', 'Target']

    df_pron = df.copy()
    df_pron_col0 = df_pron.columns
    for i, c in enumerate(radio_prons_cat):
        col_i = radio_prons_cat_dict[c]
        #df_pron = df_pron[ ( df_pron[col_i].str.split().map(len)>0 ) ] # | ( df.pronouns.str.split().map(len)>0 )
        df_pron = count_categories(dataframe = df_pron, categories_column = col_i, spliting = True, prefix_txt = '')

    df_pron['No_pronouns_plural'] = df_pron.pronouns_plural.str.split().map(len)
    df_pron['No_pronouns_singular'] = df_pron.pronouns_singular.str.split().map(len)
    she_cols = list( c for c in df_pron if c.startswith('her') or c.startswith('she') )
    he_cols = list( c for c in df_pron if c.startswith('his') or c.startswith('him') or c == 'he' )
    they_cols = list( c for c in df_pron if c.startswith('the') )
    you_cols = list( c for c in df_pron if c.startswith('you') )

    df_pron['she'] = df_pron[she_cols].sum(axis=1)
    df_pron['he'] = df_pron[he_cols].sum(axis=1)
    df_pron['they'] = df_pron[they_cols].sum(axis=1)
    df_pron['you'] = df_pron[you_cols].sum(axis=1)

    #st.stop()
    df_pron_tab = df_pron.copy()
    for i, c in enumerate(radio_prons_cat):
        col_i = radio_prons_cat_dict[c]
        #df_pron = df_pron[ ( df_pron[col_i].str.split().map(len)>0 ) ] # | ( df.pronouns.str.split().map(len)>0 )
        df_pron_tab = df_pron_tab[ df_pron_tab[col_i].str.split().map(len) > 0]

    st.write(df_pron_tab[cols_odds1].set_index('source'))
    #st.write(df_pron.shape, df_pron_tab.shape)
    df_pron_plot_avg = df_pron.groupby('ethos_label', as_index = False)[['No_pronouns_plural', 'No_pronouns_singular']].mean()
    df_pron_plot_avg.No_pronouns_plural = df_pron_plot_avg.No_pronouns_plural.astype('float').round(3)*100
    df_pron_plot_avg.No_pronouns_singular = df_pron_plot_avg.No_pronouns_singular.astype('float').round(3)*100
    df_pron_plot_avg = df_pron_plot_avg.rename(columns = {'ethos_label':'ethos'})
    df_pron_plot_avg_melt = df_pron_plot_avg.melt('ethos')
    df_pron_plot_avg_melt['variable'] = df_pron_plot_avg_melt['variable'].str.replace("No_", "")
    #st.write(df_pron_plot_avg_melt)

    #st.stop()
    df_pron_col1 = df_pron.columns[:-2]
    #df_pron_col2 = list( set(df_pron_col1).difference(set(df_pron_col0)) )
    df_pron_col2 = ['she', 'he', 'they', 'you']
    df_pron_col2.extend(['ethos_label'])
    df_pron_plot = df_pron[df_pron_col2]
    #df_pron_plot_melt = df_pron_plot.melt(['ethos_label', 'pathos_label'])
    #df_pron_plot_melt2 = df_pron_plot.groupby(['ethos_label'])[df_pron_col2[:-2]].mean().round(3)*100
    df_pron_plot_melt2 = df_pron_plot.groupby(['ethos_label'])[df_pron_col2[:-1]].mean().round(3)*100
    df_pron_plot_melt2 = df_pron_plot_melt2.reset_index()

    df_pron_plot_melt = df_pron_plot_melt2.melt(['ethos_label'])
    df_pron_plot_melt = df_pron_plot_melt.rename(columns = {'ethos_label':'ethos'})
    max_val  = df_pron_plot_melt['value'].max()
    #st.write(df_pron_plot_melt2)
    add_spacelines(2)

    sns.set(font_scale=1.45, style='whitegrid')
    st.write("**Ethos**")

    g = sns.catplot(data = df_pron_plot_avg_melt, kind = 'bar', y = 'variable', x ='value',
        col = 'ethos', hue = 'variable',
        sharey=False, dodge=False, palette = ['gold', 'purple'], legend=False, aspect = 1.4)
    plt.tight_layout(pad=2)
    g.set(xticks = np.arange(0, 101, 20), xlabel = 'percentage', ylabel = 'pronouns type')
    st.pyplot(g)
    add_spacelines(1)

    g = sns.catplot(data = df_pron_plot_melt, kind = 'bar', y = 'variable', x ='value',
        col = 'ethos', hue = 'variable',
        sharey=False, dodge=False, palette = 'hsv_r', legend=False, aspect = 1.4)
    plt.tight_layout(pad=3)
    #sns.move_legend(g, loc = 'lower left', bbox_to_anchor = (0.32, 0.98), ncol = 4, title = '')
    g.set(xticks = np.arange(0, max_val+15, 10), ylabel = 'pronoun',)
    plt.suptitle('Ethos')
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)
    plt.show()
    st.pyplot(g)



def FreqTables(df_list, rhetoric_dims = [ 'ethos']):
    st.write("### Word Frequency Tables")
    add_spacelines(2)

    selected_rhet_dim = st.selectbox("Choose a rhetoric strategy for analysis", rhetoric_dims, index=0)
    selected_rhet_dim = selected_rhet_dim+"_label"
    add_spacelines(1)
    df = df_list[0]
    df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
    #df = lemmatization(df, 'content')
    if not 'neutral' in df['ethos_label'].unique():
        df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
    if not 'pathos_label' in df['pathos_label'].unique():
        df['pathos_label'] = df['pathos_label'].map(valence_mapping)

    ddmsc = ['support', 'attack']
    if selected_rhet_dim == 'pathos_label':
        ddmsc = ['positive', 'negative']

    odds_list_of_dicts = []

    # 1 vs rest
    #num = np.floor( len(df) / 10 )
    for ddmsc1 in ddmsc:
        dict_1vsall_percent = {}
        dict_1vsall_effect_size = {}
        ddmsc2w = " ".join( df[df[selected_rhet_dim] == ddmsc1].sentence_lemmatized.fillna('').astype('str').values ).split()

        ddmsc2w = Counter(ddmsc2w).most_common()
        ddmsc2w_word = dict(ddmsc2w)
        odds_list_of_dicts.append(ddmsc2w_word)


    df_odds_pos = pd.DataFrame({
                'word':odds_list_of_dicts[0].keys(),
                'frequency':odds_list_of_dicts[0].values(),
    })
    df_odds_pos['category'] = ddmsc[0]
    df_odds_neg = pd.DataFrame({
                'word':odds_list_of_dicts[1].keys(),
                'frequency':odds_list_of_dicts[1].values(),
    })
    df_odds_neg['category'] = ddmsc[1]
    df_odds_neg = df_odds_neg.sort_values(by = ['frequency'], ascending = False)
    #df_odds_neg = df_odds_neg[df_odds_neg.frequency > 2]
    df_odds_pos = df_odds_pos.sort_values(by = ['frequency'], ascending = False)
    #df_odds_pos = df_odds_pos[df_odds_pos.frequency > 2]

    df_odds_neg = transform_text(df_odds_neg, 'word')
    df_odds_pos = transform_text(df_odds_pos, 'word')
    pos_list = ['NOUN', 'VERB', 'NUM', 'PROPN', 'ADJ', 'ADV']
    df_odds_neg['POS_tags']  = np.where(df_odds_neg.word == 'url', 'NOUN', df_odds_neg['POS_tags'])
    df_odds_pos['POS_tags']  = np.where(df_odds_pos.word == 'url', 'NOUN', df_odds_pos['POS_tags'])
    df_odds_neg = df_odds_neg[df_odds_neg.POS_tags.isin(pos_list)]
    df_odds_pos = df_odds_pos[df_odds_pos.POS_tags.isin(pos_list)]
    df_odds_neg = df_odds_neg.reset_index(drop=True)
    df_odds_pos = df_odds_pos.reset_index(drop=True)

    df_odds_neg['abusive'] = df_odds_neg.word.apply(lambda x: " ".join( set(x.lower().split()).intersection(abus_words)  ))
    df_odds_neg['abusive'] = np.where( df_odds_neg['abusive'].fillna('').astype('str').map(len) > 1 , 'abusive', 'non-abusive' )
    df_odds_pos['abusive'] = df_odds_pos.word.apply(lambda x: " ".join( set(x.lower().split()).intersection(abus_words)  ))
    df_odds_pos['abusive'] = np.where( df_odds_pos['abusive'].fillna('').astype('str').map(len) > 1, 'abusive', 'non-abusive' )

    df_odds_pos.index += 1
    df_odds_neg.index += 1

    if "sentence_lemmatized" in df.columns:
        df.sentence_lemmatized = df.sentence_lemmatized.str.replace(" pyro sick ", " pyro2sick ")

    import nltk
    oddpos_c, oddneg_c = st.columns(2, gap = 'large')
    dimm = selected_rhet_dim.split("_")[0]
    with oddpos_c:
        st.write(f'Number of **{dimm} {df_odds_pos.category.iloc[0]}** words: {len(df_odds_pos)} ')
        st.dataframe(df_odds_pos)
        add_spacelines(1)

        pos_list_freq = df_odds_pos.word.tolist()
        freq_word_pos = st.multiselect('Choose a word you would like to see data cases for', pos_list_freq, pos_list_freq[:4:2])
        df_odds_pos_words = set(freq_word_pos)
        df['freq_words_'+df_odds_pos.category.iloc[0]] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_pos_words) ))

        add_spacelines(1)
        cols_odds1 = ['source', 'sentence', 'ethos_label', 'pathos_label', 'Target',
                         'freq_words_'+df_odds_pos.category.iloc[0]]
        df01 = df[ (df['freq_words_'+df_odds_pos.category.iloc[0]].str.split().map(len) >= 1) & (df[selected_rhet_dim] == df_odds_pos.category.iloc[0]) ]
        txt_df01 = " ".join(df01.sentence_lemmatized.values)
        df['mentions'] = df.sentence.apply(lambda x: " ".join( w for w in str(x).split() if "@" in w ))

        df_targets = " ".join( df.mentions.dropna().str.replace("@", "").str.lower().unique() ).split()
        t = nltk.tokenize.WhitespaceTokenizer()
        #c = Text(t.tokenize(txt_df01))

        #st.write(freq_word_pos[0], txt_df01[:50])
        #st.write(c.concordance_list(freq_word_pos[0], width=51, lines=50))
        # Loading Libraries
        from nltk.collocations import TrigramCollocationFinder, BigramCollocationFinder
        from nltk.metrics import TrigramAssocMeasures, BigramAssocMeasures
        from nltk.corpus import stopwords
        stopset = set(stopwords.words('english'))
        filter_stops = lambda w: len(w) < 3 or w in stopset

        def get_keyword_collocations(corpus, keyword, windowsize=10, numresults=10):
            import string
            from nltk.tokenize import word_tokenize
            from nltk.collocations import BigramCollocationFinder
            from nltk.collocations import BigramAssocMeasures
            from nltk.corpus import stopwords
            nltk.download('punkt')
            #'''This function uses the Natural Language Toolkit to find collocations
            #for a specific keyword in a corpus. It takes as an argument a string that
            #contains the corpus you want to find collocations from. It prints the top
            #collocations it finds for each keyword.
            #https://github.com/ahegel/collocations/blob/master/get_collocations3.py
            #'''
            # convert the corpus (a string) into  a list of words
            tokens = word_tokenize(corpus)
            # initialize the bigram association measures object to score each collocation
            bigram_measures = BigramAssocMeasures()
            # initialize the bigram collocation finder object to find and rank collocations
            finder = BigramCollocationFinder.from_words(tokens, window_size=windowsize)
            # initialize a function that will narrow down collocates that don't contain the keyword
            keyword_filter = lambda *w: keyword not in w
            # apply a series of filters to narrow down the collocation results
            ignored_words = stopwords.words('english')
            finder.apply_word_filter(lambda w: len(w) < 2 or w.lower() in ignored_words)
            finder.apply_freq_filter(2)
            finder.apply_ngram_filter(keyword_filter)
            # calculate the top results by T-score
            # list of all possible measures: .raw_freq, .pmi, .likelihood_ratio, .chi_sq, .phi_sq, .fisher, .student_t, .mi_like, .poisson_stirling, .jaccard, .dice
            results = finder.nbest(bigram_measures.student_t, numresults)
            # print the results
            print("Top collocations for ", str(keyword), ":")
            collocations = ''
            for k, v in results:
                if k != keyword:
                    collocations += k + ' '
                else:
                    collocations += v + ' '
            #print(collocations, '\n')
            st.write(collocations, '\n')

        words_of_interest = list(freq_word_pos) # ["love", "die"]
        import nltk
        #from nltk.collocations
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        finder = nltk.collocations.BigramCollocationFinder.from_words(txt_df01.split(), window_size=5)
        finder.nbest(bigram_measures.pmi, 10)
        finder.apply_freq_filter(2)
        results=finder.nbest(bigram_measures.pmi, 10)
        scores = finder.score_ngrams(bigram_measures.pmi)
        seed = "word"
        result_term = []
        result_pmi = []
        for terms, score in scores:
            if terms [0] in freq_word_pos or terms [1] in freq_word_pos:
                if not " ".join(terms) in result_term and not str(terms [1]) + " " + str(terms [0]) in result_term:
                    result_term.append(" ".join(terms))
                    result_pmi.append(score)

        df_bigrams0 = pd.DataFrame({'bi-grams':result_term, 'PMI':result_pmi})
        df_bigrams0['len1'] = df_bigrams0['bi-grams'].astype('str').apply(lambda x: len(str(x).split()[0]) )
        df_bigrams0['len2'] = df_bigrams0['bi-grams'].astype('str').apply(lambda x: len(str(x).split()[-1]) )
        df_bigrams0 = df_bigrams0[ (df_bigrams0.len1 > 2) & (df_bigrams0.len2 > 2) ]
        df_bigrams0 = df_bigrams0[df_bigrams0.PMI > 0].reset_index(drop=True)
        df_bigrams0.PMI = df_bigrams0.PMI.round(3)

        df_bigrams0.index += 1
        #st.write(df_bigrams0)


        #for word in words_of_interest:
            #get_keyword_collocations(txt_df01, word)

        st.write(f'Bi-gram collocations with **{freq_word_pos}**')
        colBigrams = list(nltk.ngrams(t.tokenize(txt_df01), 2))
        colBigrams2 = []
        for p in colBigrams:
            for w in freq_word_pos:
                if (w == p[0] or w == p[1]) and not (p[0] in df_targets or p[1] in df_targets):
                    colBigrams2.append(" ".join(p))
        df_bigrams = pd.DataFrame({'bi-grams':colBigrams2})
        df_bigrams = df_bigrams.drop_duplicates()
        df_bigrams = df_bigrams.groupby('bi-grams', as_index=False).size()
        df_bigrams.columns = ['bi-grams', 'frequency']
        df_bigrams = df_bigrams.sort_values(by = 'frequency', ascending = False).reset_index(drop=True)
        #df_bigrams = df_bigrams[df_bigrams.duplicated()].reset_index(drop=True)
        df_bigrams.index += 1
        df_bigrams0 = df_bigrams0.drop(columns =  ['len1', 'len2'] )


        st.write(df_bigrams0)# df_bigrams0 df_bigrams

        add_spacelines(1)
        st.write(f'Cases with **{freq_word_pos}** words:')
        st.dataframe(df01[cols_odds1].set_index('source'))


    with oddneg_c:
        st.write(f'Number of **{dimm} {df_odds_neg.category.iloc[0]}** words: {len(df_odds_neg)} ')
        st.dataframe(df_odds_neg)
        add_spacelines(1)

        neg_list_freq = df_odds_neg.word.tolist()
        freq_word_neg = st.multiselect('Choose a word you would like to see data cases for', neg_list_freq, neg_list_freq[:4:2])
        df_odds_neg_words = set(freq_word_neg)
        df['freq_words_'+df_odds_neg.category.iloc[0]] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_neg_words) ))
        df02 = df[ (df['freq_words_'+df_odds_neg.category.iloc[0]].str.split().map(len) >= 1) & (df[selected_rhet_dim] == df_odds_neg.category.iloc[0]) ]
        txt_df02 = " ".join(df02.sentence_lemmatized.values)

        finder = nltk.collocations.BigramCollocationFinder.from_words(txt_df02.split(), window_size=5)
        finder.nbest(bigram_measures.pmi, 10)
        finder.apply_freq_filter(2)
        results=finder.nbest(bigram_measures.pmi, 10)
        scores = finder.score_ngrams(bigram_measures.pmi)
        seed = "word"
        result_term = []
        result_pmi = []
        for terms, score in scores:
            if terms [0] in freq_word_neg or terms [1] in freq_word_neg:
                if not " ".join(terms) in result_term and not str(terms [1]) + " " + str(terms [0]) in result_term:
                    result_term.append(" ".join(terms))
                    result_pmi.append(score)

        df_bigrams0 = pd.DataFrame({'bi-grams':result_term, 'PMI':result_pmi})
        df_bigrams0['len1'] = df_bigrams0['bi-grams'].astype('str').apply(lambda x: len(str(x).split()[0]) )
        df_bigrams0['len2'] = df_bigrams0['bi-grams'].astype('str').apply(lambda x: len(str(x).split()[-1]) )
        df_bigrams0 = df_bigrams0[ (df_bigrams0.len1 > 2) & (df_bigrams0.len2 > 2) ]
        df_bigrams0 = df_bigrams0[df_bigrams0.PMI > 0].reset_index(drop=True)
        df_bigrams0.PMI = df_bigrams0.PMI.round(3)
        df_bigrams0.index += 1

        cols_odds2 = ['source', 'sentence', 'ethos_label', 'pathos_label', 'Target',
                         'freq_words_'+df_odds_neg.category.iloc[0]]
        add_spacelines(1)
        st.write(f'Bi-gram collocations with **{freq_word_neg}**')
        colBigrams = list(nltk.ngrams(t.tokenize(txt_df02), 2))
        colBigrams2 = []
        for p in colBigrams:
            for w in freq_word_neg:
                if (w == p[0] or w == p[1]) and not (p[0] in df_targets or p[1] in df_targets):
                    colBigrams2.append(" ".join(p))
        df_bigrams = pd.DataFrame({'bi-grams':colBigrams2})
        df_bigrams = df_bigrams.drop_duplicates()#.reset_index(drop=True)
        df_bigrams = df_bigrams.groupby('bi-grams', as_index=False).size()
        df_bigrams.columns = ['bi-grams', 'frequency']
        df_bigrams = df_bigrams.sort_values(by = 'frequency', ascending = False).reset_index(drop=True)
        #df_bigrams = df_bigrams[df_bigrams.duplicated()].reset_index(drop=True)
        df_bigrams.index += 1
        df_bigrams0 = df_bigrams0.drop(columns =  ['len1', 'len2'] )
        st.write(df_bigrams0)

        add_spacelines(1)
        st.write(f'Cases with **{freq_word_neg}** words:')
        st.dataframe(df02[cols_odds2].set_index('source'))




def FreqTablesLog(df_list, rhetoric_dims = ['ethos', 'logos']):
    st.write("### Word Frequency Tables")
    add_spacelines(2)

    selected_rhet_dim = st.selectbox("Choose a rhetoric strategy for analysis", rhetoric_dims, index=0)
    if selected_rhet_dim == 'logos':
        df = df_list[1]
        df['locution_conclusion'] = df.locution_conclusion.apply(lambda x: " ".join( str(x).split(':')[1:]) )
        df['locution_premise'] = df.locution_premise.apply(lambda x: " ".join( str(x).split(':')[1:]) )
        df['sentence'] = df['locution_premise'].astype('str') + " " + df['locution_conclusion'].astype('str')
        map_naming = {
                'Default Conflict': 'attack',
                'Default Rephrase' : 'neutral',
                'Default Inference' : 'support'}
        df.connection = df.connection.map(map_naming)

    else:
        df = df_list[0]
        if not 'neutral' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)

    selected_rhet_dim = selected_rhet_dim.replace('ethos', 'ethos_label').replace('logos', 'connection')
    add_spacelines(1)

    if 'sentence_lemmatized' in df.columns:
        df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
    else:
        df = lemmatization(df, 'sentence')
        df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")


    ddmsc = ['support', 'attack']
    if selected_rhet_dim == 'pathos_label':
        ddmsc = ['positive', 'negative']

    odds_list_of_dicts = []

    # 1 vs rest
    #num = np.floor( len(df) / 10 )
    for ddmsc1 in ddmsc:
        dict_1vsall_percent = {}
        dict_1vsall_effect_size = {}
        ddmsc2w = " ".join( df[df[selected_rhet_dim] == ddmsc1].sentence_lemmatized.fillna('').astype('str').values ).split()

        ddmsc2w = Counter(ddmsc2w).most_common()
        ddmsc2w_word = dict(ddmsc2w)
        odds_list_of_dicts.append(ddmsc2w_word)


    df_odds_pos = pd.DataFrame({
                'word':odds_list_of_dicts[0].keys(),
                'frequency':odds_list_of_dicts[0].values(),
    })
    df_odds_pos['category'] = ddmsc[0]
    df_odds_neg = pd.DataFrame({
                'word':odds_list_of_dicts[1].keys(),
                'frequency':odds_list_of_dicts[1].values(),
    })
    df_odds_neg['category'] = ddmsc[1]
    df_odds_neg = df_odds_neg.sort_values(by = ['frequency'], ascending = False)
    #df_odds_neg = df_odds_neg[df_odds_neg.frequency > 2]
    df_odds_pos = df_odds_pos.sort_values(by = ['frequency'], ascending = False)
    #df_odds_pos = df_odds_pos[df_odds_pos.frequency > 2]

    df_odds_neg = transform_text(df_odds_neg, 'word')
    df_odds_pos = transform_text(df_odds_pos, 'word')
    pos_list = ['NOUN', 'VERB', 'NUM', 'PROPN', 'ADJ', 'ADV']
    df_odds_neg = df_odds_neg[df_odds_neg.POS_tags.isin(pos_list)]
    df_odds_pos = df_odds_pos[df_odds_pos.POS_tags.isin(pos_list)]

    df_odds_neg = df_odds_neg.reset_index(drop=True)
    df_odds_pos = df_odds_pos.reset_index(drop=True)
    df_odds_pos.index += 1
    df_odds_neg.index += 1

    df_odds_pos_10n = np.ceil(df_odds_pos.shape[0] * 0.1)
    df_odds_neg_10n = np.ceil(df_odds_neg.shape[0] * 0.1)

    df_odds_pos_tags_summ = df_odds_pos.iloc[:int(df_odds_pos_10n)].POS_tags.value_counts(normalize = True).round(2)*100
    df_odds_neg_tags_summ = df_odds_neg.iloc[:int(df_odds_neg_10n)].POS_tags.value_counts(normalize = True).round(2)*100
    df_odds_pos_tags_summ = df_odds_pos_tags_summ.reset_index()
    df_odds_pos_tags_summ.columns = ['POS_tags', 'percentage']

    df_odds_neg_tags_summ = df_odds_neg_tags_summ.reset_index()
    df_odds_neg_tags_summ.columns = ['POS_tags', 'percentage']

    df_odds_pos_tags_summ = df_odds_pos_tags_summ[df_odds_pos_tags_summ.percentage > 1]
    df_odds_neg_tags_summ = df_odds_neg_tags_summ[df_odds_neg_tags_summ.percentage > 1]

    if "sentence_lemmatized" in df.columns:
        df.sentence_lemmatized = df.sentence_lemmatized.str.replace(" pyro sick ", " pyro2sick ")

    import nltk
    oddpos_c, oddneg_c = st.columns(2, gap = 'large')
    dimm = selected_rhet_dim.split("_")[0]
    dimm = str(dimm).replace('connection', 'logos')
    with oddpos_c:
        st.write(f'Number of **{dimm} {df_odds_pos.category.iloc[0]}** words: {len(df_odds_pos)} ')
        st.dataframe(df_odds_pos)
        add_spacelines(1)
        st.write("Part-of-Speech analysis for the top 10% of words in the table")
        st.write(df_odds_pos_tags_summ)
        add_spacelines(1)

        pos_list_freq = df_odds_pos.word.tolist()
        freq_word_pos = st.multiselect('Choose a word you would like to see data cases for', pos_list_freq, pos_list_freq[:4:2])
        df_odds_pos_words = set(freq_word_pos)
        df['freq_words_'+df_odds_pos.category.iloc[0]] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_pos_words) ))

        add_spacelines(1)

        df01 = df[ (df['freq_words_'+df_odds_pos.category.iloc[0]].str.split().map(len) >= 1) & (df[selected_rhet_dim] == df_odds_pos.category.iloc[0]) ]
        txt_df01 = " ".join(df01.sentence_lemmatized.values)
        df['mentions'] = df.sentence.apply(lambda x: " ".join( w for w in str(x).split() if "@" in w ))

        df_targets = " ".join( df.mentions.dropna().str.replace("@", "").str.lower().unique() ).split()

        words_of_interest = list(freq_word_pos) # ["love", "die"]
        import nltk
        t = nltk.tokenize.WhitespaceTokenizer()
        #from nltk.collocations
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        finder = nltk.collocations.BigramCollocationFinder.from_words(txt_df01.split(), window_size=5)
        finder.nbest(bigram_measures.pmi, 10)
        finder.apply_freq_filter(2)
        results=finder.nbest(bigram_measures.pmi, 10)
        scores = finder.score_ngrams(bigram_measures.pmi)
        seed = "word"
        result_term = []
        result_pmi = []
        for terms, score in scores:
            if terms [0] in freq_word_pos or terms [1] in freq_word_pos:
                if not " ".join(terms) in result_term and not str(terms [1]) + " " + str(terms [0]) in result_term:
                    result_term.append(" ".join(terms))
                    result_pmi.append(score)

        df_bigrams0 = pd.DataFrame({'bi-grams':result_term, 'PMI':result_pmi})
        df_bigrams0['len1'] = df_bigrams0['bi-grams'].astype('str').apply(lambda x: len(str(x).split()[0]) )
        df_bigrams0['len2'] = df_bigrams0['bi-grams'].astype('str').apply(lambda x: len(str(x).split()[-1]) )
        df_bigrams0 = df_bigrams0[ (df_bigrams0.len1 > 2) & (df_bigrams0.len2 > 2) ]
        df_bigrams0 = df_bigrams0[df_bigrams0.PMI > 0].reset_index(drop=True)
        df_bigrams0.PMI = df_bigrams0.PMI.round(3)
        df_bigrams0.index += 1

        st.write(f'Bi-gram collocations with **{freq_word_pos}**')
        colBigrams = list(nltk.ngrams(t.tokenize(txt_df01), 2))
        colBigrams2 = []
        for p in colBigrams:
            for w in freq_word_pos:
                if (w == p[0] or w == p[1]) and not (p[0] in df_targets or p[1] in df_targets):
                    colBigrams2.append(" ".join(p))
        df_bigrams = pd.DataFrame({'bi-grams':colBigrams2})
        df_bigrams = df_bigrams.drop_duplicates()
        df_bigrams = df_bigrams.groupby('bi-grams', as_index=False).size()
        df_bigrams.columns = ['bi-grams', 'frequency']
        df_bigrams = df_bigrams.sort_values(by = 'frequency', ascending = False).reset_index(drop=True)
        #df_bigrams = df_bigrams[df_bigrams.duplicated()].reset_index(drop=True)
        df_bigrams.index += 1
        st.write(df_bigrams0)# df_bigrams0 df_bigrams

        add_spacelines(1)
        st.write(f'Cases with **{freq_word_pos}** words:')
        st.dataframe(df01)


    with oddneg_c:
        st.write(f'Number of **{dimm} {df_odds_neg.category.iloc[0]}** words: {len(df_odds_neg)} ')
        st.dataframe(df_odds_neg)
        add_spacelines(1)
        st.write("Part-of-Speech analysis for the top 10% of words in the table")
        st.write(df_odds_neg_tags_summ)
        add_spacelines(1)

        neg_list_freq = df_odds_neg.word.tolist()
        freq_word_neg = st.multiselect('Choose a word you would like to see data cases for', neg_list_freq, neg_list_freq[:4:2])
        df_odds_neg_words = set(freq_word_neg)
        df['freq_words_'+df_odds_neg.category.iloc[0]] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_neg_words) ))
        df02 = df[ (df['freq_words_'+df_odds_neg.category.iloc[0]].str.split().map(len) >= 1) & (df[selected_rhet_dim] == df_odds_neg.category.iloc[0]) ]
        txt_df02 = " ".join(df02.sentence_lemmatized.values)

        finder = nltk.collocations.BigramCollocationFinder.from_words(txt_df02.split(), window_size=5)
        finder.nbest(bigram_measures.pmi, 10)
        finder.apply_freq_filter(2)
        results=finder.nbest(bigram_measures.pmi, 10)
        scores = finder.score_ngrams(bigram_measures.pmi)
        seed = "word"
        result_term = []
        result_pmi = []
        for terms, score in scores:
            if terms [0] in freq_word_neg or terms [1] in freq_word_neg:
                if not " ".join(terms) in result_term and not str(terms [1]) + " " + str(terms [0]) in result_term:
                    result_term.append(" ".join(terms))
                    result_pmi.append(score)

        df_bigrams0 = pd.DataFrame({'bi-grams':result_term, 'PMI':result_pmi})
        df_bigrams0['len1'] = df_bigrams0['bi-grams'].astype('str').apply(lambda x: len(str(x).split()[0]) )
        df_bigrams0['len2'] = df_bigrams0['bi-grams'].astype('str').apply(lambda x: len(str(x).split()[-1]) )
        df_bigrams0 = df_bigrams0[ (df_bigrams0.len1 > 2) & (df_bigrams0.len2 > 2) ]
        df_bigrams0 = df_bigrams0[df_bigrams0.PMI > 0].reset_index(drop=True)
        df_bigrams0.PMI = df_bigrams0.PMI.round(3)
        df_bigrams0.index += 1

        add_spacelines(1)
        st.write(f'Bi-gram collocations with **{freq_word_neg}**')
        colBigrams = list(nltk.ngrams(t.tokenize(txt_df02), 2))
        colBigrams2 = []
        for p in colBigrams:
            for w in freq_word_neg:
                if (w == p[0] or w == p[1]) and not (p[0] in df_targets or p[1] in df_targets):
                    colBigrams2.append(" ".join(p))
        df_bigrams = pd.DataFrame({'bi-grams':colBigrams2})
        df_bigrams = df_bigrams.drop_duplicates()#.reset_index(drop=True)
        df_bigrams = df_bigrams.groupby('bi-grams', as_index=False).size()
        df_bigrams.columns = ['bi-grams', 'frequency']
        df_bigrams = df_bigrams.sort_values(by = 'frequency', ascending = False).reset_index(drop=True)
        #df_bigrams = df_bigrams[df_bigrams.duplicated()].reset_index(drop=True)
        df_bigrams.index += 1
        st.write(df_bigrams0)

        add_spacelines(1)
        st.write(f'Cases with **{freq_word_neg}** words:')
        st.dataframe(df02)





def OddsRatio(df_list):
    st.write("### Lexical Analysis - Odds Ratio")
    add_spacelines(2)
    rhetoric_dims = ['ethos', ] # , 'pathos'
    selected_rhet_dim = st.selectbox("Choose a rhetoric strategy for analysis", rhetoric_dims, index=0)
    selected_rhet_dim = selected_rhet_dim+"_label"
    add_spacelines(1)
    df = df_list[0]
    df = df.drop_duplicates(subset = ['source', 'sentence'])
    df['mentions'] = df.sentence.apply(lambda x: " ".join( w for w in str(x).split() if '@' in w ))
    df['sentence_lemmatized'] = df.sentence.apply(lambda x: " ".join( str(w).replace("#", "") for w in str(x).split() if not '@' in w ))
    df = lemmatization(df, 'sentence_lemmatized')
    df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
    df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str') + " " + df['mentions']

    if not 'neutral' in df['ethos_label'].unique():
        df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
    if not 'pathos_label' in df['pathos_label'].unique():
        df['pathos_label'] = df['pathos_label'].map(valence_mapping)

    ddmsc = ['support', 'attack']
    if selected_rhet_dim == 'pathos_label':
        ddmsc = ['positive', 'negative']

    odds_list_of_dicts = []
    effect_list_of_dicts = []
    freq_list_of_dicts = []
    # 1 vs rest
    #num = np.floor( len(df) / 10 )
    for ddmsc1 in ddmsc:
        dict_1vsall_percent = {}
        dict_1vsall_effect_size = {}
        dict_1vsall_freq = {}
        #all100popular = Counter(" ".join( df.lemmatized.values ).split()).most_common(100)
        #all100popular = list(w[0] for w in all100popular)

        ddmsc1w = " ".join( df[df[selected_rhet_dim] == ddmsc1].sentence_lemmatized.fillna('').astype('str').values ).split() # sentence_lemmatized
        c = len(ddmsc1w)
        #ddmsc1w = list(w for w in ddmsc1w if not w in all100popular)
        ddmsc1w = Counter(ddmsc1w).most_common() # num
        if ddmsc1 in ['positive', 'support']:
            ddmsc1w = [w for w in ddmsc1w if w[1] >= 3 ]
        else:
            ddmsc1w = [w for w in ddmsc1w if w[1] > 3 ]
        #print('**********')
        #print(len(ddmsc1w), ddmsc1w)
        #print([w for w in ddmsc1w if w[1] > 2 ])
        #print(len([w for w in ddmsc1w if w[1] > 2 ]))
        ddmsc1w_word = dict(ddmsc1w)

        ddmsc2w = " ".join( df[df[selected_rhet_dim] != ddmsc1].sentence_lemmatized.fillna('').astype('str').values ).split() # sentence_lemmatized
        d = len(ddmsc2w)
        #ddmsc2w = list(w for w in ddmsc2w if not w in all100popular)
        ddmsc2w = Counter(ddmsc2w).most_common()
        ddmsc2w_word = dict(ddmsc2w)


        ddmsc1w_words = list( ddmsc1w_word.keys() )
        for n, dim in enumerate( ddmsc1w_words ):

            a = ddmsc1w_word[dim]
            try:
                b = ddmsc2w_word[dim]
            except:
                b = 0.5

            ca = c-a
            bd = d-b

            E1 = c*(a+b) / (c+d)
            E2 = d*(a+b) / (c+d)

            g2 = 2*((a*np.log(a/E1)) + (b* np.log(b/E2)))
            g2 = round(g2, 2)

            odds = round( (a*(d-b)) / (b*(c-a)), 2)

            if odds > 1:

                if g2 > 10.83:
                    #print(f"{dim, g2, odds} ***p < 0.001 ")
                    dict_1vsall_percent[dim] = odds
                    dict_1vsall_effect_size[dim] = 0.001
                    dict_1vsall_freq[dim] = a
                elif g2 > 6.63:
                    #print(f"{dim, g2, odds} **p < 0.01 ")
                    dict_1vsall_percent[dim] = odds
                    dict_1vsall_effect_size[dim] = 0.01
                    dict_1vsall_freq[dim] = a
                elif g2 > 3.84:
                    #print(f"{dim, g2, odds} *p < 0.05 ")
                    dict_1vsall_percent[dim] = odds
                    dict_1vsall_effect_size[dim] = 0.05
                    dict_1vsall_freq[dim] = a
        #print(dict(sorted(dict_1vsall_percent.items(), key=lambda item: item[1])))
        odds_list_of_dicts.append(dict_1vsall_percent)
        effect_list_of_dicts.append(dict_1vsall_effect_size)
        freq_list_of_dicts.append(dict_1vsall_freq)

    df_odds_pos = pd.DataFrame({
                'word':odds_list_of_dicts[0].keys(),
                'odds':odds_list_of_dicts[0].values(),
                'frequency':freq_list_of_dicts[0].values(),
                'effect_size_p':effect_list_of_dicts[0].values(),
    })
    df_odds_pos['category'] = ddmsc[0]
    df_odds_neg = pd.DataFrame({
                'word':odds_list_of_dicts[1].keys(),
                'odds':odds_list_of_dicts[1].values(),
                'frequency':freq_list_of_dicts[1].values(),
                'effect_size_p':effect_list_of_dicts[1].values(),
    })
    df_odds_neg['category'] = ddmsc[1]
    df_odds_neg = df_odds_neg.sort_values(by = ['odds'], ascending = False)
    df_odds_pos = df_odds_pos.sort_values(by = ['odds'], ascending = False)


    df_odds_neg = transform_text(df_odds_neg, 'word')
    df_odds_pos = transform_text(df_odds_pos, 'word')
    pos_list = ['NOUN', 'VERB', 'NUM', 'PROPN', 'ADJ', 'ADV']
    df_odds_neg = df_odds_neg[df_odds_neg.POS_tags.isin(pos_list)]
    df_odds_pos = df_odds_pos[df_odds_pos.POS_tags.isin(pos_list)]
    df_odds_neg.loc[ df_odds_neg.word.str.startswith("@"), 'POS_tags' ]  = 'PROPN'
    df_odds_pos.loc[ df_odds_pos.word.str.startswith("@"), 'POS_tags' ]  = 'PROPN'

    df_odds_neg = df_odds_neg.reset_index(drop=True)
    df_odds_pos = df_odds_pos.reset_index(drop=True)
    df_odds_pos.index += 1
    df_odds_neg.index += 1

    df_odds_neg['abusive'] = df_odds_neg.word.apply(lambda x: " ".join( set(x.lower().split()).intersection(abus_words)  ))
    df_odds_neg['abusive'] = np.where( df_odds_neg['abusive'].fillna('').astype('str').map(len) > 1 , 'abusive', 'non-abusive' )
    df_odds_pos['abusive'] = df_odds_pos.word.apply(lambda x: " ".join( set(x.lower().split()).intersection(abus_words)  ))
    df_odds_pos['abusive'] = np.where( df_odds_pos['abusive'].fillna('').astype('str').map(len) > 1, 'abusive', 'non-abusive' )

    df_odds_pos_tags_summ = df_odds_pos.POS_tags.value_counts(normalize = True).round(2)*100
    df_odds_neg_tags_summ = df_odds_neg.POS_tags.value_counts(normalize = True).round(2)*100
    df_odds_pos_tags_summ = df_odds_pos_tags_summ.reset_index()
    df_odds_pos_tags_summ.columns = ['POS_tags', 'percentage']

    df_odds_neg_tags_summ = df_odds_neg_tags_summ.reset_index()
    df_odds_neg_tags_summ.columns = ['POS_tags', 'percentage']

    df_odds_pos_tags_summ = df_odds_pos_tags_summ[df_odds_pos_tags_summ.percentage > 1]
    df_odds_neg_tags_summ = df_odds_neg_tags_summ[df_odds_neg_tags_summ.percentage > 1]

    df_odds_pos_words = set(df_odds_pos.word.values)
    df_odds_neg_words = set(df_odds_neg.word.values)

    df_odds = pd.concat( [df_odds_pos, df_odds_neg], axis = 0, ignore_index = True )
    df_odds = df_odds.sort_values(by = ['category', 'odds'], ascending = False)
    df['odds_words_'+df_odds_pos.category.iloc[0]] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_pos_words) ))
    df['odds_words_'+df_odds_neg.category.iloc[0]] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_neg_words) ))

    df_odds_pos_abs= df_odds_pos.abusive.value_counts(normalize = True).round(3)*100
    df_odds_neg_abs = df_odds_neg.abusive.value_counts(normalize = True).round(3)*100
    df_odds_pos_abs = df_odds_pos_abs.reset_index()
    df_odds_pos_abs.columns = ['abusive', 'percentage']
    df_odds_neg_abs = df_odds_neg_abs.reset_index()
    df_odds_neg_abs.columns = ['abusive', 'percentage']

    tab_odd, tab_pos, tab_abuse = st.tabs(['Odds', 'POS', 'Abusiveness'])
    with tab_odd:
        oddpos_c, oddneg_c = st.columns(2, gap = 'large')
        cols_odds = ['source', 'sentence', 'ethos_label', 'pathos_label', 'Target',
                     'odds_words_'+df_odds_pos.category.iloc[0], 'odds_words_'+df_odds_neg.category.iloc[0]]
        dimm = selected_rhet_dim.split("_")[0]

        with oddpos_c:
            st.write(f'Number of **{dimm} {df_odds_pos.category.iloc[0]}** words: {len(df_odds_pos)} ')
            st.dataframe(df_odds_pos)
            add_spacelines(1)
            pos_list_freq = df_odds_pos.word.tolist()
            freq_word_pos = st.multiselect('Choose a word you would like to see data cases for', pos_list_freq, pos_list_freq[:4:2])
            df_odds_pos_words = set(freq_word_pos)
            df['odds_words_'+df_odds_pos.category.iloc[0]] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_pos_words) ))
            df01 = df[ (df['odds_words_'+df_odds_pos.category.iloc[0]].str.split().map(len) >= 1) & (df[selected_rhet_dim] == df_odds_pos.category.iloc[0]) ]
            st.write(f'Cases with **{freq_word_pos}** words:')
            cols = ['source', 'sentence', 'ethos_label', 'Target', 'odds_words_'+df_odds_pos.category.iloc[0]]
            st.dataframe(df01[cols].set_index('source'))
            #st.dataframe(df_odds_pos_tags_summ)
            add_spacelines(1)
            #st.write(f'Cases with **{df_odds_pos.category.iloc[0]}** words:')
            #st.dataframe(df[ (df['odds_words_'+df_odds_pos.category.iloc[0]].str.split().map(len) >= 1) &\
            #                    (df[selected_rhet_dim] == df_odds_pos.category.iloc[0])  ][cols_odds].set_index('source').drop_duplicates('sentence'))

        with oddneg_c:
            st.write(f'Number of **{dimm} {df_odds_neg.category.iloc[0]}** words: {len(df_odds_neg)} ')
            st.dataframe(df_odds_neg)
            add_spacelines(1)
            neg_list_freq = df_odds_neg.word.tolist()
            freq_word_neg = st.multiselect('Choose a word you would like to see data cases for', neg_list_freq, neg_list_freq[:4:2])
            df_odds_neg_words = set(freq_word_neg)
            df['odds_words_'+df_odds_neg.category.iloc[0]] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_neg_words) ))
            df01 = df[ (df['odds_words_'+df_odds_neg.category.iloc[0]].str.split().map(len) >= 1) & (df[selected_rhet_dim] == df_odds_neg.category.iloc[0]) ]
            cols = ['source', 'sentence', 'ethos_label', 'Target', 'odds_words_'+df_odds_neg.category.iloc[0]]
            st.write(f'Cases with **{freq_word_neg}** words:')
            st.dataframe(df01[cols].set_index('source'))
            #st.dataframe(df_odds_neg_tags_summ)
            add_spacelines(1)
            #st.write(f'Cases with **{df_odds_neg.category.iloc[0]}** words:')
            #st.dataframe(df[ (df['odds_words_'+df_odds_neg.category.iloc[0]].str.split().map(len) >= 1) &\
            #                    (df[selected_rhet_dim] == df_odds_neg.category.iloc[0]) ][cols_odds].set_index('source').drop_duplicates('sentence'))

    with tab_pos:
        sns.set(font_scale = 1.25, style = 'whitegrid')
        df_odds_pos_tags_summ['category'] = df_odds_pos.category.iloc[0]
        df_odds_neg_tags_summ['category'] = df_odds_neg.category.iloc[0]
        df_odds_pos = pd.concat([df_odds_pos_tags_summ, df_odds_neg_tags_summ], axis = 0, ignore_index=True)
        ffp = sns.catplot(kind='bar', data = df_odds_pos,
        y = 'POS_tags', x = 'percentage', hue = 'POS_tags', aspect = 1.3, height = 5, dodge=False,
        legend = False, col = 'category')
        ffp.set(ylabel = '')
        plt.tight_layout(w_pad=3)
        st.pyplot(ffp)
        add_spacelines(1)

        oddpos_cpos, oddneg_cpos = st.columns(2, gap = 'large')
        with oddpos_cpos:
            st.write(f'POS analysis of **{dimm} {df_odds_pos.category.iloc[0]}** words')
            add_spacelines(1)
            st.dataframe(df_odds_pos_tags_summ)
            add_spacelines(1)

        with oddneg_cpos:
            st.write(f'POS analysis of **{dimm} {df_odds_neg.category.iloc[0]}** words')
            add_spacelines(1)
            st.dataframe(df_odds_neg_tags_summ)
            add_spacelines(1)


    with tab_abuse:
        sns.set(font_scale = 1, style = 'whitegrid')
        df_odds_pos_abs['category'] = df_odds_pos.category.iloc[0]
        df_odds_neg_abs['category'] = df_odds_neg.category.iloc[0]
        df_odds_abs = pd.concat([df_odds_pos_abs, df_odds_neg_abs], axis = 0, ignore_index=True)
        ffp = sns.catplot(kind='bar', data = df_odds_abs,
        y = 'abusive', x = 'percentage', hue = 'abusive', aspect = 1.3, height = 3,dodge=False,
        palette = {'abusive':'darkred', 'non-abusive':'grey'}, legend = False, col = 'category')
        ffp.set(ylabel = '')
        plt.tight_layout(w_pad=3)
        st.pyplot(ffp)

        oddpos_cab, oddneg_cab = st.columns(2, gap = 'large')
        with oddpos_cab:
            st.write(f'Abusiveness analysis of **{dimm} {df_odds_pos.category.iloc[0]}** words')
            add_spacelines(1)
            st.dataframe(df_odds_pos_abs)
            add_spacelines(1)
            #ffp = sns.catplot(kind='bar', data = df_odds_pos_abs,
            #y = 'abusive', x = 'percentage', hue = 'abusive', aspect = 1.3, height = 3,dodge=False,
            #palette = {'abusive':'darkred', 'non-abusive':'grey'}, legend = False)
            #ffp.set(ylabel = '', title = f'Abusiveness of {dimm} {df_odds_pos.category.iloc[0]} words')
            #st.pyplot(ffp)
            add_spacelines(1)
            if df_odds_pos_abs.shape[0] > 1:
                st.write(df_odds_pos[df_odds_pos['abusive'] == 'abusive'])

        with oddneg_cab:
            st.write(f'Abusiveness analysis of **{dimm} {df_odds_neg.category.iloc[0]}** words')
            add_spacelines(1)
            st.dataframe(df_odds_neg_abs)
            add_spacelines(1)
            #ffn = sns.catplot(kind='bar', data = df_odds_neg_abs,
            #y = 'abusive', x = 'percentage', hue = 'abusive', aspect = 1.3, height = 3, dodge=False,
            #palette = {'abusive':'darkred', 'non-abusive':'grey'}, legend = False)
            #ffn.set(ylabel = '', title = f'Abusiveness of {dimm} {df_odds_neg.category.iloc[0]} words')
            #st.pyplot(ffn)
            add_spacelines(1)
            if df_odds_neg_abs.shape[0] > 1:
                st.write(df_odds_neg[df_odds_neg['abusive'] == 'abusive'])


################################################

#@st.cache
@st.cache_data
def PolarizingNetworksSub(df3):
    pos_n = []
    neg_n = []
    neu_n = []
    tuples_tree = {
     'pos': [],
     'neg': [],
     'neu': []}
    for i in df3.index:
     if df3.ethos_label.loc[i] == 'support':
       tuples_tree['pos'].append(tuple( [str(df3.source.loc[i]), str(df3.Target.loc[i]).replace("@", '')] ))
       pos_n.append(df3.source.loc[i])

     elif df3.ethos_label.loc[i] == 'attack':
       tuples_tree['neg'].append(tuple( [str(df3.source.loc[i]), str(df3.Target.loc[i]).replace("@", '')] ))
       neg_n.append(df3.source.loc[i])

     elif df3.ethos_label.loc[i] == 'neutral':
       tuples_tree['neu'].append(tuple( [str(df3.source.loc[i]), str(df3.Target.loc[i]).replace("@", '')] ))
       neu_n.append(df3.source.loc[i])

    G = nx.DiGraph()

    default_weight = 0.7
    for nodes in tuples_tree['neu']:
        n0 = nodes[0]
        n1 = nodes[1]
        if n0 != n1:
            if G.has_edge(n0,n1):
                if G[n0][n1]['weight'] <= 4:
                    G[n0][n1]['weight'] += default_weight
            else:
                G.add_edge(n0,n1, weight=default_weight, color='blue')

    default_weight = 0.9
    for nodes in tuples_tree['neg']:
        n0 = nodes[0]
        n1 = nodes[1]
        if n0 != n1:
            if G.has_edge(n0,n1):
                if G[n0][n1]['weight'] <= 4:
                    G[n0][n1]['weight'] += default_weight
            else:
                G.add_edge(n0,n1, weight=default_weight, color='red')

    default_weight = 0.9
    for nodes in tuples_tree['pos']:
        n0 = nodes[0]
        n1 = nodes[1]
        if n0 != n1:
            if G.has_edge(n0,n1):
                if G[n0][n1]['weight'] <= 4:
                    G[n0][n1]['weight'] += default_weight
            else:
                G.add_edge(n0,n1, weight=default_weight, color='green')

    colors_nx_node = {}
    for n0 in G.nodes():
        if not (n0 in neu_n or n0 in neg_n or n0 in pos_n):
            colors_nx_node[n0] = 'grey'
        elif n0 in neu_n and not (n0 in neg_n or n0 in pos_n):
            colors_nx_node[n0] = 'blue'
        elif n0 in pos_n and not (n0 in neg_n or n0 in neu_n):
            colors_nx_node[n0] = 'green'
        elif n0 in neg_n and not (n0 in neu_n or n0 in pos_n):
            colors_nx_node[n0] = 'red'
        else:
            colors_nx_node[n0] = 'gold'
    nx.set_node_attributes(G, colors_nx_node, name="color")
    return G



def UserProfileResponse(df_list):
    st.write("### Fellows - Devils")
    add_spacelines(1)
    meth_feldev = 'frequency' #st.radio("Choose a method of calculation", ('frequency', 'log-likelihood ratio') )
    add_spacelines(1)
    selected_rhet_dim = 'ethos'
    selected_rhet_dim = selected_rhet_dim+"_label"
    add_spacelines(1)
    df = df_list[0]
    df.source = df.source.astype('str')
    df.source = np.where(df.source == 'nan', 'user1', df.source)
    #df.source = "@" + df.source

    df = df.drop_duplicates(subset = ['source', 'sentence'])
    df.Target = df.Target.str.replace('humans', 'people')
    src = df.source.unique()
    #df.Target = df.Target.str.replace("@@", "")
    #df.Target = df.Target.str.replace("@", "")
    #df.source = df.source.str.replace("@@", "")
    #df.source = df.source.str.replace("@", "")

    df['mentions'] = df.sentence.apply(lambda x: " ".join( str(w).replace(',','') for w in str(x).split() if '@' in w ))
    df['mentions'] = df['mentions'].astype('str').apply( lambda x: x.split()[0] if len(x) > 1 else '' )
    df = df.sort_values(by = 'date')

    emo_labs = set( df.emotion.unique())
    if not 'neutral' in df['ethos_label'].unique():
        df['ethos_label'] = df['ethos_label'].map(ethos_mapping)

    col_prof1, col_prof2 = st.columns(2)
    with col_prof1:
        st.write("##### Profile 1")
        label_etho1 = st.radio("Choose a label of ethos appeals in **Profile-1**", ('attack', 'support'))
        label_emo1 = st.radio("Choose a label of emotions expressed in **Profile-1**", emo_labs, index=0)

        df_prof1 = df[ (df['ethos_label'] == label_etho1) & (df['emotion'] == label_emo1) ].sort_values(by = 'date')
        prof1_src = df_prof1.source.unique()
        prof1_mtn = df_prof1.mentions.unique()
        df_prof1_resp = df_prof1[ (df_prof1.source.isin(prof1_mtn)) & (df_prof1.mentions.isin(prof1_src)) ].sort_values(by = 'date')
        df_prof1_resp2 = df_prof1[ (df_prof1.mentions.isin(prof1_src)) ].sort_values(by = 'date')
        ids1 = set(df_prof1.index)
        ids2 = set(df_prof1_resp2.index)
        ids22 = list( ids2.difference(ids1) )
        df_prof1_resp2 = df_prof1_resp2.loc[ids22]

        st.write(df_prof1) # @sulnick1 -> @Sdg13Un
        st.write( df_prof1[df_prof1.mentions == '@sulnick1'] )
        st.write(df_prof1_resp)
        st.write(df_prof1_resp2)




    with col_prof2:
        st.write("##### Profile 2")
        label_etho2 = st.radio("Choose a label of ethos appeals in **Profile-2**", ('attack', 'support'))
        label_emo2 = st.radio("Choose a label of emotions expressed in **Profile-2**", emo_labs, index=2)

        df_prof2 = df[ (df['ethos_label'] == label_etho2) & (df['emotion'] == label_emo2) ].sort_values(by = 'date')





def FellowsDevils(df_list):
    st.write("### Fellows - Devils")
    add_spacelines(1)
    meth_feldev = 'frequency' # st.radio("Choose a method of calculation", ('frequency', 'log-likelihood ratio') )
    add_spacelines(1)
    selected_rhet_dim = 'ethos'
    selected_rhet_dim = selected_rhet_dim+"_label"
    add_spacelines(1)
    df = df_list[0]
    df.source = df.source.astype('str')
    df.source = np.where(df.source == 'nan', 'user1', df.source)
    df.source = "@" + df.source

    df = df.drop_duplicates(subset = ['source', 'sentence'])
    df.Target = df.Target.str.replace('humans', 'people')
    src = df.source.unique()
    df.Target = df.Target.str.replace("@@", "")
    df.Target = df.Target.str.replace("@", "")
    df.source = df.source.str.replace("@@", "")
    df.source = df.source.str.replace("@", "")

    df['mentions'] = df.sentence.apply(lambda x: " ".join( w for w in str(x).split() if '@' in w ))
    #df['sentence_lemmatized'] = df.sentence.apply(lambda x: " ".join( str(w).replace("#", "") for w in str(x).split() if not '@' in w ))
    #df = lemmatization(df, 'sentence_lemmatized')
    #df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
    #df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str') + " " + df['mentions']
    df['sentence_lemmatized'] = df['Target']

    if not 'neutral' in df['ethos_label'].unique():
        df['ethos_label'] = df['ethos_label'].map(ethos_mapping)

    if not meth_feldev == 'frequency':
        ddmsc = ['support', 'attack']
        odds_list_of_dicts = []
        effect_list_of_dicts = []
        freq_list_of_dicts = []
        # 1 vs rest
        #num = np.floor( len(df) / 10 )
        for ii, ddmsc1 in enumerate(ddmsc):
            dict_1vsall_percent = {}
            dict_1vsall_effect_size = {}
            dict_1vsall_freq = {}

            ddmsc12 = set(ddmsc).difference([ddmsc1])
            #all100popular = Counter(" ".join( df.lemmatized.values ).split()).most_common(100)
            #all100popular = list(w[0] for w in all100popular)

            ddmsc1w = " ".join( df[df[selected_rhet_dim] == ddmsc1 ].sentence_lemmatized.fillna('').astype('str').values ).split() # sentence_lemmatized
            c = len(ddmsc1w)
            #ddmsc1w = list(w for w in ddmsc1w if not w in all100popular)
            ddmsc1w = Counter(ddmsc1w).most_common() # num
            if ddmsc1 in ['positive', 'support']:
                ddmsc1w = [w for w in ddmsc1w if w[1] >= 3 ]
            else:
                ddmsc1w = [w for w in ddmsc1w if w[1] > 3 ]
            #print('**********')
            #print(len(ddmsc1w), ddmsc1w)
            #print([w for w in ddmsc1w if w[1] > 2 ])
            #print(len([w for w in ddmsc1w if w[1] > 2 ]))
            ddmsc1w_word = dict(ddmsc1w)

            #st.write( list(ddmsc12)[0] )

            ddmsc2w = " ".join( df[ df[selected_rhet_dim] == list(ddmsc12)[0] ].sentence_lemmatized.fillna('').astype('str').values ).split() # sentence_lemmatized
            d = len(ddmsc2w)
            #ddmsc2w = list(w for w in ddmsc2w if not w in all100popular)
            ddmsc2w = Counter(ddmsc2w).most_common()
            ddmsc2w_word = dict(ddmsc2w)


            ddmsc1w_words = list( ddmsc1w_word.keys() )
            for n, dim in enumerate( ddmsc1w_words ):

                a = ddmsc1w_word[dim]
                try:
                    b = ddmsc2w_word[dim]
                except:
                    b = 0.5

                ca = c-a
                bd = d-b

                E1 = c*(a+b) / (c+d)
                E2 = d*(a+b) / (c+d)

                g2 = 2*((a*np.log(a/E1)) + (b* np.log(b/E2)))
                g2 = round(g2, 2)

                odds = round( (a*(d-b)) / (b*(c-a)), 2)

                if odds > 1:

                    if g2 > 10.83:
                        #print(f"{dim, g2, odds} ***p < 0.001 ")
                        dict_1vsall_percent[dim] = odds
                        dict_1vsall_effect_size[dim] = 0.001
                        dict_1vsall_freq[dim] = a
                    elif g2 > 6.63:
                        #print(f"{dim, g2, odds} **p < 0.01 ")
                        dict_1vsall_percent[dim] = odds
                        dict_1vsall_effect_size[dim] = 0.01
                        dict_1vsall_freq[dim] = a
                    elif g2 > 3.84:
                        #print(f"{dim, g2, odds} *p < 0.05 ")
                        dict_1vsall_percent[dim] = odds
                        dict_1vsall_effect_size[dim] = 0.05
                        dict_1vsall_freq[dim] = a
            #print(dict(sorted(dict_1vsall_percent.items(), key=lambda item: item[1])))
            odds_list_of_dicts.append(dict_1vsall_percent)
            effect_list_of_dicts.append(dict_1vsall_effect_size)
            freq_list_of_dicts.append(dict_1vsall_freq)

        df_odds_pos = pd.DataFrame({
                    'word':odds_list_of_dicts[0].keys(),
                    'odds':odds_list_of_dicts[0].values(),
                    'frequency':freq_list_of_dicts[0].values(),
                    'effect_size_p':effect_list_of_dicts[0].values(),
        })
        df_odds_pos['category'] = 'fellows'
        df_odds_neg = pd.DataFrame({
                    'word':odds_list_of_dicts[1].keys(),
                    'odds':odds_list_of_dicts[1].values(),
                    'frequency':freq_list_of_dicts[1].values(),
                    'effect_size_p':effect_list_of_dicts[1].values(),
        })
        df_odds_neg['category'] = 'devils'

        df_odds_neg = df_odds_neg.sort_values(by = 'odds', ascending = False)
        df_odds_neg = df_odds_neg[df_odds_neg.word != 'user']

        df_odds_pos = df_odds_pos.sort_values(by = 'odds', ascending = False)
        df_odds_pos = df_odds_pos[df_odds_pos.word != 'user']

    else:
        df = df[df.Target != 'nan']
        df_odds_pos = pd.DataFrame( df[df.ethos_label == 'support'].Target.value_counts() ).reset_index()
        df_odds_pos.columns = ['word', 'size']
        df_odds_pos['category'] = 'fellows'
        df_odds_pos.index += 1

        df_odds_neg = pd.DataFrame( df[df.ethos_label == 'attack'].Target.value_counts() ).reset_index()
        df_odds_neg.columns = ['word', 'size']
        df_odds_neg['category'] = 'devils'
        df_odds_neg.index += 1
        #st.write(df_odds_neg)

    #st.write( df[ (df.Target != 'nan' ) & (df.ethos_label == 'neutral') ] )
    #st.stop()


    df_odds_neg = transform_text(df_odds_neg, 'word')
    df_odds_pos = transform_text(df_odds_pos, 'word')
    df_odds_neg.loc[ df_odds_neg.word.str.startswith("@"), 'POS_tags' ]  = 'PROPN'
    df_odds_pos.loc[ df_odds_pos.word.str.startswith("@"), 'POS_tags' ]  = 'PROPN'
    pos_list = ['NOUN', 'PROPN']
    df_odds_neg = df_odds_neg[df_odds_neg.POS_tags.isin(pos_list)]
    df_odds_pos = df_odds_pos[df_odds_pos.POS_tags.isin(pos_list)]

    targets_list = df.Target.astype('str').unique()
    df_odds_pos = df_odds_pos[df_odds_pos.word.isin(targets_list)]
    df_odds_neg = df_odds_neg[df_odds_neg.word.isin(targets_list)]

    df_odds_neg = df_odds_neg.reset_index(drop=True)
    df_odds_pos = df_odds_pos.reset_index(drop=True)
    df_odds_pos.index += 1
    df_odds_neg.index += 1

    df_odds_pos_words = set(df_odds_pos.word.values)
    df_odds_neg_words = set(df_odds_neg.word.values)


    tab_odd, tab_fellow, tab_devil = st.tabs(['Tables', 'Fellows', 'Devils'])
    with tab_odd:
        oddpos_c, oddneg_c = st.columns(2, gap = 'large')
        cols_odds = ['source', 'sentence', 'ethos_label', 'Target']

        with oddpos_c:
            st.write(f'Number of **{df_odds_pos.category.iloc[0]}**: {len(df_odds_pos)} ')
            st.dataframe(df_odds_pos)
            add_spacelines(1)
            pos_list_freq = df_odds_pos.word.tolist()
            freq_word_pos = st.multiselect('Choose entities for network analytics', pos_list_freq, pos_list_freq[:3])
            df_odds_pos_words = set(freq_word_pos)
            df0p = df[df.Target.isin(df_odds_pos_words)]

            #pos_list_freq = df_odds_pos.word.tolist()
            #freq_word_pos = st.multiselect('Choose a word you would like to see data cases for', pos_list_freq, pos_list_freq[:2])
            #df_odds_pos_words = set(freq_word_pos)
            #df[df_odds_pos.category.iloc[0]] = df.Target.apply(lambda x: " ".join( set([x]).intersection(df_odds_pos_words) ))
            #df0p = df[ (df[df_odds_pos.category.iloc[0]].str.split().map(len) >= 1) ]
            #st.write(f'Cases with **{freq_word_pos}** :')
            #st.dataframe(df0p[cols_odds])

            #df0p = df[df.Target.isin(pos_list_freq)]
            #df0p = df0p.groupby(['Target', 'source'], as_index=False).size()
            #df0p.Target = np.where(df0p.Target.duplicated(), '', df0p.Target)
            #df0p = df0p.rename(columns = {'size':'# references'})
            #st.write(df0p)
            add_spacelines(1)


        with oddneg_c:
            st.write(f'Number of **{df_odds_neg.category.iloc[0]}**: {len(df_odds_neg)} ')
            st.dataframe(df_odds_neg)
            add_spacelines(1)
            neg_list_freq = df_odds_neg.word.tolist()
            freq_word_neg = st.multiselect('Choose entities for network analytics', neg_list_freq, neg_list_freq[:3])
            df_odds_neg_words = set(freq_word_neg)
            df0n = df[df.Target.isin(df_odds_neg_words)]

            #neg_list_freq = df_odds_neg.word.tolist()
            #freq_word_neg = st.multiselect('Choose a word you would like to see data cases for', neg_list_freq, neg_list_freq[:2])
            #df_odds_neg_words = set(freq_word_neg)
            #df[df_odds_neg.category.iloc[0]] = df.Target.apply(lambda x: " ".join( set([x]).intersection(df_odds_neg_words) ))
            #df0n = df[ (df[df_odds_neg.category.iloc[0]].str.split().map(len) >= 1) ]
            #st.write(f'Cases with **{freq_word_neg}** words:')
            #st.dataframe(df0n[cols_odds])

            #df0p = df[df.Target.isin(neg_list_freq)]
            #df0p = df0p.groupby(['Target', 'source'], as_index=False).size()
            #df0p.Target = np.where(df0p.Target.duplicated(), '', df0p.Target)
            #df0p = df0p.rename(columns = {'size':'# references'})
            #st.write(df0p)
            add_spacelines(1)


    with tab_fellow:

        st.write("")
        pos_tr = df_odds_pos_words #df_odds_pos.word.unique() #df.Target.unique()
        pos_sr = df[ (df.Target.isin(pos_tr)) & (df.ethos_label == 'support') ].source.unique() #df.source.unique()

        df0p = df[ ((df.source.isin(pos_sr)) & (df.Target.isin(pos_tr))) | (df.source.isin(pos_sr)) | (df.Target.isin(pos_sr)) ] #  | (df.source.isin(pos_tr)) | (df.Target.isin(pos_sr))
        df0p.Target = df0p.Target.astype('str')
        df0p = df0p[df0p.Target != 'nan']

        #st.write(df0p)
        #st.write(df0p.shape)

        df0p_graph = df0p.groupby(['source', 'Target'], as_index=False).size()
        #G_a = nx.from_pandas_edgelist(df0p_graph,
        #                    source='source',
        #                    target='Target',
        #                    edge_attr = 'size',
        #                    create_using=nx.DiGraph()) # edge_attr='emo_src',
        #edges_g = len(G_a.edges())
        node_s = 100

        sns.set(font_scale=1.35, style='whitegrid')
        #widths = np.asarray(list(nx.get_edge_attributes(G_a,'size').values()))/5
        #widths = [ w if w < 3 else 3 for w in widths ]

        fig1, ax1 = plt.subplots(figsize = (12, 10))
        #pos = graphviz_layout(G_a, prog='dot' )
        #nx.draw_networkx(G_a, with_labels=True, pos = pos,
        #      width=widths, font_size=8,
        #      alpha=0.75, node_size = 120)# edge_color=colors, node_color = colors_nodes,
        #plt.title('')
        #plt.draw()
        #plt.show()
        #st.pyplot(fig1)

        G = PolarizingNetworksSub(df0p)
        sns.set(font_scale=1.25, style='whitegrid')
        widths = list(nx.get_edge_attributes(G,'weight').values())
        widths = [ w - 0.2 if w < 2.5 else 2.5 for w in widths ]
        colors = list(nx.get_edge_attributes(G,'color').values())
        colors_nodes = list(nx.get_node_attributes(G, "color").values())

        fig2, ax2 = plt.subplots(figsize = (14, 13))
        pos = nx.drawing.layout.spring_layout(G, k=0.75, iterations=20, seed=6)

        nx.draw_networkx(G, with_labels=False, pos = pos,
               width=widths, edge_color=colors,
               alpha=0.75, node_color = colors_nodes, node_size = 450)

        font_names = ['Sawasdee', 'Gentium Book Basic', 'FreeMono']
        family_names = ['sans-serif', 'serif', 'fantasy', 'monospace']
        pos_tr = list( x.replace("@", "") for x in pos_tr )

        text = nx.draw_networkx_labels(G, pos, font_size=10,
                labels = { n:n if not (n in pos_tr or n in pos_sr) else '' for n in nx.nodes(G) } )

        for i, nodes in enumerate(pos_tr):
            # extract the subgraph
            g = G.subgraph(pos_tr[i])
            # draw on the labels with different fonts
            nx.draw_networkx_labels(g, pos, font_size=14.5, font_weight='bold', font_color = 'darkgreen')

        for i, nodes in enumerate(pos_sr):
            # extract the subgraph
            g = G.subgraph(pos_sr[i])
            # draw on the labels with different fonts
            nx.draw_networkx_labels(g, pos, font_size=10, font_weight='bold',)
        #for _, t in text.items():
            #t.set_rotation(0)

        import matplotlib.patches as mpatches
        # add legend
        att_users_only = mpatches.Patch(color='red', label='negative')
        both_users = mpatches.Patch(color='gold', label='ambivalent')
        sup_users_only = mpatches.Patch(color='green', label='positive')
        #neu_users_only = mpatches.Patch(color='blue', label='neutral')
        targ_only = mpatches.Patch(color='grey', label='target only')
        plt.legend(handles=[att_users_only, sup_users_only, both_users, targ_only],
                    loc = 'upper center', bbox_to_anchor = (0.5, 1.045), ncol = 5, title = f'{df.corpus.iloc[0].split()[0]} Network of Fellows')
        plt.draw()
        plt.show()
        st.pyplot(fig2)
        add_spacelines(2)
        df0p_graph = df0p.groupby(['source', 'Target', 'ethos_label'], as_index=False).size()
        #df0p_graph.ethos_label = df0p_graph.ethos_label.map({'attack':'negative', 'support':'positive'})
        st.dataframe(df0p_graph.rename( columns = {'size':"# references", "ethos_label":"reference"} ))



    with tab_devil:
        st.write("")
        neg_tr = df_odds_neg_words #df_odds_neg.word.unique() #df.Target.unique()
        neg_sr = df[(df.Target.isin(neg_tr)) & (df.ethos_label == 'attack') ].source.unique() #df.source.unique()

        df0p = df[ ((df.source.isin(neg_sr)) & (df.Target.isin(neg_tr)) ) | (df.source.isin(neg_sr)) | (df.Target.isin(neg_sr)) ] #  | (df.source.isin(neg_tr)) | (df.Target.isin(neg_sr))
        df0p.Target = df0p.Target.astype('str')
        df0p = df0p[df0p.Target != 'nan']

        #st.write(df0p)
        #st.write(df0p.shape)

        df0p_graph = df0p.groupby(['source', 'Target'], as_index=False).size()
        #G_a = nx.from_pandas_edgelist(df0p_graph,
        #                    source='source',
        #                    target='Target',
        #                    edge_attr = 'size',
        #                    create_using=nx.DiGraph()) # edge_attr='emo_src',
        #edges_g = len(G_a.edges())
        node_s = 100

        sns.set(font_scale=1.35, style='whitegrid')
        #widths = np.asarray(list(nx.get_edge_attributes(G_a,'size').values()))/5
        #widths = [ w if w < 2.5 else 2.5 for w in widths ]

        #fig1, ax1 = plt.subplots(figsize = (12, 10))
        #pos = graphviz_layout(G_a, prog='dot' )
        #nx.draw_networkx(G_a, with_labels=True, pos = pos,
        #      width=widths, font_size=8,
        #      alpha=0.75, node_size = 120)# edge_color=colors, node_color = colors_nodes,
        #plt.title('')
        #plt.draw()
        #plt.show()
        #st.pyplot(fig1)

        G = PolarizingNetworksSub(df0p)
        sns.set(font_scale=1.25, style='whitegrid')
        widths = list(nx.get_edge_attributes(G,'weight').values())
        widths = [ w - 0.2 if w < 2.5 else 2.5 for w in widths ]
        colors = list(nx.get_edge_attributes(G,'color').values())
        colors_nodes = list(nx.get_node_attributes(G, "color").values())

        #st.write(df.corpus.iloc[0])

        fig2, ax2 = plt.subplots(figsize = (12, 11))
        pos = nx.drawing.layout.spring_layout(G, k=0.5, iterations=25, seed=5)

        nx.draw_networkx(G, with_labels=False, pos = pos,
               width=widths, edge_color=colors,
               alpha=0.75, node_color = colors_nodes, node_size = 450)

        font_names = ['Sawasdee', 'Gentium Book Basic', 'FreeMono']
        family_names = ['sans-serif', 'serif', 'fantasy', 'monospace']
        neg_tr = list( x.replace("@", "") for x in neg_tr )

        text = nx.draw_networkx_labels(G, pos, font_size=10,
                labels = { n:n if not (n in neg_tr or n in neg_sr) else '' for n in nx.nodes(G) } )

        for i, nodes in enumerate(neg_tr):
            # extract the subgraph
            g = G.subgraph(neg_tr[i])
            # draw on the labels with different fonts
            nx.draw_networkx_labels(g, pos, font_size=14.5, font_weight='bold', font_color = 'darkred')

        for i, nodes in enumerate(neg_sr):
            # extract the subgraph
            g = G.subgraph(neg_sr[i])
            # draw on the labels with different fonts
            nx.draw_networkx_labels(g, pos, font_size=10, font_weight='bold',)
        #for _, t in text.items():
            #t.set_rotation(0)

        import matplotlib.patches as mpatches
        # add legend
        att_users_only = mpatches.Patch(color='red', label='negative')
        both_users = mpatches.Patch(color='gold', label='ambivalent')
        sup_users_only = mpatches.Patch(color='green', label='positive')
        #neu_users_only = mpatches.Patch(color='blue', label='neutral')
        targ_only = mpatches.Patch(color='grey', label='target only')
        plt.legend(handles=[att_users_only, sup_users_only, both_users, targ_only],
                    loc = 'upper center', bbox_to_anchor = (0.5, 1.045), ncol = 5, title = f'{df.corpus.iloc[0].split()[0]} Network of Devils')
        plt.draw()
        plt.show()
        st.pyplot(fig2)
        add_spacelines(2)
        df0p_graph = df0p.groupby(['source', 'Target', 'ethos_label'], as_index=False).size()
        #df0p_graph.ethos_label = df0p_graph.ethos_label.map({'attack':'negative', 'support':'positive'})
        st.dataframe(df0p_graph.rename( columns = {'size':"# references", "ethos_label":"reference"} ))



###################################################

def generateWordCloud_log():
    selected_rhet_dim = st.selectbox("Choose a rhetoric category for a WordCloud", rhetoric_dims, index=0)
    add_spacelines(1)
    if selected_rhet_dim == 'pathos':
        label_cloud = st.radio("Choose a label for words in WordCloud", ('negative', 'positive'))
        selected_rhet_dim = selected_rhet_dim.replace("ethos", "ethos_label").replace("pathos", "pathos_label")
        label_cloud = label_cloud.replace("negative", "attack").replace("positive", "support")
    else:
        label_cloud = st.radio("Choose a label for words in WordCloud", ('attack', 'support'))
        selected_rhet_dim = selected_rhet_dim.replace("ethos", "ethos_label")
        label_cloud = label_cloud.replace("attack / negative", "attack").replace("support / positive", "support")

    add_spacelines(1)
    threshold_cloud = st.slider('Select a precision value (threshold) for words in WordCloud', 0, 100, 80)
    st.info(f'Selected precision: **{threshold_cloud}**')
    add_spacelines(1)
    st.write("**Processing the output ...**")

    generateWordCloudc1, generateWordCloudc2 = st.columns(2)
    with generateWordCloudc1:
        st.write(f"##### {corpora_list[0].corpus.iloc[0]}")
        add_spacelines(1)
        generateWordCloud_sub_log(corpora_list[:2], rhetoric_dims = ['ethos', 'logos'], an_type = contents_radio_an_cat,
            selected_rhet_dim = selected_rhet_dim, label_cloud=label_cloud, threshold_cloud=threshold_cloud)
    with generateWordCloudc2:
        st.write(f"##### {corpora_list[-1].corpus.iloc[0]}")
        add_spacelines(1)
        generateWordCloud_sub_log(corpora_list[2:], rhetoric_dims = ['ethos', 'logos'], an_type = contents_radio_an_cat,
            selected_rhet_dim = selected_rhet_dim, label_cloud=label_cloud, threshold_cloud=threshold_cloud)


def generateWordCloud_sub_log(df_list,
        selected_rhet_dim, label_cloud, threshold_cloud,
        rhetoric_dims = ['ethos', 'pathos'], an_type = 'ADU-based'):

    df = df_list[0]
    #st.write(df)
    add_spacelines(1)
    if selected_rhet_dim != 'logos':
        df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
        if not 'neutral' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        if not 'pathos_label' in df['pathos_label'].unique():
            df['pathos_label'] = df['pathos_label'].map(valence_mapping)

    elif selected_rhet_dim == 'logos':
        df = df_list[-1] #pd.concat(df_list, axis=0, ignore_index=True)
        #st.write(df)
        df = df.dropna(subset = 'premise')
        df['sentence_lemmatized'] = df['premise'].astype('str') + " " + df['conclusion'].astype('str')

        if an_type != 'Relation-based':
            df = lemmatization(df, 'sentence_lemmatized', name_column = True)
            df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str').str.lower().str.replace('ahould', 'should')

        elif an_type == 'Relation-based':
            df['premise'] = df['premise'].astype('str')
            df['conclusion'] = df['conclusion'].astype('str')
            df = df.reset_index()

            dfp = df.groupby(['id_connection', 'connection'])['premise'].apply(lambda x: " ".join(x)).reset_index()
            #st.write(dfp)
            #dfp['sentence_lemmatized'] = dfp['sentence_lemmatized'].astype('str')
            dfc = df.groupby(['id_connection', 'connection'])['conclusion'].apply(lambda x: " ".join(x)).reset_index()
            #st.write(dfc)

            dfp = dfp.merge(dfc, on = ['id_connection', 'connection']) #pd.concat([dfp, dfc.iloc[:, -1:]], axis=1) #dfp.merge(dfc, on = ['id_connection', 'connection'])
            dfp = dfp.drop_duplicates()
            #st.write(dfp)
            #st.write(dfp[dfp.id_connection == 185352])
            #st.stop()
            dfp['sentence_lemmatized'] = dfp.premise.astype('str')+ " " + dfc['conclusion'].astype('str')
            #st.write(dfp)
            import re
            dfp['sentence_lemmatized'] = dfp['sentence_lemmatized'].apply(lambda x: re.sub(r"\W+", " ", str(x)))
            dfp = lemmatization(dfp, 'sentence_lemmatized', name_column = True)
            dfp['sentence_lemmatized'] = dfp['sentence_lemmatized'].astype('str').str.lower().str.replace('ahould', 'should')
            df = dfp.copy()
            #st.write(dfc.shape, dfp.shape, df.shape)

    st.write(df.corpus.iloc[0])

    if (selected_rhet_dim == 'ethos_label'):
         df_for_wordcloud = prepare_cloud_lexeme_data(df[df[str(selected_rhet_dim)] == 'neutral'],
         df[df[str(selected_rhet_dim)] == 'support'],
         df[df[str(selected_rhet_dim)] == 'attack'])

    elif selected_rhet_dim == 'logos':
         df_for_wordcloud = prepare_cloud_lexeme_data(df[ ~(df['connection'].isin(['Default Inference', 'Default Conflict'])) ],
         df[df['connection'] == 'Default Inference'],
         df[df['connection'] == 'Default Conflict'])
    else:
        df_for_wordcloud = prepare_cloud_lexeme_data(df[df[str(selected_rhet_dim)] == 'neutral'],
        df[df[str(selected_rhet_dim)] == 'positive'],
        df[df[str(selected_rhet_dim)] == 'negative'])

    fig_cloud1, df_cloud_words1, figure_cloud_words1 = wordcloud_lexeme(df_for_wordcloud, lexeme_threshold = threshold_cloud, analysis_for = str(label_cloud))

    #_, cw2, _ = st.columns([1, 6, 1])
    #with cw2:
    st.pyplot(fig_cloud1)

    add_spacelines(2)

    st.write(f'WordCloud frequency table: ')
    if selected_rhet_dim == 'pathos_label':

        label_cloud = label_cloud.replace('attack', 'negative').replace('support', 'positive')
        df_cloud_words1 = df_cloud_words1.rename(columns = {
        'precis':'precision',
        'attack #':'negative #',
        'general #':'overall #',
        'support #':'positive #',
        })
    else:
        df_cloud_words1 = df_cloud_words1.rename(columns = {'general #':'overall #', 'precis':'precision'})

    df_cloud_words1 = df_cloud_words1.sort_values(by = 'precision', ascending = False)
    df_cloud_words1 = df_cloud_words1.reset_index(drop = True)
    df_cloud_words1.index += 1
    st.write(df_cloud_words1)


    cols_odds1 = ['source', 'sentence', 'ethos_label', 'pathos_label', 'Target',
                         'freq_words_'+label_cloud]

    if selected_rhet_dim == 'logos':
        df = df.rename(columns = {'connection':'logos'})
        #cols_odds1 = ['locution_conclusion', 'locution_premise', 'logos', 'argument_linked', 'freq_words_'+label_cloud]
        cols_odds1 = ['premise', 'conclusion', 'sentence_lemmatized', 'logos', 'freq_words_'+label_cloud]
        df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str')
        df['logos'] = df['logos'].map({'Default Inference':'support', 'Default Conflict':'attack'})

    pos_list_freq = df_cloud_words1.word.tolist()
    freq_word_pos = st.multiselect('Choose word(s) you would like to see data cases for', pos_list_freq, pos_list_freq[:2])
    df_odds_pos_words = set(freq_word_pos)
    df['freq_words_'+label_cloud] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_pos_words) ))
    #st.write(df)
    add_spacelines(1)
    st.write(f'Cases with **{freq_word_pos}** words:')
    st.dataframe(df[ (df['freq_words_'+label_cloud].str.split().map(len) >= 1) & (df[selected_rhet_dim] == label_cloud) ][cols_odds1])# .set_index('source')





def generateWordCloud(df_list, rhetoric_dims = ['ethos', 'ethos & emotion'], an_type = 'ADU-based'):
    #st.header(f" Text-Level Analytics ")

    selected_rhet_dim = st.selectbox("Choose a rhetoric category for a WordCloud", rhetoric_dims, index=0)
    add_spacelines(1)
    if selected_rhet_dim in ['pathos', 'sentiment']:
        df = df_list[0]
        label_cloud = st.radio("Choose a label for words in WordCloud", ('negative', 'positive'))
        selected_rhet_dim = selected_rhet_dim.replace("ethos", "ethos_label").replace("pathos", "pathos_label")
        label_cloud = label_cloud.replace("negative", "attack").replace("positive", "support")

    elif selected_rhet_dim == 'ethos':
        df = df_list[0]
        label_cloud = st.radio("Choose a label for words in WordCloud", ('attack', 'support'))
        selected_rhet_dim = selected_rhet_dim.replace("ethos", "ethos_label")
        label_cloud = label_cloud.replace("attack / negative", "attack").replace("support / positive", "support")
    else:
        df = df_list[0]
        label_cloud = st.radio("Choose a label of **ethos** for words in WordCloud", ('attack', 'support'))
        label_cloud_emo = st.radio("Choose a label of **emotion** for words in WordCloud", set( df.emotion.unique()))
        selected_rhet_dim = selected_rhet_dim.replace("ethos", "ethos_label")

    add_spacelines(1)
    threshold_cloud = st.slider('Select a precision value (threshold) for words in WordCloud', 0, 100, 80)
    st.info(f'Selected precision: **{threshold_cloud}**')

    add_spacelines(1)
    st.write("**Processing the output ...**")
    #my_bar = st.progress(0)
    #for percent_complete in range(100):
        #time.sleep(0.1)
        #my_bar.progress(percent_complete + 1)

    add_spacelines(1)
    if selected_rhet_dim != 'logos':

        df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
        if not 'neutral' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        if not 'pathos_label' in df['pathos_label'].unique():
            df['pathos_label'] = df['pathos_label'].map(valence_mapping)

    elif selected_rhet_dim == 'logos':
        df = df_list[-1] #pd.concat(df_list, axis=0, ignore_index=True)
        df = df.dropna(subset = 'premise')
        df['sentence_lemmatized'] = df['premise'].astype('str') + " " + df['conclusion'].astype('str')

        if an_type != 'Relation-based':
            df = lemmatization(df, 'sentence_lemmatized', name_column = True)
            df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str').str.lower().str.replace('ahould', 'should')

        elif an_type == 'Relation-based':
            df['premise'] = df['premise'].astype('str')
            df['conclusion'] = df['conclusion'].astype('str')
            df = df.reset_index()

            dfp = df.groupby(['id_connection', 'connection'])['premise'].apply(lambda x: " ".join(x)).reset_index()
            #st.write(dfp)
            #dfp['sentence_lemmatized'] = dfp['sentence_lemmatized'].astype('str')
            dfc = df.groupby(['id_connection', 'connection'])['conclusion'].apply(lambda x: " ".join(x)).reset_index()
            #st.write(dfc)

            dfp = dfp.merge(dfc, on = ['id_connection', 'connection']) #pd.concat([dfp, dfc.iloc[:, -1:]], axis=1) #dfp.merge(dfc, on = ['id_connection', 'connection'])
            dfp = dfp.drop_duplicates()
            #st.write(dfp)
            #st.write(dfp[dfp.id_connection == 185352])
            #st.stop()
            dfp['sentence_lemmatized'] = dfp.premise.astype('str')+ " " + dfc['conclusion'].astype('str')
            #st.write(dfp)
            import re
            dfp['sentence_lemmatized'] = dfp['sentence_lemmatized'].apply(lambda x: re.sub(r"\W+", " ", str(x)))
            dfp = lemmatization(dfp, 'sentence_lemmatized', name_column = True)
            dfp['sentence_lemmatized'] = dfp['sentence_lemmatized'].astype('str').str.lower().str.replace('ahould', 'should')
            df = dfp.copy()
            #st.write(dfc.shape, dfp.shape, df.shape)


    if (selected_rhet_dim == 'ethos_label'):
         df_for_wordcloud = prepare_cloud_lexeme_data(df[df[str(selected_rhet_dim)] == 'neutral'],
         df[df[str(selected_rhet_dim)] == 'support'],
         df[df[str(selected_rhet_dim)] == 'attack'])

    elif (selected_rhet_dim == 'ethos_label & emotion'):
         df_for_wordcloud = prepare_cloud_lexeme_data(df[ (df['ethos_label'] == 'neutral') & (df['emotion'] == str(label_cloud_emo)) ],
         df[ (df['ethos_label'] == 'support') & (df['emotion'] == str(label_cloud_emo)) ],
         df[ (df['ethos_label'] == 'attack') & (df['emotion'] == str(label_cloud_emo)) ])

    elif selected_rhet_dim == 'logos':
         df_for_wordcloud = prepare_cloud_lexeme_data(df[ ~(df['connection'].isin(['Default Inference', 'Default Conflict'])) ],
         df[df['connection'] == 'Default Inference'],
         df[df['connection'] == 'Default Conflict'])
    else:
        df_for_wordcloud = prepare_cloud_lexeme_data(df[df[str(selected_rhet_dim)] == 'neutral'],
        df[df[str(selected_rhet_dim)] == 'positive'],
        df[df[str(selected_rhet_dim)] == 'negative'])

    fig_cloud1, df_cloud_words1, figure_cloud_words1 = wordcloud_lexeme(df_for_wordcloud, lexeme_threshold = threshold_cloud, analysis_for = str(label_cloud))

    #_, cw2, _ = st.columns([1, 6, 1])
    #with cw2:
    st.pyplot(fig_cloud1)

    add_spacelines(2)

    st.write(f'WordCloud frequency table: ')
    if selected_rhet_dim == 'pathos_label':

        label_cloud = label_cloud.replace('attack', 'negative').replace('support', 'positive')
        df_cloud_words1 = df_cloud_words1.rename(columns = {
        'precis':'precision',
        'attack #':'negative #',
        'general #':'overall #',
        'support #':'positive #',
        })
    else:
        df_cloud_words1 = df_cloud_words1.rename(columns = {'general #':'overall #', 'precis':'precision'})

    df_cloud_words1 = df_cloud_words1.sort_values(by = 'precision', ascending = False)
    df_cloud_words1 = df_cloud_words1.reset_index(drop = True)
    df_cloud_words1.index += 1
    st.write(df_cloud_words1)


    cols_odds1 = ['source', 'sentence', 'ethos_label', 'pathos_label', 'Target',
                         'freq_words_'+label_cloud]

    if selected_rhet_dim == 'logos':
        df = df.rename(columns = {'connection':'logos'})
        #cols_odds1 = ['locution_conclusion', 'locution_premise', 'logos', 'argument_linked', 'freq_words_'+label_cloud]
        cols_odds1 = ['premise', 'conclusion', 'sentence_lemmatized', 'logos', 'freq_words_'+label_cloud]
        df['sentence_lemmatized'] = df['sentence_lemmatized'].astype('str')
        df['logos'] = df['logos'].map({'Default Inference':'support', 'Default Conflict':'attack'})

    pos_list_freq = df_cloud_words1.word.tolist()
    freq_word_pos = st.multiselect('Choose word(s) you would like to see data cases for', pos_list_freq, pos_list_freq[:2])
    df_odds_pos_words = set(freq_word_pos)
    df['freq_words_'+label_cloud] = df.sentence_lemmatized.apply(lambda x: " ".join( set(x.split()).intersection(df_odds_pos_words) ))
    #st.write(df)
    add_spacelines(1)
    st.write(f'Cases with **{freq_word_pos}** words:')

    if (selected_rhet_dim == 'ethos_label & emotion'):
        st.dataframe(df[ (df['freq_words_'+label_cloud].str.split().map(len) >= 1) & (df['ethos_label'] == label_cloud) ][cols_odds1])# .set_index('source')
    else:
        st.dataframe(df[ (df['freq_words_'+label_cloud].str.split().map(len) >= 1) & (df[selected_rhet_dim] == label_cloud) ][cols_odds1])# .set_index('source')




def ProfilesEntity_compare(data_list, selected_rhet_dim):

        up_data_dict = {}
        up_data_dicth = {}
        up_data_dictah = {}
        target_shared = {}
        up_data_dict_hist = {}
        #n = 0
        #for data in data_list:
        #heroes_tab1, heroes_tab2 = st.tabs(['Overview', 'Single Case Analysis'])
        #with heroes_tab1:
        df = data_list[0].copy()
        #st.dataframe(df)
        ds = df['corpus'].iloc[0]
        add_spacelines(1)
        st.write("##### Profiles Overview")
        dds = df.groupby('source', as_index=False).size()
        dds = dds[dds['size']>2]
        ddt = df.groupby('Target', as_index=False).size()
        ddt = ddt[ddt['size']>2]

        df = df[(df.source.isin(dds.source.values)) | (df.Target.isin(ddt.Target.values))]

        if not 'attack' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        if not 'pathos_label' in df.columns:
            df['pathos_label'] = 'neutral'
        if not 'neutral' in df['pathos_label'].unique():
            df['pathos_label'] = df['pathos_label'].map(valence_mapping)

        df["source"] = df["source"].astype('str').str.replace("@", "")
        df["Target"] = df["Target"].astype('str').str.replace("@", "")
        #df = df[ (df.Target != 'nan') & (df.Target != '') & (df.ethos_label != 'neutral') ]
        df["Target"] = df["Target"].str.replace('Government', 'government')
        selected_rhet_dim_n_cats = {'pathos_label':2, 'ethos_label':2, 'sentiment':2}
        selected_rhet_dim_subcats = {'pathos_label':['negative', 'positive'],
                                    'ethos_label': ['attack', 'support'],
                                     'sentiment':['negative', 'positive']}

        dd2_sizet = pd.DataFrame(df[df[selected_rhet_dim] != 'neutral'].groupby(['Target'])[selected_rhet_dim].value_counts(normalize=True).round(3)*100)
        #dd2_sizet = pd.DataFrame(df.groupby(['Target'])[selected_rhet_dim].value_counts(normalize=True).round(3)*100)
        dd2_sizet.columns = ['percentage']
        dd2_sizet = dd2_sizet.reset_index()
        adj_target = dd2_sizet.Target.unique()
        for t in adj_target:
            dd_adj_target =  dd2_sizet[dd2_sizet.Target == t]
            if dd_adj_target.shape[0] != selected_rhet_dim_n_cats[selected_rhet_dim]:
                if 'support' in dd_adj_target[selected_rhet_dim].values and not 'attack' in dd_adj_target[selected_rhet_dim].values:
                    dd2_sizet.loc[len(dd2_sizet)] = [t, 'attack', 0]
                elif not 'support' in dd_adj_target[selected_rhet_dim].values and 'attack' in dd_adj_target[selected_rhet_dim].values:
                    dd2_sizet.loc[len(dd2_sizet)] = [t, 'support', 0]
                elif not 'negative' in dd_adj_target[selected_rhet_dim].values and ('positive' in dd_adj_target[selected_rhet_dim].values):
                    dd2_sizet.loc[len(dd2_sizet)] = [t, 'negative', 0]
                elif not 'positive' in dd_adj_target[selected_rhet_dim].values and ('negative' in dd_adj_target[selected_rhet_dim].values):
                    dd2_sizet.loc[len(dd2_sizet)] = [t, 'positive', 0]
                #elif not 'neutral' in dd_adj_target[selected_rhet_dim].values and ('positive' in dd_adj_target[selected_rhet_dim].values or 'negative' in dd_adj_target[selected_rhet_dim].values):
                    #dd2_sizet.loc[len(dd2_sizet)] = [t, 'neutral', 0]


        #if selected_rhet_dim == 'ethos_label':
            #dd2_sizes = pd.DataFrame(df[df[selected_rhet_dim] != 'neutral'].groupby(['source'])[selected_rhet_dim].value_counts(normalize=True).round(3)*100)
        #else:
            #dd2_sizes = pd.DataFrame(df.groupby(['source'])[selected_rhet_dim].value_counts(normalize=True).round(3)*100)
        dd2_sizes = pd.DataFrame(df[df[selected_rhet_dim] != 'neutral'].groupby(['source'])[selected_rhet_dim].value_counts(normalize=True).round(3)*100)

        dd2_sizes.columns = ['percentage']
        dd2_sizes = dd2_sizes.reset_index()
        adj_source = dd2_sizes.source.unique()
        for t in adj_source:
            dd_adj_source =  dd2_sizes[dd2_sizes.source == t]
            if dd_adj_source.shape[0] != selected_rhet_dim_n_cats[selected_rhet_dim]:
                if 'support' in dd_adj_source[selected_rhet_dim].values and not 'attack' in dd_adj_source[selected_rhet_dim].values:
                    dd2_sizes.loc[len(dd2_sizes)] = [t, 'attack', 0]
                elif not 'support' in dd_adj_source[selected_rhet_dim].values and 'attack' in dd_adj_source[selected_rhet_dim].values:
                    dd2_sizes.loc[len(dd2_sizes)] = [t, 'support', 0]
                elif not 'negative' in dd_adj_source[selected_rhet_dim].values and ('positive' in dd_adj_source[selected_rhet_dim].values or 'neutral' in dd_adj_source[selected_rhet_dim].values):
                    dd2_sizes.loc[len(dd2_sizes)] = [t, 'negative', 0]
                elif not 'positive' in dd_adj_source[selected_rhet_dim].values and ('negative' in dd_adj_source[selected_rhet_dim].values or 'neutral' in dd_adj_source[selected_rhet_dim].values):
                    dd2_sizes.loc[len(dd2_sizes)] = [t, 'positive', 0]
                elif not 'neutral' in dd_adj_source[selected_rhet_dim].values and ('positive' in dd_adj_source[selected_rhet_dim].values or 'negative' in dd_adj_source[selected_rhet_dim].values):
                    dd2_sizes.loc[len(dd2_sizes)] = [t, 'neutral', 0]

        dd2_sizet.columns = ['entity', 'category', 'percentage']
        dd2_sizes.columns = ['entity', 'category', 'percentage']
        dd2_sizet['role'] = 'passive'
        dd2_sizes['role'] = 'active'

        #if selected_rhet_dim == 'ethos_label':
        dd2_sizet = dd2_sizet[dd2_sizet.category != 'neutral']
        dd2_sizes = dd2_sizes[dd2_sizes.category != 'neutral']

        dd2_size = pd.concat([dd2_sizes, dd2_sizet], axis = 0, ignore_index = True)

        #st.write(dd2_size)

        #dd2_size = pd.pivot_table(dd2_size, values='percentage', index=['entity', 'role'], columns=['category'], aggfunc=np.sum)
        cat_neg = selected_rhet_dim_subcats[selected_rhet_dim][0]
        cat_pos = selected_rhet_dim_subcats[selected_rhet_dim][1]
        plt1 = {cat_neg:'darkred', cat_pos:'darkgreen'}
        dd2_size.percentage = dd2_size.percentage.round(-1).astype('int')

        #dd2_size['profile'] = np.where( (dd2_size['role'] = 'passive') & (dd2_size[cat_neg] > dd2_size[cat_pos]) )
        #fig_pr = sns.catplot(kind = 'count', data = dd2_size, y = 'percentage',
        #        col = 'role', hue = 'category', palette = plt1, aspect = 1.3)
        #st.pyplot(fig_pr)

        dd2_size = pd.pivot_table(dd2_size, values='percentage', index=['entity', 'role'], columns=['category'], aggfunc=np.sum)
        dd2_size = dd2_size.reset_index()
        #st.write(dd2_size)


        #dd2_sizet.columns = ['entity', 'category', 'percentage']
        dd2_sizet = pd.pivot_table(dd2_sizet, values='percentage', index=['entity', 'role'], columns=['category'], aggfunc=np.sum).reset_index()
        #dd2_sizes.columns = ['entity', 'category', 'percentage']
        dd2_sizes = pd.pivot_table(dd2_sizes, values='percentage', index=['entity', 'role'], columns=['category'], aggfunc=np.sum).reset_index()
        #st.write(dd2_sizes)
        dd2_sizet['role'] = 1
        dd2_sizes['role'] = 1

        dd2_sizes = dd2_sizes.rename(columns = {'role':'active', cat_neg:cat_neg+'_active', cat_pos:cat_pos+'_active'})
        dd2_sizet = dd2_sizet.rename(columns = {'role':'passive', cat_neg:cat_neg+'_passive', cat_pos:cat_pos+'_passive'})

        dd2_size_2 = dd2_sizes.merge(dd2_sizet, on = 'entity')
        #st.write(dd2_size_2)

        dd2_size_2['profile'] = 'other'
        dd2_size_2['profile'] = np.where((dd2_size_2[cat_neg+'_active'] > dd2_size_2[cat_pos+'_active']) & (dd2_size_2[cat_neg+'_passive'] > dd2_size_2[cat_pos+'_passive']), 'angry man', dd2_size_2['profile'] )
        dd2_size_2['profile'] = np.where((dd2_size_2[cat_neg+'_active'] < dd2_size_2[cat_pos+'_active']) & (dd2_size_2[cat_neg+'_passive'] < dd2_size_2[cat_pos+'_passive']), 'positive soul', dd2_size_2['profile'] )
        dd2_size_2['profile'] = np.where((dd2_size_2[cat_neg+'_active'] > dd2_size_2[cat_pos+'_active']) & (dd2_size_2[cat_neg+'_passive'] <= dd2_size_2[cat_pos+'_passive']), 'attacker', dd2_size_2['profile'] )
        dd2_size_2['profile'] = np.where((dd2_size_2[cat_neg+'_active'] < dd2_size_2[cat_pos+'_active']) & (dd2_size_2[cat_neg+'_passive'] >=  dd2_size_2[cat_pos+'_passive']), 'supporter', dd2_size_2['profile'] )

        dd2_size_2['profile'] = np.where((dd2_size_2[cat_neg+'_active'] == dd2_size_2[cat_pos+'_active']) & (dd2_size_2[cat_neg+'_passive'] > dd2_size_2[cat_pos+'_passive']), 'negative undecided', dd2_size_2['profile'] )
        dd2_size_2['profile'] = np.where((dd2_size_2[cat_neg+'_active'] == dd2_size_2[cat_pos+'_active']) & (dd2_size_2[cat_neg+'_passive'] < dd2_size_2[cat_pos+'_passive']), 'positive undecided', dd2_size_2['profile'] )

        dd2_size_2_pr = pd.DataFrame( dd2_size_2['profile'].value_counts(normalize = True).round(3)*100)
        dd2_size_2_pr = dd2_size_2_pr.reset_index()
        dd2_size_2_pr.columns = ['profile', 'percentage']
        plt2 = {'angry man':'darkred', 'positive soul':'darkgreen', 'attacker':'darkred', 'supporter':'darkgreen',
        'negative undecided':'red', 'positive undecided':'green', 'other':'grey'}
        if dd2_size_2_pr.shape[0] != len(plt2.keys()):
            miss_pr =  set(plt2.keys()).difference( set(dd2_size_2_pr.profile.values))
            for p in miss_pr:
                dd2_size_2_pr.loc[len(dd2_size_2_pr)] = [p, 0]
        #st.write(dd2_size_2_pr)
        dd2_size_2_pr = dd2_size_2_pr.sort_values(by = 'profile')

        #st.write(dd2_size_2[dd2_size_2['profile'] == 'other'])
        fig_pr = sns.catplot(kind = 'bar', data = dd2_size_2_pr, x = 'percentage', y = 'profile',
                aspect = 1.65, palette = plt2)
        titl_fig = selected_rhet_dim.replace('_label', '')
        fig_pr.set(title = f'{titl_fig.capitalize()} profiles in {ds}', xlim = (0,100), xticks = np.arange(0, 100, 15))
        st.pyplot(fig_pr)


        dd2_size_2[[cat_pos+'_active', cat_pos+'_passive', cat_neg+'_active', cat_neg+'_passive']] = dd2_size_2[[cat_pos+'_active', cat_pos+'_passive', cat_neg+'_active', cat_neg+'_passive']].round(-1)
        #cat_pos+'_active', cat_pos+'_passive'
        dd2_size_2_melt_ac = dd2_size_2[['entity', cat_pos+'_active', cat_neg+'_active']].melt('entity')
        dd2_size_2_melt_ac['role'] = 'active'
        dd2_size_2_melt_ac['category'] = dd2_size_2_melt_ac['category'].str.replace('_active', '')

        dd2_size_2_melt_ps = dd2_size_2[['entity', cat_pos+'_passive', cat_neg+'_passive']].melt('entity')
        dd2_size_2_melt_ps['role'] = 'passive'
        dd2_size_2_melt_ps['category'] = dd2_size_2_melt_ps['category'].str.replace('_passive', '')

        dd2_size_2_melt = pd.concat([dd2_size_2_melt_ps, dd2_size_2_melt_ac], axis=0, ignore_index = True)
        dd2_size_2_melt_dd = pd.DataFrame(dd2_size_2_melt.groupby(['role', 'category'])['value'].value_counts(normalize = True).round(3)*100)
        dd2_size_2_melt_dd.columns = ['percentage']
        dd2_size_2_melt_dd = dd2_size_2_melt_dd.reset_index()
        for r in dd2_size_2_melt_dd.role.unique():
            for c in dd2_size_2_melt_dd.category.unique():
                for v in np.arange(0, 101, 10):
                    if not v in dd2_size_2_melt_dd[ (dd2_size_2_melt_dd.role == r) & (dd2_size_2_melt_dd.category == c) ]['value'].unique():
                        dd2_size_2_melt_dd.loc[len(dd2_size_2_melt_dd)] = [r, c, v, 0]
        dd2_size_2_melt_dd = dd2_size_2_melt_dd.sort_values(by = ['role', 'category', 'value'])
        dd2_size_2_melt_dd['value'] = dd2_size_2_melt_dd['value'].astype('int').astype('str')


        #st.write(dd2_size_2)
        dd2_size_2_melt['category'] = np.where(dd2_size_2_melt['value'] == 50, cat_pos + " & " + cat_neg, dd2_size_2_melt['category'])
        dd2_size_2_melt = dd2_size_2_melt.sort_values(by = ['entity', 'role', 'value'])
        dd2_size_2_melt2 = dd2_size_2_melt.drop_duplicates(subset = ['entity', 'role'], keep = 'last')
        dd2_size_2_melt2_grp = pd.DataFrame(dd2_size_2_melt2.groupby([ 'role' ]).category.value_counts(normalize = True).round(3)*100)
        dd2_size_2_melt2_grp.columns = ['percentage']
        dd2_size_2_melt2_grp = dd2_size_2_melt2_grp.reset_index()
        #st.write(dd2_size_2_melt2_grp)
        #st.write(dd2_size_2_melt2)
        colors[cat_pos + " & " + cat_neg] = '#CB7200'

        st.write("##### Roles Overview")
        sns.set(font_scale=1.6, style='whitegrid')
        fig_ac_pas = sns.catplot(kind = 'bar', data = dd2_size_2_melt2_grp, y = 'category', x = 'percentage',
                    col = 'role', hue = 'category', palette = colors,
                    dodge=False, aspect = 1.3, height = 6, legend = False)

        plt.tight_layout(pad=2)
        if max(dd2_size_2_melt2_grp.percentage) == 100:
            mm = 100
        else:
            mm = max(dd2_size_2_melt2_grp.percentage)+11

        fig_ac_pas.set(xticks = np.arange(0, mm, 10))
        st.pyplot(fig_ac_pas)

        #st.stop()
        #add_spacelines()
        st.write("*********************************************************")


        st.write("##### Single Case Analysis")
        dd2_size_2[[cat_pos+'_active', cat_pos+'_passive', cat_neg+'_active', cat_neg+'_passive']] = dd2_size_2[[cat_pos+'_active', cat_pos+'_passive', cat_neg+'_active', cat_neg+'_passive']].applymap(lambda x: 1 if x == 0 else x)
        #with heroes_tab2:
        dd2_size_2 = dd2_size_2.drop(columns = ['active', 'passive'], axis = 1)
        dd2_size_2.sort_values(by = [cat_neg+'_active', cat_pos+'_active', cat_neg+'_passive', cat_pos+'_passive'], ascending = False)
        vals = list(set(dd2_size_2.entity.values))
        select_box_prof = st.multiselect("Select an entity", vals, vals[:3])
        #st.write(dd2_size_2)
        add_spacelines(2)

        dd2_size_2_s = dd2_size_2[dd2_size_2.entity.isin(select_box_prof)]
        dd2_size_2_s = dd2_size_2_s.iloc[:, :-1].melt('entity', var_name = 'category', value_name = 'percentage')
        dd2_size_2_s['role'] = np.where(dd2_size_2_s.category.isin([cat_neg+'_active', cat_pos+'_active']), 'active', 'passive')
        dd2_size_2_s['category'] = dd2_size_2_s['category'].str.replace('_active', '').str.replace('_passive', '')

        plt3 = {cat_neg+"_active":'darkred', cat_pos+"_active":'darkgreen', cat_pos+"_passive":'green', cat_neg+"_passive":'red'}
        sns.set(font_scale=1.35, style='whitegrid')

        #st.write(dd2_size_2_s)

        fig_pr = sns.catplot(kind = 'bar', data = dd2_size_2_s, y = 'category', x = 'percentage',
                col = 'role', hue = 'category', row = 'entity', alpha = 0.9,
                palette = plt1, aspect = 1.3, dodge=False)
        for ax in fig_pr.axes.flatten():
            ax.tick_params(labelbottom=True, bottom=True)
        #labelbottom
        plt.tight_layout(pad=2.2)
        sns.move_legend(fig_pr, loc = 'upper right', bbox_to_anchor = (0.63, 1.04), ncol = 3)
        st.pyplot(fig_pr)
        add_spacelines(2)
        dd2_size_2_s2 = dd2_size_2[dd2_size_2.entity.isin(select_box_prof)]
        dd2_size_2_s2[[cat_neg+'_active', cat_pos+'_active', cat_neg+'_passive', cat_pos+'_passive']] = dd2_size_2_s2[[cat_neg+'_active', cat_pos+'_active', cat_neg+'_passive', cat_pos+'_passive']].applymap(lambda x: 0 if x == 1 else x)
        #dd2_size_2_s.percentage = np.where(dd2_size_2_s.percentage == 1, 0, dd2_size_2_s.percentage)
        st.write(dd2_size_2_s2.set_index('entity'))
        #st.write(dd2_size_2_s.sort_values(by = 'entity').set_index('entity'))
        #st.stop()
        add_spacelines(2)

        with st.expander('Profile names'):
            st.write('**Angry man**')
            st.write("""If negativity of a certain entity dominates her both active and passsive roles.
            That is, the entity is both negative towards others (i.e., uses ethotic attacks/negative emotions) and others are negative towards this entity (i.e., others also use ethotic attacks/negative emotions when mentioning this entity).""")

            st.write('**Attacker**')
            st.write("""If negativity of a certain entity dominates her both active role; a passsive role is either positive or ambivalent (both positive and negative).
            That is, the entity is negative towards others (i.e., uses ethotic attacks/negative emotions) but others are positive or both positive and negative towards this entity (e.g., someone attacks her and someone supports her).""")

            st.write('**Negative undecided**')
            st.write("""If the active role of a certain entity is equally negative and positive; a passsive role is negative.
            That is, the entity is negative towards some people and positive towards other but others are negative towards this entity (e.g., others attack her).""")

            st.write('**Positive soul**')
            st.write("""If positivity of a certain entity dominates her both active and passsive roles.
            That is, the entity is both positive towards others (i.e., uses ethotic supports/positive emotions) and others are positive towards this entity (i.e., others also use ethotic supports/positive emotions when mentioning this entity).""")

            st.write('**Supporter**')
            st.write("""If positivity of a certain entity dominates her both active role; a passsive role is either positive or ambivalent (both positive and negative).
            That is, the entity is positive towards others (i.e., uses ethotic supports/positive emotions) but others are positive or both positive and negative towards this entity (e.g., someone attacks her and someone supports her).""")

            st.write('**Positive undecided**')
            st.write("""If the active role of a certain entity is equally negative and positive; a passsive role is positive.
            That is, the entity is negative towards some people and positive towards other but others are positive towards this entity (e.g., others support her).""")

            st.write('**Other**')
            st.write("""If the profile of a certain entity could not be categorised to either of the above categories, it is classified as Other.""")


def TargetHeroScores_compare(data_list, singl_an = True):
    st.write("### (Anti)-heroes")
    add_spacelines(1)
    contents_radio_heroes = st.radio("Category of the target of ethotic statements", ("both", "direct ethos", "3rd party ethos"))

    up_data_dict = {}
    up_data_dicth = {}
    up_data_dictah = {}
    target_shared = {}
    up_data_dict_hist = {}

    n = 0
    for data in data_list:
        df = data.copy()
        ds = df['corpus'].iloc[0]
        if not 'attack' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        df["Target"] = df["Target"].astype('str')
        df = df[ (df.Target != 'nan') & (df.Target != '') & (df.ethos_label != 'neutral') ]
        df["Target"] = df["Target"].str.replace('Government', 'government')
        target_shared[n] = set(df["Target"].unique())

        if contents_radio_heroes == "direct ethos":
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if "@" in t]
            df = df[df.Target.isin(targets_limit)]
            if len(targets_limit) < 2:
                st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
                st.stop()
        elif contents_radio_heroes == "3rd party ethos":
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if not "@" in t]
            df = df[df.Target.isin(targets_limit)]
            if len(targets_limit) < 2:
                st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
                st.stop()

        dd2_size = df.groupby(['Target'], as_index=False).size()
        dd2_size = dd2_size[dd2_size['size'] > 1]
        adj_target = dd2_size['Target'].unique()

        dd = pd.DataFrame(df.groupby(['Target'])['ethos_label'].value_counts(normalize=True))
        dd.columns = ['normalized_value']
        dd = dd.reset_index()
        dd = dd[dd.Target.isin(adj_target)]
        dd = dd[dd.ethos_label != 'neutral']
        dd_hero = dd[dd.ethos_label == 'support']
        dd_antihero = dd[dd.ethos_label == 'attack']

        dd2 = pd.DataFrame({'Target': dd.Target.unique()})
        dd2_hist = dd2.copy()
        dd2anti_scores = []
        dd2hero_scores = []
        dd2['score'] = np.nan
        for t in dd.Target.unique():
            try:
                h = dd_hero[dd_hero.Target == t]['normalized_value'].iloc[0]
            except:
                h = 0
            try:
                ah = dd_antihero[dd_antihero.Target == t]['normalized_value'].iloc[0]
            except:
                ah = 0
            dd2hero_scores.append(h)
            dd2anti_scores.append(ah)
            i = dd2[dd2.Target == t].index
            dd2.loc[i, 'score'] = h - ah

        dd2 = dd2[dd2.score != 0]
        dd2['ethos_label'] = np.where(dd2.score < 0, 'anti-heroes', 'neutral')
        dd2['ethos_label'] = np.where(dd2.score > 0, 'heroes', dd2['ethos_label'])
        dd2 = dd2.sort_values(by = ['ethos_label', 'Target'])
        dd2['score'] = dd2['score'] * 100
        dd2['score'] = dd2['score'].round()
        dd2['corpus'] = ds
        up_data_dict_hist[n] = dd2
        #st.write(dd2)
        #st.stop()
        #dd2['score'] = dd2[dd2['ethos_label'] != 'neutral' ]
        dd2_dist = pd.DataFrame(dd2['ethos_label'].value_counts(normalize=True).round(3)*100).reset_index()
        dd2_dist.columns = ['heroes', 'percentage']
        dd2_dist['corpus'] = ds
        up_data_dict[n] = dd2_dist
        up_data_dicth[n] = dd2[dd2['ethos_label'] == 'heroes']['Target'].unique()
        up_data_dictah[n] = dd2[dd2['ethos_label'] == 'anti-heroes']['Target'].unique()
        n += 1

    df_dist_ethos_all = up_data_dict[0].copy()
    for k in range(int(len(up_data_dict.keys()))-1):
        k_sub = k+1
        df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

    sns.set(font_scale=1.35, style='whitegrid')
    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos_all, height=5, aspect=1.2,
                    x = 'heroes', y = 'percentage', hue = 'heroes', dodge=False, legend = False,
                    palette = {'anti-heroes':'#BB0000', 'heroes':'#026F00'},
                    col = 'corpus')
    f_dist_ethos.set(ylim=(0, 110), xlabel = '')
    for ax in f_dist_ethos.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f')+"%", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    add_spacelines(1)


    df_dist_hist_all = up_data_dict_hist[0].copy()
    for k in range(int(len(up_data_dict_hist.keys()))-1):
        k_sub = k+1
        df_dist_hist_all = pd.concat([df_dist_hist_all, up_data_dict_hist[k_sub]], axis=0, ignore_index=True)
    sns.set(font_scale=1.35, style='whitegrid')
    #st.write(df_dist_hist_all)
    #st.stop()
    #f_dist_ethoshist = sns.catplot(kind='count', data = df_dist_hist_all, height=5, aspect=1.3,
    #                x = 'score', hue = 'ethos_label', dodge=False,
    #                palette = {'anti-heroes':'#BB0000', 'heroes':'#026F00'},
    #                col = 'corpus')
    #for axes in f_dist_ethoshist.axes.flat:
    #    _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90)

    df_dist_hist_all = df_dist_hist_all.rename(columns = {'ethos_label':'label'})
    sns.set(font_scale=1, style='whitegrid')
    f_dist_ethoshist = sns.catplot(kind='strip', data = df_dist_hist_all, height=4, aspect=1.7,
                    y = 'score', hue = 'label', dodge=False, s=14, alpha=0.85,
                    palette = {'anti-heroes':'#BB0000', 'heroes':'#026F00'},
                    x = 'corpus')
    f_dist_ethoshist.set(xlabel = '', title = 'Distribution of (anti)-hero scores')

    heroes_tab1, heroes_tab2, heroes_tab3 = st.tabs(['(Anti)-heroes Plots', '(Anti)-heroes Tables', '(Anti)-heroes Single Target Analysis'])
    with heroes_tab1:
        add_spacelines(1)
        st.pyplot(f_dist_ethos)
        add_spacelines(1)
        st.pyplot(f_dist_ethoshist)

    with heroes_tab3:
        add_spacelines(2)
        if singl_an:
            st.write("### Single Target Analysis")
            add_spacelines(1)

            target_shared_list = target_shared[0]
            for n in range(int(len(data_list))-1):
                target_shared_list = set(target_shared_list).intersection(target_shared[n+1])
            selected_target = st.selectbox("Choose a target entity you would like to analyse", set(target_shared_list))

            cols_columns = st.columns(len(data_list), gap='large')
            for n, c in enumerate(cols_columns):
                with c:
                    df = data_list[n].copy()
                    ds = df['corpus'].iloc[0]
                    #st.dataframe(df)
                    if not 'neutral' in df['ethos_label'].unique():
                        df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
                    if not 'pathos_label' in df['pathos_label'].unique():
                        df['pathos_label'] = df['pathos_label'].map(valence_mapping)

                    # all df targets
                    df_target_all = pd.DataFrame(df[df.ethos_label != 'neutral']['ethos_label'].value_counts(normalize = True).round(2)*100)
                    df_target_all.columns = ['percentage']
                    df_target_all.reset_index(inplace=True)
                    df_target_all.columns = ['label', 'percentage']
                    df_target_all = df_target_all.sort_values(by = 'label')
                    df_target_all_att = df_target_all[df_target_all.label == 'attack']['percentage'].iloc[0]
                    df_target_all_sup = df_target_all[df_target_all.label == 'support']['percentage'].iloc[0]

                    # chosen target df
                    df_target = pd.DataFrame(df[df.Target == str(selected_target)]['ethos_label'].value_counts(normalize = True).round(2)*100)
                    df_target.columns = ['percentage']
                    df_target.reset_index(inplace=True)
                    df_target.columns = ['label', 'percentage']

                    if len(df_target) == 1:
                      if not ("attack" in df_target.label.unique()):
                          df_target.loc[len(df_target)] = ["attack", 0]
                      elif not ("support" in df_target.label.unique()):
                          df_target.loc[len(df_target)] = ["support", 0]
                    df_target = df_target.sort_values(by = 'label')
                    df_target_att = df_target[df_target.label == 'attack']['percentage'].iloc[0]
                    df_target_sup = df_target[df_target.label == 'support']['percentage'].iloc[0]

                    add_spacelines(1)
                    df_target.columns = ['ethos', 'percentage']
                    df_dist_ethos = df_target.sort_values(by = 'ethos')
                    df_dist_ethos['corpus'] = ds

                    sns.set(font_scale=1.35, style='whitegrid')
                    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos, height=4, aspect=1.4, legend = False,
                                    x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False, col = 'corpus',
                                    palette = {'attack':'#BB0000', 'neutral':'#949494', 'support':'#026F00'})
                    vals_senti = df_dist_ethos['percentage'].values.round(1)
                    plt.title(f"Ethos towards **{str(selected_target)}** in {df.corpus.iloc[0]} \n")
                    plt.xlabel('')
                    plt.ylim(0, 105)
                    plt.yticks(np.arange(0, 105, 20))
                    for index_senti, v in enumerate(vals_senti):
                        plt.text(x=index_senti , y = v+1 , s=f"{v}%" , fontdict=dict(ha='center'))
                    st.pyplot(f_dist_ethos)


                    st.write('**********************************************************************************')
                    #add_spacelines(1)
                    cols = ['sentence', 'ethos_label', 'source', 'Target', 'pathos_label'] #, 'date', 'conversation_id'
                    if len(df[df.Target == str(selected_target)]) == 1:
                        st.write(f"{len(df[df.Target == str(selected_target)])} case of ethotic statements towards **{selected_target}**  in {df['corpus'].iloc[0]} corpus")
                    else:
                        st.write(f"{len(df[df.Target == str(selected_target)])} cases of ethotic statements towards **{selected_target}**  in {df['corpus'].iloc[0]} corpus")
                    if not "neutral" in df['pathos_label'].unique():
                        df['pathos_label'] = df['pathos_label'].map(valence_mapping)
                    st.dataframe(df[df.Target == str(selected_target)][cols].set_index('source').rename(columns={'ethos_label':'ethos'}), width = None)
                    add_spacelines(1)


    with heroes_tab2:
        cops_names = df_dist_hist_all.corpus.unique()
        cols_columns = st.columns(len(cops_names))
        for n, c in enumerate(cols_columns):
            with c:
                df_dist_hist_all_0 = df_dist_hist_all[df_dist_hist_all.corpus == cops_names[n]]
                add_spacelines(1)

                def colorred(s):
                    if s < 0:
                        return 'background-color: red'
                    elif s > 0:
                        return 'background-color: green'
                    else:
                        return 'background-color: white'

                st.write(cops_names[n])
                df_dist_hist_all_0 = df_dist_hist_all_0.sort_values(by = 'score')
                df_dist_hist_all_0 = df_dist_hist_all_0.reset_index(drop=True)
                df_dist_hist_all_0 = df_dist_hist_all_0.set_index('Target').reset_index()
                df_dist_hist_all_0.index += 1
                #df_dist_hist_all_0['score'] = df_dist_hist_all_0['score'].style.applymap(lambda x: "background-color: red" if x > 0 else "background-color: white")
                st.write(df_dist_hist_all_0.style.applymap(colorred, subset=['score']))
                df_dist_hist_all_0.Target = df_dist_hist_all_0.Target.apply(lambda x: "_".join(x.split()))

                f_att0, _ = make_word_cloud(" ".join(df_dist_hist_all_0[df_dist_hist_all_0.label == 'anti-heroes'].Target.values), 800, 500, '#1E1E1E', 'Reds')
                f_sup0, _ = make_word_cloud(" ".join(df_dist_hist_all_0[df_dist_hist_all_0.label == 'heroes'].Target.values), 800, 500, '#1E1E1E', 'Greens')

                add_spacelines(1)
                st.pyplot(f_att0)
                add_spacelines(2)
                st.pyplot(f_sup0)



def TargetHeroScores(data_list):
  st.write("### (Anti)-heroes")
  tabheroes1, tabheroes2 = st.tabs(['(Anti)-hero Plots', '(Anti)-hero Tables'])

  add_spacelines(1)
  with tabheroes1:
    contents_radio_heroes = st.radio("Category of the target of ethotic statements", ("both", "direct ethos", "3rd party ethos"))

    up_data_dict = {}
    up_data_dicth = {}
    up_data_dictah = {}
    dd_target_table = pd.DataFrame(columns = ['Target', 'score', 'ethos_label'])
    n = 0
    corp_name = '0'
    for data in data_list:
        df = data.copy()
        ds = df['corpus'].iloc[0]
        corp_name = ds
        if not 'attack' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        df["Target"] = df["Target"].astype('str')
        df = df[ (df.Target != 'nan') & (df.Target != '') & (df.ethos_label != 'neutral') ]
        df["Target"] = df["Target"].str.replace('Government', 'government')

        if contents_radio_heroes == "direct ethos":
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if "@" in t]
            df = df[df.Target.isin(targets_limit)]
            if len(targets_limit) < 2:
                st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
                st.stop()
        elif contents_radio_heroes == "3rd party ethos":
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if not "@" in t]
            df = df[df.Target.isin(targets_limit)]
            if len(targets_limit) < 2:
                st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
                st.stop()

        dd2_size = df.groupby(['Target'], as_index=False).size()
        dd2_size = dd2_size[dd2_size['size'] > 1]
        adj_target = dd2_size['Target'].unique()

        dd = pd.DataFrame(df.groupby(['Target'])['ethos_label'].value_counts(normalize=True))
        dd.columns = ['normalized_value']
        dd = dd.reset_index()
        dd = dd[dd.Target.isin(adj_target)]
        dd = dd[dd.ethos_label != 'neutral']
        dd_hero = dd[dd.ethos_label == 'support']
        dd_antihero = dd[dd.ethos_label == 'attack']

        dd2 = pd.DataFrame({'Target': dd.Target.unique()})
        dd2_hist = dd2.copy()
        dd2anti_scores = []
        dd2hero_scores = []
        dd2['score'] = np.nan
        for t in dd.Target.unique():
            try:
                h = dd_hero[dd_hero.Target == t]['normalized_value'].iloc[0]
            except:
                h = 0
            try:
                ah = dd_antihero[dd_antihero.Target == t]['normalized_value'].iloc[0]
            except:
                ah = 0
            dd2hero_scores.append(h)
            dd2anti_scores.append(ah)
            i = dd2[dd2.Target == t].index
            dd2.loc[i, 'score'] = h - ah

        dd2 = dd2[dd2.score != 0]
        dd2['ethos_label'] = np.where(dd2.score < 0, 'anti-heroes', 'neutral')
        dd2['ethos_label'] = np.where(dd2.score > 0, 'heroes', dd2['ethos_label'])
        dd2 = dd2.sort_values(by = ['ethos_label', 'Target'])
        dd2['score'] = dd2['score'] * 100
        dd_target_table = pd.concat([dd_target_table, dd2], axis = 0, ignore_index = True)
        #dd2['score'] = dd2[dd2['ethos_label'] != 'neutral' ]
        dd2_dist = pd.DataFrame(dd2['ethos_label'].value_counts(normalize=True).round(3)*100).reset_index()
        dd2_dist.columns = ['heroes', 'percentage']
        dd2_dist['corpus'] = ds
        up_data_dict[n] = dd2_dist
        up_data_dicth[n] = dd2[dd2['ethos_label'] == 'heroes']['Target'].unique()
        up_data_dictah[n] = dd2[dd2['ethos_label'] == 'anti-heroes']['Target'].unique()
        n += 1

    df_dist_ethos_all = up_data_dict[0].copy()
    for k in range(int(len(up_data_dict.keys()))-1):
        k_sub = k+1
        df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

    sns.set(font_scale=1.25, style='whitegrid')
    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos_all, height=5, aspect=1.2,
                    x = 'heroes', y = 'percentage', hue = 'heroes', dodge=False, legend = False,
                    palette = {'anti-heroes':'#BB0000', 'heroes':'#026F00'},
                    col = 'corpus')
    f_dist_ethos.set(ylim=(0, 110), xlabel = '')
    for ax in f_dist_ethos.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f')+"%", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    add_spacelines(1)

    dd_target_table.score = dd_target_table.score.round()
    #st.write(dd_target_table)
    #st.write(corp_name)
    dd_target_table['corpus'] = corp_name
    dd_target_table = dd_target_table.sort_values(by = 'score')
    dd_target_table = dd_target_table.reset_index(drop = True)
    dd_target_table.index += 1
    dd_target_table.columns = ['Target', 'score', 'label', 'corpus']

    sns.set(font_scale=1.3, style='whitegrid')
    f_dist_ethoshist = sns.catplot(kind='strip', data = dd_target_table, height=5, aspect=1.5,
                    y = 'score', hue = 'label', dodge=False, s=33, alpha=0.85,
                    palette = {'anti-heroes':'#BB0000', 'heroes':'#026F00'},
                    x = 'corpus')
    f_dist_ethoshist.set(xlabel = 'corpus', title = 'Distribution of (anti)-hero scores')

    with st.container():
        st.pyplot(f_dist_ethos)
        add_spacelines(2)
        st.pyplot(f_dist_ethoshist)


    add_spacelines(2)
    st.write("### Single Target Analysis")
    add_spacelines(1)
    selected_target = st.selectbox("Choose a target entity you would like to analyse", set(adj_target))

    # all df targets
    df_target_all = pd.DataFrame(df[df.ethos_label != 'neutral']['ethos_label'].value_counts(normalize = True).round(2)*100)
    df_target_all.columns = ['percentage']
    df_target_all.reset_index(inplace=True)
    df_target_all.columns = ['label', 'percentage']
    df_target_all = df_target_all.sort_values(by = 'label')

    df_target_all_att = df_target_all[df_target_all.label == 'attack']['percentage'].iloc[0]
    df_target_all_sup = df_target_all[df_target_all.label == 'support']['percentage'].iloc[0]

    # chosen target df
    df_target = pd.DataFrame(df[df.Target == str(selected_target)]['ethos_label'].value_counts(normalize = True).round(2)*100)
    df_target.columns = ['percentage']
    df_target.reset_index(inplace=True)
    df_target.columns = ['label', 'percentage']

    if len(df_target) == 1:
      if not ("attack" in df_target.label.unique()):
          df_target.loc[len(df_target)] = ["attack", 0]
      elif not ("support" in df_target.label.unique()):
          df_target.loc[len(df_target)] = ["support", 0]

    df_target = df_target.sort_values(by = 'label')
    df_target_att = df_target[df_target.label == 'attack']['percentage'].iloc[0]
    df_target_sup = df_target[df_target.label == 'support']['percentage'].iloc[0]


    with st.container():
        st.info(f'Selected entity: ** {str(selected_target)} **')
        add_spacelines(1)
        col2, col1 = st.columns([3, 2])
        with col1:
            st.subheader("Positivity score")
            col1.metric(str(selected_target), str(round(df_target_sup, 1))+ str('%') + f" ({len(df[ (df.Target == str(selected_target)) & (df['ethos_label'] == 'support') ])})" ,
            str(round((df_target_sup - df_target_all_sup),  1))+ str(' p.p.'),
            help = f"Percentage (number in brackets) of texts that support ** {str(selected_target)} **") # round(((df_target_sup / df_target_all_sup) * 100) - 100, 1)

        with col2:
            st.subheader("Negativity score")
            col2.metric(str(selected_target), str(round(df_target_att, 1))+ str('%') + f" ({len(df[ (df.Target == str(selected_target)) & (df['ethos_label'] == 'attack') ])})",
            str(round((df_target_att - df_target_all_att),  1))+ str(' p.p.'), delta_color="inverse",
            help = f"Percentage (number in brackets) of texts that attack ** {str(selected_target)} **") # ((df_target_att / df_target_all_att) * 100) - 100, 1)

        add_spacelines(2)

        #if not ("neutral" in df_target.label.unique()):
            #df_target.loc[len(df_target)] = ["neutral", 0]
        df_target.columns = ['ethos', 'percentage']
        df_dist_ethos = df_target.sort_values(by = 'ethos')

        sns.set(font_scale=1.25, style='whitegrid')
        f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos, height=4, aspect=1.4,
                        x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False,  legend = False,
                        palette = {'attack':'#BB0000', 'neutral':'#949494', 'support':'#026F00'})
        vals_senti = df_dist_ethos['percentage'].values.round(1)
        plt.title(f"Ethos towards **{str(selected_target)}** in {df.corpus.iloc[0]} \n")
        plt.xlabel('')
        plt.ylim(0, 105)
        plt.yticks(np.arange(0, 105, 20))
        for index_senti, v in enumerate(vals_senti):
            plt.text(x=index_senti , y = v+1 , s=f"{v}%" , fontdict=dict(ha='center'))

        plot1, plot2, plot3 = st.columns([1, 6, 1], gap='small')
        with plot2:
            st.pyplot(f_dist_ethos)

        st.write('**********************************************************************************')
        #add_spacelines(1)
        cols = [
            'sentence', 'ethos_label', 'source', 'Target', 'pathos_label'] #, 'date', 'conversation_id'
        #st.write('#### Cases of ethotic statements towards **', selected_target, ' **')
        if len(df[df.Target == str(selected_target)]) == 1:
            st.write(f"{len(df[df.Target == str(selected_target)])} case of ethotic statements towards ** {selected_target} **  in {df['corpus'].iloc[0]} corpus")
        else:
            st.write(f"{len(df[df.Target == str(selected_target)])} cases of ethotic statements towards ** {selected_target} **  in {df['corpus'].iloc[0]} corpus")

        if not "neutral" in df['pathos_label'].unique():
            df['pathos_label'] = df['pathos_label'].map(valence_mapping)
        st.dataframe(df[df.Target == str(selected_target)][cols].set_index('source').rename(columns={'ethos_label':'ethos'}), width = None)


  with tabheroes2:
      add_spacelines(2)
      st.write("##### Anti-(heroes)")

      st.write(dd_target_table.set_index('Target'))

      dd_target_table.Target = dd_target_table.Target.apply(lambda x: "_".join(x.split()))

      f_att0, _ = make_word_cloud(" ".join(dd_target_table[dd_target_table.label == 'anti-heroes'].Target.values), 800, 500, '#1E1E1E', 'Reds')
      f_sup0, _ = make_word_cloud(" ".join(dd_target_table[dd_target_table.label == 'heroes'].Target.values), 800, 500, '#1E1E1E', 'Greens')

      add_spacelines(1)
      st.pyplot(f_att0)
      add_spacelines(2)
      st.pyplot(f_sup0)

  add_spacelines(1)





def distribution_plot(data):
    df = data.copy()

    if not 'neutral' in df['ethos_label'].unique():
        df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
    if not 'pathos_label' in df['pathos_label'].unique():
        df['pathos_label'] = df['pathos_label'].map(valence_mapping)

    st.write("### Ethos distribution")
    add_spacelines(2)
    contents_radio_targs = st.radio("Category of the target of ethotic statements", ("both", "direct ethos", "3rd party ethos"))

    df["Target"] = df["Target"].astype('str')
    df["Target"] = df["Target"].str.replace('Government', 'government')

    if contents_radio_targs == "direct ethos":
        targets_limit = df['Target'].dropna().unique()
        targets_limit = [t for t in targets_limit if "@" in t]
        targets_limit.append('nan')
        df = df[df.Target.isin(targets_limit)]
        if len(targets_limit) < 1:
            st.error(f'No cases of **{contents_radio_targs}** found in the chosen corpora.')
            st.stop()
    elif contents_radio_targs == "3rd party ethos":
        targets_limit = df['Target'].dropna().unique()
        targets_limit = [t for t in targets_limit if not "@" in t]
        targets_limit.append('nan')
        df = df[df.Target.isin(targets_limit)]
        if len(targets_limit) < 1:
            st.error(f'No cases of **{contents_radio_targs}** found in the chosen corpora.')
            st.stop()

    df_dist_ethos = pd.DataFrame(df['ethos_label'].value_counts(normalize = True).round(2)*100)
    df_dist_ethos.columns = ['percentage']
    df_dist_ethos.reset_index(inplace=True)
    df_dist_ethos.columns = ['ethos', 'percentage']
    df_dist_ethos = df_dist_ethos.sort_values(by = 'ethos')

    per = []
    eth = []
    eth.append('no ethos')
    per.append(float(df_dist_ethos[df_dist_ethos.ethos == 'neutral']['percentage'].iloc[0]))
    eth.append('ethos')
    per.append(100 - float(df_dist_ethos[df_dist_ethos.ethos == 'neutral']['percentage'].iloc[0]))
    df_dist_ethos_all0 = pd.DataFrame({'ethos':eth, 'percentage':per})

    sns.set(font_scale=1.1, style='whitegrid')
    f_dist_ethos0 = sns.catplot(kind='bar', data = df_dist_ethos_all0, height=4.5, aspect=1.4,
                    x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False,
                    palette = {'ethos':'#EA9200', 'no ethos':'#022D96'})
    f_dist_ethos0.set(ylim=(0, 110))
    plt.xlabel("")
    plt.title(f"Ethos distribution in **{contents_radio}** \n")
    vals_senti0 = df_dist_ethos_all0['percentage'].values.round(1)
    for index_senti2, v in enumerate(vals_senti0):
        plt.text(x=index_senti2, y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=13, ha='center'))

    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos, height=4.5, aspect=1.4,
                    x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False,
                    palette = {'attack':'#BB0000', 'neutral':'#949494', 'support':'#026F00'})
    vals_senti = df_dist_ethos['percentage'].values.round(1)
    f_dist_ethos.set(ylim=(0, 110))
    plt.title(f"Ethos distribution in **{contents_radio}** \n")
    for index_senti, v in enumerate(vals_senti):
        plt.text(x=index_senti , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=13, ha='center'))


    df_dist_ethos2 = pd.DataFrame(df[df['ethos_label'] != 'neutral']['ethos_label'].value_counts(normalize = True).round(2)*100)

    df_dist_ethos2.columns = ['percentage']
    df_dist_ethos2.reset_index(inplace=True)
    df_dist_ethos2.columns = ['ethos', 'percentage']
    df_dist_ethos2 = df_dist_ethos2.sort_values(by = 'ethos')

    f_dist_ethos2 = sns.catplot(kind='bar', data = df_dist_ethos2, height=4.5, aspect=1.4,
                    x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False,
                    palette = {'attack':'#BB0000', 'support':'#026F00'})
    f_dist_ethos2.set(ylim=(0, 110))
    plt.title(f"Ethos distribution in **{contents_radio}** \n")
    vals_senti2 = df_dist_ethos2['percentage'].values.round(1)
    for index_senti2, v in enumerate(vals_senti2):
        plt.text(x=index_senti2, y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=13, ha='center'))

    plot1_dist_ethos, plot2_dist_ethos, plot3_dist_ethos = st.columns([1, 8, 1])
    with plot1_dist_ethos:
        st.write('')
    with plot2_dist_ethos:
        st.pyplot(f_dist_ethos0)
        add_spacelines(1)
        st.pyplot(f_dist_ethos)
        add_spacelines(1)
        st.pyplot(f_dist_ethos2)
    with plot3_dist_ethos:
        st.write('')
    add_spacelines(2)


    with st.expander("Pathos distribution"):
        add_spacelines(1)

    if contents_radio_targs == "direct ethos":
        targets_limit = df['Target'].dropna().unique()
        targets_limit = [t for t in targets_limit if "@" in t]
        targets_limit.append('nan')
        df = df[df.Target.isin(targets_limit)]
        if len(targets_limit) < 1:
            st.error(f'No cases of **{contents_radio_targs}** found in the chosen corpora.')
            st.stop()
    elif contents_radio_targs == "3rd party ethos":
        targets_limit = df['Target'].dropna().unique()
        targets_limit = [t for t in targets_limit if not "@" in t]
        targets_limit.append('nan')
        df = df[df.Target.isin(targets_limit)]

        if not 'neutral' in df['pathos_label'].unique():
            df['pathos_label'] = df['pathos_label'].map(valence_mapping)
        df_dist_ethos = pd.DataFrame(df['pathos_label'].value_counts(normalize = True).round(2)*100)
        df_dist_ethos.columns = ['percentage']
        df_dist_ethos.reset_index(inplace=True)
        df_dist_ethos.columns = ['pathos', 'percentage']
        df_dist_ethos = df_dist_ethos.sort_values(by = 'pathos')


        per = []
        eth = []
        eth.append('no pathos')
        per.append(float(df_dist_ethos[df_dist_ethos.pathos == 'neutral']['percentage'].iloc[0]))
        eth.append('pathos')
        per.append(100 - float(df_dist_ethos[df_dist_ethos.pathos == 'neutral']['percentage'].iloc[0]))
        df_dist_ethos_all0 = pd.DataFrame({'pathos':eth, 'percentage':per})

        f_dist_ethos0 = sns.catplot(kind='bar', data = df_dist_ethos_all0, height=4.5, aspect=1.4,
                        x = 'pathos', y = 'percentage', hue = 'pathos', dodge=False,
                        palette = {'pathos':'#EA9200', 'no pathos':'#022D96'})
        f_dist_ethos0.set(ylim=(0, 110))
        plt.xlabel("")
        plt.title(f"Pathos distribution in **{contents_radio}** \n")
        vals_senti0 = df_dist_ethos_all0['percentage'].values.round(1)
        for index_senti2, v in enumerate(vals_senti0):
            plt.text(x=index_senti2, y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=13, ha='center'))

        f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos, height=4.5, aspect=1.4,
                        x = 'pathos', y = 'percentage', hue = 'pathos', dodge=False,
                        palette = {'negative':'#BB0000', 'neutral':'#949494', 'positive':'#026F00'})
        vals_senti = df_dist_ethos['percentage'].values.round(1)
        f_dist_ethos.set(ylim=(0, 110))
        plt.title(f"Pathos distribution in **{contents_radio}** \n")
        for index_senti, v in enumerate(vals_senti):
            plt.text(x=index_senti , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=12, ha='center'))


        df_dist_ethos2 = pd.DataFrame(df[df['pathos_label'] != 'neutral']['pathos_label'].value_counts(normalize = True).round(2)*100)
        df_dist_ethos2.columns = ['percentage']
        df_dist_ethos2.reset_index(inplace=True)
        df_dist_ethos2.columns = ['pathos', 'percentage']
        df_dist_ethos2 = df_dist_ethos2.sort_values(by = 'pathos')

        sns.set(font_scale=1.1, style='whitegrid')
        f_dist_ethos2 = sns.catplot(kind='bar', data = df_dist_ethos2, height=4.5, aspect=1.4,
                        x = 'pathos', y = 'percentage', hue = 'pathos', dodge=False,
                        palette = {'negative':'#BB0000', 'positive':'#026F00'})
        vals_senti2 = df_dist_ethos2['percentage'].values.round(1)
        f_dist_ethos2.set(ylim=(0, 110))
        plt.title(f"Pathos distribution in **{contents_radio}** \n")
        for index_senti2, v in enumerate(vals_senti2):
            plt.text(x=index_senti2, y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=12, ha='center'))

        plot1_dist_ethos, plot2_dist_ethos, plot3_dist_ethos = st.columns([1, 8, 1])
        with plot1_dist_ethos:
            st.write('')
        with plot2_dist_ethos:
            st.pyplot(f_dist_ethos0)
            add_spacelines(1)
            st.pyplot(f_dist_ethos)
            add_spacelines(1)
            st.pyplot(f_dist_ethos2)
        with plot3_dist_ethos:
            st.write('')
        add_spacelines(1)



def distribution_plot_compare_logos(data_list, an_type):
    contents_radio_targs = st.radio("Category of the target of ethotic statements", ("both", "direct ethos", "3rd party ethos"))

    up_data_dict = {}
    up_data_dict2 = {}
    n = 0
    naming = 'category'
    for nn, data in enumerate(data_list):
        nn = int(round(nn / 2))
        df = data.copy()
        ds = df['corpus'].iloc[0]
        #st.dataframe(df)
        if df['kind'].iloc[0] == 'ethos':
            naming_cols = 'ethos_label'

            if not 'attack' in df['ethos_label'].unique():
                df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
            df["Target"] = df["Target"].astype('str')
            df["Target"] = df["Target"].str.replace('Government', 'government')

            if contents_radio_targs == "direct ethos":
                targets_limit = df['Target'].dropna().unique()
                targets_limit = [t for t in targets_limit if "@" in t]
                if 'Hansard' in ds:
                    targets_limit = list( df['Target'].dropna().unique() )
                targets_limit.append('nan')
                df = df[df.Target.isin(targets_limit)]
                if len(targets_limit) < 1:
                    st.error(f'No cases of **{contents_radio_targs}** found in the chosen corpora.')
                    st.stop()
            elif contents_radio_targs == "3rd party ethos":
                targets_limit = df['Target'].dropna().unique()
                targets_limit = [t for t in targets_limit if not "@" in t]
                if 'Hansard' in ds:
                    targets_limit = []
                targets_limit.append('nan')
                df = df[df.Target.isin(targets_limit)]
                if len(targets_limit) < 1:
                    st.error(f'No cases of **{contents_radio_targs}** found in the chosen corpora.')
                    st.stop()


        elif df['kind'].iloc[0] == 'logos':
            naming_cols = 'connection'
            connection_cats = ['Default Conflict', 'Default Inference']
            #df = df[ df[naming_cols].isin(connection_cats) ]

            if an_type == 'Relation-based':
                ids_link = set(df[df.duplicated(['argument_linked', 'id_connection'])].index)
                ids = set(df.index)
                ids_cln = ids.difference(ids_link)
                ids_cln = list(ids_cln)
                df = df.loc[ids_cln]

            if an_type == 'ADU-based' and naming_cols == 'connection':
                cc_raw = summary_corpora_list_raw[nn].copy()
                cc_raw['locution'] = cc_raw['locution'].apply(lambda x: ":".join( str(x).split(":")[1:] ))
                cc_raw['connection'] = 'Default Rephrase'

                df = pd.concat( [df, cc_raw], axis = 0, ignore_index = True )

        map_naming = {'attack':'Ethos Attack', 'neutral':'Ethos  Neutral', 'support':'Ethos Support',
                'Default Conflict': ' Logos Attack',
                'Default Rephrase' : ' Logos  Neutral',
                'Default Inference' : ' Logos Support'}
        df_dist_ethos = pd.DataFrame(df[naming_cols].value_counts(normalize = True).round(2)*100)
        df_dist_ethos.columns = ['percentage']

        df_dist_ethos.reset_index(inplace=True)
        #st.dataframe(df_dist_ethos)
        df_dist_ethos.columns = [naming, 'percentage']
        df_dist_ethos = df_dist_ethos.sort_values(by = naming)
        df_dist_ethos['corpus'] = ds
        df_dist_ethos[naming] = df_dist_ethos[naming].map(map_naming)
        up_data_dict[n] = df_dist_ethos

        df_dist_ethos2 = pd.DataFrame(df[ ~(df[naming_cols].isin(['neutral', 'Default Rephrase'])) ][naming_cols].value_counts(normalize = True).round(2)*100)
        df_dist_ethos2.columns = ['percentage']
        df_dist_ethos2.reset_index(inplace=True)
        df_dist_ethos2.columns = [naming, 'percentage']
        df_dist_ethos2[naming] = df_dist_ethos2[naming].map(map_naming)
        df_dist_ethos2 = df_dist_ethos2.sort_values(by = naming)
        df_dist_ethos2['corpus'] = ds
        up_data_dict2[n] = df_dist_ethos2

        n += 1

    df_dist_ethos_all = up_data_dict[0].copy()
    for k in range(int(len(up_data_dict.keys()))-1):
        k_sub = k+1
        df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

    sns.set(font_scale=1.25, style='whitegrid')

    naming_cats = set( ['Ethos Attack', 'Ethos  Neutral', 'Ethos Support',
                    ' Logos Attack', ' Logos  Neutral',' Logos Support'] )
    ds = df_dist_ethos_all.corpus.iloc[0]

    df_dist_ethos_all_cat = set( df_dist_ethos_all[naming].unique() )
    df_dist_ethos_all_cat_left = naming_cats.difference( df_dist_ethos_all_cat )
    if len(df_dist_ethos_all_cat_left) >= 1:
        for c in df_dist_ethos_all_cat_left:
            df_dist_ethos_all.loc[len(df_dist_ethos_all)] = [c, 0, ds]

    df_dist_ethos_all = df_dist_ethos_all.sort_values(by = naming)
    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos_all, height=4.5, aspect=1.4,
                    x = naming, y = 'percentage', hue = naming, dodge=False,
                    palette = {'Ethos Attack':'#BB0000', 'Ethos  Neutral':'#022D96', 'Ethos Support':'#026F00',
                            ' Logos Attack':'#BB0000', ' Logos  Neutral':'#022D96', ' Logos Support':'#026F00'},
                    col = 'corpus')
    f_dist_ethos.set(ylim=(0, 110), xlabel='')

    for ax in f_dist_ethos.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f')+"%", (p.get_x() + p.get_width() / 2., p.get_height()),
            ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    for axes in f_dist_ethos.axes.flat:
        _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=60)


    df_dist_ethos_all2 = up_data_dict2[0].copy()
    for k in range(int(len(up_data_dict2.keys()))-1):
        k_sub2 = k+1
        df_dist_ethos_all2 = pd.concat([df_dist_ethos_all2, up_data_dict2[k_sub2]], axis=0, ignore_index=True)

    #st.dataframe(df_dist_ethos_all2)
    naming_cats = set( ['Ethos Attack', 'Ethos Support', ' Logos Attack', ' Logos Support'] )
    df_dist_ethos_all_cat = set( df_dist_ethos_all2[naming].unique() )
    df_dist_ethos_all_cat_left = naming_cats.difference( df_dist_ethos_all_cat )
    if len(df_dist_ethos_all_cat_left) >= 1:
        for c in df_dist_ethos_all_cat_left:
            df_dist_ethos_all2.loc[len(df_dist_ethos_all2)] = [c, 0, ds]

    df_dist_ethos_all2 = df_dist_ethos_all2.sort_values(by = naming)
    f_dist_ethos2 = sns.catplot(kind='bar', data = df_dist_ethos_all2, height=4.5, aspect=1.4,
                    x = naming, y = 'percentage', hue = naming, dodge=False,
                    palette = {'Ethos Attack':'#BB0000', ' No Ethos':'#022D96', 'Ethos Support':'#026F00',
                            ' Logos Attack':'#BB0000', ' Logos  Rephrase':'#D7A000', ' Logos Support':'#026F00'},
                    col = 'corpus')

    f_dist_ethos2.set(ylim=(0, 110), xlabel='')
    for ax in f_dist_ethos2.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f')+"%", (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    for axes in f_dist_ethos2.axes.flat:
        _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=40)


    #st.stop()

    c1, c2, c3 = st.columns([1, 8, 1], gap='small')
    with c2:
        #add_spacelines(1)
        #st.pyplot(f_dist_ethos)
        add_spacelines(2)
        if an_type == 'Relation-based':
            st.write('#### Distribution based on relations')
        elif an_type == 'Sentence-based':
            st.write('#### Distribution based on sentences')
        elif an_type == 'ADU-based':
            st.write('#### Distribution based on ADUs')
            st.pyplot(f_dist_ethos)
        st.pyplot(f_dist_ethos2)
        #add_spacelines(3)
        #st.write('#### Distribution based on sentences')
        #st.pyplot(f_dist_ethos3)
    add_spacelines(1)



colors_old= {'joy' : '#8DF903', 'anger' : '#FD7E00', 'sadness' : '#CA00B9',
          'fear' : '#000000', 'disgust' :'#840079', 'no sentiment' : '#2002B5','surprise' : '#E1CA01',
          'positive':'#097604', 'negative':'#9B0101', 'neutral':'#2002B5',
          'contains_emotion':'#B10156', 'no_emotion':'#2002B5',
          'support':'#097604', 'attack':'#9B0101'}
colors = {'anger' : '#BA0A0A', 'surprise': '#BAB50A', 'disgust': '#520ABA',
          'sadness': '#03BDB4', 'joy': '#1CAF02', 'fear': '#0A0B0B',
          'no sentiment' : '#2002B5', 'positive':'#097604', 'negative':'#9B0101', 'neutral':'#949494',
          'contains_emotion':'#B10156', 'no_emotion':'#2002B5',
          'support':'#097604', 'attack':'#9B0101'}




#plotly
def distribution_plot_compareX_sub_crossPlotly(data_list0, an_unit, dim1, dim2):
    if 'label' in dim1:
        dim10 = dim1.split("_")[0]
    else:
        dim10 = dim1

    if 'label' in dim2:
        dim20 = dim2.split("_")[0]
    else:
        dim20 = dim2

    if len(data_list0) == 1:
        add_spacelines(1)
        up_data_dict = {}
        up_data_dict2 = {}
        n = 0
        for data in data_list0:
            df = data.copy()
            ds = df['corpus'].iloc[0]

            if not 'attack' in df['ethos_label'].unique():
                df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
            if not 'positive' in df['pathos_label'].unique():
                df['pathos_label'] = df['pathos_label'].map(valence_mapping)

            if an_unit == 'number':
                col_unit = 'number'
                df_dist_ethos = pd.DataFrame(df.groupby([dim1])[dim2].value_counts())

            else:
                col_unit = 'percentage'
                df_dist_ethos = pd.DataFrame(df.groupby([dim1])[dim2].value_counts(normalize = True).round(2)*100)


            df_dist_ethos.columns = [col_unit]
            df_dist_ethos.reset_index(inplace=True)
            df_dist_ethos.columns = [dim10, dim20, col_unit]
            df_dist_ethos = df_dist_ethos.sort_values(by = dim10)
            up_data_dict[n] = df_dist_ethos
            df_dist_ethos['corpora'] = ds

            n += 1

        df_dist_ethos_all = up_data_dict[0].copy()
        for k in range(int(len(up_data_dict.keys()))-1):
            k_sub = k+1
            df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

        sns.set(font_scale=1.2, style='whitegrid')
        maxval = df_dist_ethos_all[col_unit].max()


        import plotly.express as px
        fg1 = px.bar(df_dist_ethos_all, x=dim10, y=dim10, color=dim20,
                     pattern_shape=dim20, title= f'{dim10.capitalize()} x {dim20.capitalize()}'  )
        fg1.update_yaxes(range = [0, 100])

        fg1.show()



        fg2 = px.bar(df_dist_ethos_all, x=col_unit, y=dim20, color=dim10,
                     pattern_shape=dim10, title = f'{dim20.capitalize()} x {dim10.capitalize()}',  )
        fg2.update_yaxes(range = [0, 100])

        fg2.show()

        return fg1, fg2

    else:
        st.info("Function not supported for multiple corpora comparison.")





def distribution_plot_compareX_sub_cross(data_list0, an_unit, dim1, dim2):
    if 'label' in dim1:
        dim10 = dim1.split("_")[0]
    else:
        dim10 = dim1

    if 'label' in dim2:
        dim20 = dim2.split("_")[0]
    else:
        dim20 = dim2

    if len(data_list0) == 1:
        add_spacelines(1)
        up_data_dict = {}
        up_data_dict2 = {}
        n = 0
        for data in data_list0:
            df = data.copy()
            ds = df['corpus'].iloc[0]

            if not 'attack' in df['ethos_label'].unique():
                df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
            if not 'positive' in df['pathos_label'].unique():
                df['pathos_label'] = df['pathos_label'].map(valence_mapping)

            if an_unit == 'number':
                col_unit = 'number'
                df_dist_ethos = pd.DataFrame(df.groupby([dim1])[dim2].value_counts())

            else:
                col_unit = 'percentage'
                df_dist_ethos = pd.DataFrame(df.groupby([dim1])[dim2].value_counts(normalize = True).round(2)*100)


            df_dist_ethos.columns = [col_unit]
            df_dist_ethos.reset_index(inplace=True)
            df_dist_ethos.columns = [dim10, dim20, col_unit]
            df_dist_ethos = df_dist_ethos.sort_values(by = dim10)
            up_data_dict[n] = df_dist_ethos
            df_dist_ethos['corpora'] = ds

            n += 1

        df_dist_ethos_all = up_data_dict[0].copy()
        for k in range(int(len(up_data_dict.keys()))-1):
            k_sub = k+1
            df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

        sns.set(font_scale=1.2, style='whitegrid')
        maxval = df_dist_ethos_all[col_unit].max()
        fg1=sns.catplot(kind='bar', data=df_dist_ethos_all, y = dim10, x = col_unit,
                        hue=dim20, dodge=True, palette = colors, legend = True )
        if col_unit == 'percentage':
            plt.xlim(0, 100)
            plt.xticks(np.arange(0, 100, 10))
        else:
            plt.xlim(0, maxval+111)
            plt.xticks(np.arange(0, maxval+111, 100))
        plt.title(f'{dim10.capitalize()} x {dim20.capitalize()}')

        fg2=sns.catplot(kind='bar', data=df_dist_ethos_all, y = dim20, x = col_unit,
                        hue=dim10, dodge=True, palette = colors)
        if col_unit == 'percentage':
            plt.xlim(0, 100)
            plt.xticks(np.arange(0, 100, 10))
        else:
            plt.xlim(0, maxval+111)
            plt.xticks(np.arange(0, maxval+111, 100))
        plt.title(f'{dim20.capitalize()} x {dim10.capitalize()}')

        return fg1, fg2

    else:
        st.info("Function not supported for multiple corpora comparison.")




def distribution_plot_compareX_sub_single(data_list0, an_unit, dim1):
    if 'label' in dim1:
        dim0 = dim1.split("_")[0]
    else:
        dim0 = dim1

    up_data_dict = {}
    up_data_dict2 = {}
    n = 0
    for data in data_list0:
        df = data.copy()
        ds = str(df['corpus'].iloc[0])
        #st.dataframe(df)
        if not 'attack' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        if not 'positive' in df['pathos_label'].unique():
            df['pathos_label'] = df['pathos_label'].map(valence_mapping)

        if an_unit == 'number':
            col_unit = 'number'
            df_dist_ethos = pd.DataFrame(df[dim1].value_counts())
            df_dist_ethos2 = pd.DataFrame(df[df[dim1] != 'neutral'][dim1].value_counts())
        else:
            col_unit = 'percentage'
            df_dist_ethos = pd.DataFrame(df[dim1].value_counts(normalize = True).round(2)*100)
            df_dist_ethos2 = pd.DataFrame(df[df[dim1] != 'neutral'][dim1].value_counts(normalize = True).round(2)*100)

        df_dist_ethos.columns = [col_unit]
        df_dist_ethos.reset_index(inplace=True)
        df_dist_ethos.columns = [dim0, col_unit]
        df_dist_ethos = df_dist_ethos.sort_values(by = dim0)

        df_dist_ethos['corpora'] = ds
        up_data_dict[n] = df_dist_ethos

        df_dist_ethos2.columns = [col_unit]
        df_dist_ethos2.reset_index(inplace=True)
        df_dist_ethos2.columns = [dim0, col_unit]
        df_dist_ethos2 = df_dist_ethos2.sort_values(by = dim0)
        df_dist_ethos2['corpora'] = ds
        up_data_dict2[n] = df_dist_ethos2

        n += 1

    df_dist_ethos_all = up_data_dict[0].copy()
    for k in range(int(len(up_data_dict.keys()))-1):
        k_sub = k+1
        df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

    neu = []
    eth = []
    corp = []
    for cor in df_dist_ethos_all.corpora.unique():
        corp.append(cor)
        nn = df_dist_ethos_all[(df_dist_ethos_all.corpora == cor) & (df_dist_ethos_all[dim0] == 'neutral')][col_unit].iloc[0]
        neu.append(nn)
        eth.append(f'no {dim0}')
        nn2 = df_dist_ethos_all[(df_dist_ethos_all.corpora == cor)][col_unit].sum() - nn
        neu.append(nn2)
        eth.append(dim0)
        corp.append(cor)
    #st.dataframe(df_dist_ethos_all)
    #st.stop()
    df_dist_ethos_all0 = pd.DataFrame({dim0 : eth, col_unit:neu, 'corpora':corp})

    sns.set(font_scale=1.6, style='whitegrid')
    f_dist_ethos0 = sns.catplot(kind='bar', data = df_dist_ethos_all0, height=5.4, aspect=1.2,
                    x = dim0, y = col_unit, hue = dim0, dodge=False, legend = False,
                    palette = {dim0:'#EA9200', f'no {dim0}':'#022D96'},
                    col = 'corpora')
    if an_unit != 'number':
        f_dist_ethos0.set(ylim=(0, 110))
    f_dist_ethos0.set(xlabel="")
    #plt.title(f"Ethos distribution")
    for ax in f_dist_ethos0.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    sns.set(font_scale=1.4, style='whitegrid')
    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos_all, height=5.4, aspect=1.2,
                    x = dim0, y = col_unit, hue = dim0, dodge=False, legend = False,
                    palette = {'attack':'#BB0000', 'neutral':'#022D96', 'support':'#026F00',
                                'negative':'#BB0000', 'positive':'#026F00',
                                'anger': '#BA0A0A', 'surprise': '#BAB50A', 'disgust': '#520ABA', 'sadness': '#03BDB4', 'joy': '#1CAF02', 'fear': '#0A0B0B'
                                },
                    col = 'corpora')
    #plt.title(f"Ethos distribution")
    if an_unit != 'number':
        f_dist_ethos.set(ylim=(0, 110))
    for ax in f_dist_ethos.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
            ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

    df_dist_ethos_all2 = up_data_dict2[0].copy()
    for k in range(int(len(up_data_dict2.keys()))-1):
        k_sub2 = k+1
        df_dist_ethos_all2 = pd.concat([df_dist_ethos_all2, up_data_dict2[k_sub2]], axis=0, ignore_index=True)

    f_dist_ethos2 = sns.catplot(kind='bar', data = df_dist_ethos_all2, height=5.4, aspect=1.2,
                    x = dim0, y = col_unit, hue = dim0, dodge=False, legend = False,
                    palette = {'attack':'#BB0000', 'neutral':'#949494', 'support':'#026F00',
                                'negative':'#BB0000', 'positive':'#026F00',
                                'anger': '#BA0A0A', 'surprise': '#BAB50A', 'disgust': '#520ABA', 'sadness': '#03BDB4', 'joy': '#1CAF02', 'fear': '#0A0B0B'},
                    col = 'corpora')
    #plt.title(f"Ethos distribution")
    if an_unit != 'number':
        f_dist_ethos2.set(ylim=(0, 110))
    for ax in f_dist_ethos2.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

    return f_dist_ethos0, f_dist_ethos, f_dist_ethos2



def distribution_plot_compareX(data_list):
    st.write("### Distribution")
    add_spacelines(2)
    contents_radio_unit = st.radio("Unit of analysis", ("percentage", "number"))


    add_spacelines(1)
    c1, c3, c2, c5, c4 = st.tabs(['Ethos', 'Sentiment', 'Emotion', "Ethos x Sentiment", "Ethos x Emotion"])
    with c1:
        f_dist_0, f_dist_1, f_dist_2 = distribution_plot_compareX_sub_single(data_list0 = data_list,
                                    an_unit = contents_radio_unit, dim1 = 'ethos_label')
        add_spacelines(1)
        st.pyplot(f_dist_0)
        add_spacelines(1)
        st.pyplot(f_dist_1)
        add_spacelines(1)
        st.pyplot(f_dist_2)
        add_spacelines(1)

    with c3:
        f_dist_0, f_dist_1, f_dist_2 = distribution_plot_compareX_sub_single(data_list0 = data_list,
                                    an_unit = contents_radio_unit, dim1 = 'sentiment')
        add_spacelines(1)
        st.pyplot(f_dist_0)
        add_spacelines(1)
        st.pyplot(f_dist_1)
        add_spacelines(1)
        st.pyplot(f_dist_2)
        add_spacelines(1)

    with c2:
        f_dist_0, f_dist_1, f_dist_2 = distribution_plot_compareX_sub_single(data_list0 = data_list,
                                    an_unit = contents_radio_unit, dim1 = 'emotion')
        add_spacelines(1)
        st.pyplot(f_dist_0)
        add_spacelines(1)
        st.pyplot(f_dist_1)
        add_spacelines(1)
        st.pyplot(f_dist_2)
        add_spacelines(1)


    with c5:
        if len(data_list) == 1:
            fg1x, fg2x = distribution_plot_compareX_sub_cross(data_list0 = data_list,
                        an_unit = contents_radio_unit, dim1 = 'ethos_label', dim2 = 'sentiment')
            ff1, ff2, = st.columns(2)
            with ff1:
                st.pyplot(fg1x)
                #st.plotly_chart(fg1x)
            with ff2:
                st.pyplot(fg2x)
                #st.plotly_chart(fg2x)

    with c4:
        if len(data_list) == 1:
            fg1x, fg2x = distribution_plot_compareX_sub_cross(data_list0 = data_list,
                        an_unit = contents_radio_unit, dim1 = 'ethos_label', dim2 = 'emotion')
            ff1, ff2, = st.columns(2)
            with ff1:
                st.pyplot(fg1x)
                #st.plotly_chart(fg1x)
            with ff2:
                st.pyplot(fg2x)
                #st.plotly_chart(fg2x)


        else:
            add_spacelines(2)
            st.info("Function not supported for multiple corpora comparison.")




def distribution_plot_compare(data_list):
    st.write("### Compare distributions")
    add_spacelines(2)
    contents_radio_targs = st.radio("Category of the target of ethotic statements", ("both", "direct ethos", "3rd party ethos"))

    up_data_dict = {}
    up_data_dict2 = {}
    n = 0
    for data in data_list:
        df = data.copy()
        ds = df['corpus'].iloc[0]
        #st.dataframe(df)
        if not 'attack' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        df["Target"] = df["Target"].astype('str')
        df["Target"] = df["Target"].str.replace('Government', 'government')

        if contents_radio_targs == "direct ethos":
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if "@" in t]
            targets_limit.append('nan')
            df = df[df.Target.isin(targets_limit)]
            if len(targets_limit) < 1:
                st.error(f'No cases of **{contents_radio_targs}** found in the chosen corpora.')
                st.stop()
        elif contents_radio_targs == "3rd party ethos":
            targets_limit = df['Target'].dropna().unique()
            targets_limit = [t for t in targets_limit if not "@" in t]
            targets_limit.append('nan')
            df = df[df.Target.isin(targets_limit)]
            if len(targets_limit) < 1:
                st.error(f'No cases of **{contents_radio_targs}** found in the chosen corpora.')
                st.stop()

        df_dist_ethos = pd.DataFrame(df['ethos_label'].value_counts(normalize = True).round(2)*100)
        df_dist_ethos.columns = ['percentage']
        df_dist_ethos.reset_index(inplace=True)
        #st.dataframe(df_dist_ethos)
        df_dist_ethos.columns = ['ethos', 'percentage']
        df_dist_ethos = df_dist_ethos.sort_values(by = 'ethos')
        df_dist_ethos['corpus'] = ds
        up_data_dict[n] = df_dist_ethos

        df_dist_ethos2 = pd.DataFrame(df[df['ethos_label'] != 'neutral']['ethos_label'].value_counts(normalize = True).round(2)*100)
        df_dist_ethos2.columns = ['percentage']
        df_dist_ethos2.reset_index(inplace=True)
        df_dist_ethos2.columns = ['ethos', 'percentage']
        df_dist_ethos2 = df_dist_ethos2.sort_values(by = 'ethos')
        df_dist_ethos2['corpus'] = ds
        up_data_dict2[n] = df_dist_ethos2

        n += 1

    df_dist_ethos_all = up_data_dict[0].copy()
    for k in range(int(len(up_data_dict.keys()))-1):
        k_sub = k+1
        df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

    neu = []
    eth = []
    corp = []
    for cor in df_dist_ethos_all.corpus.unique():
        corp.append(cor)
        nn = df_dist_ethos_all[(df_dist_ethos_all.corpus == cor) & (df_dist_ethos_all['ethos'] == 'neutral')]['percentage'].iloc[0]
        neu.append(nn)
        eth.append('no ethos')
        neu.append(100 - nn)
        eth.append('ethos')
        corp.append(cor)
    df_dist_ethos_all0 = pd.DataFrame({'ethos':eth, 'percentage':neu, 'corpus':corp})

    sns.set(font_scale=1.55, style='whitegrid')
    f_dist_ethos0 = sns.catplot(kind='bar', data = df_dist_ethos_all0, height=4.5, aspect=1.4,
                    x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False, legend = False,
                    palette = {'ethos':'#EA9200', 'no ethos':'#022D96'},
                    col = 'corpus')
    f_dist_ethos0.set(ylim=(0, 110))
    for ax in f_dist_ethos0.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f')+"%", (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    f_dist_ethos0.set(xlabel="")

    sns.set(font_scale=1.25, style='whitegrid')
    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos_all, height=4.5, aspect=1.4,
                    x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False, legend = False,
                    palette = {'attack':'#BB0000', 'neutral':'#022D96', 'support':'#026F00',
                    'anger': '#BA0A0A', 'surprise': '#BAB50A', 'disgust': '#520ABA', 'sadness': '#03BDB4', 'joy': '#1CAF02', 'fear': '#0A0B0B'},
                    col = 'corpus')
    f_dist_ethos.set(ylim=(0, 110))
    for ax in f_dist_ethos.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f')+"%", (p.get_x() + p.get_width() / 2., p.get_height()),
            ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

    df_dist_ethos_all2 = up_data_dict2[0].copy()
    for k in range(int(len(up_data_dict2.keys()))-1):
        k_sub2 = k+1
        df_dist_ethos_all2 = pd.concat([df_dist_ethos_all2, up_data_dict2[k_sub2]], axis=0, ignore_index=True)

    #st.dataframe(df_dist_ethos_all2)
    f_dist_ethos2 = sns.catplot(kind='bar', data = df_dist_ethos_all2, height=4.5, aspect=1.4,
                    x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False, legend = False,
                    palette = {'attack':'#BB0000', 'support':'#026F00', 'neutral':'#949494'},
                    col = 'corpus')

    f_dist_ethos2.set(ylim=(0, 110))
    for ax in f_dist_ethos2.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f')+"%", (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

    c1, c2, c3 = st.columns([1, 8, 1], gap='small')
    with c2:
        add_spacelines(1)
        st.pyplot(f_dist_ethos0)
        add_spacelines(1)
        st.pyplot(f_dist_ethos)
        add_spacelines(1)
        st.pyplot(f_dist_ethos2)
    add_spacelines(1)


    add_spacelines(1)
    with st.expander("Pathos distribution"):
        add_spacelines(1)
        up_data_dict = {}
        up_data_dict2 = {}
        n = 0
        for data in data_list:
            df = data.copy()
            ds = df['corpus'].iloc[0]

            if not 'neutral' in df['pathos_label'].unique():
                df['pathos_label'] = df['pathos_label'].map(valence_mapping)

            if not 'attack' in df['ethos_label'].unique():
                df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
            df["Target"] = df["Target"].astype('str')
            df["Target"] = df["Target"].str.replace('Government', 'government')

            if contents_radio_targs == "direct ethos":
                targets_limit = df['Target'].dropna().unique()
                targets_limit = [t for t in targets_limit if "@" in t]
                targets_limit.append('nan')
                df = df[df.Target.isin(targets_limit)]
                if len(targets_limit) < 1:
                    st.error(f'No cases of **{contents_radio_targs}** found in the chosen corpora.')
                    st.stop()
            elif contents_radio_targs == "3rd party ethos":
                targets_limit = df['Target'].dropna().unique()
                targets_limit = [t for t in targets_limit if not "@" in t]
                targets_limit.append('nan')
                df = df[df.Target.isin(targets_limit)]

            df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")

            df_dist_ethos = pd.DataFrame(df['pathos_label'].value_counts(normalize = True).round(2)*100)
            df_dist_ethos.columns = ['percentage']
            df_dist_ethos.reset_index(inplace=True)
            df_dist_ethos.columns = ['pathos', 'percentage']
            df_dist_ethos = df_dist_ethos.sort_values(by = 'pathos')
            up_data_dict[n] = df_dist_ethos
            df_dist_ethos['corpus'] = ds
            #st.dataframe(df_dist_ethos)

            df_dist_ethos2 = pd.DataFrame(df[df['pathos_label'] != 'neutral']['pathos_label'].value_counts(normalize = True).round(2)*100)
            df_dist_ethos2.columns = ['percentage']
            df_dist_ethos2.reset_index(inplace=True)
            df_dist_ethos2.columns = ['pathos', 'percentage']
            df_dist_ethos2['corpus'] = ds
            df_dist_ethos2 = df_dist_ethos2.sort_values(by = 'pathos')
            up_data_dict2[n] = df_dist_ethos2
            #st.dataframe(df_dist_ethos2)

            n += 1

        df_dist_ethos_all = up_data_dict[0].copy()
        for k in range(int(len(up_data_dict.keys()))-1):
            k_sub = k+1
            df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

        neu = []
        eth = []
        corp = []
        for cor in df_dist_ethos_all.corpus.unique():
            corp.append(cor)
            nn = df_dist_ethos_all[(df_dist_ethos_all.corpus == cor) & (df_dist_ethos_all['pathos'] == 'neutral')]['percentage'].iloc[0]
            neu.append(nn)
            eth.append('no pathos')
            neu.append(100 - nn)
            eth.append('pathos')
            corp.append(cor)
        df_dist_ethos_all0 = pd.DataFrame({'pathos':eth, 'percentage':neu, 'corpus':corp})

        sns.set(font_scale=1.4, style='whitegrid')
        f_dist_ethos0 = sns.catplot(kind='bar', data = df_dist_ethos_all0, height=4.5, aspect=1.4,
                        x = 'pathos', y = 'percentage', hue = 'pathos', dodge=False,
                        palette = {'pathos':'#EA9200', 'no pathos':'#022D96'},
                        col = 'corpus')
        f_dist_ethos0.set(ylim=(0, 110))
        f_dist_ethos0.set(xlabel="")
        for ax in f_dist_ethos0.axes.ravel():
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.0f')+"%", (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

        f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos_all, height=4.5, aspect=1.4,
                        x = 'pathos', y = 'percentage', hue = 'pathos', dodge=False,
                        palette = {'negative':'#BB0000', 'neutral':'#949494', 'positive':'#026F00'},
                        col = 'corpus')
        f_dist_ethos.set(ylim=(0, 110))
        for ax in f_dist_ethos.axes.ravel():
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.0f')+"%", (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

        df_dist_ethos_all2 = up_data_dict2[0].copy()
        for k in range(int(len(up_data_dict2.keys()))-1):
            k_sub2 = k+1
            df_dist_ethos_all2 = pd.concat([df_dist_ethos_all2, up_data_dict2[k_sub2]], axis=0, ignore_index=True)

        f_dist_ethos2 = sns.catplot(kind='bar', data = df_dist_ethos_all2, height=4.5, aspect=1.4,
                        x = 'pathos', y = 'percentage', hue = 'pathos', dodge=False,
                        palette = {'negative':'#BB0000', 'positive':'#026F00', 'neutral':'#949494'},
                        col = 'corpus')
        f_dist_ethos2.set(ylim=(0, 110))
        for ax in f_dist_ethos2.axes.ravel():
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.0f')+"%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

        cc1, cc2, cc3 = st.columns([1, 8, 1], gap='small')
        with cc2:
            add_spacelines(1)
            st.pyplot(f_dist_ethos0)
            add_spacelines(1)
            st.pyplot(f_dist_ethos)
            add_spacelines(1)
            st.pyplot(f_dist_ethos2)
        add_spacelines(1)






import time

##################### page config  #####################
st.set_page_config(page_title="Analytics", layout="centered") # centered wide



import re
abuse=load_data(abslex)
abus_words = set(abuse.word.values)


summary_corpora_list = []
summary_corpora_list_raw = []
summary_corpora_dict_raw = {}
summary_corpora_list_raw_len = []

#  *********************** sidebar  *********************
with st.sidebar:
    st.title("Parameters of analysis")
    #st.write('Analysis Interface')
    contents_radio_rhetoric_category_ethos = True
    contents_radio_rhetoric_category_logos = False

    if contents_radio_rhetoric_category_ethos and not contents_radio_rhetoric_category_logos:
        contents_radio_type = st.radio("Type of analysis", ('Single Corpus', 'Compare Corpora'))

        if contents_radio_type == 'Compare Corpora':
            add_spacelines(1)
            contents_radio_type_compare = st.radio("Type of comparison", ('1 vs. 1', 'Group comparison'))
            add_spacelines(1)

            if contents_radio_type_compare == '1 vs. 1':
                st.write('Corpora')
                box_pol1 = st.checkbox("Covid-Vaccines Reddit")
                box_pol2 = st.checkbox("Covid-Vaccines Twitter", value=True)
                box_pol3 = st.checkbox("Climate-Change Reddit")
                box_pol4 = st.checkbox("Climate-Change Twitter", value=True)
                box_pol5 = st.checkbox("US-2016-Elections Reddit", value=False)

                if not (int(box_pol1) + int(box_pol2) + int(box_pol4) + int(box_pol3) + int(box_pol5) > 1):
                    st.error('Choose at least 2 corpora')
                    st.stop()
                corpora_list = []
                if box_pol1:
                    cor11 = load_data(vac_red)
                    cor1 = cor11.copy()
                    cor1_src = cor1['source'].unique()
                    cor1['conversation_id'] = 0
                    cor1_src = [str(s).replace('@', '') for s in cor1_src]
                    cor1['Target'] = cor1['Target'].astype('str')
                    cor1['source'] = cor1['source'].astype('str').apply(lambda x: "@"+str(x) if not "@" in x else x)
                    cor1['Target'] = cor1['Target'].apply(lambda x: "@"+str(x) if (not "@" in x and x in cor1_src) else x)
                    cor1['corpus'] = "Covid-Vaccines Reddit"
                    corpora_list.append(cor1)
                    summary_corpora_list.append(cor1)

                if box_pol2:
                    cor22 = load_data(vac_tw)
                    cor2 = cor22.copy()
                    cor2_src = cor2['source'].unique()
                    cor2_src = [str(s).replace('@', '') for s in cor2_src]
                    cor2['Target'] = cor2['Target'].astype('str')
                    cor2['Target'] = cor2['Target'].apply(lambda x: ["@"+str(x) if (not "@" in x and x in cor2_src) else x][0])
                    cor2['corpus'] = "Covid-Vaccines Twitter"
                    cor2['source'] = cor2['source'].astype('str').apply(lambda x: ["@"+str(x) if not "@" in x else x][0])
                    corpora_list.append(cor2)
                    summary_corpora_list.append(cor2)

                if box_pol3:
                    cor33 = load_data(cch_red)
                    cor3 = cor33.copy()
                    cor3_src = cor3['source'].unique()
                    cor3_src = [str(s).replace('@', '') for s in cor3_src]
                    cor3['Target'] = cor3['Target'].astype('str')
                    cor3['Target'] = cor3['Target'].apply(lambda x: ["@"+str(x) if (not "@" in x and x in cor3_src) else x][0])
                    cor3['corpus'] = "Climate-Change Reddit"
                    cor3['source'] = cor3['source'].astype('str').apply(lambda x: ["@"+str(x) if not "@" in x else x][0])
                    corpora_list.append(cor3)
                    summary_corpora_list.append(cor3)

                if box_pol4:
                    cor44 = load_data(cch_tw)
                    cor4 = cor44.copy()
                    cor4_src = cor4['source'].unique()
                    cor4_src = [str(s).replace('@', '') for s in cor4_src]
                    cor4['Target'] = cor4['Target'].astype('str')
                    cor4['Target'] = cor4['Target'].apply(lambda x: ["@"+str(x) if (not "@" in x and x in cor4_src) else x][0])
                    cor4['corpus'] = "Climate-Change Twitter"
                    cor4['source'] = cor4['source'].astype('str').apply(lambda x: ["@"+str(x) if not "@" in x else x][0])
                    corpora_list.append(cor4)
                    summary_corpora_list.append(cor4)

                if box_pol5:
                    cor55 = load_data(us16)
                    cor5 = cor55.copy()
                    cor5['Target'] = cor5['Target'].astype('str')
                    cor5['corpus'] = "US-2016-Elections Reddit"
                    cor5['source'] = cor5['source'].astype('str')
                    corpora_list.append(cor5)
                    summary_corpora_list.append(cor5)

            elif contents_radio_type_compare == 'Group comparison':
                st.write('Corpora')
                corpora_list_names = ["Covid-Vaccines Reddit", "Covid-Vaccines Twitter",
                                     "Climate-Change Reddit", "Climate-Change Twitter",
                                     "US-2016-Elections Reddit"]
                corpora_paths = {"Covid-Vaccines Reddit": vac_red,
                "Covid-Vaccines Twitter": vac_tw,
                "Climate-Change Reddit": cch_red,
                "Climate-Change Twitter": cch_tw,
                "US-2016-Elections Reddit": us16}

                options1 = st.multiselect('First group of corpora', corpora_list_names, corpora_list_names[:2])
                corpora_list_names_grp2 = set(corpora_list_names) - set(options1)
                corpora_list_names_grp2 = list(corpora_list_names_grp2)
                options2 = st.multiselect('Second group of corpora', corpora_list_names_grp2, corpora_list_names_grp2[0])

                shared_cols = ['sentence', 'source', 'Target', 'ethos_label', 'pathos_label',
                                'corpus'] # 'full_text_id', 'conversation_id'
                data1 = load_data(corpora_paths[options1[0]])


                data1['corpus'] = " &\n ".join(options1)
                data1 = data1#[shared_cols]
                if len(options1) > 1:
                    for nn in range(int(len(options1))-1):
                        n = nn+1
                        data1_2 = load_data(corpora_paths[options1[int(n)]])

                        data1_2['corpus'] = " &\n ".join(options1)
                        data1_2 = data1_2#[shared_cols]
                        data1 = pd.concat( [data1, data1_2], axis=0, ignore_index=True )

                data2 = load_data(corpora_paths[options2[0]])
                data2['corpus'] = " &\n ".join(options2)

                data2 = data2#[shared_cols]
                if len(options2) > 1:
                    for nn in range(int(len(options2))-1):
                        n = nn+1
                        data2_2 = load_data(corpora_paths[options2[int(n)]])

                        data2_2['corpus'] = " &\n ".join(options2)
                        data2_2 = data2_2#[shared_cols]
                        data2 = pd.concat( [data2, data2_2], axis=0, ignore_index=True )

                corpora_list = []
                data1_src = data1['source'].unique()
                data1_src = [str(s).replace('@', '') for s in data1_src]
                data1['Target'] = data1['Target'].astype('str')
                data1['source'] = data1['source'].astype('str').apply(lambda x: ["@"+str(x) if not "@" in x else x][0])
                data1['Target'] = data1['Target'].apply(lambda x: "@"+str(x) if (not "@" in x and x in data1_src) else x)

                data2_src = data2['source'].unique()
                data2_src = [str(s).replace('@', '') for s in data2_src]
                data2['Target'] = data2['Target'].astype('str')
                data2['source'] = data2['source'].astype('str').apply(lambda x: ["@"+str(x) if not "@" in x else x][0])
                data2['Target'] = data2['Target'].apply(lambda x: "@"+str(x) if (not "@" in x and x in data2_src) else x)
                corpora_list.append(data1)
                corpora_list.append(data2)

            add_spacelines(1)
            contents_radio_an_cat = st.radio("Analytics Units", ('Sentence-based', 'Entity-based'))#, 'Time-based'
            add_spacelines(1)
            if contents_radio_an_cat == 'Entity-based':
                contents_radio3 = st.radio("Analytics", {'(Anti)-heroes'})
            else:
                contents_radio3 = st.radio("Analytics", ('Distribution', 'WordCloud'))
            #add_spacelines(1)


        elif contents_radio_type == 'Single Corpus':
            add_spacelines(1)
            st.write('Corpora Aspect')
            bool1 = False
            bool2 = False
            bool3 = False
            bool4 = False
            bool5 = False

            if 'boxTopic' not in st.session_state and "boxPlatform" not in st.session_state:
                st.session_state['boxTopic'] = False
                st.session_state['boxPlatform'] = False

            box_topic = st.checkbox("Topic-based", disabled=st.session_state.boxPlatform, key="boxTopic")
            box_platform = st.checkbox("Platform-based", disabled=st.session_state.boxTopic, key="boxPlatform")

            label_visibility_box_num = "visible"

            if box_topic:
                st.write('Choose topic')
                box_topic_cc = st.checkbox("Climate-Change", value = True)
                box_topic_vacc = st.checkbox("Covid-Vaccines")
                box_topic_elect = st.checkbox("Elections")
                add_spacelines(1)

                if box_topic_cc and box_topic_vacc and box_topic_elect:
                    bool3 = True
                    bool4 = True
                    bool1 = True
                    bool2 = True
                    bool5 = True

                elif box_topic_cc and box_topic_vacc and not box_topic_elect:
                    bool3 = True
                    bool4 = True
                    bool1 = True
                    bool2 = True
                    bool5 = False

                elif box_topic_cc and not box_topic_vacc and box_topic_elect:
                    bool3 = True
                    bool4 = True
                    bool1 = False
                    bool2 = False
                    bool5 = True
                elif not box_topic_cc and box_topic_vacc and box_topic_elect:
                    bool3 = False
                    bool4 = False
                    bool1 = True
                    bool2 = True
                    bool5 = True

                elif box_topic_cc and not box_topic_vacc and not box_topic_elect:
                    bool3 = True
                    bool4 = True
                    bool1 = False
                    bool2 = False
                    bool5 = False

                elif box_topic_vacc and not box_topic_cc and not box_topic_elect:
                    bool1 = True
                    bool2 = True
                    bool3 = False
                    bool4 = False
                    bool5 = False

                elif box_topic_elect and not box_topic_vacc and not box_topic_cc:
                    bool5 = True
                    bool1 = False
                    bool2 = False
                    bool3 = False
                    bool4 = False

                if not box_topic_elect:
                    bool5 = False
                if not box_topic_vacc:
                    bool1 = False
                    bool2 = False
                if not box_topic_cc:
                    bool3 = False
                    bool4 = False

            elif box_platform:
                st.write('Choose platform')
                box_platform_reddit = st.checkbox("Reddit")
                box_platform_twitter = st.checkbox("Twitter", value = True)
                add_spacelines(1)

                if box_platform_reddit and box_platform_twitter:
                    bool2 = True
                    bool4 = True
                    bool1 = True
                    bool3 = True
                    bool5 = True

                elif box_platform_reddit and not box_platform_twitter:
                    bool1 = True
                    bool3 = True
                    bool5 = True
                    bool2 = False
                    bool4 = False
                elif box_platform_twitter and not box_platform_reddit:
                    bool2 = True
                    bool4 = True
                    bool1 = False
                    bool3 = False
                    bool5 = False

            if not box_platform and not box_topic:
                add_spacelines(1)
                st.write('Corpora')
                bool4 = True
            else:
                st.write('Corpora')

            if box_topic:
                if not box_topic_elect:
                    st.session_state['US-2016-Elections Reddit'] = False
                if not box_topic_vacc:
                    st.session_state['Covid-Vaccines Reddit'] = False
                    st.session_state['Covid-Vaccines Twitter'] = False
                if not box_topic_cc:
                    st.session_state['Climate-Change Reddit'] = False
                    st.session_state['Climate-Change Twitter'] = False
            elif box_platform:
                if not box_platform_reddit:
                    st.session_state['Climate-Change Reddit'] = False
                    st.session_state['US-2016-Elections Reddit'] = False
                    st.session_state['Covid-Vaccines Reddit'] = False

                if not box_platform_twitter:
                    st.session_state['Climate-Change Twitter'] = False
                    st.session_state['Covid-Vaccines Twitter'] = False

            box_pol1 = st.checkbox("Covid-Vaccines Reddit", value=bool1, label_visibility = label_visibility_box_num, key="Covid-Vaccines Reddit")
            box_pol2 = st.checkbox("Covid-Vaccines Twitter", value=bool2, label_visibility = label_visibility_box_num, key="Covid-Vaccines Twitter")
            box_pol3 = st.checkbox("Climate-Change Reddit", value=bool3, label_visibility = label_visibility_box_num, key= "Climate-Change Reddit")
            box_pol4 = st.checkbox("Climate-Change Twitter", value=bool4, label_visibility = label_visibility_box_num, key="Climate-Change Twitter")
            box_pol5 = st.checkbox("US-2016-Elections Reddit", value=bool5, label_visibility = label_visibility_box_num, key="US-2016-Elections Reddit")

            corpora_list = []
            if box_pol1:
                cor11 = load_data(vac_red)
                cor1 = cor11.copy()
                cor1_src = cor1['source'].unique()
                cor1['conversation_id'] = 0
                cor1_src = [str(s).replace('@', '') for s in cor1_src]
                cor1['Target'] = cor1['Target'].astype('str')
                cor1['source'] = cor1['source'].astype('str').apply(lambda x: ["@"+str(x) if not "@" in x else x][0])
                cor1['Target'] = cor1['Target'].apply(lambda x: ["@"+str(x) if (not "@" in x and x in cor1_src) else x][0])
                cor1['corpus'] = "Covid-Vaccines Reddit"
                corpora_list.append(cor1)
                summary_corpora_list.append(cor1)

            if box_pol2:
                cor22 = load_data(vac_tw)
                cor2 = cor22.copy()
                cor2_src = cor2['source'].unique()
                cor2_src = [str(s).replace('@', '') for s in cor2_src]
                cor2['Target'] = cor2['Target'].astype('str')
                cor2['Target'] = cor2['Target'].apply(lambda x: ["@"+str(x) if (not "@" in x and x in cor2_src) else x][0])
                cor2['corpus'] = "Covid-Vaccines Twitter"
                cor2['source'] = cor2['source'].astype('str').apply(lambda x: ["@"+str(x) if not "@" in x else x][0])
                corpora_list.append(cor2)
                summary_corpora_list.append(cor2)

            if box_pol3:
                cor33 = load_data(cch_red)
                cor3 = cor33.copy()
                cor3_src = cor3['source'].unique()
                cor3_src = [str(s).replace('@', '') for s in cor3_src]
                cor3['Target'] = cor3['Target'].astype('str')
                cor3['Target'] = cor3['Target'].apply(lambda x: ["@"+str(x) if (not "@" in x and x in cor3_src) else x][0])
                cor3['corpus'] = "Climate-Change Reddit"
                cor3['source'] = cor3['source'].astype('str').apply(lambda x: ["@"+str(x) if not "@" in x else x][0])
                corpora_list.append(cor3)
                summary_corpora_list.append(cor3)

            if box_pol4:
                cor44 = load_data(cch_tw)
                cor4 = cor44.copy()
                cor4_src = cor4['source'].unique()
                cor4_src = [str(s).replace('@', '') for s in cor4_src]
                cor4['Target'] = cor4['Target'].astype('str')
                cor4['Target'] = cor4['Target'].apply(lambda x: ["@"+str(x) if (not "@" in x and x in cor4_src) else x][0])
                cor4['corpus'] = "Climate-Change Twitter"
                cor4['source'] = cor4['source'].astype('str').apply(lambda x: ["@"+str(x) if not "@" in x else x][0])
                corpora_list.append(cor4)
                summary_corpora_list.append(cor4)

            if box_pol5:
                cor55 = load_data(us16)
                cor5 = cor55.copy()
                cor5['Target'] = cor5['Target'].astype('str')
                cor5['corpus'] = "US-2016-Elections Reddit"
                cor5['source'] = cor5['source'].astype('str')
                corpora_list.append(cor5)
                summary_corpora_list.append(cor5)


            data = corpora_list[0]
            if len(corpora_list) > 1:
                corp_names = data['corpus'].iloc[0]
                for nn in range(int(len(corpora_list))-1):
                    n = nn+1
                    data1_2 = corpora_list[int(n)]
                    corp_names = corp_names + " &\n " + data1_2['corpus'].iloc[0]
                    data = pd.concat( [data, data1_2], axis=0, ignore_index=True )
            if len(corpora_list) > 1:
                data['corpus'] = corp_names
                corpora_list = []
                corpora_list.append(data)

            add_spacelines(1)
            contents_radio_an_cat = st.radio("Analytics Units", ('Sentence-based', 'Entity-based'))
            add_spacelines(1)
            if contents_radio_an_cat == 'Entity-based':
                contents_radio3 = st.radio("Analytics", ['(Anti)-heroes',  "Profiles", "Fellows-Devils", ]) # "Polarising Tendency" "Rhetoric Strategies",
            else:
                contents_radio3 = st.radio("Analytics", ("Corpora Summary", 'Distribution', 'WordCloud', 'Frequency Tables',
                                            'Odds ratio', 'Pronouns', 'Explore corpora'))
            #add_spacelines(1)

    if not contents_radio_rhetoric_category_ethos and contents_radio_rhetoric_category_logos:
        contents_radio_type = st.radio("Type of analysis", ('Single Corpora', 'Compare Corpora'))
        add_spacelines(1)

        if contents_radio_type == 'Compare Corpora':
            st.write('Choose corpora')
            box_pol1_log = st.checkbox("Covid-Vaccines Reddit", value=False)
            box_pol5_log = st.checkbox("US-2016-Elections Reddit", value=True)
            box_polh_log = st.checkbox("Hansard UK", value=True)

            corpora_list = []
            add_spacelines(1)

            if not (int(box_pol1_log) + int(box_pol5_log) + int(box_polh_log) > 1):
                st.error('Choose at least 2 corpora')
                st.stop()


            if box_pol1_log:
                    cor11 = load_data(vac_red)
                    cor1 = cor11.copy()
                    cor1_src = cor1['source'].unique()
                    cor1['conversation_id'] = 0
                    cor1_src = [str(s).replace('@', '') for s in cor1_src]
                    cor1['Target'] = cor1['Target'].astype('str')
                    cor1['source'] = cor1['source'].astype('str').apply(lambda x: ["@"+str(x) if not "@" in x else x][0])
                    cor1['Target'] = cor1['Target'].apply(lambda x: ["@"+str(x) if (not "@" in x and x in cor1_src) else x][0])
                    cor1['corpus'] = "Covid-Vaccines Reddit"
                    cor1['kind'] = "ethos"
                    corpora_list.append(cor1)
                    summary_corpora_list.append(cor1)

                    cor11 = load_data(vac_red_log)
                    cor1 = cor11.copy()
                    cor11_lp = set( cor11.locution_premise.unique() )
                    cor11_lc = set( cor11.locution_conclusion.unique() )
                    cor11_l = cor11_lc.union(cor11_lp)

                    cor11_2 = load_data(vac_red_log2, indx=False)
                    cor11_2 = cor11_2.drop_duplicates(sunset = ['columns'])
                    cor11_2['corpus'] = "Covid-Vaccines Reddit"
                    summary_corpora_dict_raw["Covid-Vaccines Reddit"] = cor11_2.shape[0]
                    cor11_2['locution'] = cor11_2['locution'].apply(lambda x: ":".join( str(x).split(":")[1:] ))
                    cor11_2['nwords'] = cor11_2.locution.astype('str').str.split().map(len)
                    summary_corpora_list_raw_len.append(cor11_2[['nwords', 'corpus']])
                    cor11_2_loc = cor11_2.locution.unique()
                    cor11_2_loc = set(cor11_2_loc).difference(cor11_l)
                    cor11_2 = cor11_2[cor11_2.locution.isin(list(cor11_2_loc))]
                    #cor11 = pd.concat( [cor11, cor11_2], axis = 0, ignore_index = True )
                    cor1['kind'] = "logos"
                    cor1['corpus'] = "Covid-Vaccines Reddit"
                    corpora_list.append(cor1)
                    summary_corpora_list.append(cor1)
                    summary_corpora_list_raw.append(cor11_2)

            if box_pol5_log:
                    cor55 = load_data(us16)
                    cor5 = cor55.copy()
                    cor5['Target'] = cor5['Target'].astype('str')
                    cor5['corpus'] = "US-2016-Elections Reddit"
                    cor5['source'] = cor5['source'].astype('str')
                    cor5['kind'] = "ethos"
                    corpora_list.append(cor5)
                    summary_corpora_list.append(cor5)
                    #cor = pd.concat([cor1, cor5], axis=0, ignore_index=True)
                    #cor['corpus'] = "Covid-Vaccines Reddit &\n US-2016-Elections Reddit" #Ethos
                    #corpora_list.append(cor)

                    cor55 = load_data(us16_log, indx=False)
                    cor11_2d = pd.read_excel(us16_log2, sheet_name = 'dr1')
                    cor11_2g = pd.read_excel(us16_log2, sheet_name = 'gr1')
                    cor11_2r = pd.read_excel(us16_log2, sheet_name = 'rr1')
                    cor11_2 = pd.concat( [cor11_2d, cor11_2g, cor11_2r], axis = 0, ignore_index = True )
                    cor11_2 = cor11_2.drop_duplicates(subset = ['locution'], keep = 'last')
                    summary_corpora_dict_raw["US-2016-Elections Reddit"] = cor11_2.shape[0]
                    cor11_2['corpus'] = "US-2016-Elections Reddit"
                    cor11_2['locution'] = cor11_2['locution'].apply(lambda x: ":".join( str(x).split(":")[1:] ))
                    cor11_2['nwords'] = cor11_2.locution.astype('str').str.split().map(len)
                    summary_corpora_list_raw_len.append(cor11_2[['nwords', 'corpus']])

                    cor11_lp = set( cor55.locution_premise.unique() )
                    cor11_lc = set( cor55.locution_conclusion.unique() )
                    cor11_l = cor11_lc.union(cor11_lp)

                    cor11_2_loc = cor11_2.locution.unique()
                    cor11_2_loc = set(cor11_2_loc).difference(cor11_l)
                    cor11_2 = cor11_2[cor11_2.locution.isin(list(cor11_2_loc))]
                    cor11_2 = cor11_2.rename(columns = {'id_L-node':'id_L_node', 'id_I-node':'id_I_node'})
                    #cor55 = pd.concat( [cor55, cor11_2], axis = 0, ignore_index = True )
                    cor5 = cor55.copy()
                    cor5['kind'] = "logos"
                    cor5['corpus'] = "US-2016-Elections Reddit"
                    corpora_list.append(cor5)
                    summary_corpora_list.append(cor5)
                    summary_corpora_list_raw.append(cor11_2)

            if box_polh_log:
                    corhh = load_data(hans)
                    corh = corhh.copy()
                    corh['Target'] = corh['Target'].astype('str')
                    corh['corpus'] = "Hansard UK"
                    corh['source'] = corh['source'].astype('str')
                    corh['kind'] = "ethos"
                    corpora_list.append(corh)
                    summary_corpora_list.append(corh)

                    corhh = load_data(hans_log)
                    cor11_2 = load_data(hans_log2, indx=False)
                    cor11_2 = cor11_2.drop_duplicates(subset = ['locution'])
                    summary_corpora_dict_raw["Hansard UK"] = cor11_2.shape[0]
                    cor11_2['corpus'] = "Hansard UK"
                    cor11_2['locution'] = cor11_2['locution'].apply(lambda x: ":".join( str(x).split(":")[1:] ))
                    cor11_2['nwords'] = cor11_2.locution.astype('str').str.split().map(len)
                    summary_corpora_list_raw_len.append(cor11_2[['nwords', 'corpus']])

                    cor11_lp = set( corhh.locution_premise.unique() )
                    cor11_lc = set( corhh.locution_conclusion.unique() )
                    cor11_l = cor11_lc.union(cor11_lp)

                    cor11_2_loc = cor11_2.locution.unique()
                    cor11_2_loc = set(cor11_2_loc).difference(cor11_l)
                    cor11_2 = cor11_2[cor11_2.locution.isin(list(cor11_2_loc))]
                    cor11_2 = cor11_2.rename(columns = {'L-node_id':'id_L_node', 'I-node_id':'id_I_node',
                                                        'iillocution_id':'id_illocution'})
                    #corhh = pd.concat( [corhh, cor11_2], axis = 0, ignore_index = True )
                    corh = corhh.copy()
                    corh['kind'] = "logos"
                    corh['corpus'] = "Hansard UK"
                    corpora_list.append(corh)
                    summary_corpora_list.append(corh)
                    summary_corpora_list_raw.append(cor11_2)

        else:
            st.write('Choose corpora')
            box_pol1_log = st.checkbox("Covid-Vaccines Reddit", value=False)
            box_pol5_log = st.checkbox("US-2016-Elections Reddit", value=True)
            box_polh_log = st.checkbox("Hansard UK", value=False)
            corpora_list = []
            corpora_list_et = {}
            corpora_list_log = {}


            if box_polh_log:
                cor11 = load_data(hans)
                cor1 = cor11.copy()
                cor1_src = cor1['source'].unique()
                cor1['conversation_id'] = 0
                cor1['corpus'] = "Hansard UK"
                cor1['Target'] = cor1['Target'].astype('str')
                cor1['source'] = cor1['source'].astype('str')
                cor1['kind'] = "ethos"
                corpora_list_et[cor1['corpus'].iloc[0]] = cor1
                summary_corpora_list.append(cor1)

                cor11 = load_data(hans_log)
                cor11_2 = load_data(hans_log2, indx=False)
                cor11_lp = set( cor11.locution_premise.unique() )
                cor11_lc = set( cor11.locution_conclusion.unique() )
                cor11_l = cor11_lc.union(cor11_lp)
                cor11_2 = cor11_2.rename(columns = {'L-node_id':'id_L_node', 'I-node_id':'id_I_node',
                                                    'iillocution_id':'id_illocution'})
                cor11_2 = cor11_2.drop_duplicates(subset = ['locution'])
                summary_corpora_dict_raw["Hansard UK"] = cor11_2.shape[0]
                cor11_2['corpus'] = "Hansard UK"
                cor11_2['locution'] = cor11_2['locution'].apply(lambda x: ":".join( str(x).split(":")[1:] ))
                cor11_2['nwords'] = cor11_2.locution.astype('str').str.split().map(len)
                summary_corpora_list_raw_len.append(cor11_2[['nwords', 'corpus']])

                cor11_2_loc = cor11_2.locution.unique()
                cor11_2_loc = set(cor11_2_loc).difference(cor11_l)
                cor11_2 = cor11_2[cor11_2.locution.isin(list(cor11_2_loc))]

                #cor11 = pd.concat( [cor11, cor11_2], axis = 0, ignore_index = True )
                cor1 = cor11.copy()
                cor1['kind'] = "logos"
                cor1['corpus'] = "Hansard UK"
                corpora_list_log[cor1['corpus'].iloc[0]] = cor1
                summary_corpora_list.append(cor1)
                summary_corpora_list_raw.append(cor11_2)


            if box_pol1_log:
                cor11 = load_data(vac_red)
                cor1 = cor11.copy()
                cor1_src = cor1['source'].unique()
                cor1['conversation_id'] = 0
                cor1_src = [str(s).replace('@', '') for s in cor1_src]
                cor1['Target'] = cor1['Target'].astype('str')
                cor1['source'] = cor1['source'].astype('str').apply(lambda x: ["@"+str(x) if not "@" in x else x][0])
                cor1['Target'] = cor1['Target'].apply(lambda x: ["@"+str(x) if (not "@" in x and x in cor1_src) else x][0])
                cor1['corpus'] = "Covid-Vaccines Reddit" # Ethos
                cor1['kind'] = "ethos"
                corpora_list_et[cor1['corpus'].iloc[0]] = cor1
                summary_corpora_list.append(cor1)

                cor11 = load_data(vac_red_log)
                cor11_2 = load_data(vac_red_log2, indx=False)
                cor11_lp = set( cor11.locution_premise.unique() )
                cor11_lc = set( cor11.locution_conclusion.unique() )
                cor11_l = cor11_lc.union(cor11_lp)
                cor11_2 = cor11_2.drop_duplicates(subset = ['locution'])
                cor11_2['corpus'] = "Covid-Vaccines Reddit"
                summary_corpora_dict_raw["Covid-Vaccines Reddit"] = cor11_2.shape[0]
                cor11_2['locution'] = cor11_2['locution'].apply(lambda x: ":".join( str(x).split(":")[1:] ))
                cor11_2['nwords'] = cor11_2.locution.astype('str').str.split().map(len)
                summary_corpora_list_raw_len.append(cor11_2[['nwords', 'corpus']])

                cor11_2_loc = cor11_2.locution.unique()
                cor11_2_loc = set(cor11_2_loc).difference(cor11_l)
                cor11_2 = cor11_2[cor11_2.locution.isin(list(cor11_2_loc))]
                #cor11 = pd.concat( [cor11, cor11_2], axis = 0, ignore_index = True )
                cor1 = cor11.copy()
                cor1['kind'] = "logos"
                cor1['corpus'] = "Covid-Vaccines Reddit"
                corpora_list_log[cor1['corpus'].iloc[0]] = cor1
                summary_corpora_list.append(cor1)
                summary_corpora_list_raw.append(cor11_2)


            if box_pol5_log:
                cor55 = load_data(us16)
                cor5 = cor55.copy()
                cor5['Target'] = cor5['Target'].astype('str')
                cor5['corpus'] = "US-2016-Elections Reddit"
                cor5['source'] = cor5['source'].astype('str')
                cor5['kind'] = "ethos"
                corpora_list_et[cor5['corpus'].iloc[0]] = cor5
                summary_corpora_list.append(cor5)

                cor55 = load_data(us16_log, indx=False)
                cor11_2d = pd.read_excel(us16_log2, sheet_name = 'dr1')
                cor11_2g = pd.read_excel(us16_log2, sheet_name = 'gr1')
                cor11_2r = pd.read_excel(us16_log2, sheet_name = 'rr1')
                cor11_2 = pd.concat( [cor11_2d, cor11_2g, cor11_2r], axis = 0, ignore_index = True )
                cor11_2 = cor11_2.rename(columns = {'id_L-node':'id_L_node', 'id_I-node':'id_I_node'})
                cor11_2 = cor11_2.drop_duplicates(subset = ['locution'], keep = 'last')
                summary_corpora_dict_raw["US-2016-Elections Reddit"] = cor11_2.shape[0]
                cor11_2['corpus'] = "US-2016-Elections Reddit"
                cor11_2['locution'] = cor11_2['locution'].apply(lambda x: ":".join( str(x).split(":")[1:] ))
                cor11_2['nwords'] = cor11_2.locution.astype('str').str.split().map(len)
                summary_corpora_list_raw_len.append(cor11_2[['nwords', 'corpus']])

                cor11_lp = set( cor55.locution_premise.unique() )
                cor11_lc = set( cor55.locution_conclusion.unique() )
                cor11_l = cor11_lc.union(cor11_lp)

                cor11_2_loc = cor11_2.locution.unique()
                cor11_2_loc = set(cor11_2_loc).difference(cor11_l)
                cor11_2 = cor11_2[cor11_2.locution.isin(list(cor11_2_loc))]
                #cor55 = pd.concat( [cor55, cor11_2], axis = 0, ignore_index = True )
                cor5 = cor55.copy()
                cor5['kind'] = "logos"
                cor5['corpus'] = "US-2016-Elections Reddit"
                corpora_list_log[cor5['corpus'].iloc[0]] = cor5
                summary_corpora_list.append(cor5)
                summary_corpora_list_raw.append(cor11_2)


            if len(corpora_list_log.keys()) > 1:
                df_log = pd.concat(corpora_list_log.values(), axis = 0, ignore_index = True)
                df_let = pd.concat(corpora_list_et.values(), axis = 0, ignore_index = True)
                ds = " &\n ".join( list(corpora_list_et.keys()) )
                #print(ds)
                #print(len(corpora_list_log.keys()))
                #print(corpora_list_log.keys())
                #cor = pd.concat([cor1, cor5], axis=0, ignore_index=True)
                df_let['corpus'] = ds
                df_log['corpus'] = ds
                corpora_list.append(df_let)
                corpora_list.append(df_log)
            else:
                df_log = corpora_list_log[list(corpora_list_log.keys())[0]]
                df_let = corpora_list_et[list(corpora_list_et.keys())[0]]
                corpora_list.append(df_let)
                corpora_list.append(df_log)

        add_spacelines(1)
        contents_radio_an_cat = st.radio("Analytics Units", ('ADU-based', 'Relation-based', 'Entity-based'))
        add_spacelines(1)
        if contents_radio_an_cat == 'Entity-based':
            contents_radio3 = st.radio("Analytics", ['(Anti)-heroes', "Profiles"])
        else:
            contents_radio3 = st.radio("Analytics", ("Corpora Summary", 'Distribution', 'WordCloud', 'Frequency Tables', 'Odds Ratio', 'Lexical Stats'))# , 'Odds ratio', 'Explore corpora'
        #add_spacelines(1)




#####################  page content  #####################
st.title(f"Polarising Moral Panic Analytics")
add_spacelines(1)

@st.cache_data
def SumCorpEthosTable(dataframe, group_column):
    n_corps = dataframe[group_column].nunique()
    dataframe_desc = dataframe.groupby(group_column)['nwords'].sum().reset_index()
    dataframe_desc.columns = [group_column, '#-words']#, 'avg-#-words'
    dataframe_desc_c = dataframe.groupby(group_column)['nwords_content'].sum().reset_index()
    dataframe_desc_c.columns = [group_column, '#-content words']#, 'avg-#-content words'
    dataframe_desc = dataframe_desc.merge(dataframe_desc_c, on = group_column)

    dataframe.source = dataframe.source.astype('str')
    dataframe_src = dataframe.groupby(group_column, as_index = False)['source'].nunique()
    dataframe_src.columns = [group_column, '# speakers']
    dataframe.Target = dataframe.Target.astype('str')
    dataframe_trg = dataframe[dataframe.Target != 'nan'].groupby(group_column, as_index = False)['Target'].nunique()
    dataframe_trg.columns = [group_column, '# targets']

    dataframe_s = dataframe.groupby(group_column, as_index = False).size()
    dataframe_desc = dataframe_desc.merge(dataframe_s, on = group_column)

    dataframe_desc = dataframe_desc.merge(dataframe_src, on = group_column)
    dataframe_desc = dataframe_desc.merge(dataframe_trg, on = group_column)

    dataframe_ed = dataframe.groupby(group_column, as_index = False)[['ethos density', 'E+', 'E-']].mean()
    dataframe_ed[['ethos density', 'E+', 'E-']] = dataframe_ed[['ethos density', 'E+', 'E-']] * 100
    dataframe_desc = dataframe_desc.merge(dataframe_ed, on = group_column)

    dataframe_desc.loc[len(dataframe_desc)] = ['AVERAGE', dataframe_desc['#-words'].mean(),
                            dataframe_desc['#-content words'].mean(), dataframe_desc['size'].mean(),
                            dataframe_desc['# speakers'].mean(), dataframe_desc['# targets'].mean(),
                            dataframe['ethos density'].mean()* 100, dataframe['E+'].mean()* 100, dataframe['E-'].mean()* 100]
    dataframe_desc = dataframe_desc.round(1)
    dataframe_desc.loc[len(dataframe_desc)] = ['TOTAL', dataframe_desc['#-words'].iloc[:int(n_corps)].sum(),
                            dataframe_desc['#-content words'].iloc[:int(n_corps)].sum(),
                            dataframe_desc['size'].iloc[:int(n_corps)].sum().round(0),
                            dataframe['source'].nunique(), dataframe['Target'].nunique(),
                            ' n/a ', ' n/a ', ' n/a ']
    dataframe_desc[['#-words', '#-content words', '# speakers', '# targets', 'size']] = dataframe_desc[['#-words', '#-content words', '# speakers', '# targets', 'size']].astype('int')
    return dataframe_desc



#####################  page content  #####################
if contents_radio3 == "Corpora Summary":
    st.write("## Corpora Summary")

    if contents_radio_rhetoric_category_ethos:
        cc = pd.concat( summary_corpora_list, axis = 0, ignore_index = True )
        cc['ethos density'] = np.where(cc.ethos_label != 0, 1, 0)
        cc['E+'] = np.where(cc.ethos_label == 1, 1, 0)
        cc['E-'] = np.where(cc.ethos_label == 2, 1, 0)
        cc['topic'] = cc.corpus.apply(lambda x: str(x).split()[0] )
        cc['platform'] = cc.corpus.apply(lambda x: str(x).split()[-1] )
        n_corps = cc['corpus'].nunique()

        len_corpora = {}
        cc['nwords_content'] = cc['content'].astype("str").str.split().map(len)
        len_corpora[cc['corpus'].iloc[0]] = cc.shape[0]
        cc_desc = cc.groupby('corpus')['nwords'].sum().reset_index()
        cc_desc.columns = ['corpus', '#-words']#, 'avg-#-words'
        cc_desc_c = cc.groupby('corpus')['nwords_content'].sum().reset_index()
        cc_desc_c.columns = ['corpus', '#-content words']#, 'avg-#-content words'
        cc_desc = cc_desc.merge(cc_desc_c, on = 'corpus')

        cc.source = cc.source.astype('str')
        cc_src = cc.groupby('corpus', as_index = False)['source'].nunique()
        cc_src.columns = ['corpus', '# speakers']
        cc.Target = cc.Target.astype('str')
        cc_trg = cc[cc.Target != 'nan'].groupby('corpus', as_index = False)['Target'].nunique()
        cc_trg.columns = ['corpus', '# targets']

        cc_s = cc.groupby('corpus', as_index = False).size()
        cc_desc = cc_desc.merge(cc_s, on = 'corpus')

        cc_desc = cc_desc.merge(cc_src, on = 'corpus')
        cc_desc = cc_desc.merge(cc_trg, on = 'corpus')

        cc_ed = cc.groupby('corpus', as_index = False)[['ethos density', 'E+', 'E-']].mean()
        cc_ed[['ethos density', 'E+', 'E-']] = cc_ed[['ethos density', 'E+', 'E-']] * 100
        cc_desc = cc_desc.merge(cc_ed, on = 'corpus')

        #print(cc_desc.columns)
        # 'corpus', 'sum-#-words', 'avg-#-words', 'sum-#-content words', 'avg-#-content words', '# speakers', '# targets'
        #cc_desc.loc[len(cc_desc)] = ['TOTAL', cc_desc['sum-#-words'].sum(), cc.nwords.mean(),
        #                        cc_desc['sum-#-content words'].sum(), cc.nwords_content.mean(),
        #                        cc['source'].nunique(), cc['Target'].nunique(),
        #                        cc['ethos density'].mean()* 100, cc['E+'].mean()* 100, cc['E-'].mean()* 100]
        #cc_desc.loc[len(cc_desc)] = ['AVERAGE', cc_desc['sum-#-words'].sum(), cc.nwords.mean(),
        #                        cc_desc['sum-#-content words'].sum(), cc.nwords_content.mean(),
        #                        cc['source'].nunique(), cc['Target'].nunique(),
        #                        cc['ethos density'].mean()* 100, cc['E+'].mean()* 100, cc['E-'].mean()* 100]

        cc_desc.loc[len(cc_desc)] = ['AVERAGE', cc_desc['#-words'].mean(),
                                cc_desc['#-content words'].mean(), cc_desc['size'].mean(),
                                cc_desc['# speakers'].mean(), cc_desc['# targets'].mean(),
                                cc['ethos density'].mean()* 100, cc['E+'].mean()* 100, cc['E-'].mean()* 100]
        cc_desc = cc_desc.round(1)
        cc_desc.loc[len(cc_desc)] = ['TOTAL', cc_desc['#-words'].iloc[:int(n_corps)].sum(),
                                cc_desc['#-content words'].iloc[:int(n_corps)].sum(),
                                cc_desc['size'].iloc[:int(n_corps)].sum().round(0),
                                cc['source'].nunique(), cc['Target'].nunique(),
                                ' n/a ', ' n/a ', ' n/a ']
        #st.write(cc_ed)
        cc_desc[['#-words', '#-content words', '# speakers', '# targets', 'size']] = cc_desc[['#-words', '#-content words', '# speakers', '# targets', 'size']].astype('int')
        #st.write(cc_desc)
        #add_spacelines(2)
        cc_desc2 = SumCorpEthosTable(dataframe = cc, group_column = 'corpus')
        st.write(cc_desc2)

        # download button 2 to download dataframe as xlsx
        import io
        buffer = io.BytesIO()
        @st.cache_data
        def convert_to_csv(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv(index=False, sep = '\t').encode('utf-8')

        csv_cc_desc2 = convert_to_csv(cc_desc2)
        # download button 1 to download dataframe as csv
        download1 = st.download_button(
            label="Download data as TSV",
            data=csv_cc_desc2,
            file_name='summary_df_corpus.tsv',
            mime='text/csv'
        )

        add_spacelines(2)

        cc_desc_top = SumCorpEthosTable(dataframe = cc, group_column = 'topic')
        st.write(cc_desc_top)

        csv_cc_desc3 = convert_to_csv(cc_desc_top)
        # download button 1 to download dataframe as csv
        download1 = st.download_button(
            label="Download data as TSV",
            data=csv_cc_desc3,
            file_name='summary_df_topic.tsv',
            mime='text/csv'
        )

        add_spacelines(2)

        cc_desc_platform = SumCorpEthosTable(dataframe = cc, group_column = 'platform')
        st.write(cc_desc_platform)
        add_spacelines(2)

        csv_cc_desc4 = convert_to_csv(cc_desc_platform)
        # download button 1 to download dataframe as csv
        download1 = st.download_button(
            label="Download data as TSV",
            data=csv_cc_desc4,
            file_name='summary_df_platform.tsv',
            mime='text/csv'
        )


    else:
        rels = ['Default Conflict', 'Default Rephrase', 'Default Inference',
                'Logos Neutral', 'Logos Attack', 'Logos Rephrase', 'Logos Support']
        cc = pd.concat( summary_corpora_list[1::2], axis = 0, ignore_index = True )
        cc_raw = pd.concat( summary_corpora_list_raw, axis = 0, ignore_index = True )
        cc_len = pd.concat( summary_corpora_list_raw_len, axis = 0, ignore_index = True )
        cc_raw['locution'] = cc_raw['locution'].apply(lambda x: ":".join( str(x).split(":")[1:] ))
        cce = pd.concat( summary_corpora_list[::2], axis = 0, ignore_index = True )
        cce['ethos density'] = np.where(cce.ethos_label != 0, 1, 0)
        cce['E+'] = np.where(cce.ethos_label == 1, 1, 0)
        cce['E-'] = np.where(cce.ethos_label == 2, 1, 0)
        cc_eth = cce.groupby('corpus', as_index = False)[['ethos density','E+','E-']].mean()
        cc_eth[['ethos density','E+','E-']] = cc_eth[['ethos density','E+','E-']].round(3)*100

        cc['RepSp'] = 0
        cc['locution_premise'] = cc['locution_premise'].apply(lambda x: ":".join( str(x).split(":")[1:] ))
        cc['locution_conclusion'] = cc['locution_conclusion'].apply(lambda x: ":".join( str(x).split(":")[1:] ))
        cc['locution'] = cc['locution_premise'].astype('str') + " " + cc['locution_conclusion'].astype('str')
        #st.write(cc_len)
        cc_raw['connection'] = 'Logos Neutral'
        cc_raw['RepSp'] = np.where(cc_raw[['RepSp_int', 'RepSp_ext']].any(axis=1), 1, 0)
        cc.connection = cc.connection.map({'Default Conflict': 'Logos Attack',
                                        'Default Rephrase' : 'Logos Rephrase', 'Default Inference' : 'Logos Support'})

        cc = pd.concat( [cc, cc_raw], axis = 0, ignore_index = True )
        cc['nwords'] = cc.locution.astype('str').str.split().map(len)
        #st.write(cc)

        cc = cc[cc.connection.isin(rels)]
        cc_desc = pd.DataFrame( cc.groupby('corpus')['connection'].value_counts(normalize=True).round(3)*100 )
        cc_desc = cc_desc.rename(columns = {'connection':'%'})
        cc_desc = cc_desc.reset_index()

        cc_desc = cc_desc.pivot(index = 'corpus', columns = 'connection', values = '%')
        cc_descrs = cc.groupby('corpus').RepSp.mean().round(3)*100
        cc_descrs = cc_descrs.reset_index()
        cc_desc = cc_desc.merge(cc_descrs, on = 'corpus')

        cc_desc = cc_desc.merge(cc_eth, on = 'corpus')

        cc_descnw = cc_len.groupby('corpus', as_index = False).nwords.sum()
        cc_descnw.columns = ['corpus', '#-words']
        cc_desc = cc_desc.merge(cc_descnw, on = 'corpus')

        cc_sz = pd.DataFrame({'corpus':summary_corpora_dict_raw.keys(), 'size':summary_corpora_dict_raw.values()})
        cc_desc = cc_desc.merge(cc_sz, on = 'corpus')

        av = ['AVERAGE']
        av.extend(cc_desc.mean(axis=0).values)
        cc_desc.loc[len(cc_desc)] = av
        cc_desc = cc_desc.round(1)

        cc_desc.loc[len(cc_desc)] = ['TOTAL', ' n/a ', ' n/a ', ' n/a ', ' n/a ', ' n/a ', ' n/a ', ' n/a ', ' n/a ',
                                    cc_desc['#-words'].iloc[:int(len(summary_corpora_list[1::2]))].sum(),
                                    cc_desc['size'].iloc[:int(len(summary_corpora_list[1::2]))].sum() ]

        cc_desc = cc_desc.fillna(' n/a ')
        cc_desc[['size', '#-words']] = cc_desc[['size', '#-words']].astype('int')
        cc_desc = cc_desc.round(1)
        st.write(cc_desc)

        add_spacelines(2)
        # download button 2 to download dataframe as xlsx
        import io
        buffer = io.BytesIO()
        @st.cache_data
        def convert_to_csv(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv(index=False, sep = '\t').encode('utf-8')

        csv_cc_desc = convert_to_csv(cc_desc)
        # download button 1 to download dataframe as csv
        download1 = st.download_button(
            label="Download data as TSV",
            data=csv_cc_desc,
            file_name='summary_df.tsv',
            mime='text/csv'
        )



elif not contents_radio_rhetoric_category_ethos and contents_radio_rhetoric_category_logos and contents_radio3 == 'Distribution':
    st.write("### Compare distributions")
    add_spacelines(2)
    distribution_plot_compare_logos(data_list = corpora_list, an_type = contents_radio_an_cat)


elif not contents_radio_rhetoric_category_ethos and contents_radio_type == 'Compare Corpora' and contents_radio_rhetoric_category_logos and contents_radio3 == 'Odds Ratio':
    st.write("### Lexical Analysis - Odds Ratio")
    add_spacelines(2)
    rhetoric_dims = ['ethos', 'logos']
    selected_rhet_dim = st.selectbox("Choose a rhetoric strategy for analysis", rhetoric_dims, index=0)
    selected_rhet_dim = selected_rhet_dim.replace('ethos', 'ethos_label').replace('logos', 'logos_label')
    corpora_list_names1 = corpora_list[0].corpus.iloc[0]
    corpora_list_names2 = corpora_list[-1].corpus.iloc[0]

    add_spacelines(1)
    OddsRatioLog_comparec1, OddsRatioLog_comparec2 = st.tabs([corpora_list_names1, corpora_list_names2])
    with OddsRatioLog_comparec1:
        OddsRatioLog_compare(corpora_list[:2], selected_rhet_dim = selected_rhet_dim, an_type = contents_radio_an_cat)
    with OddsRatioLog_comparec2:
        OddsRatioLog_compare(corpora_list[-2:], selected_rhet_dim = selected_rhet_dim, an_type = contents_radio_an_cat)


elif not contents_radio_rhetoric_category_ethos and contents_radio_rhetoric_category_logos and contents_radio3 == 'Odds Ratio':
    OddsRatioLog(corpora_list, an_type = contents_radio_an_cat)

elif contents_radio_type != 'Compare Corpora' and contents_radio_rhetoric_category_logos and contents_radio3 == 'Frequency Tables':
    FreqTablesLog(corpora_list)

elif not contents_radio_rhetoric_category_ethos and contents_radio_type == 'Compare Corpora' and contents_radio_rhetoric_category_logos and contents_radio3 == 'Lexical Stats':
    StatsLog_compare(corpora_list, an_type = contents_radio_an_cat)


elif not contents_radio_rhetoric_category_ethos and contents_radio_rhetoric_category_logos and contents_radio3 == 'Lexical Stats':
    StatsLog(corpora_list, an_type = contents_radio_an_cat)



elif contents_radio_rhetoric_category_ethos and contents_radio_type == 'Single Corpus' and contents_radio3 == "Fellows-Devils":
    #FellowsDevils(corpora_list)
    FellowsDevils_new(corpora_list)

elif contents_radio_rhetoric_category_ethos and contents_radio_type == 'Single Corpus' and contents_radio3 == "Polarising Tendency":
    UserProfileResponse(corpora_list)



elif contents_radio_rhetoric_category_ethos and contents_radio_type == 'Single Corpus' and contents_radio3 == 'Profiles':
    st.write("### Profiles")
    add_spacelines(2)
    rhetoric_dims = ['ethos', 'sentiment'] # 'pathos',
    selected_rhet_dim = st.selectbox("Choose a rhetoric category of profiles", rhetoric_dims, index=0)
    add_spacelines(1)
    if selected_rhet_dim != 'logos':
        selected_rhet_dim = selected_rhet_dim.replace("ethos", "ethos_label").replace("pathos", "pathos_label")
    ProfilesEntity_compare(data_list = corpora_list, selected_rhet_dim = selected_rhet_dim)

elif contents_radio_type == 'Compare Corpora' and contents_radio_rhetoric_category_logos and contents_radio3 == 'Profiles':
    st.write("### Profiles")
    rhetoric_dims = ['ethos']
    add_spacelines(2)
    selected_rhet_dim = st.selectbox("Choose a rhetoric category of profiles", rhetoric_dims, index=0)
    add_spacelines(1)
    if selected_rhet_dim != 'logos':
        selected_rhet_dim = selected_rhet_dim.replace("ethos", "ethos_label").replace("pathos", "pathos_label")

    if len(corpora_list) == 6:
        cols_columns1, cols_columns2, cols_columns3 = st.tabs([corpora_list[0].corpus.iloc[0], corpora_list[2].corpus.iloc[0], corpora_list[-2].corpus.iloc[0]])
        with cols_columns1:
            #st.write(corpora_list[0])
            ProfilesEntity_compare(data_list = corpora_list[:1], selected_rhet_dim = selected_rhet_dim)
        with cols_columns2:
            ProfilesEntity_compare(data_list = corpora_list[2:3], selected_rhet_dim = selected_rhet_dim)
        with cols_columns3:
            ProfilesEntity_compare(data_list = corpora_list[-2:-1], selected_rhet_dim = selected_rhet_dim)

    elif len(corpora_list) == 4:
        cols_columns1, cols_columns2 = st.tabs([corpora_list[0].corpus.iloc[0], corpora_list[-2].corpus.iloc[0]])
        with cols_columns1:
            #st.write(corpora_list[0])
            ProfilesEntity_compare(data_list = corpora_list[:1], selected_rhet_dim = selected_rhet_dim)
        with cols_columns2:
            ProfilesEntity_compare(data_list = corpora_list[-2:-1], selected_rhet_dim = selected_rhet_dim)


elif not contents_radio_rhetoric_category_ethos and contents_radio_type == 'Compare Corpora' and contents_radio_rhetoric_category_logos and contents_radio3 == '(Anti)-heroes':
    if len(corpora_list) == 6:
        corpora_list0 = corpora_list[0]
        corpora_list1 = corpora_list[2]
        corpora_list2 = corpora_list[-2]
        corpora_list = [corpora_list0, corpora_list1, corpora_list2]
    elif len(corpora_list) == 4:
        corpora_list0 = corpora_list[0]
        corpora_list1 = corpora_list[-2]
        corpora_list = [corpora_list0, corpora_list1]
    elif len(corpora_list) == 2:
        corpora_list0 = corpora_list[0]
        corpora_list = corpora_list0
    TargetHeroScores_compare(data_list = corpora_list, singl_an = False)

elif not contents_radio_rhetoric_category_ethos and contents_radio_type != 'Compare Corpora' and contents_radio_rhetoric_category_logos and contents_radio3 == 'Profiles':
    corpora_list_ent = []
    df_user_et = corpora_list[0]
    df_user_log = corpora_list[1]
    #st.write(df_user_log)
    df_user_et_src = set( df_user_et.source.dropna().astype('str').str.replace("@", "").str.strip().unique() )
    df_user_log_src = set( df_user_log.speaker_premise.dropna().astype('str').str.replace(":", "").str.replace("@", "").str.strip().unique() )
    df_user_src = df_user_et_src.intersection(df_user_log_src)

    df_user_et = df_user_et[df_user_et.source.dropna().astype('str').str.replace("@", "").str.strip().isin(df_user_src) ]
    df_user_log = df_user_log[df_user_log.speaker_premise.dropna().astype('str').str.replace(":", "").str.replace("@", "").str.strip().isin(df_user_src) ]
    df_ents = pd.concat([df_user_log, df_user_et], axis = 0, ignore_index = True)

    src_logp = df_ents.groupby('speaker_premise', as_index=False).size()
    #src_logp = src_logp[src_logp['size'] > 2]
    src_logp = df_ents.speaker_premise.dropna().astype('str').str.strip().unique()
    src_et = df_ents.groupby('source', as_index=False).size()
    #src_et = src_et[src_et['size'] > 2]
    src_et = src_et.source.dropna().astype('str').str.strip().unique()
    src_list = list( set(src_et).union(set(src_logp) ) )
    src_list = list(set(str(e).replace("@", "") for e in src_list))
    if 'look' in src_list:
        src_list.remove('look')
    st.write("### Speaker Analysis")
    add_spacelines(2)
    src = st.selectbox("Choose an entity for analysis", src_list, index=0)

    df_user_et.source = df_user_et.source.dropna().astype('str').str.replace("@", "").str.strip()
    df_user_log.speaker_premise = df_user_log.speaker_premise.dropna().astype('str').str.replace(":", "").str.replace("@", "").str.strip()
    df_user_et = df_user_et[df_user_et.source == src]
    df_user_log = df_user_log[(df_user_log.speaker_premise == src) ] #  (df_user_log.speaker_conclusion == src) |

    try:
        ds = df_user_et.corpus.iloc[0]
    except:
        ds = df_user_log.corpus.iloc[0]
    ds = ds + " - **" + str(src) +"**"
    df_user_et.corpus = ds
    df_user_log.corpus = ds
    if len(df_user_et) > 0:
        corpora_list_ent.append(df_user_et)
    if len(df_user_log) > 0:
        corpora_list_ent.append(df_user_log)
    distribution_plot_compare_logos(data_list = corpora_list_ent, an_type = contents_radio_an_cat)



elif not contents_radio_rhetoric_category_ethos and contents_radio_rhetoric_category_logos and contents_radio3 == 'WordCloud' and contents_radio_type == 'Compare Corpora':
    #generateWordCloud_log(corpora_list, rhetoric_dims = ['ethos', 'logos'], an_type = contents_radio_an_cat)
    rhetoric_dims = ['ethos', 'logos']
    selected_rhet_dim = st.selectbox("Choose a rhetoric category for a WordCloud", rhetoric_dims, index=0)
    add_spacelines(1)

    if selected_rhet_dim == 'pathos':
        label_cloud = st.radio("Choose a label for words in WordCloud", ('negative', 'positive'))
        selected_rhet_dim = selected_rhet_dim.replace("ethos", "ethos_label").replace("pathos", "pathos_label")
        label_cloud = label_cloud.replace("negative", "attack").replace("positive", "support")
    else:
        label_cloud = st.radio("Choose a label for words in WordCloud", ('attack', 'support'))
        selected_rhet_dim = selected_rhet_dim.replace("ethos", "ethos_label")
        label_cloud = label_cloud.replace("attack / negative", "attack").replace("support / positive", "support")

    add_spacelines(1)
    threshold_cloud = st.slider('Select a precision value (threshold) for words in WordCloud', 0, 100, 80)
    st.info(f'Selected precision: **{threshold_cloud}**')
    add_spacelines(1)
    st.write("**Processing the output ...**")

    if 'ethos' in selected_rhet_dim:
        corpora_list = corpora_list[::2]
    elif 'logos' in selected_rhet_dim:
        corpora_list = corpora_list[1::2]

    cols_columns = st.columns(len(corpora_list))
    dict_cond = {}
    for n, c in enumerate(cols_columns):
        with c:
            add_spacelines(1)
            generateWordCloud_sub_log(corpora_list[n], rhetoric_dims = ['ethos', 'logos'], an_type = contents_radio_an_cat,
                selected_rhet_dim = selected_rhet_dim, label_cloud=label_cloud, threshold_cloud=threshold_cloud)
    #with generateWordCloudc2:
        #st.write(f"##### {corpora_list[-1].corpus.iloc[0]}")
        #add_spacelines(1)
        #generateWordCloud_sub_log(corpora_list[1::2], rhetoric_dims = ['ethos', 'logos'], an_type = contents_radio_an_cat,
            #selected_rhet_dim = selected_rhet_dim, label_cloud=label_cloud, threshold_cloud=threshold_cloud)


elif not contents_radio_rhetoric_category_ethos and contents_radio_rhetoric_category_logos and contents_radio3 == 'WordCloud':
    generateWordCloud(corpora_list, rhetoric_dims = ['ethos', 'logos'], an_type = contents_radio_an_cat)





elif contents_radio_type == 'Single Corpus' and contents_radio3 == 'WordCloud':
    st.write("### WordCloud - High Precision Words")
    add_spacelines(1)
    generateWordCloud(corpora_list)


elif contents_radio_type == 'Single Corpus' and contents_radio3 == 'Pronouns':
    PronousLoP(corpora_list)

elif contents_radio_type == 'Single Corpus' and contents_radio3 == 'Frequency Tables':
    FreqTables(corpora_list)

elif contents_radio_type == 'Single Corpus' and contents_radio3 == 'Odds ratio':
    OddsRatio(corpora_list)

elif contents_radio_type == 'Single Corpus' and contents_radio3 == 'Explore corpora':
    #st.dataframe(df)
    df = corpora_list[0]
    #st.write(df)
    st.write('### Explore corpora')
    dff_columns = [ 'sentence', 'source', 'ethos_label', 'emotion', 'sentiment', 'Target' ]# , 'conversation_id','date', 'pathos_label'

    if not 'neutral' in df['ethos_label'].unique():
        df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
    if not 'neutral' in df['pathos_label'].unique():
        df['pathos_label'] = df['pathos_label'].map(valence_mapping)
    dff = df.copy()
    dff['sentence'] = dff['sentence'].astype('str').str.replace('amp;', '')
    select_columns = st.multiselect("Choose columns for specifying conditions", dff_columns, dff_columns[-4:-2])
    cols_columns = st.columns(len(select_columns))
    dict_cond = {}
    for n, c in enumerate(cols_columns):
        with c:
            cond_col = st.multiselect(f"Choose condition for *{select_columns[n]}*",
                                   (dff[select_columns[n]].unique()), (dff[select_columns[n]].unique()[-1]))
            dict_cond[select_columns[n]] = cond_col
    dff_selected = dff.copy()
    for i, k in enumerate(dict_cond.keys()):
        dff_selected = dff_selected[ dff_selected[str(k)].isin(dict_cond[k]) ]
    add_spacelines(2)
    st.dataframe(dff_selected[dff_columns].set_index("source"), width = None)
    st.write(f"No. of cases: {len(dff_selected)}.")


elif contents_radio_type in ['Compare Corpora', 'Single Corpus']:
    if contents_radio_type == 'Single Corpus':
        if contents_radio3 == '(Anti)-heroes':
            TargetHeroScores_compare(data_list = corpora_list)
        elif contents_radio3 == 'Rhetoric Strategies':
            UserRhetStrategy(data_list = corpora_list)
        else:
            st.write(corpora_list)
            distribution_plot_compareX(data_list = corpora_list)


    elif contents_radio_type == 'Compare Corpora':
        if contents_radio3 == '(Anti)-heroes':
            TargetHeroScores_compare(data_list = corpora_list)

        elif contents_radio3 == 'WordCloud':
            st.write('### Compare WordCloud')
            data_list = corpora_list.copy()

            add_spacelines(1)
            up_data_dict = {}
            texts_all_sup = []
            texts_all_att = []
            texts_all_pos = []
            texts_all_neg = []

            txt_sup = ''
            txt_att = ''
            txt_pos = ''
            txt_neg = ''

            n = 0

            for data in data_list:
                df = data.copy()
                ds = df['corpus'].iloc[0]

                df = clean_text(df, 'sentence_lemmatized', text_column_name = "sentence_lemmatized")
                #df = lemmatization(df, 'content')

                if not 'negative' in df['pathos_label'].unique():
                    df['pathos_label'] = df['pathos_label'].map(valence_mapping)
                if not 'neutral' in df['ethos_label'].unique():
                    df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
                up_data_dict[n] = df
                n += 1

                text1 = df[df.ethos_label == 'support']['sentence_lemmatized'].values
                text1 = " ".join(text1).lower().replace(' amp ', ' ').replace(' url ', ' ')
                #text1 = " ".join([w for w in text1.split() if text1.split().count(w) > 1])
                txt_sup += text1
                text1 = set(text1.split())
                texts_all_sup.append(text1)
                text2 = df[df.ethos_label == 'attack']['sentence_lemmatized'].values
                text2 = " ".join(text2).lower().replace(' amp ', ' ').replace(' url ', ' ')
                #text2 = " ".join([w for w in text2.split() if text2.split().count(w) > 1])
                txt_att += text2
                text2 = set(text2.split())
                texts_all_att.append(text2)

                text11 = df[df.pathos_label == 'positive']['sentence_lemmatized'].values
                text11 = " ".join(text11).lower().replace(' amp ', ' ').replace(' url ', ' ')
                #text11 = " ".join([w for w in text11.split() if text11.split().count(w) > 1])
                txt_pos += text11
                text11 = set(text11.split())
                texts_all_pos.append(text11)
                text22 = df[df.pathos_label == 'negative']['sentence_lemmatized'].values
                text22 = " ".join(text22).lower().replace(' amp ', ' ').replace(' url ', ' ')
                #text22 = " ".join([w for w in text22.split() if text22.split().count(w) > 1])
                txt_neg += text22
                text22 = set(text22.split())
                texts_all_neg.append(text22)

            #shared ethos
            shared_all_sup = []
            shared_all_att = []
            shared_all_pos = []
            shared_all_neg = []

            shared_sup = texts_all_sup[0]
            shared_att = texts_all_att[0]
            shared_pos = texts_all_pos[0]
            shared_neg = texts_all_neg[0]

            for n in range(int(len(data_list))-1):
                shared1 = shared_sup.intersection(texts_all_sup[n+1])
                shared_all_sup.extend(list(shared1))

                shared11 = shared_att.intersection(texts_all_att[n+1])
                shared_all_att.extend(list(shared11))

                shared2 = shared_pos.intersection(texts_all_pos[n+1])
                shared_all_pos.extend(list(shared2))

                shared22 = shared_neg.intersection(texts_all_neg[n+1])
                shared_all_neg.extend(list(shared22))


            textatt_a = " ".join(shared_all_att)
            for word11 in shared_all_att:
                word_cnt = txt_att.split().count(str(word11))
                word_cnt_l = " ".join([str(word11)] * int(word_cnt))
                textatt_a += word_cnt_l
            textatt_a = textatt_a.split()
            textatt_a = [w for w in textatt_a if txt_att.split().count(w) > 4]
            random.shuffle(textatt_a)
            if len(textatt_a) < 1:
                textatt_a = ['empty', 'empty']
                textatt_a = " ".join(textatt_a)
            elif len(textatt_a) == 1:
                textatt_a = str(textatt_a[0])
            else:
                textatt_a = " ".join(textatt_a)

            textpos_a = " ".join(shared_all_pos)
            for word2 in shared_all_pos:
                word_cnt = txt_pos.split().count(str(word2))
                word_cnt_l = " ".join([str(word2)] * int(word_cnt))
                textpos_a += word_cnt_l
            textpos_a = textpos_a.split()
            textpos_a = [w for w in textpos_a if txt_pos.split().count(w) > 4]
            random.shuffle(textpos_a)
            if len(textpos_a) < 1:
                textpos_a = ['empty', 'empty']
                textpos_a = " ".join(textpos_a)
            elif len(textpos_a) == 1:
                textpos_a = str(textpos_a[0])
            else:
                textpos_a = " ".join(textpos_a)

            textneg_a = " ".join(shared_all_neg)
            for word in shared_all_neg:
                word_cnt = txt_neg.split().count(str(word))
                word_cnt_l = " ".join([str(word)]* int(word_cnt))
                textneg_a += word_cnt_l
            textneg_a = textneg_a.split()
            textneg_a = [w for w in textneg_a if txt_neg.split().count(w) > 4]
            random.shuffle(textneg_a)
            if len(textneg_a) < 1:
                textneg_a = ['empty', 'empty']
                textneg_a = " ".join(textneg_a)
            elif len(textneg_a) == 1:
                textneg_a = str(textneg_a[0])
            else:
                textneg_a = " ".join(textneg_a)

            textsup_a = shared_all_sup.copy()
            for word in shared_all_sup:
                word_cnt = txt_sup.split().count(str(word))
                word_cnt_l = [str(word)] * int(word_cnt)
                textsup_a.extend(word_cnt_l)
            #textsup_a = textsup_a.split()
            textsup_a = [w for w in textsup_a if txt_sup.split().count(w) > 4]
            #random.shuffle(textsup_a)
            if len(textsup_a) < 1:
                textsup_a = ['empty', 'empty']
                textsup_a = " ".join(textsup_a)
            elif len(textsup_a) == 1:
                if len(str(textsup_a[0])) > 15:
                    textsup_a = str(textsup_a[0])[:15]
                else:
                    textsup_a = str(textsup_a[0])
            else:
                textsup_a = " ".join(textsup_a)


            f_sup, words_sup = make_word_cloud(textsup_a, 800, 500, '#1E1E1E', 'Greens')
            f_att, words_att = make_word_cloud(textatt_a, 800, 500, '#1E1E1E', 'Reds')
            f_pos, words_pos = make_word_cloud(textpos_a, 800, 500, '#1E1E1E', 'Greens')
            f_neg, words_neg = make_word_cloud(textneg_a, 800, 500, '#1E1E1E', 'Reds')

            col1_cl_et, col2_cl_et = st.columns(2, gap='large')# [5, 4]
            with col1_cl_et:
                st.write('Shared **ethos support** words')
                st.pyplot(f_sup)
                st.write(f"There are {len(set(textsup_a.split()))} shared words.")

                add_spacelines(1)
                st.write('Shared **pathos positive** words')
                st.pyplot(f_pos)
                st.write(f"There are {len(set(textpos_a.split()))} shared words.")

            with col2_cl_et:
                st.write('Shared **ethos attack** words')
                st.pyplot(f_att)
                st.write(f"There are {len(set(textatt_a.split()))} shared words.")

                add_spacelines(1)
                st.write('Shared **pathos negative** words')
                st.pyplot(f_neg)
                st.write(f"There are {len(set(textneg_a.split()))} shared words.")
            add_spacelines(2)

            st.write("#### Cases")
            add_spacelines(1)
            exp_data = up_data_dict[0]
            for n in range(int(len(data_list))-1):
                exp_data = pd.concat([exp_data, up_data_dict[n+1]], axis=0, ignore_index=True)

            look_dict_exp = {
            "ethos support":textsup_a, "ethos attack":textatt_a,
            "pathos positive":textpos_a, "pathos negative":textneg_a}

            for exp_cat in ["ethos support", "ethos attack", "pathos positive", "pathos negative"]:
                with st.expander(exp_cat):
                    cols_all = ['sentence', 'source', 'Target', 'ethos_label', 'pathos_label', 'corpus']
                    c1 = f"shared_{exp_cat.split()[-1]}_words"
                    c1 = str(c1)
                    c2 = f"shared_{exp_cat.split()[-1]}"
                    c2 = str(c2)
                    cols_all.append(c1)
                    #cols_all.append(c2)

                    exp_sup = exp_data.copy()
                    exp_sup[c1] = exp_sup.sentence_lemmatized.apply(lambda x: set(x.split()).intersection(set(look_dict_exp[exp_cat].split())))
                    exp_sup[c2] = exp_sup[c1].map(len)
                    exp_sup = exp_sup[exp_sup[c2] > 0]

                    st.dataframe(exp_sup[cols_all], width = None)
            st.stop()

        else:
            distribution_plot_compare(data_list = corpora_list)
