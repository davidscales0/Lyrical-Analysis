"""
David Scales
Professor Rush
DS2500
8/16/2024

Final Project- Playlist Algorithm using LyricsGenius API, sentiment analysis, 
clustering, WCSS, silhouette score, pandas, ect.
"""

import json
import pandas as pd
from lyricsgenius import Genius
import numpy as np
import time #Documentation reviewed outside class
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import PCA

#various imports are set, to be used

API_token = 'WfA8bl5UbERlW33extb6RsDZoX0vA1285B0od0H-skxY7joMvQq5XQvqP6hefXP2'

#API token for lyricsgenius API is defined as constant


def main():
    """
    Main function is defined. Only if needed, song_df and lyric_csv 
    is made. Establishing lyrics directly as a df leads to API over-request
    issues, if made as csv, only needs to be run once.
    
    song_lyric_df reads lyric_data_full_rev(revised).csv, to load lyric
    data
    
    song_lyric_df calls find_sent_scores, to add columns for sentiment scores
    for each song
    
    X is established as independent variable for various KMean uses, using
    sentiment results.
    
    find_plot_elbow and find_plot_silh excecute functions to plot find best
    k value
    
    song_lyric_df calls find_clusters, using predefined X, and best k=4, to
    add clusters to df.
    
    Test run of input songs is run, and printed.
    """
    
    
    run = False
    if run == True:
        song_df = make_song_df()
        make_lyric_csv(song_df)

    song_lyric_df = (pd.read_csv('lyric_data_full_rev.csv')).dropna()

    song_lyric_df = find_sent_scores(song_lyric_df)
    
    X = song_lyric_df[['Neg', 'Neu', 'Pos']]
    
    find_plot_elbow(X)

    find_plot_silh(X)


    song_lyric_df = find_clusters(X, song_lyric_df, 4)
    
    

    input_songs = ['Electric Relaxation', 'Evil Ways', 'Brown Sugar', 'Wave Gods', 'Violet']
    song_recs = find_song_recs(input_songs, song_lyric_df)
    
    print(input_songs)
    print(song_recs)
    
    find_plot_pca(song_lyric_df)
    
def make_song_df():
    """
    Function designed to construct the song_df, with columns for artist name,
    and song name. Function draws data from 7 files recieved from Spotify
    about my music listening instances.
    
    Program appends these into one dataset, and sums up all listening 
    instances per song.
    
    Songs without 30 minutes of playtime are removed, to reduce strain on API
    and distinguish between once-off or so listens.
    
    Various special characters are removed, due to issues with writing into
    json files, in other functions.
    
    index in reset, for organizational purposes, and song_df is returned
    """

    dfs = []
    x = 0
    while x < 7:
        file = 'StreamingHistory_music_' + str(x) + '.json'
        df = pd.read_json(file)
        dfs.append(df)
        x += 1  
    
    song_df = pd.concat(dfs, ignore_index=True)
    song_df = song_df.groupby(['artistName', 'trackName'], as_index=False)['msPlayed'].sum()
    
    song_df = song_df[song_df['msPlayed'] >  1800000]
    
    song_df['trackName'] = song_df['trackName'].str.replace(r'\s*\(.*?\)', '', regex=True)
    df['artistName'] = df['artistName'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

    song_df = song_df.reset_index()
    
    return song_df


def lyric_find(song, artist):
    """ 
    Function designed to interact with LyricsGenius API to source lyric data
    for songs in list.
    
    Function uses API to search for given song and artist, checking to see
    if the file is empty (ex. instrumental song). Due to frequent timeouts,
    timeout limit set to 240s.
    
    Again, for issues with writing into json file, song and artist name are
    cleaned, and song lyrics are written into a json file, and saved.
    
    Filename is returned for further use
    """
    
    genius = Genius(API_token, timeout=240)
    song_file = genius.search_song(song, artist)

    if song_file is None or not song_file.lyrics:  
        return np.nan  

    song = clean_text(song)
    artist = clean_text(artist)

    filename = artist + song + '.json'
    song_file.save_lyrics(filename=filename, overwrite=True)
    
    return filename

def make_lyric_csv(song_df):
    """ 
    Designed to be used in conjunction with lyric_find, and for creating the
    csv file that serves as the basis of the program.
    
    Defines song_lyric_df, using song_df's artist and song columns, while 
    adding a blank column for lyrics.
    
    Due to over request issues with API, batches of data, and a wait time (Using
    time module) are used.
    
    For each song in the dataset, API is called using lyric_find function,
    song and artist name are cleaned, and json file written by lyric_find
    is opened, and added respectively to song_lyric_df's Lyrics column.
    
    After every batch, the relevant csv file is saved, just in case the API
    request limit is met during the excecution of the loop.
    """
    
    song_lyric_df = pd.DataFrame()
    song_lyric_df['Artist'] = song_df['artistName']
    song_lyric_df['Song'] = song_df['trackName']
    song_lyric_df['Lyrics'] = [None] * len(song_df)

    for n in range(0, len(song_df), 50):
        end = min(n + 50, len(song_df))
        for x in range(n, end):
            song = song_df['trackName'][x]
            artist = song_df['artistName'][x]
            
            filename = lyric_find(song, artist) 
            
            if type(filename) != float:
                
                song = clean_text(song)
                artist = clean_text(artist)
                song_lyric_df.at[x, 'Lyrics'] = read_json(filename)['lyrics']
            else:
                song_lyric_df.at[x, 'Lyrics'] = np.nan  
        
                
        song_lyric_df.to_csv('lyric_data_rev.csv', index=False)
        
        time.sleep(60)


def find_song_recs(input_songs, song_lyric_df):
    """ 
    Using set input songs, function finds each song's assigned cluster, and
    finds every song in the same cluster besides itself. .sample function (
    built in python function), to choose a relatively random random option.
    A list of all of these song_recs is returned
    """
    
    song_recs = []
    
    for song_title in input_songs:
        song_cluster = song_lyric_df[song_lyric_df['Song'] == song_title]['Cluster'].values[0]
        
        similar_songs = song_lyric_df[(song_lyric_df['Cluster'] == song_cluster) & 
                                      (song_lyric_df['Song'] != song_title)]
        
        song_rec = similar_songs.sample(1)
        song_recs.append(song_rec[['Song', 'Artist']])
        
    return song_recs




def find_plot_elbow(X):
    """
    Function used to find WCSS (inertia), and plot accordingly.
    
    For k values from 1-10, kmeans is defined and fit with inputted X data.
    Note KMeans function isn't used, as data is not predicting, merely fitting
    KMeans model. 
    
    inertia is found using Sickit-Learn's .inertia feature, and appended to 
    WCSS list.
    
    WCSS list is plotted against k values, and vertical axis line showing
    likely elbow point is marked.
    
    labeled, saved, shown
    """
    
    wcss = []
    
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.axvline(x=4, color='red', linestyle=':', linewidth=2, label='Elbow Point')
    plt.title('Amount of Clusters by WCSS for Determing Elbow Point')
    plt.xlabel('Amount of Clusters')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.legend()
    plt.savefig('elbow.png',bbox_inches='tight')
    plt.show()

def find_plot_silh(X):
    """
    Function used to find and plot silhouette scores.
    
    for k values 3-10, kmean function is sued and silhouette scores are 
    calculeted based on X, and the output data, y.
    
    Scores are plotted against k values, and the maximum value is labelled
    with a vertical axis line.
    
    labeled, saved, shown.
    """
    
    silhouette_scores = []
    
    for k in range(3, 11): 
        y = kmean(X, k)
        silhouette_scores.append(silhouette_score(X, y))
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(3, 11), silhouette_scores, marker='o')
    plt.axvline(x=4, color='red', linestyle=':', linewidth=2, label='Maximum')
    plt.title('Amount of Clusters by Silhouette Score')
    plt.xlabel('Amount of Clusters')
    plt.ylabel('Silhouette Score')
    plt.legend()
    plt.savefig('silhouette.png',bbox_inches='tight')
    plt.show()


def kmean(X, k):
    """
    Function designed for generalized use of KMeans, accepts k value, and 
    X, computes KMeans for random_state 0, and n_init=10 (10 is default value,
                                                          set for clarity)
    """
    
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    y = kmeans.fit_predict(X)
    
    return y

def find_clusters(X, df, k):
    """
    Function accepts independent variable X, a dataframe, and a k value. 
    a new column in the df is created based on KMeans clustering of X.
    """

    df['Cluster'] = kmean(X, k)

    return df    



def sentiment(text):
    """
    Function implements Vader sentiment analysis, returning neg, neu, and pos
    as three metrics based on inputted text.
    """
    
    analyzer = SentimentIntensityAnalyzer()

    vader_scores = analyzer.polarity_scores(text)
    
    return vader_scores


def find_sent_scores(song_lyric_df):
    """
    Function defines three new columns in song_lyric_df, Neg, Neu, and Pos,
    representing the three scores provided by sentiment function. Using a 
    while loop, scores for each song are calculated, and appended to these
    columns. New song_lyric_df is returned.
    """
    
    song_lyric_df['Neg'] = None
    song_lyric_df['Neu'] = None
    song_lyric_df['Pos'] = None


    x = 0
    
    while x < len(song_lyric_df['Song']):
        
        scores = sentiment(song_lyric_df['Lyrics'][x])
            
        song_lyric_df.loc[x, 'Neg'] = scores['neg']
        song_lyric_df.loc[x, 'Neu'] = scores['neu']
        song_lyric_df.loc[x, 'Pos'] = scores['pos']

        x += 1
        
    return song_lyric_df

def find_plot_pca(song_lyric_df):
    """
    Designed for performing PCA on data used in clustering, and plotting it
    accordingly.
    
    std_scl function is used to scale X data
    
    PCA model is defined and fit with scaled data, which is used to create
    a dataframe with columns PC1 and PC2, 'Principal Component 1', and 'Principal
    Component 2', the two outputs of the PCA.
    
    This is plotted with a colormap relating to relevant clusters.
    
    labelled, saved, shown
    """
    
    s_X = std_scl(song_lyric_df[['Neg', 'Neu', 'Pos']])
    
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(s_X)
    pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], c=song_lyric_df['Cluster'], marker='o')
    plt.title('Cluster Visualization from PCA, at k=4 Clusters')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(label='Cluster')
    plt.savefig('pca.png',bbox_inches='tight')
    plt.show()


def std_scl(X):
    """ 
    Function solely for use in PCA function, scales the inputted data by
    calculating mean, standard deviation, then computing and returning scaled data.
    """
    
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    s_X = (X - mean) / std
    
    return s_X
    
def clean_text(text):
    """
    for use in cleaning song and artists names, accepts text, removes
    special characters, and returns cleaned text
    """
    
    cleaned_text = text.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
    cleaned_text = cleaned_text.lower()
    return cleaned_text


def read_json(filename):
    """
    Solely for reading json files, uses json module to open and read file,
    returning enclosed data.
    """

    with open(filename) as file:
        data = json.load(file)

    return data






main()    

