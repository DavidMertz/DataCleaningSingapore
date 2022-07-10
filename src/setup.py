import os
import io
import sys
import gzip
import re
import sqlite3
import dbm
from glob import glob
from datetime import datetime, date, timedelta
from pprint import pprint
from math import nan, inf, pi as π, e
import math
from random import seed, choice, randint, sample
from collections import namedtuple
from collections import Counter
from itertools import islice
from textwrap import fill
from dataclasses import dataclass, astuple, asdict, fields
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import seaborn as sns

from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFECV, RFE

from IPython.display import Image as Show

# Do not display warnings
import warnings
warnings.simplefilter('ignore')

# Only show 8 rows from large DataFrame
pd.options.display.max_rows = 8
pd.options.display.min_rows = 8
        

# Utility function
def random_phone(reserved=True):
    digits = '0123456789'
    area = '000'
    while area.startswith('0'):
        area = ''.join(sample(digits, 3))
    # DO NOT REMOVE prefix code
    # it is not used now, but random seed assumes
    # the exact same sequence of calls
    prefix = '000'
    while prefix.startswith('0'):
        prefix = ''.join(sample(digits, 3))
    suffix = ''.join(sample(digits, 4))
    #-------------------------------------
    if reserved:
        prefix = "555"
    return f"+1 {area} {prefix} {suffix}"


def pprint_json(jstr):
    from json import dumps, loads
    print(dumps(loads(jstr),indent=2))

    
def print_err(err):
    print(err.__class__.__name__)
    print(fill(str(err)))


def show_boxplots(df, cols, whis=1.5):
    # Create as many horizontal plots as we have columns
    fig, axes = plt.subplots(len(cols), 1, figsize=(10, 2*len(cols)))
    # For each one, plot the non-null data inside it
    for n, col in enumerate(cols):
        data = df[col][df[col].notnull()]
        axes[n].set_title(f'{col} Distribution')
        # Extend whiskers to specified IQR multiplier
        axes[n].boxplot(data, whis=whis, vert=False, sym='x')
        axes[n].set_yticks([])
    # Fix spacing of subplots at the end
    fig.tight_layout()
    plt.savefig(f"img/boxplot-{'_'.join(cols)}.png")
    
    
# Load the digit data into namespace
digits = np.load('data/digits.npy')


def show_digits(digits=digits, x=3, y=3, title="Digits"):
    "Display of 'corrupted numerals'"
    if digits.min() >= 0:
        newcm = cm.get_cmap('Greys', 17)
    else:
        gray = cm.get_cmap('Greys', 18)
        newcolors = gray(np.linspace(0, 1, 18))
        newcolors[:1, :] = np.array([1.0, 0.9, 0.9, 1])
        newcm = ListedColormap(newcolors)

    fig, axes = plt.subplots(x, y, figsize=(x*2.5, y*2.5),
                      subplot_kw={'xticks':(), 'yticks': ()})
    
    for ax, img in zip(axes.ravel(), digits):
        ax.imshow(img, cmap=newcm)
        for i in range(8):
            for j in range(8):
                if img[i, j] == -1:
                    s = "╳"
                    c = "k"
                else:
                    s = str(img[i, j])
                    c = "k" if img[i, j] < 8 else "w"
                text = ax.text(j, i, s, color=c,
                               ha="center", va="center")
    fig.suptitle(title, y=0)
    fig.tight_layout()    
    plt.savefig(f'img/{title}.png')
                

kryptonite = pd.read_fwf('data/excited-kryptonite.fwf')

def plot_kryptonite(df=kryptonite, 
                    independent='Wavelength_nm',
                    logx=True,
                    imputed=False):
    fig, ax = plt.subplots(figsize=(10,3))

    (df[df.Kryptonite_type == "Green"]
        .plot(kind='scatter', color="green", marker="o", s=20,
              x=independent, y='candela_per_m2', 
              logx=logx, ax=ax, label="Green"))
    (df[df.Kryptonite_type == "Gold"]
        .plot(kind='scatter', color="goldenrod", marker="s", s=30,
              x=independent, y='candela_per_m2', 
              logx=logx, ax=ax, label="Gold"))
    (df[df.Kryptonite_type == "Red"]
        .plot(kind='scatter', color="darkred", marker="x", s=25,
              x=independent, y='candela_per_m2', 
              logx=logx, ax=ax, label="Red"))
    ax.legend()
    title = f"Luminance response of kryptonite types by {independent}"
    if imputed:
        title = f"{title} (imputed)"
    ax.set_title(title)
    plt.savefig(f"img/{title}.png")

DTI = pd.DatetimeIndex
date_series = pd.Series(
            [-10, 1, 2, np.nan, 4], 
            index=DTI(['2001-01-01',
                       '2001-01-05',
                       '2001-01-10',
                       '2001-02-01',
                       '2001-02-05']))


def plot_filled_trend(s=None):
    n = len(s)
    line = pd.Series(np.linspace(s[0], s[-1], n), 
                     index=s.index)
    filled = s.fillna(pd.Series(line))
    plt.plot(range(n), filled.values, 'o', range(n), line, ":")
    plt.grid(axis='y', color='lightgray', linewidth=0.5)
    plt.xticks(range(n), 
               labels=['Actual', 'Actual',
                       'Actual', 'Imputed (Feb 1)',
                       'Actual'])
    plt.plot([3], [3.69], 'x', 
             label="Time interpolated value")
    plt.legend()
    title = "Global imputation from linear trend"
    plt.title(title)
    plt.savefig(f"img/{title}.png")


def plot_univariate_trends(df, Target='Target'):
    df = df.sort_values(Target)
    target = df[Target]
    X = df.drop(Target, axis=1)
    n_feat = len(X.columns)
    fig, axes = plt.subplots(n_feat, 1,
                             figsize=(10, n_feat*2))
    for ax, col in zip(axes, X.columns):
        ax.plot(target, X[col])
        ax.set_title(f"{col} as a function of {Target}")
    fig.tight_layout()
    plt.savefig(f'img/univariate-{"_".join(X.columns)}:{Target}.png')
    

def read_glarp(cleanup=True):
    df = pd.DataFrame()
    # The different thermometers
    places = ['basement', 'lab', 'livingroom', 'outside']
    for therm in places:
        with gzip.open('data/glarp/%s.gz' % therm) as f:
            readings = dict()
            for line in f:
                Y, m, d, H, M, temp = line.split()
                readings[datetime(*map(int, 
                         (Y, m, d, H, M)))] = float(temp)
        df[therm] = pd.Series(readings)

    if cleanup:
        # Add in the relatively few missing times
        df = df.asfreq('3T').interpolate()
        
        # Remove reading with implausible jumps
        diffs = df.diff()
        for therm in places:
            errs = diffs.loc[diffs[therm].abs() > 5,
                             therm].index
            df.loc[errs, therm] = None
            
        # Backfill missing temperatures (at start)
        df = df.interpolate().bfill()

    # Sort by date but remove from index
    df = df.sort_index().reset_index()
    df = df.rename(columns={'index': 'timestamp'})
    
    return df


def get_digits():
    from sklearn.datasets import load_digits
    digits = load_digits()

    fig, axes = plt.subplots(2, 5,
                    figsize=(10, 5),
                    subplot_kw={'xticks':(), 
                                'yticks': ()})
    for ax, img in zip(axes.ravel(),
                       digits.images):
        ax.imshow(img, cmap=plt.get_cmap('Greys'))
    fig.tight_layout()
    fig.savefig("img/first-10-digits.png")
    return digits


grays10 = """
#000000 #DCDCDC #D3D3D3 #C0C0C0 #A9A9A9
#808080 #696969 #778899 #708090 #2F4F4F
""".split()

vivid = """
#476A2A #7851B8 #BD3430 #4A2D4E #875525
#A83683 #4E655E #853541 #3A3120 #535D8E
""".split()

def plot_digits(data, digits, 
                decomp="Unknown", colors=grays10):
    plt.figure(figsize=(8, 8))
    plt.xlim(data[:, 0].min(), 
             data[:, 0].max() + 1)
    plt.ylim(data[:, 1].min(), 
             data[:, 1].max() + 1)
    for i in range(len(digits.data)):
        # plot digits as text not using scatter
        plt.text(data[i, 0], 
                 data[i, 1],
                 str(digits.target[i]),
                 color = colors[
                     digits.target[i]],
                 fontdict={'size': 9})
    plt.title(f"{decomp} Decomposition")
    plt.savefig(f"img/{decomp}-decomposition.png")
