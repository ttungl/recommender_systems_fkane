# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:11:13 2018

@author: Frank
"""

from MovieLens import MovieLens
from surprise import KNNBasic, KNNWithZScore, KNNWithMeans, KNNBaseline
from surprise import NormalPredictor
from Evaluator import Evaluator

import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

###### KNNBasic
# User-based KNN
UserKNN1 = KNNBasic(sim_options = {'name': 'cosine', 'user_based': True})
evaluator.AddAlgorithm(UserKNN1, "User KNNBasic")
# Item-based KNN
ItemKNN1 = KNNBasic(sim_options = {'name': 'cosine', 'user_based': False})
evaluator.AddAlgorithm(ItemKNN1, "Item KNNBasic")
###############
###### KNNWithZScore
# User-based KNN
UserKNN2 = KNNWithZScore(sim_options = {'name': 'cosine', 'user_based': True})
evaluator.AddAlgorithm(UserKNN2, "User KNNWithZScore")
# Item-based KNN
ItemKNN2 = KNNWithZScore(sim_options = {'name': 'cosine', 'user_based': False})
evaluator.AddAlgorithm(ItemKNN2, "Item KNNWithZScore")
###############
###### KNNWithMeans
# User-based KNN
UserKNN3 = KNNWithMeans(sim_options = {'name': 'cosine', 'user_based': True})
evaluator.AddAlgorithm(UserKNN3, "User KNNWithMeans")
# Item-based KNN
ItemKNN3 = KNNWithMeans(sim_options = {'name': 'cosine', 'user_based': False})
evaluator.AddAlgorithm(ItemKNN3, "Item KNNWithMeans")
###############
###### KNNBaseline
# User-based KNN
UserKNN4 = KNNBaseline(sim_options = {'name': 'cosine', 'user_based': True})
evaluator.AddAlgorithm(UserKNN4, "User KNNBaseline")
# Item-based KNN
ItemKNN4 = KNNBaseline(sim_options = {'name': 'cosine', 'user_based': False})
evaluator.AddAlgorithm(ItemKNN4, "Item KNNBaseline")
###############


# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# Fight!
evaluator.Evaluate(False)

evaluator.SampleTopNRecs(ml)
