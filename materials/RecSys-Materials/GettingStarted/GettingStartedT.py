#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 16:18:02 2020

@author: Tung Le
"""

from MovieLens import MovieLens
from surprise import SVD
from surprise import accuracy
from surprise import KNNBasic

def BuildAntiTestSetForUser(testSubject, trainset):
    fill = trainset.global_mean

    anti_testset = []
    
    u = trainset.to_inner_uid(str(testSubject))
    
    user_items = set([j for (j, _) in trainset.ur[u]])
    
    anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                             i in trainset.all_items() if
                             i not in user_items]
    
    return anti_testset


# Pick an arbitrary test subject # a user id that we want to see the model recommends to this id.
testSubject = 81

# create instance ml from MovieLens class.
ml = MovieLens()

print("Loading movie ratings...")
data = ml.loadMovieLensLatestSmall()

userRatings = ml.getUserRatings(testSubject) # return the list of movieID and associated ratings of this user.
loved = []
hated = []

# userRatings = (movieID, rating)
# if rating > 4 then love it, else hate it.
for ratings in userRatings:
    if (float(ratings[1]) > 4.0):
        loved.append(ratings)
    if (float(ratings[1]) < 3.0):
        hated.append(ratings)


print("\nUser ", testSubject, " loved these movies:")
for ratings in loved:
    print(ml.getMovieName(ratings[0]))
print("\n...and didn't like these movies:")
for ratings in hated:
    print(ml.getMovieName(ratings[0]))

print("\nBuilding recommendation model...")
trainSet = data.build_full_trainset() # this method is from the Dataset in surprise package.

# algo = KNNBasic()
algo = SVD() # singular value decomposition
algo.fit(trainSet) # fit for the whole train dataset

print("Computing recommendations...")
testSet = BuildAntiTestSetForUser(testSubject, trainSet) # get test data for userID in testSubject. 
predictions = algo.test(testSet) # predict on the testset.

recommendations = []

print ("\nWe recommend:")
for userID, movieID, actualRating, estimatedRating, _ in predictions:
    intMovieID = int(movieID)
    recommendations.append((intMovieID, estimatedRating))

recommendations.sort(key=lambda x: x[1], reverse=True) # sort the result by estimatedRating.

for ratings in recommendations[:10]:
    print(ml.getMovieName(ratings[0]))

print("Test MAE score: ", accuracy.mae(predictions))
print("Test RMSE score: ", accuracy.rmse(predictions))

