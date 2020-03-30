# ContentRecsT
"""
Modifier: Tung Le
Date: 03/29/2020
"""

from MovieLensT import MovieLens
from ContentKNNAlgorithmT import ContentKNNAlgorithm
from EvaluatorT import Evaluator
from surprise import NormalPredictor

import random
import numpy as np

class ContentBasedFiltering:
	def LoadMovieLensData(self):
		ml = MovieLens()
		print("Loading movie ratings...")
		data = ml.loadMovieLensLatestSmall()
		print("\nComputing movie popularity ranks so we can measure novelty later...")
		rankings = ml.getPopularityRanks()
		return (ml, data, rankings)

    
def main():

	cbf = ContentBasedFiltering()
	np.random.seed(0)
	random.seed(0)

	print("\nLoading common dataset for the recommender algorithms...")
	(ml, evaluationData, rankings) = cbf.LoadMovieLensData()

	# print("\nConstruct an evaluator to evaluate the algorithms")
	evaluator = Evaluator(evaluationData, rankings)

	print("\nConstructing the ContentKNN algorithm...")
	contentKNN = ContentKNNAlgorithm()
	evaluator.AddAlgorithm(contentKNN, "contentKNN")

	# baseline random recommendations
	Random = NormalPredictor()
	evaluator.AddAlgorithm(Random, "Random")

	evaluator.Evaluate(False)

	evaluator.SampleTopNRecs(ml)


if __name__ == '__main__':
	main()


