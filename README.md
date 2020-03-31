# Building Recommender Systems with Machine Learing and AI

## Section 1: Getting Started

	+ Youtube's candidate generation:
		```
			top-N <--- Knn index <---video vectors---- softmax ---> class probabilities
					^				^	
					|				|				
				user vector 				| 		
					|				|				
					|__________ ReLU _______________|
					 	 	^
					 	 	|
					 __________ ReLU ____________
					 	 	^
					 	 	|
					 __________ ReLU ____________	
					 	 	^
					 	 	|
		watch vector  search vector	 geographic  age  gender  ...
		     ^		 ^
		     |		 |
		  average	average

		  |||||..||	|||||..||
	   video watches   search tokens		 
	   ```

	+ (one) anatomy of a top-N recommender
		```
		individual interests --> candidate generation <--> item similarities
									|
									candidate ranking
									|
									filtering
									|
									output
		```
	+ autoencoders for recommendations ("autorec")
		```
		R1i	R2i	R3i ... Rmi		(+1)
			M1i M2i...
		R1i R2i R3i ... Rmi     (+1)
		```
	+ Frameworks: Tensorflow, DSSTNE, Sagemaker, ApacheSpark.

	+ Install Anaconda: 
		+ Notes: To overcome the PermissionError issue, after creating the RecSys environment, open its terminal, then run the command `nohup sudo spyder &` to get through this error. You may do the same if you use JupyterLab or others, just replace spyder by that application. 

	+ Getting Started
		+ What is a recommender system?
			+ RecSys is NOT a system that recommends arbitrary values, that describes machine learning in general.
				+ For example,
					+ A system that recommends prices for a house you're selling is NOT a recommender system.
					+ A system that recommends whether a transaction is fraudulent is NOT a recommender system.
				+ These are general ML problems, where you'd apply techniques such as regression, deep learning, xgboost, etc.

			+ A recommender system that predicts ratings or preferences a user might give to an item. RecSys recommends on things based on the people's past behaviors. Often these are sorted and presented as top-N recommendations, aka, recommender engines or platforms.

			+ Customers don't want to see your ability to predict their rating for an item, they just want to see things they're likely to love.


			+ implicit ratings: purchase data, video viewing data, click data. (by product of user's natural behavior)
			+ explicit ratings: star reviews. (ask user reviewing on something)


		+ `GettingStarted.py`				
			+ Use `SVD` singular value decomposition. [explained](https://www.youtube.com/watch?v=P5mlg91as1c)
				+ A[m x n] = U[m x r] * sigma{r x r}(V[n x r])^T
					+ A[m x n]: (rows x cols)		
						+ documents x terms(different words)
							* a given document (in a row) contains a list of terms/words (in columns)
						+ users x movies:
							* a given user (in a row) watches a list of movies (in columns)  
					+ U: Left singular vectors (i.e. user-to-concept similarity matrix)
						+ m x r matrix (m documents, r concepts)
					+ sigma: singular values (i.e. its diagonal elements: strength of each concept)
						+ r x r diagonal matrix (strength of each 'concept' in r)
						+ r: rank of the matrix A
					+ V: right singular vectors (i.e. movie-to-concept similarity matrix)
						+ n x r matrix (n terms, r concepts)

					A = U*sigma*V^T = sum{i}sigma_i*u_i*v_i

				+ [SVD case study](https://www.youtube.com/watch?v=K38wVcdNuFc)

				+ Compare the SVD algorithm to KNNBasic:
					+ test MAE score: SVD(0.2731) KNN(0.6871)
					+ test RSME score: SVD(0.3340) KNN(0.9138)

					+ SVD (user 81)
						We recommend:
							Gladiator (1992)
							Lord of the Rings: The Fellowship of the Ring, The (2001)
							To Kill a Mockingbird (1962)
							Ghost in the Shell (KÃ´kaku kidÃ´tai) (1995)
							Godfather: Part II, The (1974)
							Seven Samurai (Shichinin no samurai) (1954)
							African Queen, The (1951)
							Memento (2000)
							Band of Brothers (2001)
							General, The (1926)

					+ KNN (user 81) 
						We recommend:
							One Magic Christmas (1985)
							Art of War, The (2000)
							Taste of Cherry (Ta'm e guilass) (1997)
							King Is Alive, The (2000)
							Innocence (2000)
							MaelstrÃ¶m (2000)
							Faust (1926)
							Seconds (1966)
							Amazing Grace (2006)
							Unvanquished, The (Aparajito) (1957)

					+ Take-away note: Apparently, SVD performs better than KNN in this scenario.

## Section 2: Intro to Python 

## Section 3: Evaluating Recommender Systems

	+ Train/test/crossvalidation
		+ full data -> train and test
		+ trainset -> machine learning -> fit 
		+ use trained model for testing on test data.

		+ K-fold cross validation
			+ Bagging: full data -> fold i-th -> ML -> measure accuracy -> take average 

	+ Accurate metrics RMSE/MAE
		+ MAE: lower is better
			sum{1..n}|yi - xi|/n
		+ RMSE: lower is better
			+ it penalizes you more when your prediction is way off, and penalizes you less when you are reasonably close. It inflates the penalty for larger errors.

	+ Top-N hit rate - many ways
		+ evaluating top-n recommenders:
			+ Hit rate: you generate top-n recommendations for all of the users in your test set, if one of top-n recommendations is rated, it's a hit. Thus, hit rate = count of all hits per total users 
		+ leave-one-out cross validation:
			+ compute the top-n recommendations for each user in our training data, 
			+ then intentionally remove one of those items from that user's training data. 
			+ then test our recommender system's ability to recommend that item that was left out in the top-n results it creates for that user in the testing phase.
		+ Notes: hit rate with leave-one-out are working better with a very large dataset.

		+ Average reciprocal hit rate (ARHR):
			+ sum{1..n}(1/rank(i))/users
			+ it measures our ability to recommend items that actually appeared in a user's top-n highest rated movies, it gives more weight to these hits when they appear near the top of the top-n list.

		+ cumulative hit rate (cHR):
			+ throw away hits if our predicted rating is below some threshold. The idea is that we shouldn't get credit for recommending items to a user that we think they won't actually enjoy.

		+ rating hit rate (rHR)
			+ break it down by predicted rating score. The idea is that we recommend movies that they actually liked and breaking down the distribution gives you some sense of how well you're doing in more detail.
		+ Take-away: RMSE and hit rate are not always related.

	+ Coverage, Diversity, and Novelty Metrics
		+ Coverage: 
			+ the percentage of possible recommendations that your system is able to provide.
				`% of <user,item> pair that can be predicted.`
			+ Can be important to watch bcos it gives you a sense of how quickly new items in your catalog will start to appear in recommendations. i.e., when a new book comes out on Amazon, it won't appear in recommendations until at least few people buy it. Therefore, establishing patterns with the purchase of other items. Until those patterns exist, that new book will reduce Amazon's coverage metric.

		+ Diversity:
			+ How broad a variety of items your recommender system is putting in front of people. Low diversity indicates it recommends next books in the series that you've started reading, but doesn't recommend books from different authors, or movies related to what you've read/watched. 
			+ We can use similarity scores to measure diversity.
			+ If we look at the similarity scores of every possible pair in a list of top-n recommendations, we can average them to get a measure of how similar the recommended items in the list are to each other, called S.
			+ Diversity = 1 - S
				+ S: avg similarity between recommendation pairs.

		+ Novelty:
			+ mean popularity rank of recommended items.



	+ Churn, Responsiveness, and A/B Tests
		+ How often do recommendations change?
		+ perceived quality: rate your recommendations.
		+ The results of online A/B tests are the metric matters more than anything. 

	+ Review ways to measure your recommender
	+ Recommender Metrics
		+ Surprise package is about making rating predictions, and we need a method to get top-n recommendations out of it. 
	+ Test Metrics
		+ run the test.
			```
			Loading movie ratings...

			Computing movie popularity ranks so we can measure novelty later...

			Computing item similarities so we can measure diversity later...
			Estimating biases using als...
			Computing the pearson_baseline similarity matrix...
			Done computing similarity matrix.

			Building recommendation model...

			Computing recommendations...

			Evaluating accuracy of model...
			RMSE:  0.9033701087151801
			MAE:  0.6977882196132263

			Evaluating top-10 recommendations...
			Computing recommendations with leave-one-out...
			Predict ratings for left-out set...
			Predict all missing ratings...
			Compute top 10 recs per user...

			Hit Rate:  0.029806259314456036

			rHR (Hit Rate by Rating value): 
			3.5 0.017241379310344827
			4.0 0.0425531914893617
			4.5 0.020833333333333332
			5.0 0.06802721088435375

			cHR (Cumulative Hit Rate, rating >= 4):  0.04960835509138381

			ARHR (Average Reciprocal Hit Rank):  0.0111560570576964

			Computing complete recommendations, no hold outs...

			User coverage:  0.9552906110283159 [this is good]
			Computing the pearson_baseline similarity matrix...
			Done computing similarity matrix.

			Diversity:  0.9665208258150911 [this is not good, too high]

			Novelty (average popularity rank):  491.5767777960256
									[this is not good, too high][as long tail in distribution]				
			```
	+ Measure the performance of SVD recommender.

## Section 4: A Recommender Engine Framework 
	+ Build a recommender engine:
		+ use `surpriselib` algorithm (base class)
		+ AlgoBase: SVD, KNNBasic, SVDpp, Custom

	+ Creating a custom algorithm:
		+ implement an estimate function.
		```
		class myOwnAlgorithm(AlgoBase):
			def __init__(self):
				AlgoBase.__init__(self)

			def estimate(self, user, item):
				return 3
		```
	+ Building on top of surpriselib:
		+ create a new class, EvaluatedAlgorithm(AlgoBase)
			+ algorithm: AlgoBase
			+ Evaluate(EvaluationData)
			+ RecommenderMetrics
			+ EvaluationData(Dataset):
				+ GetTrainSet()
				+ GetTestSet()
			+ algorithm bake-offs
				+ Evaluator(DataSet):
					+ AddAlgorithm(algorithm)
					+ Evaluate()
					+ dataset: EvaluatedDataSet
					+ algorithms: EvaluatedAlgorithm[]
	+ Implementation:
		```
		# load up common dataset for the recommender algos.
		(evaluationData, rankings) = LoadMovieLensData()

		# construct an evaluator to evaluate them
		evaluator = Evaluator(evaluationData, rankings)

		# Throw in an SVD recommender.
		SVDAlgo = SVD(random_state=10)
		evaluator.AddAlgorithm(SVDAlgorithm, "SVD")

		# Just make random recommendations
		Random = NormalPredictor()
		evaluator.AddAlgorithm(Random, "Random")

		# Evaluate
		evaluator.Evaluate(True)

		``` 

	+ Code:
		```
		Loading movie ratings...

		Computing movie popularity ranks so we can measure novelty later...
		Estimating biases using als...
		Computing the cosine similarity matrix...
		Done computing similarity matrix.
		Evaluating  SVD ...
		Evaluating accuracy...
		Evaluating top-N with leave-one-out...
		Computing hit-rate and rank metrics...
		Computing recommendations with full data set...
		Analyzing coverage, diversity, and novelty...
		Computing the cosine similarity matrix...
		Done computing similarity matrix.
		Analysis complete.
		Evaluating  Random ...
		Evaluating accuracy...
		Evaluating top-N with leave-one-out...
		Computing hit-rate and rank metrics...
		Computing recommendations with full data set...
		Analyzing coverage, diversity, and novelty...
		Computing the cosine similarity matrix...
		Done computing similarity matrix.
		Analysis complete.


		Algorithm  RMSE       MAE        HR         cHR        ARHR       Coverage   Diversity  Novelty   
		SVD        0.9034     0.6978     0.0298     0.0298     0.0112     0.9553     0.0445     491.5768  
		Random     1.4385     1.1478     0.0089     0.0089     0.0015     1.0000     0.0719     557.8365  

		Legend:

		RMSE:      Root Mean Squared Error. Lower values mean better accuracy.
		MAE:       Mean Absolute Error. Lower values mean better accuracy.
		HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better.
		cHR:       Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. Higher is better.
		ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better.
		Coverage:  Ratio of users for whom recommendations above a certain threshold exist. Higher is better.
		Diversity: 1-S, where S is the average similarity score between every possible pair of recommendations
		           for a given user. Higher means more diverse.
		Novelty:   Average popularity rank of recommended items. Higher means more novel.
		```

## Section 5: Content-Based Filtering 

	+ Cosin similarity:
		+ Assume that movie1 vs movie2 form an angle of 45 degree, so cosine helps to identify if cosine(angle)=0 (cosine(90)) means no similar at all, and cosine(angle)=1 (cosine(0)) means totally the same thing.  


	+ multi-dimensional space:
		+ convert genres to dimensions (i.e. multiple one-hot encoding).
		+ compute multi-dimensional cosines:
			```
			CosSim(x, y) = sum{1..n} xi*yi / (sqrt(sum{1..n}xi^2) * sqrt(sum{1..n}yi^2))
			```
			+ code:
			```
			def computeCosineSimilarity(self, movie1, movie2, genres):
				genres1 = genres[movie1]
				genres2 = genres[movie2]
				sumxx, sumxy, sumyy = 0, 0, 0
				# go through all genres.
				for i in range(len(genres1)):
					x, y = genres1[i], genres2[i]
					sumxx += x*x
					sumyy += y*y
					sumxy += x*y
				return sumxy / math.sqrt(sumxx * sumyy)
			```
		+ compute time similarity:
			```
			def computeYearSimilarity(self, movie1, movie2, years):
				diff = abs(years[movie1] - years[movie2])
				sim = math.exp(-diff/10.0)
				return sim
			```
		+ K-nearest-neighbors:
			+ similarity scores between this movie and all others the user rated 
				=> sort top 40 nearest movies 
				=> weighted average (weighting them by the rating the user gave them)
				=> rating prediction.
			+ knn code:
			```
			# Build up similarity scores between this item and everything the user rated.
			neighbors = []
			for rating in self.trainset.ur[u]:
				genreSimilarity = self.similarities[i, rating[0]]
				neighbors.append((genreSimilarity, rating[1]))

			# Extract the top-k most-similar ratings
			k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

			# Compute average sim score of K neighbors weighted by user ratings.
			simTotal = weightedSum = 0
			for (simScore, rating) in k_neighbors:
				if (simScore > 0):
					simTotal += simScore
					weightedSum += simScore * rating
			if (simTotal == 0):
				raise PredictionImpossible('No neighbors')

			predictedRating = weightedSum/simTotal
			return predictedRating
			```

	+ Producing and evaluating content-based filtering movies recommendation.

		+ `getPopularityRanks`
			Algorithm  RMSE       MAE       
			contentKNN 0.9375     0.7263    
			Random     1.4385     1.1478    

			Legend:

			RMSE:      Root Mean Squared Error. Lower values mean better accuracy.
			MAE:       Mean Absolute Error. Lower values mean better accuracy.

			Using recommender  contentKNN

				We recommend:
				Presidio, The (1988) 3.841314676872932
				Femme Nikita, La (Nikita) (1990) 3.839613347087336
				Wyatt Earp (1994) 3.8125061475551796
				Shooter, The (1997) 3.8125061475551796
				Bad Girls (1994) 3.8125061475551796
				The Hateful Eight (2015) 3.812506147555179
				True Grit (2010) 3.812506147555179
				Open Range (2003) 3.812506147555179
				Big Easy, The (1987) 3.7835412549266985
				Point Break (1991) 3.764158410102279

			Using recommender  Random

				We recommend:
				Sleepers (1996) 5
				Beavis and Butt-Head Do America (1996) 5
				Fear and Loathing in Las Vegas (1998) 5
				Happiness (1998) 5
				Summer of Sam (1999) 5
				Bowling for Columbine (2002) 5
				Babe (1995) 5
				Birdcage, The (1996) 5
				Carlito's Way (1993) 5
				Wizard of Oz, The (1939) 5

		+ `mixed years and genres`
			Algorithm  RMSE       MAE       
			contentKNN 0.9441     0.7310    
			Random     1.4385     1.1478    
			--
			Using recommender  contentKNN
				True Grit (2010) 3.81250614755518
				Open Range (2003) 3.81250614755518
				The Hateful Eight (2015) 3.8125061475551796
				Wyatt Earp (1994) 3.8125061475551796
				Shooter, The (1997) 3.8125061475551796
				Bad Girls (1994) 3.8125061475551796
				Romeo Must Die (2000) 3.771364493375906
				Femme Nikita, La (Nikita) (1990) 3.7678571120506548
				RoboCop (1987) 3.7594365328860415
				Die Hard (1988) 3.75840413236323

## Section 6: Neighborhood-Based Collaborative Filtering 
	+ Finding other people like you and recommending stuff they liked.
	Recommending stuff based on the things people liked that you haven't seen yet. That's why we call it collaborative filtering.
	It's recommending stuff based on other people's collaborative behavior.
	+ One anatomu of top-N recommender.
		```
		individual interests --> candidate generation <--> items similarity
										|
										candidate ranking
										|
										filtering
										|
										data output
		```

	+ First step in collaborative filtering is finding ways to measure similarity.
		+ Cosine similarity
		+ Big Challenge on the behavior data is the sparsity of the data. It's tough for collaborative filtering to work well unless you have a lot of user behavior data. We can't compute cosine similarity of two people who have nothing in common, or between two items when they have no people in common. That's why collaborative filtering at Netflix and Amazon working well because they have millions of users so they have enough data to generate meaningful relations in spite of the data sparsity.

		+ Sparsity is also challenging as you don't want to waste time storing and processing all of that missing data, so we use like sparse arrays that avoid storing all that empty space in this matrix.
		
		+ The quantity and quality of the data that you have to work with is often way more important than the algorithm that you choose. It doesn't matter what method you use to measure similarity if you don't have enough data to begin with.

		+ Adjusted cosine:
			```
			CosSim(x,y) = sum{i}((xi-x\)(yi-y\)) / sqrt(sum{i}(xi - x\)^2) * sqrt(sum{i}(yi - y\)^2)
			```
			
			CosSim attempts to normalize these differences. Instead of measuring similarities between people based on their raw rating values, we instead measure similarity based on the difference between a user rating for an item and their average rating for all items.
			x\: means the average of all user x's ratings.
			
			Notes: you can get only a meaningful average or a baseline of an individuals ratings if they have rated a lot of stuff for you to take the average of in the first place.
		+ item-based pearson similarity:
			```
			CosSim(x,y) = sum{i}((xi - I\)(yi - I\))/sqrt(sum{i}(xi - I\)^2) * sqrt(sum{i}(yi-I\)^2)
			```
			This method looks at the difference between ratings and the average of all users for that given item. 
			
			I\: the average rating of the item across all users.

			Pearson similarity measures similarity between people by how much they diverge from the average person's behavior. Imagine a film that most peolpe love like Star Wars. People who hate it are going to get a very strong similarity score from Pearson's similarity because they share opinions that are not mainstream.

		+ Spearman rank correlation:
			+ Pearson similarity based on ranks, not ratings as in original pearson.

			+ Main advantage of this method is that it can deal with ordinal data effectively. For example, you had a rating scale where the difference in meaning between different rating values were not the same. It's not usually used in industry.

		+ Mean squared difference
			``` MSD(x,y) = sum{i in Ixy} (xi - yi)^2 / |Ixy| ```
				Notes: 
					numerator: sum up for every item i of difference of squared btw users x and y. 

					denominator: Ixy: the number of items each user had in common
					
					we summed across to get the mean.

			``` MSDsim(x,y) = 1/MSD(x,y) + 1```
					we measure how similar they are (not how different), to do that, we take the inverse of MSD and plus 1 is to avoid zero when two users have identical behavior.


		+ Jaccard similarity:


	+ User-based collaborative filtering
		+ recommend stuff they liked that you haven't seen yet.

		+ First step is to collect the data we need.
			+ A table of all of the ratings for everyone in our system.
			+ Think of it as a 2D array with movies on one axis and users on the other, and rating values in each cell. 
			```
				Indiana Jones	Star Wars	Empires		Incred.		Casa
			Bob	 4			5 					
			Ted 									1
			Ann 					5		5		
			```
			=>cosine similarity.
			```
			   Bob  Ted  Ann
			Bob 1   0   1
			Ted 0   1   0
			Ann 1   0   1
			```
			Notes: 100% similar doesn't mean they like the same thing, it could mean they hate the same thing. 1 and 0 in table show we have little data to work with. If we have large data, the number will be more meaningful.

		+ Process:
			+ user -> item rating matrix
			+ user -> user similarity matrix
			+ look up similar users
			+ candidate generation
			+ candidate scoring
			+ candidate filtering

		+ run code:
			```
				Inception (2010) 3.3
				Star Wars: Episode V - The Empire Strikes Back (1980) 2.4
				Bourne Identity, The (1988) 2.0
				Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 2.0
				Dark Knight, The (2008) 2.0
				Good, the Bad and the Ugly, The (Buono, il brutto, il cattivo, Il) (1966) 1.9
				Departed, The (2006) 1.9
				Dark Knight Rises, The (2012) 1.9
				Back to the Future (1985) 1.9
				Gravity (2013) 1.8
				Fight Club (1999) 1.8
			```
		+ Modifications:
			+ `rating > 0.95`
				Star Wars: Episode IV - A New Hope (1977) 228.48498846248853
				Matrix, The (1999) 203.5299981419994
				Star Wars: Episode V - The Empire Strikes Back (1980) 185.7852060377671
				Toy Story (1995) 177.69199360501636
				Fargo (1996) 176.92324562879384
				Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981) 172.37282124063051
				American Beauty (1999) 170.08491076233042
				Back to the Future (1985) 168.63039437454358
				Godfather, The (1972) 164.6732633306926
				Usual Suspects, The (1995) 163.79259985557516
				Star Wars: Episode VI - Return of the Jedi (1983) 163.54809920010618

			

	+ Item-based collaborative filtering 
			+ 2D matrix mapping similarity score is between every item in your catalog will be much smaller than a 2D matrix mapping similarity between every user that visits your site. It makes faster to compute when dealing with massive systems.

			+ compute cossim between two items.
			```
				James Dean Story, The (1957) 10.0
				Get Real (1998) 9.987241120712646
				Kiss of Death (1995) 9.966881877751941
				Set It Off (1996) 9.963732215657119
				How Green Was My Valley (1941) 9.943984081065269
				Amos & Andrew (1993) 9.93973694500253
				My Crazy Life (Mi vida loca) (1993) 9.938290487546041
				Grace of My Heart (1996) 9.926255896645218
				Fanny and Alexander (Fanny och Alexander) (1982) 9.925699671455906
				Wild Reeds (Les roseaux sauvages) (1994) 9.916226404418774
				Edge of Seventeen (1998) 9.913028764691676
			```

			+ Modifications:
				+ Build recommendation candidates from items above a certain rating or similarity threshold, instead of the top 10.
					+ `score >= 0.5`
						Tender Mercies (1983) 66.00517680033703
						Crucible, The (1996) 65.85413695486129
						The Falcon and the Snowman (1985) 65.75879125610763
						Scent of a Woman (1992) 65.70507335169201
						Rocky II (1979) 65.7038806173855
						Queen Margot (Reine Margot, La) (1994) 65.62189595122997
						True Crime (1999) 65.60014950711003
						Seventh Seal, The (Sjunde inseglet, Det) (1957) 65.59691475660786
						Night Shift (1982) 65.57493667740741
						Young Sherlock Holmes (1985) 65.5367919327825
						Walk in the Clouds, A (1995) 65.52503694150758

					+ `rating > 4.0`
						Kiss of Death (1995) 16.910437073265502
						Amos & Andrew (1993) 16.861270021975354
						Edge of Seventeen (1998) 16.853845983977223
						Get Real (1998) 16.840092759084882
						Grace of My Heart (1996) 16.83866418909583
						Relax... It's Just Sex (1998) 16.825893097731395
						My Crazy Life (Mi vida loca) (1993) 16.825163372963015
						Set It Off (1996) 16.820045947032426
						Bean (1997) 16.81043113102984
						Joe's Apartment (1996) 16.804698282071367
						Lost & Found (1999) 16.78956315445952



















## Section 7: Matrix Factorization Methods 

## Section 8: Intro to Deep Learning 

## Section 9: Deep Learning for Recommender Systems

## Section 10: Scaling it Up 

## Section 11: Real-World Challenges of Recommender Systems 

## Section 12: Case Studies

## Section 13: Hybrid Approach 

## Section 14: Wrap Up
