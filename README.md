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


## Section 2: Intro to Python 

## Section 3: Evaluating Recommender Systems

	+ Train/test/crossvalidation
	+ Accurate metrics RMSE/MAE
	+ Top-N hit rate - many ways
	+ Coverage, Diversity, and Novelty
	+ Churn, Responsiveness, and A/B Tests
	+ Review ways to measure your recommender
	+ Recommender Metrics
	+ Test Metrics
	+ Measure the performance of SVD recommender.
	 

## Section 4: A Recommender Engine Framework 

## Section 5: Content-Based Filtering 

## Section 6: Neighborhood-Based Collaborative Filtering 

## Section 7: Matrix Factorization Methods 

## Section 8: Intro to Deep Learning 

## Section 9: Deep Learning for Recommender Systems

## Section 10: Scaling it Up 

## Section 11: Real-World Challenges of Recommender Systems 

## Section 12: Case Studies

## Section 13: Hybrid Approach 

## Section 14: Wrap Up
