# Recommendation Systems
One of the most important applications of machine learning is recommendation system. In this repo, I compare the performance of 2 different recommender models on 
[MovieLens](https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset) dataset. The first model is a basic Collaborative Filtering model, and the second one is more 
complex [DeepFM](https://www.ijcai.org/proceedings/2017/0239.pdf) model.
## MovieLens 1M Dataset
MovieLens dataset is used for movie recommendation task. Dataset contains 1 million records as user movie ratings 1 to 5. Also, it includes contextual information of users such as gender, occupation, age etc., and genres for the movies. In this project, I used the dataset as binary classification task. Ratings below 4 are labeled as dislike(0), remaining ones are labeled as like(1). 
## Collaborative Filtering
Collaborative filtering uses the similarities between users and items to provide recommendations. It learns users and movie embeddings, so that there is no need to manual
feature engineering. Single emebedding vector can be thought as (1xm) feature vector where m is hyperparameter of the model. Collaborative filtering is classified into 2 categories as memory-based and model-based. <br/>

**Memory Based:** Memory based approaches rely on similarity measures(cosine similarity, euclidean distance, jaccard similarity, etc.). For example, if we have a huge sparse matrix that has the users as rows and item ratings as columns. Similar users can be detected from the matrix by calculating similarity, and items can be recommended between the similar users.  <br/>

**Model Based:** Model based approach tries to fill the empty cells of the feedback matrix A(mxn) where m is the number of users and n is the number of items. They rely on matrix factorization.
* Learn user embedding matrix U(mxd) where i th row is the user embedding of the i th user and d is the embedding dimension.
* Learn item embedding matrix V(nxd) where i th row is the item embedding of the i th item and d is the embedding dimension.
* Finally, feedback matrix A can be factorized as A = U*V<sup>T</sup>

In this project, I conduct my experiments with a simple model based collaborative filtering. There 2 embedding layers for users and movies with a dimension of 64. In order to make a prediction, cosine similarity between user and movie embeddings are calculated. Normally, cosine similarity is between -1 and 1, since this is a classification task values are scaled to 0 and 1. If the similarity between the user and the movie embedding is close to 1, that means user will like the particular movie. Log loss is used as optimization objective. Model is trained with a learning rate of 0.5 for 5 epochs. Below figure shows the training metrics and auc of the test set. <br/>

![alt text for screen readers](images/cf-loss.png "Loss")
![alt text for screen readers](images/cf-auc.png "AUC")

AUC score is calculated as 0.81 which is not that bad. 
## DeepFM
