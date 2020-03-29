# gann
Solving a binary classification problem, using a Genetic Algorithm to optimize a Neural Network's architecture.


## Strategy 
### Data Preprocessing
1. Converted non numeric data to NaN 
2. For rows with misintroduced data 
  * If they were in the majority class, remove the lines (contribute to dataset balancing through undersampling)
  * If not, just convert the value to NaN 
3. Use Neares Neighbours to impute NaN values 
4. Visualize data to check if everything is in order, also to get an idea of what the relationship between features and the target might be 
5. The dataset was already almos balanced, so the undersampling used before left it completely balanced: no more balancing techniques needed! 
6. Scaled the dataset (so it is more adequate to use with neural nets)
