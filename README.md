# gann
Solving a binary classification problem, using a Genetic Algorithm to optimize a Neural Network's architecture.

## Tools
* [Tensorflow](https://www.tensorflow.org/) - Machine Learning platform
* [DEAP](https://deap.readthedocs.io/en/master/) - Evolutionary Computation framework

## Strategy 
### Data Preprocessing
1. Converted non numeric data to NaN 
2. For rows with misintroduced data 
  * If they were in the majority class, remove the lines (contribute to dataset balancing through undersampling)
  * If not, just convert the value to NaN 
3. Use Nearest Neighbours to impute NaN values 
4. Visualize data to check if everything is in order, also to get an idea of what the relationship between features and the target might be 
5. The dataset was already almost balanced, so the undersampling used before left it completely balanced: no more balancing techniques needed! 
6. One-hot encoded categorical columns to avoid loss of information
7. Scaled the age column (so it is more adequate to use with neural nets)

### Genetic Algorithm
