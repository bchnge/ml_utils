# ml_utils
#### Convenience utilities for machine learning / predictive modeling.
---

## Guideline
1. Define a Dataset
    - Define base classifier
    - Preprocess data
        - Define construction rules
            - Merge multiple tables to the level of unit of observation by key identifiers
        - Define variable elimination rules
            - Missing frequency
            - Near-zero variance
            - Sparse features
        - Treatment of outliers
        - Feature standardization
    - Build preliminary model for baseline variable importance
    - Randomized pruning based on preliminary variable importance
    - Feature engineering: Create new variables automatically or by hand
    - (Optional): Determine minimum required amount of data needed for training to reduce time needed for tuning hyperparameters next.
    - Export preprocessing rules to Models and weighted feature list.

2. Define Model
    - Pipe preprocessing rules with initial model parameters
    - Cross validate against training data using prescribed feature list.

3. Define Tuner
    - Use Bayesian Optimization to tune each model parameter. (bayes_opt)

4. Define ModelCollection
    - Repeat steps 2 and 3, using randomized sets of features and different model types. Can be autogenerated.

5. Define ensemble model
    - Stacking (mlxtend)
    - Voting (sklearn)

