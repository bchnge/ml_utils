

class Model:
    '''
    Model serves as a container for documenting, defining, training, and validating predictive models.
    '''
    def __init__(self, clf = None, name = None, desc = None):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.ensemble import RandomForestClassifier

        self.name = name
        self.desc = desc

        if clf is None:
            self.clf = RandomForestClassifier()

        self.cv = StratifiedKFold(n_splits=5, random_state=123)

    def set_model(self, clf):
        self.clf = clf
        
    def train(self, X, y):
        self.clf.fit(X, y)
        
    def validate_model(self, X, y):
        from sklearn.model_selection import cross_val_score

        if hasattr(self, 'clf') is False:
            print('Stop. You must define a model before validating.')
        else:
            self.validation_scores = cross_val_score(self.clf, X, y, scoring = 'roc_auc', cv = self.cv)

class Dataset:
    '''
    Datasets contain methods for maintaining train/test matrices, preprocessing, feature elimination, and automatic feature engineering. Transformers can be saved as pipeline models.
    '''
    def __init__(self, train_data, test_data):
        train_features = train_data.columns
        test_features = test_data.columns
        features = list(set(train_features) & set(test_features))
        
        print('...creating training matrix')
        self.X_train, self.y_train = get_design_matrix_lbl(train_data, 
                                                           features, train = True, train_test_split = False, 
                                                           convert_categorical = True)
        
        print('...creating test matrix')
        self.X_test = get_design_matrix_lbl(test_data, features, train = False, train_test_split = False,
                                           convert_categorical = True)
        
        self.features_initial = list(self.X_test.columns)
        self.features_initial_categorical = list(test_data.columns[test_data.dtypes == object])
        
        self.ae_discovery_ratios = []
        
    
    def preprocess(self):
        # Remove nonvariant column
        yield
    
    def ae_train_model(self, model):
        model.fit(self.X_train, self.y_train)        
        self.ae_feature_importances_dict = dict(zip(self.X_train.columns, model.feature_importances_))
        self.ae_feature_importances = model.feature_importances_        
    
    def autoengineer_ratios(self, n_iter = 1000):
        ae_params = {'boosting_type': 'gbdt',
                  'max_depth' : -1,
                  'objective': 'binary',
                  'learning_rate': 0.0212,
                  'reg_alpha': 0.8,
                  'reg_lambda': 0.4,
                  'subsample': 1,
                  'feature_fraction': 0.3,
                  'device_type': 'gpu',
                  'metric' : 'auc',
                  'random_state': 123,
                  'n_estimators': 300, 
                  'num_leaves': 40, 
                  'max_bin': 255,
                  'min_data_in_leaf': 2400,
                  'min_data_in_bin': 5}
        
        def _fn_column_selector(X, k):
            ''' 
                select up to kth column
            '''
            return X[:,:k]

        ColumnSelector = FunctionTransformer(_fn_column_selector, validate = False)

        importance_weights = self.ae_feature_importances / self.ae_feature_importances.sum()
        
        kfold = StratifiedKFold(n_splits=5, random_state=123)
        
        model = Pipeline([('selector', ColumnSelector),
                  ('clf', LGBMClassifier(**ae_params))])
        
        for i in range(n_iter):
            random_vars = list(np.random.choice(self.X_train.columns, 
                                                size = 2, p = importance_weights, 
                                                replace= False))

            X_tmp = self.X_train.loc[:, random_vars ]
            X_tmp['_DIV_'.join(random_vars)] = X_tmp.iloc[:,0] / (X_tmp.iloc[:,1] + 1)

            gs = GridSearchCV(estimator = model,
                              param_grid = {'selector__kw_args': [{'k':2},{'k':3}]},
                              scoring = 'roc_auc',
                              cv = kfold)
            gs.fit(X_tmp.values, self.y_train)
            perf_1, perf_2 = gs.cv_results_.get('mean_test_score')
            if perf_2 > perf_1:
                self.ae_discovery_ratios.append((random_vars[0], random_vars[1], perf_2/perf_1))

class Tuner:
    def __init__(self, clf, X, y):
        self.clf = clf
        self.X = X
        self.y = y

    def tune(self, kappa, pbounds, n_iters):
        from bayes_opt import BayesianOptimization

        integer_params = [p for p in pbounds.keys() if type(pbounds.get(p)[0]) == int and type(pbounds.get(p)[1])]

        print('params detected as integers: \n')
        print(integer_params)


        def _fn(**kwargs):
            # Each tuning requires a custom function to measure success

            from sklearn.model_selection import cross_val_score
            bo_model = self.clf
            for p in integer_params:
                if p in kwargs.keys():
                    kwargs[p] = int(kwargs.get(p))

            bo_model.set_params(kwargs)

            score = cross_val_score(self.clf, self.X, self.y, scoring = 'roc_auc', cv = 5)

            return(score.mean())

        self.bo = BayesianOptimization(f = _fn, pbounds = pbounds,random_state = 123)
        self.bo.maximize(kappa = kappa, n_iters = n_iters)

class ModelCollection():
    ''' A collection of Models. Contains methods for ensemble classification and comparing models
    '''
    def __init__(self, models_dict):
        '''
        models_dict should be a list of tuples [(Model, list_of_features)]
        '''
        from mlxtend.classifier import StackingClassifier
        from mlxtend.feature_selection import ColumnSelector
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold

        self.cv = StratifiedKFold(n_splits=5, random_state=123)

        self.models_dict = models_dict
        models = [(i, Pipeline([('ColumnSelect', ColumnSelector(v[1])), ('Model', v[0].clf)])) for i,v in models_dict]
        self.models = models
        self.clf_stack = Model(clf = StackingClassifier(classifiers = models, meta_classifier = LogisticRegression(), name = 'Stacked ensemble'))
        
    def validate_model(self, X, y):
        from sklearn.model_selection import cross_validate

        if hasattr(self, 'clf_stack') is False:
            print('Stop. You must define a model before validating.')
        else:
            validation_results_dict = {}
            self.validation_scores_stack = cross_validate(self.clf_stack, X, y, scoring = 'roc_auc', cv = self.cv)

            self.validation_scores_individual = [(i, cross_validate(clf, X, y, scoring = 'roc_auc', cv = self.cv)) for i, clf in enumerate(self.models)]
            
    def compare_models(self):
            stack_df = pd.DataFrame(self.validation_scores_stack)
            individual_df = pd.concat([pd.DataFrame(score) for score in self.validation_scores_individual)])
            result_df = pd.concat([stack_df, individual_df])
            return(result_df)
