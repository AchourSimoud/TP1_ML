def data_process(data):
    Y=data['label']
    Y=Y.apply(lambda y: 1 if y=='normal' else -1)
    X=data.drop(columns=['label'])
    return X,Y


def onehotencoder(X, columns_to_encode):
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd

    encoder = OneHotEncoder(sparse=False)

    encoded_data = encoder.fit_transform(X[columns_to_encode])

    columns_names = encoder.get_feature_names_out(columns_to_encode)

    encoded_df = pd.DataFrame(encoded_data, columns=columns_names)

    X = X.drop(columns=columns_to_encode)

    res = pd.concat([X, encoded_df], axis=1)

    return res


def normalisation(X):
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
    scaler.fit(X)

    scaled_data = pd.DataFrame(scaler.transform(X), columns=X.columns)

    return scaled_data


def grid_search_parameters(model, X, Y,param_grid, scoring='balanced_accuracy', cv=5):
    from sklearn.model_selection import GridSearchCV

    # Utilisez le GridSearchCV pour effectuer la recherche sur grille
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv)
    
    # Effectuez la recherche sur grille sur les données
    grid_search.fit(X,Y)

    # Recuperer les meilleurs paramètres
    best_params = grid_search.best_params_

    return best_params

def roc(model,X,Y):
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, balanced_accuracy_score
    if type(model).__name__=='IsolationForest':
        # transformer les score entre 0 et 1
        decision=model.decision_function(X)
        y_pred = 1 / (1 + np.exp(decision))
        scaler = MinMaxScaler()
        y_score = scaler.fit_transform(y_pred.reshape(-1, 1)).ravel()
    elif type(model).__name__=='LocalOutlierFactor':
         # transformer les score entre 0 et 1
         # prendre en compte le cas de la detection de nouveauté
         if model.novelty==True :
               lof_scores=model.decision_function(X)
               y_pred = 1 / (1 + np.exp(lof_scores))
         else :
               lof_scores=model.negative_outlier_factor_
               y_pred = 1 / (1 + np.exp(-lof_scores)) 
         scaler = MinMaxScaler()
         y_score = scaler.fit_transform(y_pred.reshape(-1, 1)).ravel()
    else: 
         y_score=model.predict_proba(X)[:,1]

    fpr, tpr, seuils = roc_curve(Y, y_score)
    
    #Point le plus proche du coin supérieur gauche
    distances = np.sqrt((1 - tpr)**2 + fpr**2)
    best_threshold = seuils[np.argmin(distances)]
    
    predictions=y_score.copy()
    predictions[y_score<=best_threshold]=0
    predictions[y_score>best_threshold]=1
    balanced_accuracy=balanced_accuracy_score(Y,predictions)
    roc_auc = auc(fpr, tpr)

    # Tracer la courbe ROC
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Courbe ROC (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire')
    plt.xlabel('Taux de faux positifs (FPR)')
    plt.ylabel('Taux de vrais positifs (TPR)')
    plt.title('Courbe ROC '+type(model).__name__)
    plt.legend(loc='lower right')
    plt.show()
    
    return best_threshold, balanced_accuracy