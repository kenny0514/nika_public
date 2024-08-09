from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss,accuracy_score, precision_recall_fscore_support
from Utilities.backtest_tools import *
import pandas as pd
import numpy as np
from itertools import product, zip_longest

class PurgedKFold(KFold):
    """
    Extend KFold class to work with labels that span intervals.
    The train is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (shuffle=False), without training samples in between. 
    """
    def __init__(self, n_splits=3, t1=None, pctEmbargo=0.):
        if not isinstance(t1, pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None, rightTrain = True):
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)
        test_starts = [(i[0], i[-1]+1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]

        for i, j in test_starts:
            t0 = self.t1.index[i]  # start of test set
            test_indices = indices[i:j]
            maxT1Idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            if rightTrain:
                if maxT1Idx < X.shape[0]:  # right train (with embargo)
                    train_indices = np.concatenate((train_indices, indices[maxT1Idx + mbrg:]))
            yield train_indices, test_indices

def cvScore(clf, X, y, sample_weight, scoring='neg_log_loss', t1=None, cv=None, cvGen=None, pctEmbargo=None, cls=1):
    if scoring not in ['neg_log_loss', 'accuracy', 'f1']:
        raise Exception('wrong scoring method.')

    if cvGen is None:
        cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged

    score = []
    for train, test in cvGen.split(X=X):
        fit = clf.fit(X=X.iloc[train, :], y=y.iloc[train], sample_weight=sample_weight.iloc[train].values)
        
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X.iloc[test, :])
            score_ = -log_loss(y.iloc[test], prob, sample_weight=sample_weight.iloc[test].values, labels=clf.classes_)
        elif scoring == 'accuracy':
            pred = fit.predict(X.iloc[test, :])
            score_ = accuracy_score(y.iloc[test], pred, sample_weight=sample_weight.iloc[test].values)
        elif scoring == 'f1':
            pred = fit.predict(X.iloc[test, :])
            precision, recall, f1, _ = precision_recall_fscore_support(y.iloc[test], pred, sample_weight=sample_weight.iloc[test].values, average=None)
            score_= f1[cls]
        score.append(score_)
    
    return np.array(score)

def score_by_threshold(y_true, y_pred_proba, score, threshold):
    """Custom F1 scoring function that applies a threshold to prediction probabilities."""
    # Check if y_pred_proba is 1-dimensional
    if y_pred_proba.ndim == 1:
        y_pred = np.where(y_pred_proba >= threshold, 1, 0)
    else:
        # Assuming the positive class probabilities are in the second column
        y_pred = np.where(y_pred_proba[:, 1] >= threshold, 1, 0)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    if score == 'f1':
        return f1
    elif score == 'precision':
        return precision
    elif score == 'recall':
        return recall
    elif score == 'f2':
        return (precision*10+recall*1)/(10+1)
    else:
        raise ValueError("Wrong score type.")

def f2(y_true, y_pred_proba, thresholds, sample_weight):
    sample_weight = sample_weight[y_true.index]
    control = (y_true*sample_weight).sum()/sample_weight.sum()
    score = []
    for thres in thresholds:
        y_pred = ((y_pred_proba >= thres)&(y_pred_proba < thres + 0.05))
        if y_pred.sum() == 0: continue
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', sample_weight = sample_weight.values)
        surplus_precision = precision - control    
        score.append(surplus_precision*recall)
    score = np.sum(score)
    return score

def getTestData(n_features=40, n_informative=10, n_redundant=10, n_samples=10000, random_state = 0, anonymize = False):
    # Generate a random dataset for a classification problem
    trnsX, cont = make_classification(n_samples=n_samples, n_features=n_features,
                                      n_informative=n_informative, n_redundant=n_redundant,
                                      random_state=random_state, shuffle=False)
    
    df0 = pd.date_range(start=pd.Timestamp.today() - pd.tseries.offsets.BDay(n_samples), end=pd.Timestamp.today(), freq='B')[:n_samples]

    trnsX, cont = pd.DataFrame(trnsX, index=df0), pd.Series(cont, index=df0).to_frame('bin')
    
    if anonymize is False:
        df0 = ['I_' + str(i) for i in range(n_informative)] + \
            ['R_' + str(i) for i in range(n_redundant)]
        df0 += ['N_' + str(i) for i in range(n_features - len(df0))]
        trnsX.columns = df0
    else:
        df0 = [str(i) for i in range(trnsX.shape[1])]
        trnsX = trnsX.sample(frac=1, axis=1) # Shuffle columns
        trnsX.columns = df0
        
    
    cont['w'] = 1. / cont.shape[0]
    cont['t1'] = pd.Series(cont.index, index=cont.index)
    
    return trnsX, cont

def applyPCA(Xtrain, Xtest = None ,varThres=0.95):
    scaler = StandardScaler()
    Xtrain_scaled = scaler.fit_transform(Xtrain)

    pca = PCA(n_components=varThres)
    dfP_train = pca.fit_transform(Xtrain_scaled)
    dfP_train = pd.DataFrame(dfP_train, columns=[f'PC_{i+1}' for i in range(dfP_train.shape[1])], index=Xtrain.index)
    
    # dfC = pd.DataFrame(pca.components_.T, columns = dfP_train.columns, index = Xtrain.columns)

    if Xtest is not None:
        Xtest_scaled = scaler.transform(Xtest)
        dfP_test = pca.transform(Xtest_scaled)
        dfP_test = pd.DataFrame(dfP_test, columns=[f'PC_{i+1}' for i in range(dfP_test.shape[1])], index=Xtest.index)
        return dfP_train, dfP_test
    else:
        return dfP_train

def getIndMatrix(barIx, t1):
    # Using Sparse Matrix to reduce the size
    indM = lil_matrix((len(barIx), len(t1)), dtype=int)
    for i, (t0, t1) in enumerate(t1.items()):
        start_iloc, end_iloc = barIx.get_loc(t0), barIx.get_loc(t1)+1 # Convert datetime to integer
        indM[start_iloc:end_iloc, i] = 1
    return indM

def getOvMap(events):
    start_times = np.array(events.index)
    end_times = np.array(events.t1.values)

    overlaps = {}

    for i in range(len(start_times)):
        # For each sample, find samples that end after this sample's start
        ends_after_start = end_times >= start_times[i]
        
        # And find samples that start before this sample's end
        starts_before_end = start_times <= end_times[i]
        
        # Both conditions must be true for intervals to overlap
        overlap_conditions = ends_after_start & starts_before_end
        
        # Identify the indices (samples) that satisfy the overlap condition
        overlapping_samples = np.where(overlap_conditions)[0]
        
        overlaps[i] = overlapping_samples

    return overlaps

def getAvgUniqueness(indM):
    indM_ = indM.tocsr() # CSR for arithmetic ops
    c = indM_.sum(axis=1)  # concurrency, sum over rows
    u = csr_matrix(indM_.multiply(1 / c))  # uniqueness matrix, still sparse
    avgU = np.array(u.sum(axis=0) / u.getnnz(axis=0)).flatten()  # average uniqueness for each feature
    return avgU

def seqBootstrap(indM, ovMap, sLength=None):
    # Kenny's upgraded version.
    # Marcos's code too slow
    indM = indM.tocsc()
    if sLength is None:
        sLength = indM.shape[1]
    phi = []
    avgU = np.full(indM.shape[1],1.)
    while len(phi) < sLength:
        prob = avgU / avgU.sum()
        new = np.random.choice(indM.shape[1], p=prob)
        phi.append(new)
        affected = ovMap[new]
        indM_ = indM[:,affected]
        avgU[affected] = getAvgUniqueness(indM_)
    return phi

def featImpMDI(fit, featNames):
    # feat importance based on IS mean impurity reduction
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = featNames
    df0 = df0.replace(0, np.nan)  # because max_features=1
    imp = pd.concat({'mean': df0.mean(), 'std': df0.std() * df0.shape[0] ** -.5}, axis=1)
    imp /= imp['mean'].sum()
    return imp

def featImpMDI_XGB(XGB, columns):
    # Get feature importances directly
    feature_importances = XGB.feature_importances_

    # Map feature importances to their corresponding feature names
    feature_importance_dict = dict(zip(columns, feature_importances))

    # Optionally, convert to a pandas DataFrame for easier manipulation and visualization
    feature_importance_df = pd.DataFrame(list(feature_importance_dict.items()), columns=['Feature', 'Importance'])

    # Now you can work with feature_importance_df, for example, sorting the features by their importance
    feature_importance_df_sorted = feature_importance_df.sort_values(by='Importance', ascending=False)
    feature_importance_df_sorted.set_index('Feature',inplace=True, drop=True)
    return feature_importance_df_sorted

def featImpMDA(clf, X, y, cv, sample_weight, t1, pctEmbargo, scoring='neg_log_loss', cls = 1):
    # feat importance based on OOS score reduction
    if scoring not in ['neg_log_loss', 'accuracy', 'f1']:
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss, accuracy_score

    cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged cv
    scr0, scr1 = pd.Series(dtype=float), pd.DataFrame(columns=X.columns, dtype=float)

    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight.iloc[train]
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight.iloc[test]
        fit = clf.fit(X=X0, y=y0, sample_weight=w0.values)

        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(y1, prob, sample_weight=w1.values, labels=clf.classes_)
        elif scoring == 'accuracy':
            pred = fit.predict(X1)
            scr0.loc[i] = accuracy_score(y1, pred, sample_weight=w1.values)
        elif scoring == 'f1':
            pred = fit.predict(X1)
            precision, recall, f1, _ = precision_recall_fscore_support(y1, pred, sample_weight=w1.values, average=None)
            scr0.loc[i] = f1[cls]

        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values)  # permutation of a single column
            if scoring == 'neg_log_loss':
                prob = fit.predict_proba(X1_)
                scr1.loc[i, j] = -log_loss(y1, prob, sample_weight=w1.values, labels=clf.classes_)
            elif scoring == 'accuracy':
                pred = fit.predict(X1_)
                scr1.loc[i, j] = accuracy_score(y1, pred, sample_weight=w1.values)
            elif scoring == 'f1':
                pred = fit.predict(X1_)
                precision, recall, f1, _ = precision_recall_fscore_support(y1, pred, sample_weight=w1.values, average=None)
                scr1.loc[i, j] = f1[cls]            


    imp = (-scr1).add(scr0, axis=0)
    if scoring == 'neg_log_loss':
        imp = imp / -scr1
    elif scoring =='accuracy':
        imp = imp / (1. - scr1)
    elif scoring =='f1': 
        imp = imp / (1. - scr1)
    imp = pd.concat({'mean': imp.mean(), 'std': imp.std() * imp.shape[0] ** -.5}, axis=1)
    return imp

def auxFeatImpSFI(featNames, clf, trnsX, cont, sample_weight, scoring, cv, pctEmbargo, t1):
    imp = pd.DataFrame(columns=['mean', 'std'])
    for featName in featNames:
        df0 = cvScore(clf, X=trnsX[[featName]], y=cont['bin'], sample_weight=sample_weight, scoring=scoring, cv=cv, pctEmbargo = pctEmbargo, t1 = t1)
        imp.loc[featName, 'mean'] = df0.mean()
        imp.loc[featName, 'std'] = df0.std() * df0.shape[0]**-.5
    return imp

def featImportance(trnsX, cont, method, n_estimators=1000, cv=5, max_samples=1., numThreads=8, pctEmbargo=0, scoring='accuracy', minWLeaf=0., **kargs):
    # feature importance from a random forest
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    
    # run 1 thread with ht_helper in dirac1
    n_jobs = (-1 if numThreads > 1 else 1)  
    
    # 1) prepare classifier,cv. max_features=1, to prevent masking
    clf = DecisionTreeClassifier(criterion='entropy', max_features=1, class_weight='balanced', min_weight_fraction_leaf=minWLeaf)
    clf = BaggingClassifier(base_estimator=clf, n_estimators=n_estimators, max_samples=max_samples, oob_score=True, n_jobs=n_jobs)
    fit = clf.fit(X=trnsX, y=cont['bin'], sample_weight=cont['w'].values)
    oob = fit.oob_score_

    if method == 'MDI':
        imp = featImpMDI(fit, featNames=trnsX.columns)
        oos = cvScore(clf, X=trnsX, y=cont['bin'], cv=cv, sample_weight=cont['w'], t1=cont['t1'], pctEmbargo=pctEmbargo, scoring=scoring).mean()
    elif method == 'MDA':
        imp, oos = featImpMDA(clf, X=trnsX, y=cont['bin'], cv=cv, sample_weight=cont['w'], t1=cont['t1'], pctEmbargo=pctEmbargo, scoring=scoring)
    elif method == 'SFI':
        cvGen = PurgedKFold(n_splits=cv, t1=cont['t1'], pctEmbargo=pctEmbargo)
        oos = cvScore(clf, X=trnsX, y=cont['bin'], sample_weight=cont['w'], scoring=scoring, cvGen=cvGen).mean()
        clf.n_jobs = 1 # Parallelize auxFeatImpSFI rather than clf
        clf.n_estimators = 20 # Feature당 샘플 20개 하면 충분.
        imp = mpPandasObj(auxFeatImpSFI, trnsX.columns, numThreads, clf=clf, trnsX=trnsX, cont=cont, scoring=scoring, cvGen=cvGen)
    
    return imp, oob, oos

def testFunc(n_features=40, n_informative=10, n_redundant=10, n_estimators=1000, n_samples=10000, cv=5):
    # test the performance of the feat importance functions on artificial data
    # Nr noise features = n_features - n_informative - n_redundant
    trnsX, cont = getTestData(n_features, n_informative, n_redundant, n_samples)
    dict0 = {
        'minWLeaf': [0.],
        'scoring': ['accuracy'],
        'method': ['MDI', 'MDA', 'SFI'],
        'max_samples': [1.]
    }
    jobs, out = (dict(zip_longest(dict0, i)) for i in product(*dict0.values())), []
    kargs = {
        'pathOut': 'results/',
        'n_estimators': n_estimators,
        'tag': 'testFunc',
        'cv': cv
    }
    for job in jobs:
        job['simNum'] = job['method'] + '_' + job['scoring'] + '_' + '%.2f' % job['minWLeaf'] + '_' + str(job['max_samples'])
        print(job['simNum'])
        kargs.update(job)
        imp, oob, oos = featImportance(trnsX=trnsX, cont=cont, **kargs)
        plotFeatImportance(imp=imp, oob=oob, oos=oos, **kargs)
        df0 = imp[['mean']] / imp['mean'].abs().sum()
        df0['type'] = [i[0] for i in df0.index]
        df0 = df0.groupby('type')['mean'].sum().to_dict()
        df0.update({'oob': oob, 'oos': oos})
        df0.update(job)
        out.append(df0)
    out = pd.DataFrame(out).sort_values(['method', 'scoring', 'minWLeaf', 'max_samples'])
    out = out[['method', 'scoring', 'minWLeaf', 'max_samples', 'I', 'R', 'N', 'oob', 'oos']]
    out.to_csv(kargs['pathOut'] + 'stats.csv')
    return out

def plotFeatImportance(pathOut, imp, oob, oos, method, tag=0, simNum=0, **kargs):
    # plot mean imp bars with std
    plt.figure(figsize=(10, imp.shape[0] / 5.))
    imp = imp.sort_values('mean', ascending=True)
    ax = imp['mean'].plot(kind='barh', color='b', alpha=.25, xerr=imp['std'], error_kw={'ecolor': 'r'})
    
    if method == 'MDI':
        plt.xlim([0, imp.sum(axis=1).max()])
        plt.axvline(1. / imp.shape[0], linewidth=1, color='r', linestyle='dotted')
    ax.get_yaxis().set_visible(False)
    
    for i, j in zip(ax.patches, imp.index):
        ax.text(i.get_width() / 2, i.get_y() + i.get_height() / 2, j, ha='center', va='center', color='black')
    
    plt.title('tag=' + str(tag) + ' | simNum=' + str(simNum) + ' | oob=' + str(round(oob, 4)) + ' | oos=' + str(round(oos, 4)))
    plt.savefig(pathOut + 'featImportance_' + str(simNum) + '.png', dpi=100)
    plt.clf()
    plt.close()
    return
