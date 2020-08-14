import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors, metrics
from sklearn.model_selection import train_test_split, cross_validate
from web.models import Myrating
import scipy.optimize


def Myrecommend():
    def normalizeRatings(myY, myR):
        # The mean is only counting movies that were rated
        Ymean = np.sum(myY, axis=1)/np.sum(myR, axis=1)
        Ymean = Ymean.reshape((Ymean.shape[0], 1))
        return myY-Ymean, Ymean

    def flattenParams(myX, myTheta):
        return np.concatenate((myX.flatten(), myTheta.flatten()))

    def reshapeParams(flattened_XandTheta, mynm, mynu, mynf):
        assert flattened_XandTheta.shape[0] == int(mynm*mynf+mynu*mynf)
        reX = flattened_XandTheta[:int(mynm*mynf)].reshape((mynm, mynf))
        reTheta = flattened_XandTheta[int(mynm*mynf):].reshape((mynu, mynf))
        return reX, reTheta

    def cofiCostFunc(myparams, myY, myR, mynu, mynm, mynf, mylambda=0.):
        myX, myTheta = reshapeParams(myparams, mynm, mynu, mynf)
        term1 = myX.dot(myTheta.T)
        term1 = np.multiply(term1, myR)
        cost = 0.5 * np.sum(np.square(term1-myY))
    # for regularization
        cost += (mylambda/2.) * np.sum(np.square(myTheta))
        cost += (mylambda/2.) * np.sum(np.square(myX))
        return cost

    def cofiGrad(myparams, myY, myR, mynu, mynm, mynf, mylambda=0.):
        myX, myTheta = reshapeParams(myparams, mynm, mynu, mynf)
        term1 = myX.dot(myTheta.T)
        term1 = np.multiply(term1, myR)
        term1 -= myY
        Xgrad = term1.dot(myTheta)
        Thetagrad = term1.T.dot(myX)
    # Adding Regularization
        Xgrad += mylambda * myX
        Thetagrad += mylambda * myTheta
        return flattenParams(Xgrad, Thetagrad)

    df = pd.DataFrame(list(Myrating.objects.all().values()))
    mynu = df.user_id.unique().shape[0]
    mynm = df.movie_id.unique().shape[0]
    mynf = 10
    Y = np.zeros((mynm, mynu))
    for row in df.itertuples():
        Y[row[2]-1, row[4]-1] = row[3]
    R = np.zeros((mynm, mynu))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if Y[i][j] != 0:
                R[i][j] = 1

    Ynorm, Ymean = normalizeRatings(Y, R)
    X = np.random.rand(mynm, mynf)
    Theta = np.random.rand(mynu, mynf)
    myflat = flattenParams(X, Theta)
    mylambda = 12.2
    result = scipy.optimize.fmin_cg(cofiCostFunc, x0=myflat, fprime=cofiGrad, args=(
        Y, R, mynu, mynm, mynf, mylambda), maxiter=40, disp=True, full_output=True)
    resX, resTheta = reshapeParams(result[0], mynm, mynu, mynf)
    prediction_matrix = resX.dot(resTheta.T)
    return prediction_matrix, Ymean


def MyrecommendWithKNN(current_user_id):
    def get_mse(preds, actuals):
        preds = preds[actuals.nonzero()].flatten()
        actuals = actuals[actuals.nonzero()].flatten()
        return metrics.mean_squared_error(preds, actuals)

    f = list(Myrating.objects.all().values())
    for tdict in f:    # To iterate over all dictionaries present in the list
        # print tdict
        for key in tdict:    # To iterate over all the keys in current dictionary
            # print key
            if key == 'rating' and tdict[key] == 0:
                f.remove(tdict)
            else:
                tdict[key] = tdict[key]

    df = pd.DataFrame(f)

    df = df.drop(["id"], 1)

    n_users = df.user_id.unique().shape[0]
    n_movies = df.movie_id.unique().shape[0]

    ratings = np.zeros((n_users, n_movies))

    for row in df.itertuples():
        ratings[row[3]-1, row[1]-1] = row[2]

    sparsity = float(len(ratings.nonzero()[0]))
    sparsity /= (ratings.shape[0]*ratings.shape[1])
    sparsity *= 100

    rating_train, rating_test = train_test_split(
        ratings, test_size=0.3, random_state=43)
    similarity_matrix = metrics.pairwise.cosine_distances(rating_train)

    k = 11
    neigh = neighbors.NearestNeighbors(k, 'cosine')
    neigh.fit(rating_train.T)
    top_k_distances_movies, top_k_users_movies = neigh.kneighbors(
        rating_train.T, return_distance=True)

    movies_predict_k = np.zeros(rating_train.T.shape)
    # Para cada película del conjunto de entrenamiento
    for i in range(rating_train.T.shape[0]):
        movies_predict_k[i, :] = top_k_distances_movies[i].dot(
            rating_train.T[top_k_users_movies][i]) / np.array([np.abs(top_k_distances_movies[i]).sum(axis=0)]).T

    print(get_mse(movies_predict_k, rating_train.T))

    # Y = df["movie_id"].values
    # X = df[["rating", "user_id"]].values

    # print(df['movie_id'].value_counts())

    # X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     X, Y, test_size=0.2, random_state=4)

    # print('Set de Entrenamiento Movies Recommend:',
    #       X_train.shape,  Y_train.shape)
    # print('Set de Prueba Movies Recommend:', X_test.shape,  Y_test.shape)
    # # Entrenar el Modelo y Predecir
    # k = 2
    # neigh = neighbors.KNeighborsClassifier(n_neighbors=k).fit(X_train, Y_train)
    #clf.fit(X_train, Y_train)
    # print(neigh)
    #accuracy = clf.score(X_test, Y_test)
    # print(accuracy)
    # yhat = neigh.predict(X_test)
    # print("Entrenar el set de Certeza: ", metrics.accuracy_score(
    #     Y_train, neigh.predict(X_train)))
    # print("Probar el set de Certeza: ", metrics.accuracy_score(
    #     Y_test, yhat))
    # print(yhat[0:5])

    print(top_k_users_movies.T)
    print('Convert to DataFrame: ')
    converted = pd.DataFrame(top_k_users_movies.T)
    convert = np.array(converted[current_user_id - 1])
    print(convert)

    return convert[0:5]
