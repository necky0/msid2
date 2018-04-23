# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------


from __future__ import division
import numpy as np


def hamming_distance(X, X_train):
    """
    :param X: zbior porownwanych obiektow N1xD
    :param X_train: zbior obiektow do ktorych porownujemy N2xD
    Funkcja wyznacza odleglosci Hamminga obiektow ze zbioru X od
    obiektow X_train. Odleglosci obiektow z jednego i drugiego
    zbioru zwrocone zostana w postaci macierzy
    :return: macierz odleglosci pomiedzy obiektami z X i X_train N1xN2
    """

    def hamming(vector_x):
        return np.sum(np.logical_xor(vector_x, X_train.toarray()), axis=1)

    return np.apply_along_axis(hamming, axis=1, arr=X.toarray())


def sort_train_labels_knn(Dist, y):
    """
    Funkcja sortujaca etykiety klas danych treningowych y
    wzgledem prawdopodobienstw zawartych w macierzy Dist.
    Funkcja zwraca macierz o wymiarach N1xN2. W kazdym
    wierszu maja byc posortowane etykiety klas z y wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist
    :param Dist: macierz odleglosci pomiedzy obiektami z X
    i X_train N1xN2
    :param y: wektor etykiet o dlugosci N2
    :return: macierz etykiet klas posortowana wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist. Uzyc algorytmu mergesort.
    """

    def sorted_labels(dist):
        sorted_dist = np.argsort(dist, kind='margesort')
        return y[sorted_dist]

    return np.apply_along_axis(sorted_labels, axis=1, arr=Dist)


def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszuch sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """

    length = np.max(y)+1

    def probability(labels):
        env = np.bincount(labels[:k], minlength=length)
        return env/k

    return np.apply_along_axis(probability, axis=1, arr=y)


def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """

    y = p_y_x.shape[1] - np.argmax(np.flip(p_y_x, axis=1), axis=1) - 1
    bool_y = np.not_equal(y, y_true)
    return np.mean(bool_y)


def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: zbior danych walidacyjnych N1xD
    :param Xtrain: zbior danych treningowych N2xD
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartosci parametru k, ktore maja zostac sprawdzone
    :return: funkcja wykonuje selekcje modelu knn i zwraca krotke (best_error,best_k,errors), gdzie best_error to najnizszy
    osiagniety blad, best_k - k dla ktorego blad byl najnizszy, errors - lista wartosci bledow dla kolejnych k z k_values
    """

    dist = hamming_distance(Xval, Xtrain)
    y = sort_train_labels_knn(dist, ytrain)

    errors = []
    for k in k_values:
        p_y_x = p_y_x_knn(y, k)
        err = classification_error(p_y_x, yval)
        errors.append(err)

    index = np.argmin(errors)
    best_k = k_values[int(index % len(k_values))]
    best_error = errors[int(index)]

    return best_error, best_k, errors


def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: etykiety dla dla danych treningowych 1xN
    :return: funkcja wyznacza rozklad a priori p(y) i zwraca p_y - wektor prawdopodobienstw a priori 1xM
    """

    length = np.max(ytrain)+1
    env = np.bincount(ytrain, minlength=length)

    return env/len(ytrain)


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: dane treningowe NxD
    :param ytrain: etykiety klas dla danych treningowych 1xN
    :param a: parametr a rozkladu Beta
    :param b: parametr b rozkladu Beta
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(x|y) (czy p(x=1|y)?) zakladajac, ze x przyjmuje wartosci binarne i ze elementy
    x sa niezalezne od siebie. Funkcja zwraca macierz p_x_y (czy p_x_1_y?)  o wymiarach MxD.
    """

    length = np.max(ytrain) + 1
    denominator = np.bincount(ytrain, minlength=length) + a + b - 2

    def quotient(x):
        numerator = np.bincount(np.extract(x, ytrain), minlength=length) + a - 1
        return numerator/denominator

    return np.apply_along_axis(quotient, axis=0, arr=Xtrain.toarray())


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: wektor prawdopodobienstw a priori o wymiarach 1xM
    :param p_x_1_y: rozklad prawdopodobienstw p(x=1|y) - macierz MxD
    :param X: dane dla ktorych beda wyznaczone prawdopodobienstwa, macierz NxD
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla kazdej z klas z wykorzystaniem klasyfikatora Naiwnego
    Bayesa. Funkcja zwraca macierz p_y_x o wymiarach NxM.
    """
    X = X.toarray()

    def p_y_x(x):
        prod = np.prod((p_x_1_y ** x) * ((1 - p_x_1_y) ** (1 - x)), axis=1)*p_y
        return prod/np.sum(prod, axis=0)

    return np.apply_along_axis(p_y_x, axis=1, arr=X)


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: zbior danych treningowych N2xD
    :param Xval: zbior danych walidacyjnych N1xD
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrow a do sprawdzenia
    :param b_values: lista parametrow b do sprawdzenia
    :return: funkcja wykonuje selekcje modelu Naive Bayes - wybiera najlepsze wartosci parametrow a i b. Funkcja zwraca
    krotke (error_best, best_a, best_b, errors) gdzie best_error to najnizszy
    osiagniety blad, best_a - a dla ktorego blad byl najnizszy, best_b - b dla ktorego blad byl najnizszy,
    errors - macierz wartosci bledow dla wszystkich par (a,b)
    """

    errors = []
    for a in a_values:
        error = []
        for b in b_values:
            p_y = estimate_a_priori_nb(ytrain)
            p_x_1_y = estimate_p_x_y_nb(Xtrain, ytrain, a, b)

            p_y_x = p_y_x_nb(p_y, p_x_1_y, Xval)
            err = classification_error(p_y_x, yval)

            error.append(err)
        errors.append(error)

    index = np.argmin(errors)
    index_a = int(index/len(b_values))
    index_b = int(index % len(a_values))

    error_best = errors[index_a][index_b]
    best_a = a_values[index_a]
    best_b = b_values[index_b]

    return error_best, best_a, best_b, errors
