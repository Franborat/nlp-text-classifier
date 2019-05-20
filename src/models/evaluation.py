import click
import sys
import pickle
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB

sys.path.append('src')
from data import read_processed_corpus, read_processed_category


@click.command()
@click.argument('input_directory', type=click.Path(exists=True, readable=True, dir_okay=True))
@click.argument('output_directory', type=click.Path(exists=True, readable=True, dir_okay=True))
@click.option('--long', is_flag=True)
def main(input_directory, output_directory, long):

    X = read_processed_corpus(input_directory + '/corpus.pickle')
    y = read_processed_category(input_directory + '/category.pickle')

    print('Train and test split: 80-20')
    print()
    # Separate train test and test test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if long:
        # For SVM, AdaBoost, KNN

        # Inputs:
        # X: Sparse matrix
        # y: numpy list

        ## SVM Linear
        clf_svc_linear = svm.SVC(kernel='linear', gamma='auto')
        # Accuracy
        print("Accuracy of SVC Linear: ")
        print(cross_val_score(clf_svc_linear, X_train, y_train, cv=5, scoring='accuracy').mean())
        print()

        ## AdaBoost
        from sklearn.ensemble import AdaBoostClassifier
        clf_ab = AdaBoostClassifier(n_estimators=100)
        # Accuracy
        print("Accuracy of AdaBoost: ")
        print(cross_val_score(clf_ab, X_train, y_train, cv=5, scoring='accuracy').mean())
        print()

        ## KNN
        from sklearn.neighbors import KNeighborsClassifier
        clf_knn = KNeighborsClassifier()
        # Accuracy
        print("Accuracy of KNN: ")
        print(cross_val_score(clf_knn, X_train, y_train, cv=5, scoring='accuracy').mean())
        print()

        # For NaiveBayes

        # Inputs:
        # X: dense matrix
        # y: numpy list

        ## Naive Bayes GAUSSIAN
        from sklearn.naive_bayes import GaussianNB
        clf_gnb = GaussianNB()
        # Accuracy
        print("Accuracy of nb Gaussian: ")
        print(cross_val_score(clf_gnb, X_train.toarray(), y_train, cv=5, scoring='accuracy').mean())
        print()

        ## Naive Bayes MULTINOMIAL
        clf_mnb = MultinomialNB()
        # Accuracy
        print("Accuracy of nb Multinomial: ")
        print(cross_val_score(clf_mnb, X_train.toarray(), y_train, cv=5, scoring='accuracy').mean())
        print()

        ## Naive Bayes BERNOULLI
        from sklearn.naive_bayes import BernoulliNB
        clf_bnb = BernoulliNB()
        # Accuracy
        print("Accuracy of nb Bernoulli: ")
        print(cross_val_score(clf_bnb, X_train.toarray(), y_train, cv=5, scoring='accuracy').mean())
        print()

        # For Multi-layer perceptron:

        # Inputs:
        # X: dense matrix
        # y: hot-encoded

        # One-hot encoding of y

        lb = preprocessing.LabelBinarizer()
        lb.fit(y)

        y_mlp = lb.transform(y)

        from sklearn.model_selection import train_test_split
        _X_train, _X_test, _y_train, _y_test = train_test_split(X, y_mlp, test_size=0.2)

        from sklearn.neural_network import MLPClassifier
        clf_mlp = MLPClassifier()
        # Accuracy
        print("Accuracy of MLP: ")
        print(cross_val_score(clf_mlp, _X_train, _y_train, cv=5, scoring='accuracy').mean())
        print()

        ## Hyperparameters-tuning for the 2 best models

        ## SVM -->
        print("Hyperparameter-tuning for SVM model: Kernel and C")
        Cs = [0.001, 0.01, 0.1, 1, 10]
        kernels = ['linear', 'poly', 'rbf']

        params_comb_svm = []
        for kernel in kernels:
            for c in Cs:
                params_comb_svm.append([kernel, c])

        for params in params_comb_svm:
            clf_svc = svm.SVC(kernel=params[0], C=params[1], gamma='scale')
            # Accuracy
            accuracy = cross_val_score(clf_svc, X_train, y_train, cv=5, scoring='accuracy').mean()
            print("Accuracy of SVM kernel-->{} C-->{}: {}".format(params[0], params[1], accuracy))
            params.append(accuracy)
            print()

    ## NB Multinomial -->
    print("Hyperparameter-tuning for SVM model: Alpha")
    print()
    alphas = [0.01, 0.1, 1, 10]

    params_comb_nb = []
    for alpha in alphas:
        params_comb_nb.append([alpha])

    max_accuracy = [0, 0]
    for params in params_comb_nb:
        clf_mnb = MultinomialNB(alpha=params[0])
        # Accuracy
        accuracy = cross_val_score(clf_mnb, X_train.toarray(), y_train, cv=5, scoring='accuracy').mean()
        if accuracy>max_accuracy[0]:
            max_accuracy = [accuracy, params[0]]
        print("Accuracy of SVC Linear alpha-->{}: {}".format(params[0], accuracy))


    print()
    print('Final Model: NB Multinomial with alpha-->{}. Accuracy = {}'.format(max_accuracy[1], max_accuracy[0]))

    ## Best Model Train and Save

    clf_mnb_final = MultinomialNB(alpha=max_accuracy[1])
    clf_mnb_final.fit(X, y)

    # pickle - saving
    with open(output_directory+'/nb_model.pickle', 'wb') as f:
        pickle.dump(clf_mnb_final, f)


if __name__ == '__main__':
    main()
