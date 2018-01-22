
def SVMclassify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
    
    ### your code goes here!
    from sklearn.svm import SVC
    clf =SVC(C=100000,kernel='rbf')
    clf.fit(features_train,labels_train)
    return clf


def SVMAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.svm import SVC
    

    ### create classifier
    clf =SVC(C=10,kernel='rbf')

    ### fit the classifier on the training features and labels
    #TODO
    clf.fit(features_train,labels_train)

    accuracy=clf.score(features_test,labels_test,sample_weight=None)
    return accuracy