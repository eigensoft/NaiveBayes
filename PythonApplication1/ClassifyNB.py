def NBclassify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
    
    ### your code goes here!
    
    from sklearn.naive_bayes import GaussianNB
    clf= GaussianNB()
    clf.fit(features_train,labels_train)
    return clf


def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB

    ### create classifier
    clf =GaussianNB()

    ### fit the classifier on the training features and labels
    #TODO
    clf.fit(features_train,labels_train)
    ### use the trained classifier to predict labels for the test features
    correct_count=0
    pred = clf.predict(features_test)
    for ii in range(0,len(features_test)):
        if pred[ii]==labels_test[ii]:
            correct_count=correct_count+1



    



    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    accuracy = correct_count/len(features_test)
    accuracy2=clf.score(features_test,labels_test,sample_weight=None)
    return accuracy2