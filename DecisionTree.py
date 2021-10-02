def DecisionTree():
    

    from sklearn import tree

    model2 = tree.DecisionTreeClassifier()   # empty model of the decision tree
    model2 = model2.fit(X1, np.ravel(Y))
    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=model2.predict(x1_test)
    acc2 = accuracy_score(y, y_pred)
    print(f"Accuracy of the Decision Tree algorithm is {acc2}")
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l5)):
        # print (k,)
        for z in psymptoms:
            if(z==l5[k]):
                l6[k]=1

    inputtest = [l6]
    predict = model2.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t2.delete("1.0", END)
        t2.insert(END, disease[a])
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")
        