# Create your views here.

from django.shortcuts import render
from projects.models import Project
from projects.forms import InputForm

def project_index(request):
    projects = Project.objects.all()
    context = {
        'projects': projects}
    return render(request, 'project_index.html', context)

# Create your views here.
def home_view(request):
    context ={}
    context['form']= InputForm()
    return render(request, "home.html", context)

def corona_view(request):
    eyesymptom="null"
    context = {}
    metrics={}
    if request.method == "POST":
        # Get the posted form
        form = InputForm(request.POST)
        if form.is_valid():
            eyesymptom = form.cleaned_data.get('eye_symptom')
            chestsymptom=form.cleaned_data.get('chest_symptom')
            soarsymptom = form.cleaned_data.get('soar_symptom')
            runnynosesymptom = form.cleaned_data.get('runnynose_symptom')
            weaknesssymptom=form.cleaned_data.get('weakness_symptom')
            achessymptom = form.cleaned_data.get('aches_symptom')
            headachesymptom = form.cleaned_data.get('headache_symptom')
            coughsymptom = form.cleaned_data.get('cough_symptom')
            breathingsymptom=form.cleaned_data.get('breathing_symptom')
            sleepsymptom=form.cleaned_data.get('sleep_symptom')
            meetppl=form.cleaned_data.get('meetppl')
            coronatest=form.cleaned_data.get('corona_test')
            corona14=form.cleaned_data.get('corona14')
            disease=form.cleaned_data.get('disease')
            gone_out=form.cleaned_data.get('gone_out')
            gone_out1 = form.cleaned_data.get('gone_out1')
            gone_out2 = form.cleaned_data.get('gone_out2')
            spent_time=form.cleaned_data.get('spent_time')
            more_meet=form.cleaned_data.get('more_meet')
            crowd = form.cleaned_data.get('crowd')
            transport=form.cleaned_data.get('transport')
            home=form.cleaned_data.get('home')
            any=form.cleaned_data.get('any')
            prec=form.cleaned_data.get('precautions')
            times=form.cleaned_data.get('times')
            gen=form.cleaned_data.get('gender')
            smoke=form.cleaned_data.get('smoke')
            alc=form.cleaned_data.get('alc')
            handwash=form.cleaned_data.get('handwash')
            handwash2 = form.cleaned_data.get('handwash2')
            age=form.cleaned_data.get('age')
            context = {'eyesymptom': eyesymptom,'chestsymptom':chestsymptom,'soarsymptom':soarsymptom,'runnynosesymptom':runnynosesymptom,'weaknesssymptom':weaknesssymptom,'achessymptom':achessymptom,'headachesymptom':headachesymptom
                       ,'coughsymptom':coughsymptom,'breathingsymptom':breathingsymptom,'sleepsymptom':sleepsymptom
                       ,'meetppl':meetppl,'coronatest':coronatest,'corona14':corona14,'disease':disease
                       ,'gone_out':gone_out,'gone_out1':gone_out1,'gone_out2':gone_out2,'spent_time':spent_time,'more_meet':more_meet
                       ,'crowd':crowd,'transport':transport,'home':home,'any':any,'prec':prec,'times':times,'gen':gen,'smoke':smoke
                       ,'alc':alc,'handwash':handwash,'handwash2':handwash2,'age':age}

            print("context['eyesymptom']", context['eyesymptom'])
            metrics=covid_prediction(context)

            context['eyesymptom']="No" if context['eyesymptom'] == 0 else "Yes"
            #print(context['eyesymptom'])
            context['chestsymptom'] = "No" if context['chestsymptom'] == 0 else "Yes"
            context['soarsymptom'] = "No" if context['soarsymptom'] == 0 else "Yes"
            context['runnynosesymptom'] = "No" if context['runnynosesymptom'] == 0 else "Yes"
            context['weaknesssymptom'] = "No" if context['weaknesssymptom'] == 0 else "Yes"
            context['achessymptom'] = "No" if context['achessymptom'] == 0 else "Yes"
            context['headachesymptom'] = "No" if context['headachesymptom'] == 0 else "Yes"
            context['coughsymptom'] = "No" if context['coughsymptom'] == 0 else "Yes"
            context['breathingsymptom'] = "No" if context['breathingsymptom'] == 0 else "Yes"
            context['sleepsymptom'] = "No" if context['sleepsymptom'] == 0 else "Yes"
            context['meetppl'] = "No" if context['meetppl'] == 0 else "Yes"
            context['coronatest'] = "No" if context['coronatest'] == 0 else "Yes"
            context['corona14'] = "No" if context['corona14'] == 0 else "Yes"
            if context['disease'] ==2:
                context['disease'] = 'High BP-2'
            elif context['disease'] == 3:
                context['disease'] = 'Asthama-3'
            else:
                context['disease'] = "None"
            context['gone_out'] = "No" if context['gone_out'] == 0 else "Yes"
            context['gone_out1'] = "No" if context['gone_out1'] == 0 else "Yes"
            context['gone_out2'] = "No" if context['gone_out2'] == 0 else "Yes"
            context['spent_time'] = "No" if context['spent_time'] == 0 else "Yes"
            context['more_meet'] = "No" if context['more_meet'] == 0 else "Yes"
            context['crowd'] = "No" if context['crowd'] == 0 else "Yes"
            context['transport'] = "No" if context['transport'] == 0 else "Yes"
            context['home'] = "No" if context['home'] == 0 else "Yes"
            context['any'] = "No" if context['any'] == 0 else "Yes"
            if context['prec'] == 2:
                context['prec'] = 'Sometimes'
            elif context['prec'] == 3:
                context['prec'] = 'Often'
            elif context['prec'] == 4:
                context['prec'] = 'Always'
            else:
                context['prec'] = "Rarely"

            if context['times'] == 0:
                context['times'] = 'None of the time'
            elif context['times'] == 3:
                context['times'] = 'Most of the time'
            elif context['times'] == 4:
                context['times'] = 'All of the time'
            else:
                context['times'] = "Some of the time"
            context['gen'] = "Female" if context['any'] == 0 else "Male"
            context['smoke'] = "No" if context['smoke'] == 0 else "Yes"
            context['alc'] = "No" if context['alc'] == 0 else "Yes"
            context['handwash'] = "No"  if context['handwash'] == 0 else "Yes"
            context['handwash2'] = "No"  if context['handwash2'] == 0 else "Yes"
            if context['age'] == 0:
                context['age'] = '18-25'
            elif context['age'] == 1:
                context['age'] = '26-32'
            elif context['age'] == 2:
                context['age'] = '33-45'
            else:
                context['age'] = ">45"
            metrics['testSetValues1'] = "Potential No" if metrics['testSetValues1'] == 0 else "Potential Yes"
            metrics['testSetValues2'] = "Potential No" if metrics['testSetValues2'] == 0 else "Potential Yes"
            metrics['testSetValues3'] = "Potential No" if metrics['testSetValues3'] == 0 else "Potential Yes"
            metrics['testSetValues4'] = "Potential No" if metrics['testSetValues4'] == 0 else "Potential Yes"

            context.update(metrics)
            print(metrics)
            print(context)
            print("context['eyesymptom']",context['eyesymptom'])
        return render(request, "poutput.html", context)
    else:
        context['form'] = InputForm()
    return render(request, "home.html",context)


def covid_prediction(pcontext):
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    #%matplotlibinline
    import time
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, fbeta_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from random import randint

    metrics = {}
    link='https://raw.githubusercontent.com/PraseedaSaripalle/Symptoms_Cleaned_Dataset_/main/DATASET_Symptoms_CovidPred.csv'
    #link2='https://www.kaggle.com/jayasreethyadi/covid19-symptoms'
    #df=pd.read_csv("/datasets/DATASET_Symptoms_CovidPred.csv")
    df=pd.read_csv(link)
    #df=pd.read_csv(link2)
    # Split dataset in training and test datasets. in this X_train, X_test,are testing related feature set and label set. y_train, y_test are testing feature set and label set
    size = df.shape[0]
    testsize = 0.2
    print("test size of records", testsize)
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(['If you have been tested, did this test find that you had coronavirus (COVID-19)?'], axis=1),
        df['If you have been tested, did this test find that you had coronavirus (COVID-19)?'], test_size=testsize)
    gnb = GaussianNB()
    # Train classifier
    gnb.fit(X_train, y_train)
    # predict using X_Test

    y_pred = gnb.predict(X_test)
    print(y_pred)
    ypredlenth = len(y_pred)
    print("Last person contacted covid or not", y_pred[ypredlenth - 1])
    print("Last but 1 person contacted covid or not", y_pred[ypredlenth - 2])
    print("Last but 2 person contacted covid or not", y_pred[ypredlenth - 3])
    # to know the lenght of predicted
    print(len(y_pred))
    accuracyScore = round(accuracy_score(y_test, y_pred), 2)
    confusion_matrix(y_test, y_pred)
    precisionScore = round(precision_score(y_test, y_pred), 2)
    recallScore = round(recall_score(y_test, y_pred), 2)
    f1Score = round(f1_score(y_test, y_pred), 2)
    fbetaScore = round(fbeta_score(y_test, y_pred, beta=0.5, average='micro'), 2)

    print("precisionScore", precisionScore)
    print("recallScore", recallScore)
    print("f1Score", f1Score)
    print("accuracyScore", accuracyScore)
    print("fbetaScore", fbetaScore)

    dtc = DecisionTreeClassifier(max_depth=None, random_state=None)
    dtc.fit(X_train, y_train)
    # predict using X_Test
    y_pred = dtc.predict(X_test)
    print(y_pred)
    ypredlenth = len(y_pred)
    # print("Last Project Success or Fail", y_pred[ypredlenth-1])
    # print("Last but 1 Project Success or Fail", y_pred[ypredlenth - 2])
    # print("Last but 2 Project Success or Fail", y_pred[ypredlenth - 3])
    # to know the lenght of predicted
    # print(len(y_pred))

    # now find out precision_score, recall_score, f1_score
    # to build confusion matrix actual vs predicted
    daccuracyScore = round(accuracy_score(y_test, y_pred), 2)
    confusion_matrix(y_test, y_pred)
    dprecisionScore = round(precision_score(y_test, y_pred), 2)
    drecallScore = round(recall_score(y_test, y_pred), 2)
    df1Score = round(f1_score(y_test, y_pred), 2)
    dfbetaScore = round(fbeta_score(y_test, y_pred, beta=0.5), 2)

    print(daccuracyScore, dprecisionScore, drecallScore, df1Score, dfbetaScore)

    rfc = RandomForestClassifier(max_depth=None, random_state=None)
    rfc.fit(X_train, y_train)
    # predict using X_Test
    y_pred = rfc.predict(X_test)
    print(y_pred)
    ypredlenth = len(y_pred)
    # print("Last Project Success or Fail", y_pred[ypredlenth-1])
    # print("Last but 1 Project Success or Fail", y_pred[ypredlenth - 2])
    # print("Last but 2 Project Success or Fail", y_pred[ypredlenth - 3])
    # to know the lenght of predicted
    # print(len(y_pred))
    # brining back to normal shape
    # now find out precision_score, recall_score, f1_score
    # to build confusion matrix actual vs predicted
    raccuracyScore = round(accuracy_score(y_test, y_pred), 2)
    confusion_matrix(y_test, y_pred)
    rprecisionScore = round(precision_score(y_test, y_pred), 2)
    rrecallScore = round(recall_score(y_test, y_pred), 2)
    rf1Score = round(f1_score(y_test, y_pred), 2)
    rfbetaScore = round(fbeta_score(y_test, y_pred, beta=0.5, average='micro'), 2)

    print(raccuracyScore, rprecisionScore, rrecallScore, rf1Score, rfbetaScore)

    lrs = LogisticRegression()
    lrs.fit(X_train, y_train)
    # predict using X_Test
    y_pred = lrs.predict(X_test)
    print(y_pred)
    ypredlenth = len(y_pred)
    # print("Last Project Success or Fail", y_pred[ypredlenth-1])
    # print("Last but 1 Project Success or Fail", y_pred[ypredlenth - 2])
    # print("Last but 2 Project Success or Fail", y_pred[ypredlenth - 3])
    # to know the lenght of predicted
    # print(len(y_pred))
    # now find out precision_score, recall_score, f1_score
    # to build confusion matrix actual vs predicted
    laccuracyScore = round(accuracy_score(y_test, y_pred), 2)
    confusion_matrix(y_test, y_pred)
    lprecisionScore = round(precision_score(y_test, y_pred), 2)
    lrecallScore = round(recall_score(y_test, y_pred), 2)
    lf1Score = round(f1_score(y_test, y_pred), 2)
    lfbetaScore = round(fbeta_score(y_test, y_pred, beta=0.5, average='micro'), 2)
    print(laccuracyScore, lprecisionScore, lrecallScore, lf1Score, lfbetaScore)

    pcontext['eyesymptom']=0 if pcontext['eyesymptom']=="No" else 1
    print(pcontext['eyesymptom'])
    pcontext['chestsymptom'] = 0 if pcontext['chestsymptom'] == "No" else 1
    pcontext['soarsymptom'] = 0 if pcontext['soarsymptom'] == "No" else 1
    pcontext['runnynosesymptom'] = 0 if pcontext['runnynosesymptom'] == "No" else 1
    pcontext['weaknesssymptom'] = 0 if pcontext['weaknesssymptom'] == "No" else 1
    pcontext['achessymptom'] = 0 if pcontext['achessymptom'] == "No" else 1
    pcontext['headachesymptom'] = 0 if pcontext['headachesymptom'] == "No" else 1
    pcontext['coughsymptom'] = 0 if pcontext['coughsymptom'] == "No" else 1
    pcontext['breathingsymptom'] = 0 if pcontext['breathingsymptom'] == "No" else 1
    pcontext['sleepsymptom'] = 0 if pcontext['sleepsymptom'] == "No" else 1
    pcontext['meetppl'] = 0 if pcontext['meetppl'] == "No" else 1
    pcontext['coronatest'] = 0 if pcontext['coronatest'] == "No" else 1
    pcontext['corona14'] = 0 if pcontext['corona14'] == "No" else 1
    if pcontext['disease']=='High BP-2':
        pcontext['disease']=2
    elif pcontext['disease']=='Asthama-3':
        pcontext['disease']=3
    else:
        pcontext['disease']=1
    pcontext['gone_out'] = 0 if pcontext['gone_out'] == "No" else 1
    pcontext['gone_out1'] = 0 if pcontext['gone_out1'] == "No" else 1
    pcontext['gone_out2'] = 0 if pcontext['gone_out2'] == "No" else 1
    pcontext['spent_time'] = 0 if pcontext['spent_time'] == "No" else 1
    pcontext['more_meet'] = 0 if pcontext['more_meet'] == "No" else 1
    pcontext['crowd'] = 0 if pcontext['crowd'] == "No" else 1
    pcontext['transport'] = 0 if pcontext['transport'] == "No" else 1
    pcontext['home'] = 0 if pcontext['home'] == "No" else 1
    pcontext['any'] = 0 if pcontext['any'] == "No" else 1
    if pcontext['prec']=='Sometimes':
        pcontext['prec']=2
    elif pcontext['prec']=='Often':
        pcontext['prec']=3
    elif pcontext['prec']=='Always':
        pcontext['prec']=4
    else:
        pcontext['prec']=0

    if pcontext['times'] == 'None of the time':
        pcontext['times'] = 0
    elif pcontext['times'] == 'Most of the time':
        pcontext['times'] = 3
    elif pcontext['times'] == 'All of the time':
        pcontext['times'] = 4
    else:
        pcontext['times'] = 2
    pcontext['gen'] = 0 if pcontext['any'] == "Female" else 1
    pcontext['smoke'] = 0 if pcontext['smoke'] == "No" else 1
    pcontext['alc'] = 0 if pcontext['alc'] == "No" else 1
    pcontext['handwash'] = 0 if pcontext['handwash'] == "No" else 1
    pcontext['handwash2'] = 0 if pcontext['handwash2'] == "No" else 1
    if pcontext['age']=='18-25':
        pcontext['age']=0
    elif pcontext['age']=='26-32':
        pcontext['age']=1
    elif pcontext['age']=='33-45':
        pcontext['age']=2
    else:
        pcontext['age']=3

    # trying to predict new projects sucess using this algorithm
    testSet = [[[pcontext['eyesymptom'],pcontext['chestsymptom'],pcontext['soarsymptom'],
                 pcontext['runnynosesymptom'],pcontext['weaknesssymptom'],pcontext['achessymptom'],
                 pcontext['headachesymptom'],pcontext['coughsymptom'],pcontext['breathingsymptom'],
                 pcontext['sleepsymptom'],pcontext['meetppl'],pcontext['coronatest'],
                 pcontext['corona14'],pcontext['disease'],pcontext['gone_out'],
                 pcontext['gone_out1'],pcontext['gone_out2'],pcontext['spent_time'],
                 pcontext['more_meet'],pcontext['crowd'],pcontext['transport'],
                 pcontext['home'],pcontext['any'],pcontext['prec'],pcontext['times'],pcontext['gen'],
                 pcontext['smoke'],pcontext['alc'],pcontext['handwash'],pcontext['handwash2'],pcontext['age']
               ]]]
    # print(testSet[0][0][0])
    print(testSet)
    predTest = []
    predTest = testSet
    testSetValues = []
    print("project scores are ")
    newy_pred1 = lrs.predict(testSet[0])
    testSetValues.append(newy_pred1[0])
    print("lrs newy_pred", newy_pred1)
    newy_pred1 = rfc.predict(testSet[0])
    testSetValues.append(newy_pred1[0])
    print("rfc newy_pred", newy_pred1)
    newy_pred1 = gnb.predict(testSet[0])
    testSetValues.append(newy_pred1[0])
    print("gnb newy_pred", newy_pred1)
    newy_pred1 = dtc.predict(testSet[0])
    testSetValues.append(newy_pred1[0])
    print("dtc newy_pred", newy_pred1)

    metrics =  {'precisionScore': precisionScore, 'recallScore': recallScore, 'f1Score': f1Score,
                   'accuracyScore': accuracyScore, 'fbetaScore': fbetaScore, 'daccuracyScore': daccuracyScore,
                   'dprecisionScore': dprecisionScore, 'drecallScore': drecallScore, 'df1Score': df1Score,
                   'dfbetaScore': dfbetaScore, 'raccuracyScore': raccuracyScore, 'rprecisionScore': rprecisionScore,
                   'rrecallScore': rrecallScore, 'rf1Score': rf1Score, 'rfbetaScore': rfbetaScore,
                   'laccuracyScore': laccuracyScore, 'lprecisionScore': lprecisionScore, 'lrecallScore': lrecallScore,
                   'lf1Score': lf1Score, 'lfbetaScore': lfbetaScore, 'predTest': predTest, 'testSetValues1': testSetValues[0],
                    'testSetValues2': testSetValues[1],'testSetValues3': testSetValues[2],'testSetValues4': testSetValues[3]
                   }
    return metrics
