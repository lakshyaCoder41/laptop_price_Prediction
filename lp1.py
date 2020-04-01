import pandas as pd
import numpy as np
import pickle
import os
import xgboost as xgb
from flask import Flask, flash, redirect, render_template,request, url_for
from flask_wtf import FlaskForm
from wtforms import StringField

app = Flask(__name__)

@app.route('/')
@app.route('/index1')
def index1():
    return render_template('index1.html')
@app.route('/test',methods=['GET','POST'])
def test():
    X,Y,independent_variables,mini,maxi,X_test,Y_test=data_set()
    if request.method=='GET':
        processor = request.args.get('processor')
        os = request.args.get('os')
        gpu = request.args.get('gpu')
        memtype = request.args.get('memtype')
        wght = request.args.get('weight')
        ramval = request.args.get('ram')
        memsize = request.args.get('size1')
        memsize2 = request.args.get('size2')
        dim = request.args.get('dimension')
    dictx= {}
    for i in independent_variables:
        dictx[i] = 0

    #mapping for vendors
    print(type(processor))
    if processor[0]== '1':
        dictx['cpu_Intel'] = 1
    else:
        pass
        dictx['cpu_AMD'] = 1
    dictx['cpu'] = int(processor[1:])

    # mapping for operating system

    if os[0] == '1':
        dictx['os_windows'] = 1
    elif os[0] == '2':
        dictx['os_mac'] = 1
    else:
        dictx['os_other'] = 1
    dictx['version'] = int(os[1:])

     # mapping for gpu

    if gpu[0] == '1':
        dictx['gpu_Intel'] = 1
    elif gpu[0] == '2':
        dictx['gpu_AMD'] = 1
    else:
        dictx['gpu_Nvidia'] = 1

    dictx['gputype_1'] = -int(gpu[1:])
    # mapping for memory type:
    if len(memtype) == 2:
        dictx['memory_type2'] = int(memtype[1])
    dictx['memory_type1'] = int(memtype[0])

    # mapping textfield values
    dictx['ram'] = int(ramval)
    dictx['mem_size1'] = int(memsize)
    dictx['mem_size2'] = int(memsize2)
    dictx['weight'] = float(wght)
    dictx['dimension_inch'] = float(dim)
    price=prediction(dictx,X,Y,independent_variables,mini,maxi,X_test,Y_test)
    #print(dictx)
    return render_template('index1.html',price=int(price*65),spec = dictx)


def data_set():
        data=pd.read_excel(r'newdata.xlsx')
        from sklearn.model_selection import train_test_split
        independent_variables = data.columns

        independent_variables = independent_variables.delete(24)
        Y=data['price']
        X = data[independent_variables[6:]]
        mini = 1.0
        maxi = -.30
        #for i in range(100):
        print(independent_variables)
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size = 0.1)
        return X,Y,independent_variables,mini,maxi,X_test,Y_test

def prediction(test_input,X,Y,independent_variables,mini,maxi,X_test,Y_test):
    pickle_in = open('xgboost.pickle','rb')
    clf = pickle.load(pickle_in)
    res = float(clf.score(X_test,Y_test))
    if maxi < res :
        maxi = res
    if mini> res:
        mini = res
    df = pd.DataFrame(data = test_input,index = [0], columns=independent_variables[6:])
    price=clf.predict(df)
    return price[0]

if __name__=='__main__':
    app.run(debug=True)
