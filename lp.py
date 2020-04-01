import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from flask import Flask, flash, redirect, render_template,request, url_for

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template(
        'index.html',
        dim=[{'name':'13.5'},{'name':'14.5'},{'name':'15.5'},{'name':'17.5'}],
        ram=[{'name':'4'}, {'name':'8'}, {'name':'16'}],
        memory=[{'name':'SSD'},{'name':'HDD'},{'name':'Flash'}],
        cpu=[{'name':'Intel'},{'name':'AMD'}],
        gpu=[{'name':'Intel'},{'name':'AMD'},{'name':'Nvidia'}],
        os=[{'name':'Mac'},{'name':'Windows'},{'name':'Others'}],
        ver=[{'name':'0'},{'name':'1'},{'name':'7'},{'name':'10'}],
        weight=[{'name':'weight'}])

@app.route("/test", methods=['GET', 'POST'])
def test():
    if request.method=='POST':
        dim=request.form.get('dim')
        ram =request.form.get('RAM')
        mem = request.form.get('mem')
        cpu=  request.form.get('cpu')
        gpu= request.form.get('gpu')
        os= request.form.get('os')
        ver=request.form.get('ver')
        weight=request.form.get('weight')
        li=[ram,mem,dim,cpu,gpu,os,ver,weight]
    if mem=='SSD' :
        ssd=1
        hdd=0
        flash=0
    elif mem=='HDD':
        ssd=0
        hdd=1
        flash=0
    else:
        flash=1
        ssd=0
        hdd=0
    if cpu=='Intel':
        intel=1
        amd=0
    elif cpu=='AMD':
        amd=0
        intel=1
    if gpu=='Intel':
        igpu=1
        agpu=0
        ngpu=0
    elif gpu=='AMD':
        agpu=1
        igpu=0
        ngpu=0
    elif gpu=='Nvidia':
        ngpu=1
        igpu=0
        agpu=0
    if os=='Mac':
        mac=1
        win=0
        oth=0
    elif os=='Windows':
        win=1
        mac=0
        oth=0
    else:
        oth=1
        win=0
        mac=0
    if ver=='0':
        ver=0
    elif ver=='1':
        ver=1
    elif ver=='7':
        ver=7
    else:
        ver=10
    res=[dim,intel,amd,17,ram,ssd,hdd,256,0,igpu,agpu,ngpu,-275,mac,oth,win,ver,17.5]
    price,mini,maxi=data_loading_and_prediction(res)
    #res=dim+' '+ram+' '+mem+' '+cpu+' '+gpu+' '+os+' '+ver+' '+weight
    return render_template('index.html',price=str(price),res=li,mini=mini,maxi=maxi)
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
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size = 0.1)
    return X,Y,independent_variables,mini,maxi,X_test,Y_test
#     clf = xgb.XGBRegressor(
#     max_depth=5,
#     learning_rate=0.05,
#     gamma=0.0,
#     min_child_weight=0.0,
#     max_delta_step=0.0,
#     subsample=0.75,
#     colsample_bytree=0.75,
#     colsample_bylevel=1.0,
#     reg_alpha=0.0,
#     reg_lambda=1.0,
#     n_estimators=300,
#     silent=0,
#     nthread=4,
#     scale_pos_weight=1.0,
#     base_score=0.5,
#     seed=1337,
#     missing=None,
#     )
#     clf.fit(X_train,Y_train)
#     ac=clf.score(X_test,Y_test)
#     print(ac)
#     if ac>=0.90:
#         break
#     with open('xgboost.pickle','wb') as f:
#         pickle.dump(clf,f)
def data_loading_and_prediction(test_data):
    X,Y,independent_variables,mini,maxi,X_test,Y_test=data_set()
    pickle_in = open('xgboost.pickle','rb')
    clf = pickle.load(pickle_in)
    res = float(clf.score(X_test,Y_test))
    if maxi < res :
        maxi = res
    if mini> res:
        mini = res
    #print(mini,maxi)
    dicx={}
    index=0
    test_data[0]=float(test_data[0])
    test_data[4]=int(test_data[4])
    for feature in independent_variables[6:]:
        #temp=float(input("Enter "+feature+": "))
        dicx[feature]=test_data[index]
        dicx_df=pd.DataFrame(data=dicx,index=[0], columns=independent_variables[6:])
        index+=1
    price=clf.predict(dicx_df)
    return price,mini,maxi
            #print('\nPredicted Price: ',int(round(price[0])))






if __name__=='__main__':
    app.run(debug=True)
