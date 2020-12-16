from flask import Flask
from flask import request
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def fun():
        value=request.form['value']
        value=request.form['value']
        company=value
        import urllib.request 
        from pprint import pprint       
        from html_table_parser import HTMLTableParser
        import pandas as pd
        
##        return "nitin"
        
        pd.set_option('display.max_columns', None)

        url="https://www.moneycontrol.com/india/stockpricequote/food-processing/dfmfoods/DFM"
        url1="https://www.moneycontrol.com/india/stockpricequote/computers-software/infosys/IT"

        from googlesearch import search 
        x=company
        y="share"
        z="moneycontrol"
        query = x+y+z
        for j in search(query, tld="co.in"): 
            print(j)
            url=j 
            break

        req = urllib.request.Request(url) 
        f = urllib.request.urlopen(req) 
        xhtml=f.read().decode('utf-8')
        p = HTMLTableParser()  
        p.feed(xhtml) 


        x1=pd.DataFrame(p.tables[15])
        x1=x1.set_index([0])
        x1.columns = [''] * len(x1.columns)
        print("x=",x1.index[1])

        if(x1.index[1]=="Sales"):
            print(x1)
        else:
            print("i")
            i=1
            while(True):
              x1=pd.DataFrame(p.tables[i])
              x1=x1.set_index([0])
              x1.columns = [''] * len(x1.columns)
              if(x1.index[1]=="Sales"):
                print(x1)
                break
              i=i+1
          
        print("k")
        x2=pd.DataFrame(p.tables[28])
        x2=x2.set_index([0])
        x2.columns = [''] * len(x2.columns)
        print("x2=",x2.index[1])
        if(x2.index[1]=="Promoters"):
            print(x2)
        else:
            i=1
            print("k2")
            while(True):
                  x2=pd.DataFrame(p.tables[i])
                  x2=x2.set_index([0])
                  x2.columns = [''] * len(x2.columns)
                  print(x2.index[1])
                  if(x2.index[1]=="Promoters"):
                    print(x2)
                    break
                  i=i+1
            x1=x1.replace(',','',regex=True)
            x2=x2.replace(',','',regex=True)
            print(x1,x2)

    
        def funda(x1,x2):
          score=0.0

          #promoters---------------------------------------------------------------------------
          print(float((x2.iloc[1])[0]))
          if(int(float((x2.iloc[1])[0])) > 50):
            score=score+1
            if(int(float((x2.iloc[1])[0])) > 70):
              score=score+1
          k=0.75
          print(type(x2.iloc[1]))
          for i in range(3):
            if(float((x2.iloc[1])[i])-float((x2.iloc[1])[i+1])>0):
              #print(int(float((x2.iloc[1])[i]))," ",int(float((x2.iloc[1])[i+1])))
              score=score+k
            k=k-0.25
          print("score -pro:",score)

          #pleging---------------------------------------------------------------------------
          p=int(float((x2.iloc[2])[0]))
          if(p==0):
            score=score+1
            if(p<25):
              score=score+0.5
              if(p<50):
                score=score+0.25
          k=0.75
          for i in range(3):
            if(float((x2.iloc[2])[i])-float((x2.iloc[2])[i+1])<0):
              #print(int(float((x2.iloc[2])[i]))," ",int(float((x2.iloc[2])[i+1])))
              score=score+k
            k=k-0.25
          print("score-pleg:",score)

          #fii---------------------------------------------------------------------------
          p=int(float((x2.iloc[3])[0]))
          print(p)
          if(int(float((x2.iloc[3])[0])) > 0):
            score=score+0.5
          k=0.50
          for i in range(3):
            if(float((x2.iloc[3])[i])-float((x2.iloc[3])[i+1])>0):
              print(float((x2.iloc[3])[i])," ",float((x2.iloc[3])[i+1]))
              score=score+k
            k=k-0.15
          print("score-fii:",score)

          #mf---------------------------------------------------------------------------
          p=int(float((x2.iloc[7])[0]))
          if(int(float((x2.iloc[7])[0])) > 0):
            score=score+0.5
          k=0.50
          for i in range(3):
            if(float((x2.iloc[7])[i])-float((x2.iloc[7])[i+1])>0):
              print(float((x2.iloc[7])[i])," ",float((x2.iloc[7])[i+1]))
              score=score+k
            k=k-0.15
          print("score-mf:",score)


          #netprofit
          print((x1.iloc[8])[0])
          ((x1.iloc[8])[0])=((x1.iloc[8])[0]).replace(",","")
          print((x1.iloc[8])[0])
          p=int(float((x1.iloc[8])[0]))
          print(p)
          if(int(float((x1.iloc[8])[0])) > 0):
            score=score+0.5
          k=0.50
          for i in range(4):
            if(float((x1.iloc[8])[i])-float((x1.iloc[8])[i+1])>0):
              print(float((x1.iloc[8])[i])," ",float((x1.iloc[8])[i+1]))
              score=score+k
            k=k-0.15
          print("score-mf:",score)

          
            
          print(p)
          print("    ")    
          print("Fundamental score:",score)
          print("    ")
          return score

        from datetime import date
        from nsepy import get_history
        df = get_history(symbol=company, start=date(2019,1,1), end=date(2020,12,4))
        import pandas as pd
        df1=df.reset_index()['Close']
        import numpy as np
        li=[]
        li=df['Close']
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        #feature_range=(0,1)
        scaler=MinMaxScaler()
        #.reshape(-1,1))
        df1=scaler.fit_transform(np.array(li).reshape(-1,1))
        #invers=scaler.inverse_transform(li)
        
        from sklearn import linear_model
        from sklearn.model_selection import train_test_split
        data=df
        data = data.iloc[:3]
        data1=data[:-1]
        data2=data[1:]

        x=data[["Open","High","Low","Volume","Turnover","Trades","Deliverable Volume","VWAP"]]
        y=data[["Close"]]
        print(data.head())
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20)
        print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

        regr = linear_model.LinearRegression()
        regr.fit(X_train,Y_train)
        
        Y_pred = (regr.predict(X_test))
        last=data["Close"].iloc[-1]
        print(Y_pred[0][0],last)
        if int(last)>int(Y_pred[0][0]):
                xx="sell"
        else:
                xx="buy"
        kk=funda(x1,x2)
        kk=float(kk)
        fundascore=str(round(kk, 2))
        Y_pred[0][0]=float(Y_pred[0][0])
        dd=str(round(Y_pred[0][0],2))
        tup=(xx,",",fundascore,",",dd)
        str1 =  ''.join(tup)
        
        
        
        return str1
        '''

        ##splitting dataset into train and test split
        training_size=int(len(df1)*0.65)
        test_size=len(df1)-training_size
        train_data,test_data = df1[0:training_size,:],df1[training_size:len(df1),:1]
        training_size,test_size

        import numpy
        # convert an array of values into a dataset matrix
        def create_dataset(dataset, time_step=1):
                dataX, dataY = [], []
                for i in range(len(dataset)-time_step-1):
                        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
                        dataX.append(a)
                        dataY.append(dataset[i + time_step, 0])
                return numpy.array(dataX), numpy.array(dataY)

        # reshape into X=t,t+1,t+2,t+3 and Y=t+4
        time_step = 100
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, ytest = create_dataset(test_data, time_step)
        print(X_train.shape), print(y_train.shape)

        # reshape input to be [samples, time steps, features] which is required for LSTM
        X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

        # Create the Stacked LSTM model
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import LSTM



        model=Sequential()
        model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
        model.add(LSTM(50,return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',optimizer='adam')

        invers = scaler.inverse_transform(df1)
        invers

        model.summary()

        model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=2,batch_size=6,verbose=1)

        import tensorflow as tf
        tf.__version__

        invers = scaler.inverse_transform(df1)
        invers

        ### Lets Do the prediction and check performance metrics
        train_predict=model.predict(X_train)
        test_predict=model.predict(X_test)

        ##Transformback to original form
        train_predict=scaler.inverse_transform(train_predict)
        test_predict=scaler.inverse_transform(test_predict)

        ### Calculate RMSE performance metrics
        import math
        from sklearn.metrics import mean_squared_error
        (math.sqrt(mean_squared_error(y_train,train_predict)))

        import matplotlib.pyplot as plt
        look_back=100
        trainPredictPlot = numpy.empty_like(df1)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
        # shift test predictions for plotting
        testPredictPlot = numpy.empty_like(df1)
        testPredictPlot[:, :] = numpy.nan
        testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
        # plot baseline and predictions
##        plt.plot(scaler.inverse_transform(df1))
##        plt.plot(trainPredictPlot)
##        plt.plot(testPredictPlot)
##        plt.show()

        len(test_data)

        x_input=test_data[len(test_data)-100:].reshape(1,-1)
        x_input.shape

        temp_input=(list(x_input))
        temp_input=temp_input[0].tolist()
        (temp_input)

        # demonstrate prediction for next 60 days
        from numpy import array

        lst_output=[]
        n_steps=100
        i=0
        while(i<60):
            
            if(len(temp_input)>100):
                #print(temp_input)
                x_input=np.array(temp_input[1:])
                print("{} day input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                #print(x_input)
                yhat = model.predict(x_input, verbose=0)
                print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                #print(temp_input)
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = model.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i=i+1
            

        #print(lst_output)

        day_new=np.arange(1,101)
        day_pred=np.arange(101,161)

        len(df1)

        ##plt.plot(day_new,scaler.inverse_transform(df1[(len(df1)-100):]))
        #plt.plot(day_new,scaler.inverse_transform(df1[4916:]))
        ##plt.plot(day_pred,scaler.inverse_transform(lst_output))

        df3=df1.tolist()
        df3.extend(lst_output)
        plt.plot(df3[(len(df1)-1000):])
        plt.plot(df1[(len(df1)-1000):])

        print(scaler.inverse_transform(df1[(len(df1)-100):]))

        df3=scaler.inverse_transform(df3).tolist()

        df1=scaler.inverse_transform(df1)

        ##plt.plot((df3))
        #___________df3 has future predictions______________
        ##plt.plot((df1))

        #df3=scaler.inverse_transform(df3).tolist()
        #plt.plot(scaler.inverse_transform(df3))
        #plt.plot(scaler.inverse_transform(df1))

        ### Calculate RMSE performance metrics
        import statistics 
        #import math
        #from sklearn.metrics import mean_squared_error
        #(math.sqrt(mean_squared_error(y_train,train_predict)))
        df4=[]
        #__________________df4 is future prediction of 60 days__________________
        for i in range(60):
          df4.append(df3[-(1+i)])
        ##d=(np.array(df4).reshape(1,60))
        df5=[]
        #__________________df5 is last 60 days of data__________________
        for i in range(60):
          df5.append(df1[-(1+i)])
        df4

        np.sum(df4)

        if np.sum(df4)>=np.sum(df5):
          xx="Buy"
        else:
          xx="Sell"
        return xx
        ##return {'data': df4,'call':xx}
        '''
if __name__ == '__main__':
     app.run(host='0.0.0.0')





#WIPRO,LTI,DFM,TATAMOTORS,INFOSYS
















































