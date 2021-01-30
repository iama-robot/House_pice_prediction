#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from flask import Flask, render_template, request,jsonify
import pickle


# In[3]:


app = Flask(__name__) #referencing this file


# In[4]:


model = pickle.load(open("model.pkl", 'rb'))


# In[5]:


@app.route('/', methods=['GET', 'POST'])
def basic():
    print('A')
    return render_template('pred.html')


# In[6]:


@app.route('/predict', methods = ['POST','GET'])
def predict():
        print('B')
        rooms = int(request.form['rooms']) 
        parking_spaces = int(request.form['parking spaces']) 
        city = request.form['city']
        if(city=='Belo Horizonte'):
            Belo_Horizonte=1
            Campinas= Porto_Alegre= Rio_de_Janeiro= São_Paulo=0
        elif(city=='Campinas'):
            Campinas=1
            Belo_Horizonte= Porto_Alegre= Rio_de_Janeiro= São_Paulo=0
        elif(city=='Porto Alegre'):
            Porto_Alegre=1
            Belo_Horizonte= Campinas= Rio_de_Janeiro= São_Paulo=0
        elif(city=='Rio de Janeiro'):
            Rio_de_Janeiro=1
            Belo_Horizonte= Campinas= Porto_Alegre= São_Paulo=0
        else:
            São_Paulo=1
            Belo_Horizonte= Campinas=Porto_Alegre=Rio_de_Janeiro=0
        bathroom = int(request.form['bathrooms'])
        furniture = request.form['furniture']
        if(furniture=='furnished'):
            furnished = 1
            not_furnished=0
        elif(furniture=='not furnished'):
            furnished = 0
            not_furnished=1
        fire_insurance = float(request.form['fire insurance'])
        pred = model.predict([[Belo_Horizonte,Campinas, Porto_Alegre, Rio_de_Janeiro, São_Paulo, rooms, parking_spaces, furnished, not_furnished, fire_insurance, bathroom]])
        pred=round(pred[0],2)
        return render_template('index.html',prediction="You Can Sell The House at {}".format(pred))


# In[12]:


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




