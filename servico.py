# -*- coding: utf-8 -*-
from flask import Flask
app = Flask(__name__)
from tensorflow import keras
import numpy as np
import pickle
from datetime import datetime
import tensorflow as tf
from flask import jsonify
from flask import request
import limpar_texto as lt
import train_ngram_model as tr

app = Flask(__name__)

cache = {};


def loadModel():
	
	cache['new_model'] = keras.models.load_model('documentos_mpl_model.h5')
	cache['selector'] = pickle.load(open("selector.pickle", "rb"))
	cache['vectorizer'] = pickle.load(open("vectorizer.pickle", "rb"))

	cache['labels'] = pickle.load(open("labels.pickle", "rb"))

@app.route("/predict", methods=['POST', 'GET'])
def predict():
	if request.method == 'POST':
		loadModel()

		val_texts = request.json['dados']
		val_texts = list(map(lt.cleanhtml,val_texts))

		print("=========########################==========")
		for v in val_texts:
			print(v)
		
		print("========####################===========")
		print(type(val_texts))
		#cache['new_model'].summary()


			
		
		x_val = cache['vectorizer'].transform(val_texts)
		x_val = cache['selector'].transform(x_val)
		x_val = x_val.astype('float32')
		results = cache['new_model'].predict(x_val)
		
		retorno = {}
		retorno['sucesso'] = 'true'
		
		retorno['dados'] = []

		for result in results:
			r = {}
			#r['classe'] = labels[np.argmax(result)]
			#r['percentuais'] = result.tolist()
			retorno['dados'].append(cache['labels'][np.argmax(result)])
		
		return jsonify(retorno)
    
@app.route("/treinar", methods=['POST', 'GET'])
def treinar():
	if request.method == 'GET':
		acuracia,labels,num_classes = tr.treinar()
		retorno = {}
		retorno['sucesso'] = 'true'
		retorno['acuracia'] = str(acuracia)
		retorno['num_reg'] = str(num_classes)
		retorno['labels'] = list(labels)
		print (retorno)
		return jsonify(retorno)    

if __name__ == '__main__':	
	app.run(debug=True,host='0.0.0.0')