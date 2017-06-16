import pandas as pd
import numpy as np
import requests, json
from flask import Flask, jsonify, request
import cPickle as pickle
from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adadelta, Adagrad
import os
import random
from flask_cors import CORS, cross_origin

# closer = ((df['pre_HDist'] < df['pos_HDist']).astype(int) - 0.5) * 2
# move = nums[:,[0,1]] - nums[:,[4,5]]
# df['MoveV'] = np.sqrt((move ** 2).sum(axis = 1)) * closer

def features(df):
    """
    Modified copy of script from coordinator.py to prepare data for modeling.
    """
    """
    #1.0 Sorts by offense.
    """
    df.sort_values(by = 'Off', inplace = True)

    """
    #2.0 Separates numeric columns for #math operations.
    """
    numcols = ['x', 'y']

    """
    #2.1 Separates out shooter specifically for comparison columns.
    """
    shooter = df[df['isShoot'] == True][numcols].values

    """
    #3.0 Computes various numeric stats including how they move, their
         cosine similarity to the shooter, and also how they are boxing out.
    """

    nums = df[numcols].values
    hoop = np.array([41.65, 25])
    diff = nums - hoop
    df['HDist'] = np.sqrt((diff ** 2).sum(axis = 1))
    angle = np.arctan2(diff[:,0], diff[:,1])
    df['Angle'] = angle
    df['CosSim'] = np.cos(angle - angle[df['isShoot'] == True])
    df['Box'] = boxgen(nums)
    df.sort_index(inplace = True)
    return df

def boxgen(arr):
    """
    Uses a single iteration of K-means to identify who is boxing out whom.
    Basically, uses one team as sample centroids, and then looks at the number
    of elements in a centroid to determine how many players one player is boxing
    out. Ideally will also add column that indicates how many are closer to the
    hoop.
    """
    bdist = np.sqrt(((arr - np.array([41.65,25])) ** 2).sum(axis = 1)).reshape(10,1)
    arr = np.concatenate((arr, bdist), axis = 1)
    arr1 = arr[:5, :]
    arr2 = arr[5:,:]
    dists = np.array([np.sqrt(((arr1[:,:2] - row)**2).sum(axis = 1)) for row in arr2[:,:2]])
    d = np.argmin(dists, axis = 1)
    o = np.argmin(dists, axis = 0)
    dbox = [np.sum(o == i) for i in xrange(5)]
    obox = [np.sum(d == i) for i in xrange(5)]
    boxes = np.array(dbox + obox)
    return boxes

class normer(object):

    def __init__(self):
        pass

    def fit(self, arr):
        self.ms = arr.ms(axis = 0)
        self.sds = arr.sds(axis = 0)

    def transform(self, arr):
        return (arr - self.ms)/self.sds

    def fit_transform(self, arr):
        self.fit(arr)
        return self.transform(arr)



class inputDecode(object):

    def __init__(self, posModel, model, normer):
        self.posModel = posModel
        self.model = model
        self.normer = normer

    def CreatePre(self, df):
        self.pre = features(df)
        self.preArr = self.pre[['x','y']]

    def CreatePos(self):
        self.posArr = self.posModel.predict(self.normer.transform(self.pre.values), batch_size = 40)
        self.pos = self.pre.copy()
        self.pos['x'] = self.posArr[:,0]
        self.pos['y'] = self.posArr[:,1]
        self.pos['newy'] = self.posArr[:,0]
        self.pos['newx'] = self.posArr[:,1]
        self.pos = features(self.pos)
        move = self.posArr - self.preArr
        closer = ((self.pre['HDist'] < self.pos['HDist']).astype(int) - 0.5) * 2
        self.pos.pop('Off')
        self.pos.pop('isShoot')
        self.pos['MoveV'] = np.sqrt((move ** 2).sum(axis = 1)) * closer
        res = decoder.pos[['newx', 'newy']]
        res['probability'] = np.random.rand(10,1)
        return res

    def Modeling(self, fitModel):
        self.modIn = np.concatenate([self.pre.values, self.pos[['x', 'y', 'HDist', 'Angle', 'CosSim', 'Box', 'MoveV']].values], axis = 1)
        self.model = fitModel
        self.pos['probability'] = self.model.predict(self.modIn)
        return self.pos[['newx', 'newy', 'probability']]

app = Flask(__name__)
CORS(app)

posnn = load_model('posnn.h5')
norm = pickle.load(open('./normer.pkl', 'rb'))
# Model = pickle.load(open('./FinalModel.pkl', 'rb'))

script = inputDecode(posModel = posnn, model = 'Model', normer = norm)

@app.route('/predict', methods = ["GET", "POST"])
def predict():
    d = request.form
    print d.keys()
    print 'boop'
    print d.getlist('name[]')
    print 'boop'
    print d['bench']
    data = json.loads(d)
    print data
    df = pd.DataFrame(data['bench'])
    df.columns = ['Off', 'isShoot', 'y', 'x']
    script.CreatePre(df)
    res = script.CreatePos()

    # pre = decoder.pos
    # res = fitter.predict(playarray).reshape(10,1)
    # results = pd.DataFrame(np.concatenate([pre, res], axis = 1), columns = ['NewX', 'NewY', 'Probability'])
    # d = results.to_dict(orient = 'index').values
    # j = jsonify(d)
    test = res.values()
    print 'received'
    return jsonify(test)


if __name__ == '__main__':

    port = int(os.getenv('PORT', 9099))
    app.run(host='0.0.0.0', port=port)
