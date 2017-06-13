import pandas as pd
import numpy as np

class inputDecode(object):

    def __init__(self, json):
        self.data = self.decode(json)

    def CreatePre(self):
        shooter = self.data[self.data[:,0] == 1,:]
        playerxy = self.data[:,2:4]
        hoop = np.array([41.65,25])
        diff = playerxy - hoop
        hdist = np.sqrt(((diff)**2).sum(axis = 1))
        angle = np.arctan2(diff[:,0], diff[:,1])
        cossim = np.cos(angle - angle[shotind])
        boxes = boxgen(playerxy)
        frm = pd.DataFrame(np.concatenate([self.data, hdist, boxes, angle, cossim], axis = 1), columns=cols)
        frm.sort_values(by = ['Off', 'pre_HDist'], ascending = ['False', 'False'], inplace = True)
        self.frame = frm.reset_index(inplace = True)
