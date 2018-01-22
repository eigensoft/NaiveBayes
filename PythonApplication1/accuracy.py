
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from ClassifyNB import NBAccuracy
from ClassifySVM import SVMAccuracy

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()
print('NB:' + str(NBAccuracy(features_train, labels_train, features_test, labels_test)))
print('SVM:' + str(SVMAccuracy(features_train, labels_train, features_test, labels_test)))