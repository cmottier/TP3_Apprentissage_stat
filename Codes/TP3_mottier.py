#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from svm_source import *
from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

scaler = StandardScaler()

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')

#%%
###############################################################################
#               Iris Dataset
###############################################################################
# import iris dataset
np.random.seed = 20

iris = datasets.load_iris()
X = iris.data
X = scaler.fit_transform(X)
y = iris.target
X = X[y != 0, :2]
y = y[y != 0]

# split train test
X, y = shuffle(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# visualization
plt.figure()
plot_2d(X, y)
plt.title("iris dataset")

#%%
#############
# Question 1 
#############

# fit the model with linear kernel
clf_lin = SVC(kernel='linear')
clf_lin.fit(X_train, y_train)

# predict labels for the test data base
y_pred = clf_lin.predict(X_test)

# score
print('Generalization score for linear kernel: %s, %s' %
      (clf_lin.score(X_train, y_train),
       clf_lin.score(X_test, y_test)))

# display the frontiere
def f_lin(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_lin.predict(xx.reshape(1, -1))

plt.figure()
frontiere(f_lin, X_train, y_train, w=None, step=50, alpha_choice=1)


#%%
#############
# Question 2
#############

# fit the model with polynomial kernel
clf_poly = SVC(kernel='poly')
clf_poly.fit(X_train, y_train)

# predict labels for the test data base
y_pred = clf_poly.predict(X_test)

# score
print('Generalization score for polynomial kernel: %s, %s' %
      (clf_poly.score(X_train, y_train),
       clf_poly.score(X_test, y_test)))

# display the frontiere
def f_poly(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_poly.predict(xx.reshape(1, -1))

plt.figure()
frontiere(f_poly, X_train, y_train, w=None, step=50, alpha_choice=1)

#%%
# Display the results with linear and polynomial kernel

plt.ion()
plt.figure(figsize=(15, 5))
plt.subplot(131)
plot_2d(X, y)
plt.title("iris dataset")

plt.subplot(132)
frontiere(f_lin, X, y)
plt.title("linear kernel")

plt.subplot(133)
frontiere(f_poly, X, y)

plt.title("polynomial kernel")
plt.tight_layout()
plt.draw()

#%%
#############
# Remarque
#############

# fit the linear model with a parameter grid search
parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 300))}
clf_linear = SVC()
clf_linear_grid = GridSearchCV(clf_linear, parameters, n_jobs=-1)
clf_linear_grid.fit(X_train, y_train)

# score
print(clf_linear_grid.best_params_)
print('Best score : %s' % clf_linear_grid.score(X_test, y_test))

# display the frontiere
def f_lin_grid(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_linear_grid.predict(xx.reshape(1, -1))

plt.figure()
frontiere(f_lin_grid, X_train, y_train, w=None, step=50, alpha_choice=1)


# fit the polynomial model with a parameter grid search
Cs = list(np.logspace(-3, 3, 5))
gammas = 10. ** np.arange(1, 2)
degrees = np.r_[1, 2, 3]

parameters = {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree': degrees}
clf_polyn = SVC()
clf_polyn_grid = GridSearchCV(clf_polyn, parameters, n_jobs=-1)
clf_polyn_grid.fit(X_train, y_train)

# score
print(clf_polyn_grid.best_params_)
print('Best score : %s' % clf_polyn_grid.score(X_test, y_test))

# display the frontiere
def f_poly_grid(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_polyn_grid.predict(xx.reshape(1, -1))

plt.figure()
frontiere(f_poly_grid, X_train, y_train, w=None, step=50, alpha_choice=1)






###############################################################################
#               SVM GUI
###############################################################################

# please open a terminal and run python svm_gui.py
# Then, play with the applet : generate various datasets and observe the
# different classifiers you can obtain by varying the kernel




#%%
###############################################################################
#               Face Recognition Task
###############################################################################
"""
The dataset used in this example is a preprocessed excerpt
of the "Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  _LFW: http://vis-www.cs.umass.edu/lfw/
"""

# Download the data and unzip; then load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              color=True, funneled=False, slice_=None,
                              download_if_missing=True)
# data_home='.'

# introspect the images arrays to find the shapes (for plotting)
images = lfw_people.images
n_samples, h, w, n_colors = images.shape

# the label to predict is the id of the person
target_names = lfw_people.target_names.tolist()

# Pick a pair to classify such as
names = ['Tony Blair', 'Colin Powell']

idx0 = (lfw_people.target == target_names.index(names[0]))
idx1 = (lfw_people.target == target_names.index(names[1]))
images = np.r_[images[idx0], images[idx1]]
n_samples = images.shape[0]
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(int)

# plot a sample set of the data
plot_gallery(images, np.arange(12))
plt.show()

#%%
# Extract features

# features using only illuminations
X = (np.mean(images, axis=3)).reshape(n_samples, -1)

# # or compute features using colors (3 times more features)
# X = images.copy().reshape(n_samples, -1)

# Scale features
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

# Split data into a half training and half test set

indices = np.random.permutation(X.shape[0])
train_idx, test_idx = indices[:X.shape[0] // 2], indices[X.shape[0] // 2:]
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
images_train, images_test = images[
    train_idx, :, :, :], images[test_idx, :, :, :]

# plot_gallery(images_train, np.arange(12))
# plt.show()

# plot_gallery(images_test, np.arange(12))
# plt.show()

#%%
##############
# Question 4
##############

# fit a classifier (linear) and test all the Cs
Cs = 10. ** np.arange(-5, 6)
error = []
for C in Cs:
    clf = SVC(kernel='linear', C=C)
    clf.fit(X_train, y_train)
    error.append(1-clf.score(X_test, y_test))

# display the results
plt.figure()
plt.plot(Cs, error)
plt.xlabel("Parametres de regularisation C")
plt.ylabel("Erreur de prédiction")
plt.xscale("log")
plt.tight_layout()
plt.show()

#%%
# Best result
ind = np.argmin(error)
print("Best C: {}".format(Cs[ind]))
print("Best accuracy: {}".format(1-np.min(error)))
print("Predicting the people names on the testing set")

# predict labels for the X_test images with the best classifier
clf =  SVC(kernel='linear', C=Cs[ind])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = clf.score(X_test, y_test)

# The chance level is the accuracy that will be reached when constantly predicting the majority class.
print("Chance level : %s" % max(np.mean(y), 1. - np.mean(y)))

#%%
# Qualitative evaluation of the predictions using matplotlib

prediction_titles = [title(y_pred[i], y_test[i], names)
                     for i in range(y_pred.shape[0])]

plot_gallery(images_test, prediction_titles)
plt.show()

# Look at the coefficients
plt.figure()
plt.imshow(np.reshape(clf.coef_, (h, w)))
plt.show()


#%%
##############
# Question 5
##############

def run_svm_cv(_X, _y):
    # split data
    _indices = np.random.permutation(_X.shape[0])
    _train_idx, _test_idx = _indices[:_X.shape[0] // 2], _indices[_X.shape[0] // 2:]
    _X_train, _X_test = _X[_train_idx, :], _X[_test_idx, :]
    _y_train, _y_test = _y[_train_idx], _y[_test_idx]
    # fit the linear model with a parameter grid search
    _parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
    _svr = svm.SVC()
    _clf_linear = GridSearchCV(_svr, _parameters)
    _clf_linear.fit(_X_train, _y_train)
    # score
    print('Generalization score for linear kernel: %s, %s \n' %
          (_clf_linear.score(_X_train, _y_train), _clf_linear.score(_X_test, _y_test)))

print("Score sans variable de nuisance")
run_svm_cv(X,y)

#%%
# ajout des variables de bruit
sigma = 5
noise = sigma * np.random.randn(n_samples, 1000, )
X_noisy = np.concatenate((X, noise), axis=1)
X_noisy = X_noisy[:,np.random.permutation(X_noisy.shape[1])]

print("Score avec variable de nuisance")
run_svm_cv(X_noisy,y)

#%%
##############
# Question 6
##############

# Scale features
X_noisy -= np.mean(X_noisy, axis=0)
X_noisy /= np.std(X_noisy, axis=0)

# PCA
n_components = 50
X_pca = PCA(n_components=n_components).fit_transform(X_noisy)

print("Score apres reduction de dimension")
run_svm_cv(X_pca,y)


