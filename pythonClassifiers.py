from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.naive_bayes import GaussianNB #Naive Bayes
from sklearn.neural_network import MLPClassifier #Artificial Neural Network
from sklearn import svm #Support Vector Machine
from sklearn.ensemble import AdaBoostClassifier #Ensemble Classifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import csv
import numpy
import math
import random
from statistics import mean

def fMeasure(confusion_matrix):
	tp = confusion_matrix[0][0]
	fp = confusion_matrix[1][0]
	fn = confusion_matrix[0][1]
	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	return (2*((precision*recall)/(precision+recall)))

def classifiers(records, classes):
	best_classifier = ""
	accuracy = 0
	f_measure = 0

	decision_tree = DecisionTreeClassifier()
	dt_prediction = cross_val_predict(decision_tree, records, classes)
	dt_accuracy = metrics.accuracy_score(classes, dt_prediction)
	dt_confusion_matrix = metrics.confusion_matrix(classes, dt_prediction)
	dt_fmeasure = fMeasure(dt_confusion_matrix)

	'''print("Decision Tree Accuracy:", dt_accuracy)
	print("Confusion Matrix:", dt_confusion_matrix)
	print("F-measure:", dt_fmeasure)'''

	if dt_accuracy > accuracy and dt_fmeasure > f_measure:
		best_classifier = "Decision Tree"
		accuracy = dt_accuracy
		f_measure = dt_fmeasure

	naive_bayes = GaussianNB()
	nb_prediction = cross_val_predict(naive_bayes, records, classes)
	nb_accuracy = metrics.accuracy_score(classes, nb_prediction)
	nb_confusion_matrix = metrics.confusion_matrix(classes, nb_prediction)
	nb_fmeasure = fMeasure(nb_confusion_matrix)

	'''print("Naive Bayes Accuracy:", nb_accuracy)
	print("Confusion Matrix:", nb_confusion_matrix)
	print("F-measure:", nb_fmeasure)'''

	if nb_accuracy > accuracy and nb_fmeasure > f_measure:
		best_classifier = "Naive Bayes"
		accuracy = nb_accuracy
		f_measure = nb_fmeasure

	ann = MLPClassifier()
	ann_prediction = cross_val_predict(ann, records, classes)
	ann_accuracy = metrics.accuracy_score(classes, ann_prediction)
	ann_confusion_matrix = metrics.confusion_matrix(classes, ann_prediction)
	ann_fmeasure = fMeasure(ann_confusion_matrix)

	'''print("Artificial Neural Net Accuracy:", ann_accuracy)
	print("Confusion Matrix:", ann_confusion_matrix)
	print("F-measure:", ann_fmeasure)'''

	if ann_accuracy > accuracy and ann_fmeasure > f_measure:
		best_classifier = "Artificial Neural Net"
		accuracy = ann_accuracy
		f_measure = ann_fmeasure

	support_vm = svm.SVC()
	svm_prediction = cross_val_predict(support_vm, records, classes)
	svm_accuracy = metrics.accuracy_score(classes, svm_prediction)
	svm_confusion_matrix = metrics.confusion_matrix(classes, svm_prediction)
	svm_fmeasure = fMeasure(svm_confusion_matrix)

	'''print("Support Vector Machine", svm_accuracy)
	print("Confusion Matrix:", svm_confusion_matrix)
	print("F-measure:", svm_fmeasure)'''

	if svm_accuracy > accuracy and svm_fmeasure > f_measure:
		best_classifier = "Support Vector Machine"
		accuracy = svm_accuracy
		f_measure = svm_fmeasure

	ensemble = AdaBoostClassifier()
	ensemble_prediction = cross_val_predict(ensemble, records, classes)
	ensemble_accuracy = metrics.accuracy_score(classes, ensemble_prediction)
	ensemble_confusion_matrix = metrics.confusion_matrix(classes, ensemble_prediction)
	ensemble_fmeasure = fMeasure(ensemble_confusion_matrix)

	'''print("Ensemble (Adaboost) Accuracy:", ensemble_accuracy)
	print("Confusion Matrix:", ensemble_confusion_matrix)
	print("F-measure:", ensemble_fmeasure)'''

	if ensemble_accuracy > accuracy and ensemble_fmeasure > f_measure:
		best_classifier = "Ensemble Classifier"
		accuracy = ensemble_accuracy
		f_measure = ensemble_fmeasure

	print("Most Accurate Classifier:", best_classifier)
	print("Accuracy:",accuracy)
	print("F-Measure",f_measure)

#Open CSV file containing data set
with open('2016_results.csv', "rt") as FourFactors_data:
	FourFactors = csv.reader(FourFactors_data)
	FourFactors = list(FourFactors)

	records = []
	classes = []
	for record in FourFactors[1:]:
		records.append(numpy.array(record[1:6]).astype(numpy.float))
		classes.append(record[6])

	min_max = MinMaxScaler(feature_range=(0, 1), copy=False)
	records = min_max.fit_transform(records)

	print("Four Factors:")
	classifiers(records, classes)

#Open CSV file containing data set
with open('2016_results_opp_team.csv', "rt") as FourFactorsOpp_data:
	FourFactorsOpp = csv.reader(FourFactorsOpp_data)
	FourFactorsOpp = list(FourFactorsOpp)

	records = []
	classes = []
	for record in FourFactorsOpp[1:]:
		attributes = numpy.append(numpy.array(record[1:5]).astype(numpy.float),numpy.array(record[6:10]).astype(numpy.float))
		records.append(attributes)
		classes.append(record[10])

	min_max = MinMaxScaler(feature_range=(0, 1), copy=False)
	records = min_max.fit_transform(records)

	print("Four Factors with Opposing Team Stats:")
	classifiers(records, classes)

#Open CSV file containing data set
with open('2016_advanced_statistics.csv', "rt") as Advanced_data:
	Advanced = csv.reader(Advanced_data)
	Advanced = list(Advanced)

	lb = LabelBinarizer()
	lb.fit([' away',' home'])

	records = []
	classes = []
	for record in Advanced[1:]:
		attributes = numpy.array(record[2:6])
		attributes = numpy.append(attributes, lb.transform([record[6]]))
		attributes = numpy.append(attributes,numpy.array(record[7:10]).astype(numpy.float))
		records.append(attributes)
		classes.append(record[10])

	min_max = MinMaxScaler(feature_range=(0, 1), copy=False)
	records = min_max.fit_transform(records)

	print("Advanced Stats:")
	classifiers(records, classes)

#Open CSV file containing data set
with open('2016_advanced_results_opp_team.csv', "rt") as Advanced_data:
	Advanced = csv.reader(Advanced_data)
	Advanced = list(Advanced)

	lb = LabelBinarizer()
	lb.fit(['  away','  home'])

	records = []
	classes = []
	for record in Advanced[1:]:
		attributes = numpy.array(record[1:5])
		attributes = numpy.append(attributes, lb.transform([record[5]]))
		attributes = numpy.append(attributes,numpy.array(record[6:9]).astype(numpy.float))
		attributes = numpy.append(attributes,numpy.array(record[10:17]).astype(numpy.float))
		records.append(attributes)
		classes.append(record[17])

	min_max = MinMaxScaler(feature_range=(0, 1), copy=False)
	records = min_max.fit_transform(records)

	print("Advanced Stats with Opposing Team Stats:")
	classifiers(records, classes)