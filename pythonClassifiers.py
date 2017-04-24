from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.naive_bayes import GaussianNB #Naive Bayes
from sklearn.neural_network import MLPClassifier #Artificial Neural Network
from sklearn import svm #Support Vector Machine
#Ensemble Classifiers:
from sklearn.ensemble import AdaBoostClassifier #Adaboost
from sklearn.ensemble import VotingClassifier #Voting
#Preprocessing:
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer

import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_predict
import csv
import numpy
import math
import random
from statistics import mean

#Calculate F-Measure based on a provided confusion matrix
def fMeasure(confusion_matrix):
	tp = confusion_matrix[0][0]
	fp = confusion_matrix[1][0]
	fn = confusion_matrix[0][1]
	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	return (2*((precision*recall)/(precision+recall)))

#Given a set of records and true classes, determine the optimal classifier based on 10-fold cross-validation
def classifiers(records, classes):
	#Initialize variables to store best classifier's info
	best_classifier = ""
	accuracy = 0
	f_measure = 0

	#Decision Tree Classifier
	decision_tree = DecisionTreeClassifier(criterion="entropy")
	#Predict the class of each training record through 10-fold cross-validation
	dt_prediction = cross_val_predict(decision_tree, records, classes, cv=10)
	#Calculate performance measures
	dt_accuracy = metrics.accuracy_score(classes, dt_prediction)
	dt_confusion_matrix = metrics.confusion_matrix(classes, dt_prediction)
	dt_fmeasure = fMeasure(dt_confusion_matrix)

	'''print("Decision Tree Accuracy:", dt_accuracy)
	print("Confusion Matrix:", dt_confusion_matrix)
	print("F-measure:", dt_fmeasure)'''

	#Determine if classifier is optimal of those tested so far
	if dt_accuracy > accuracy and dt_fmeasure > f_measure:
		best_classifier = "Decision Tree"
		accuracy = dt_accuracy
		f_measure = dt_fmeasure

	#Naive Bayes Classifier
	naive_bayes = GaussianNB()
	#Predict the class of each training record through 10-fold cross-validation
	nb_prediction = cross_val_predict(naive_bayes, records, classes, cv=10)
	#Calculate performance measures
	nb_accuracy = metrics.accuracy_score(classes, nb_prediction)
	nb_confusion_matrix = metrics.confusion_matrix(classes, nb_prediction)
	nb_fmeasure = fMeasure(nb_confusion_matrix)

	'''print("Naive Bayes Accuracy:", nb_accuracy)
	print("Confusion Matrix:", nb_confusion_matrix)
	print("F-measure:", nb_fmeasure)'''

	#Determine if classifier is optimal of those tested so far
	if nb_accuracy > accuracy and nb_fmeasure > f_measure:
		best_classifier = "Naive Bayes"
		accuracy = nb_accuracy
		f_measure = nb_fmeasure

	#Artificial Neural Net
	ann = MLPClassifier()
	#Predict the class of each training record through 10-fold cross-validation
	ann_prediction = cross_val_predict(ann, records, classes, cv=10)
	#Calculate performance measures
	ann_accuracy = metrics.accuracy_score(classes, ann_prediction)
	ann_confusion_matrix = metrics.confusion_matrix(classes, ann_prediction)
	ann_fmeasure = fMeasure(ann_confusion_matrix)

	'''print("Artificial Neural Net Accuracy:", ann_accuracy)
	print("Confusion Matrix:", ann_confusion_matrix)
	print("F-measure:", ann_fmeasure)'''

	#Determine if classifier is optimal of those tested so far
	if ann_accuracy > accuracy and ann_fmeasure > f_measure:
		best_classifier = "Artificial Neural Net"
		accuracy = ann_accuracy
		f_measure = ann_fmeasure

	#Support Vector Machine
	support_vm = svm.SVC(kernel='linear')
	#Predict the class of each training record through 10-fold cross-validation
	svm_prediction = cross_val_predict(support_vm, records, classes, cv=10)
	#Calculate performance measures
	svm_accuracy = metrics.accuracy_score(classes, svm_prediction)
	svm_confusion_matrix = metrics.confusion_matrix(classes, svm_prediction)
	svm_fmeasure = fMeasure(svm_confusion_matrix)

	'''print("Support Vector Machine", svm_accuracy)
	print("Confusion Matrix:", svm_confusion_matrix)
	print("F-measure:", svm_fmeasure)'''

	#Determine if classifier is optimal of those tested so far
	if svm_accuracy > accuracy and svm_fmeasure > f_measure:
		best_classifier = "Support Vector Machine"
		accuracy = svm_accuracy
		f_measure = svm_fmeasure

	#Adaboost Classifier
	ensemble = AdaBoostClassifier(base_estimator=support_vm,algorithm='SAMME')
	#Predict the class of each training record through 10-fold cross-validation
	ensemble_prediction = cross_val_predict(ensemble, records, classes, cv=10)
	#Calculate performance measures
	ensemble_accuracy = metrics.accuracy_score(classes, ensemble_prediction)
	ensemble_confusion_matrix = metrics.confusion_matrix(classes, ensemble_prediction)
	ensemble_fmeasure = fMeasure(ensemble_confusion_matrix)

	'''print("Ensemble (Adaboost) Accuracy:", ensemble_accuracy)
	print("Confusion Matrix:", ensemble_confusion_matrix)
	print("F-measure:", ensemble_fmeasure)'''

	#Determine if classifier is optimal of those tested so far
	if ensemble_accuracy > accuracy and ensemble_fmeasure > f_measure:
		best_classifier = "Adaboost Classifier"
		accuracy = ensemble_accuracy
		f_measure = ensemble_fmeasure

	#Majority Voting Ensemble Classifier
	voting = VotingClassifier(estimators=[('dt',decision_tree),('gnb',naive_bayes),('svm',support_vm),('ann',ann)])
	#Predict the class of each training record through 10-fold cross-validation
	voting_prediction = cross_val_predict(voting, records, classes, cv=10)
	#Calculate performance measures
	voting_accuracy = metrics.accuracy_score(classes, voting_prediction)
	voting_confusion_matrix = metrics.confusion_matrix(classes, voting_prediction)
	voting_fmeasure = fMeasure(voting_confusion_matrix)

	'''print("Ensemble (Voting) Accuracy:", voting_accuracy)
	print("Confusion Matrix:", voting_confusion_matrix)
	print("F-measure:", voting_fmeasure)'''

	#Determine if classifier is optimal of those tested so far
	if voting_accuracy > accuracy and voting_fmeasure > f_measure:
		best_classifier = "Voting Classifier"
		accuracy = voting_accuracy
		f_measure = voting_fmeasure

	print("Most Accurate Classifier:", best_classifier)
	print("Accuracy:",accuracy)
	print("F-Measure",f_measure)

#Four Factors training set
with open('2016_results.csv', "rt") as FourFactors_data:
	#Read data from CSV file
	FourFactors = csv.reader(FourFactors_data)
	FourFactors = list(FourFactors)

	#Generate training records and classes
	records = []
	classes = []
	for record in FourFactors[1:]:
		#Isolate desired attributes and add list of attributes to list of records
		records.append(numpy.array(record[1:6]).astype(numpy.float))
		#Add class to list of classes
		classes.append(record[6])

	#Preformat the training data by scaling to a range of [0,1]
	min_max = MinMaxScaler(feature_range=(0, 1), copy=False)
	records = min_max.fit_transform(records)

	print("Four Factors:")
	classifiers(records, classes)

#Four Factors w/ Opposing Teams training set
with open('2016_results_opp_team.csv', "rt") as FourFactorsOpp_data:
	#Read data from CSV file
	FourFactorsOpp = csv.reader(FourFactorsOpp_data)
	FourFactorsOpp = list(FourFactorsOpp)

	#Generate training records and classes
	records = []
	classes = []
	for record in FourFactorsOpp[1:]:
		#Isolate desired attributes
		attributes = numpy.append(numpy.array(record[1:5]).astype(numpy.float),numpy.array(record[6:10]).astype(numpy.float))
		#Add list of attributes to list of records
		records.append(attributes)
		#Add class to list of classes
		classes.append(record[10])

	#Preformat the training data by scaling to a range of [0,1]
	min_max = MinMaxScaler(feature_range=(0, 1), copy=False)
	records = min_max.fit_transform(records)

	print("Four Factors with Opposing Team Stats:")
	classifiers(records, classes)

#Advanced Stats training set
with open('2016_advanced_statistics.csv', "rt") as Advanced_data:
	#Read data from CSV file
	Advanced = csv.reader(Advanced_data)
	Advanced = list(Advanced)

	#Initialize preformatting for categorical attribute
	lb = LabelBinarizer()
	lb.fit([' away',' home'])

	#Generate training records and classes
	records = []
	classes = []
	#For each record in training data set,
	for record in Advanced[1:]:
		#Isolate desired attributes
		attributes = numpy.array(record[2:6])
		#Binarize categoirical attribute and add to list of record's attributes
		attributes = numpy.append(attributes, lb.transform([record[6]]))
		attributes = numpy.append(attributes,numpy.array(record[7:10]).astype(numpy.float))
		#Add list of attributes to list of records
		records.append(attributes)
		#Add class to list of classes
		classes.append(record[10])

	#Preformat the training data by scaling to a range of [0,1]
	min_max = MinMaxScaler(feature_range=(0, 1), copy=False)
	records = min_max.fit_transform(records)

	print("Advanced Stats:")
	classifiers(records, classes)

#Advanced Stats w/ Opposing Teams training set
with open('2016_advanced_results_opp_team.csv', "rt") as AdvancedOpp_data:
	#Read data from CSV file
	AdvancedOpp = csv.reader(AdvancedOpp_data)
	AdvancedOpp = list(AdvancedOpp)

	#Initialize preformatting for categorical attribute
	lb = LabelBinarizer()
	lb.fit(['  away','  home'])

	#Generate training records and classes
	records = []
	classes = []
	#For each record in training data set,
	for record in AdvancedOpp[1:]:
		#Isolate desired attributes
		attributes = numpy.array(record[1:5])
		#Binarize categoirical attribute and add to list of record's attributes
		attributes = numpy.append(attributes, lb.transform([record[5]]))
		attributes = numpy.append(attributes,numpy.array(record[6:9]).astype(numpy.float))
		attributes = numpy.append(attributes,numpy.array(record[10:17]).astype(numpy.float))
		#Add list of attributes to list of records
		records.append(attributes)
		#Add class to list of classes
		classes.append(record[17])

	#Preformat the training data by scaling to a range of [0,1]
	min_max = MinMaxScaler(feature_range=(0, 1), copy=False)
	records = min_max.fit_transform(records)

	print("Advanced Stats with Opposing Team Stats:")
	classifiers(records, classes)

#Testing on Playoff Data
'''The optimal algorithm was determined through analysis of the results obtained through the "classifiers" function'''
with open('playoff_statistics_predictions.csv', "rt") as Playoffs_data:
	#Open CSV file
	Playoffs = csv.reader(Playoffs_data)
	Playoffs = list(Playoffs)

	#Generate test datasets
	test_records = []
	test_classes = []
	for record in Playoffs[1:]:
		attributes = numpy.append(numpy.array(record[1:5]).astype(numpy.float),numpy.array(record[6:10]).astype(numpy.float))
		test_records.append(attributes)
		test_classes.append(record[10])

	#Preformat the test data by scaling to a range of [0,1] and converting to a list
	test_min_max = MinMaxScaler(feature_range=(0, 1), copy=False)
	test_records = test_min_max.fit_transform(test_records)
	test_records = test_records.tolist()

	#Training data set: Four Factors w/ Opposing Teams
	with open('2016_results_opp_team.csv', "rt") as training_data:
		#Read data from CSV file
		training = csv.reader(training_data)
		training = list(training)

		#Generate training datasets
		training_records = []
		training_classes = []
		for record in training[1:]:
			attributes = numpy.append(numpy.array(record[1:5]).astype(numpy.float),numpy.array(record[6:10]).astype(numpy.float))
			training_records.append(attributes)
			training_classes.append(record[10])

		#Preformat the training data by scaling to a range of [0,1] and converting to a list
		training_min_max = MinMaxScaler(feature_range=(0, 1), copy=False)
		training_records = training_min_max.fit_transform(training_records)
		training_records = training_records.tolist()

		#Initialize Support Vector Machine (optimal algorithm based on analysis)
		support_vm = svm.SVC(kernel='linear')

		#Fit SVM to training data set
		support_vm.fit(training_records, training_classes)

		#Use model to predict the outcome of each game in the test dataset
		predictions = support_vm.predict(test_records)
		predictions = [classifier.strip(' ') for classifier in predictions]
		
		#Calculate performance measures
		accuracy = metrics.accuracy_score(test_classes, predictions)
		confusion_matrix = metrics.confusion_matrix(test_classes, predictions)
		fmeasure = fMeasure(confusion_matrix)

		#Print results
		print("Playoffs Accuracy:", accuracy)
		print("Confusion Matrix:", confusion_matrix)
		print("F-measure:", fmeasure)

		#Generate CSV file for analysis of outcomes
		with open('playoffs_results.csv', 'w') as output_file:
			#csv writers
			writer = csv.writer(output_file)
			writer.writerow(['Team Name','Opposing Team','Result','Prediction'])
			for i in range(0,len(test_records)):
				row = [Playoffs[i+1][0],Playoffs[i+1][5],test_classes[i],predictions[i]]
				writer.writerow(row)