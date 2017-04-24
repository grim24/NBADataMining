Directory name: final_project_weka_results
Author: Claudia Moeller

The files in this directory are the output files created by the Weka Explorer GUI.  The output files include all of the data used for evaluation and analysis.  The files are labeled as follows:
	-<advanced/basic>_<no_opp/opp>_<model name><parameter version>
		-advanced: additional attributes were used
		-basic: only the four factors were used as attributes
		-no_opp: no data on the opponent was in the training data
		-opp: data on the opponent was in the training data
		-model_name: terms and their corresponding weka library
			-'bayes' = weka.classifiers.bayes.NaiveBayes
			-'svn1' =  weka.classifiers.functions.SMO
			-'j48' =  weka.classifiers.trees.J48
			 	-parameter version:
			 		-a: unpruned= false, reducedErrorPruning = true
			 		-b: unpruned = false, reducedErrorPruning = false
			 		-c: unpruned = true, reducedErrorPruning = false
			-'enemble' = weka.classifiers.meta.AdaBoostM1
				-parameter version:
					-1: uses SVN's
					-2: uses neural networks
			-'nn'=  weka.classifiers.functions.MultilayerPerceptron
			 	-parameter version:
			 		-1: learningRate = .3
			 		-2: learningRate = .1

The confusion matrix is at the bottom of the file and the evaluation statistics are above it.
