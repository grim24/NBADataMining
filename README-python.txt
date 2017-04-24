File name: pythonClassifiers.py
Author: Jennifer Barry

Dependencies:
	This program expects python3 to be installed.
	This program requires the "CSV" module for Python.
	This program requires the "NumPy" library for Python.
		To install, use the following commands:
		- "sudo apt-get install python-pip"
		- "pip3 install numpy"
	This program requires the "sklearn" library for Python.
		To install, use the command "pip3 install scikit-learn"

Input Data:
	Input data files should be placed in the same directory as the program file.

	The following files are required:
	- 2016_results.csv
	- 2016_results_opp_team.csv
	- 2016_advanced_statistics.csv
	- 2016_advanced_results_opp_team.csv
	- playoff_statistics_predictions.csv

Execution:
	Once dependencies have been resolved, the program can be run from a linux command line with the following command:
		python3 pythonClassifiers.py

Behavior:
	The program takes as input 4 datasets: Four Factors (2016_results.csv), Four Factors w/ Opposing Teams (2016_results_opp_team.csv), Advanced Stats (2016_advanced_statistics.csv), Advanced Stats w/ Opposing Teams (2016_advanced_results_opp_team.csv).

	For each of these training datasets, a number of classification algorithms are tested using 10-fold cross-validation, and the optimal algorithm's name, accuracy, and f-measure are printed to the console.

	Based on the analysis provided on the results of the various algorithms tested, it was determined that the Support Vector Machine (SVM) algorithm yielded optimal results on the Four Factors w/ Opposing Teams dataset. Upon completion, the program will have trained the SVM using the aforementioned dataset, and tested its predictions on the playoff_statistics_predictions.csv dataset.

	Based on these predictions, the program will write its accuracy and f-measure to the console, and output a data set containing the following information:
	- Team Name
	- Opposing Team Name
	- Result (the actual outcome of the game)
	- Prediction (the predicted outcome of the game)

Output Data:
	Output data files will be stored in CSV format in the same directory as the program file.
	The output file will be named "playoffs_results.csv"