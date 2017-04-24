# NBADataMining
Description: DataMining final project for class. Parses basketball-reference.com for scores and statistics for games

Author: Landon Grim

To run, one needs to install Nokogiri using `gem install nokogiri`

To obtain the csv file for the 4 factors run: `ruby nba_scraper.rb > 2016_results.csv`

For advanced statistics run: `ruby nba_extra_cat_scraper.rb > 2016_avanced_statistics.csv`

This produces the first 2 datasets, 2016_results.csv and 2016_advanced_statistics.csv

To get the other 2 datasets run `python basic_data_transform.py` and `python advanced_data_transform.py`

This produces the 2016_results_opp_team.csv and 2016_advanced_results_opp_team.csv

The playoffs_statistics_predictions.csv file used as the test dataset was generated manually.

For the Python classification algorithms, view README-python.txt.
For the Weka classification algorithsm, view README-weka.txt.
