#!/usr/bin/env python

from __future__ import division
import csv
import sys
import os
import math


#main start
file_path = '2016_results.csv'

# Output file setup for Euclidean measure
writer= open('2016_results_opp_team',"w+")


writer.write(
       'Team, ' + 
       'Efg_pct, ' +
       'Turnover_PCT, ' + 
       'orb_pct, ' + 
       'ft_rate, ' + 
       'opp_Team, ' + 
       'opp_Efg_pct, ' +
       'opp_Turnover_PCT, ' + 
       'opp_orb_pct, ' + 
       'opp_ft_rate, ' + 
       'class\n'
        )


with open(file_path, 'r+') as f:
    lines = f.readlines()
    del lines[0]
    for i in xrange (0, len(lines), 2):
        away_line = lines[i].split(',')
        home_line = lines[i+1].split(',')

        
        away_string = (away_line[0] + ", " + away_line[2] + ", " + away_line[3]+  ", " 
        + away_line[4] + ", " + away_line[5] + ", " + home_line[0] + ", " + home_line[2] + ", " + 
        home_line[3] + ", " + home_line[4] + ", " + home_line[5] + ", " + away_line[6] 
        )
        home_string = (home_line[0] + ", " + home_line[2] + ", " + home_line[3] + ", "
        + home_line[4]  + ", "+ home_line[5]  + ", "+ away_line[0]  + ", "+ away_line[2]  + ", "+ 
        away_line[3]  + ", "+ away_line[4]  + ", "+ away_line[5]  + ", "+ home_line[6] 
        )
        writer.write(
               away_string 
                )
        writer.write(
                home_string
                )





