import requests
from datetime import datetime
import csv
import itertools
import os

#testing file
def teststrftime():
	default_start = "Jun 1 2008 00:00"
	date = datetime.strptime(default_start, '%b %d %Y %H:%M')
	print(default_start)
	print(date)
	#YYYY-MM-DDTHH:MM:SSZ
	print(datetime.strftime(date, "%Y-%m-%dT%H:%M:%SZ"))

teststrftime()