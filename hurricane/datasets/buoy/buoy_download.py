import requests
from datetime import datetime
import csv
import itertools
import os

BASE_URL = "http://data.gcoos.org/get_gcoos_data.php?bbox={0}&start={1}&stop={2}&obs={3}&source={4}&fmt={5}&sortBy={6}&qc={7}"
BASE_OUTPUT = "../../data"
standard_area = ["-98.0","23.5","-81.0","31.0"]
default_start = "Jun 1 2008 00:00"
default_end = "Mar 1 2019 00:00"
#date range is a datedelta obj
#output_dir is appended onto ../../data/_____

def make_date_range(start = default_start, end = default_end):
	#example input:  'Jun 1 2005  13:33'
	#check datetime strptime docs for the flags
	date_start_obj = datetime.strptime(default_start, '%b %d %Y %H:%M')
	date_end_obj = datetime.strptime(default_end, '%b %d %Y %H:%M')
	return date_end_obj - date_start_obj

def download_data(
	date_range, output_dir = "",
	type_exclusions = [],
	station_group_exclusions = [],
	area = standard_area,
	sort_by = "dates",
	qc = "on"):
	os.makedirs(BASE_OUTPUT + output_dir, exist_ok = True)
	#determining data types to take
	all_data_types = [
		"air_pressure",
		"air_temperature",
		"chlorophyll",
		"current",
		"do",
		"relHumidity",
		"salinity",
		"turbidity",
		"water_level",
		"water_temperature",
		"waves",
		"winds"
		]
	data_types_list = [d for d in all_data_types if d not in type_exclusions]
	#to create a list of all HTTP requests as strings
	reqs = []
	for d in data_types_list:
		reqs.append([
			','.join(area),
			datetime.strftime(date_range[0], "%Y-%m-%dT%H:%M:%SZ"),
			datetime.strftime(date_range[1], "%Y-%m-%dT%H:%M:%SZ"),
			d,
			"All", #I can edit this later to disclude certain station groups.
			"csv",
			"dates",
			qc])
	#Make the actual requests, and save them
	for r in reqs:
		print("downloading: " + str(r))
		print(BASE_URL.format(*r))
		req = requests.get(BASE_URL.format(*r))
		filename = BASE_OUTPUT + output_dir + "/" + r[3] + "_" + r[1] + "_to_" + r[2] + ".csv"
		print("stored: " + filename)
		print(filename)
		with open(filename, 'wb') as f:
			for chunk in req.iter_content(chunk_size = 128):
				f.write(chunk) #break it into chunks.
		f.close()

def main(argv):
	print("RUNNING MAIN")
	seasons_only = True
	#do some console input stuff.
	#check flags to edit defaults
	# d_range = make_date_range()
	start = datetime.strptime(default_start, '%b %d %Y %H:%M')
	end = datetime.strptime(default_end, '%b %d %Y %H:%M')
	if seasons_only:
		# we include only hurricane seasons:
		for i in range(start.year, end.year + 1):
			start_string = "May 01 {0!s}  00:00".format(i)
			season_start = datetime.strptime(start_string, '%b %d %Y %H:%M')
			end_string = "Dec 31 {0!s}  00:00".format(i)
			season_end = datetime.strptime(end_string, '%b %d %Y %H:%M')
			download_data((season_start, season_end))
	else:
		d_range = make_date_range(input1, input2)
		download_data(d_range)

main(None)