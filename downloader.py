
'''
Downloader of datasets needed to generate model's training data.

## Usage:
Script must be executed from its directory using the command
   python3 downloader.py

## Log:
Downloader supports logging on console and on "./downloader.log" file.

## Multiple executions:
During runtime script looks for already downloaded files avoiding unnecessary operations.
No versioning is done so if parameters of the current execution changed from the previous ones
or you may want to update your data, you need to delete older files and run the script again.
Log file is not preserved and it is overwritten every time.
'''


import sys
import os
import gc
import pickle
import json as js
from urllib.request import urlopen
from io import StringIO
from functools import partial
import logging

import pandas as pd
import numpy as np
from tqdm.auto import tqdm, trange
import joblib as jb


def save_df(df, path, ext="pkl", log_level="info", **kwargs):

	if ext == "pkl":
		df.to_pickle(path, **kwargs)
	elif ext == "csv":
		df.to_csv(path, **kwargs)
	else:
		raise NotImplementedError("unable to save dataframe with extension {}".format(ext))

	if type(log_level) == str and log_level == "debug":
		logging.getLogger(__name__).debug("new file \"{}\"".format(path))
	else:
		logging.getLogger(__name__).info("new file \"{}\"".format(path))


def save_pkl(var, path, log_level="info"):

	with open(path, "wb") as file:
		pickle.dump(var,file)

	if type(log_level) == str and log_level == "debug":
		logging.getLogger(__name__).debug("new file \"{}\"".format(path))
	else:
		logging.getLogger(__name__).info("new file \"{}\"".format(path))



def read_pkl(path):
	with open(path, "rb") as file:
		return pickle.load(file)


# API call and return parsed result
def API_call(url, max_retries=5, ext="json", **kwargs):

	logging.getLogger(__name__).debug("API call: {}".format(url))

	# call
	attempt = 0
	while attempt < max_retries:
		try:
			api_result = urlopen(url).read()
		except Exception as ex:
			logging.getLogger(__name__).debug("attempt {} for API call".format(attempt))
			logging.getLogger(__name__).debug(ex)
			attempt += 1
			if attempt == max_retries:
				logging.getLogger(__name__).debug("all attetmpts failed, throwing last exception")
				raise ex
		else:
			break

	logging.getLogger(__name__).debug("API call successful")
	logging.getLogger(__name__).debug("API call returned result parsed as {}".format(ext))

	# parse
	if ext == "json":
		api_result = js.loads(api_result, **kwargs)
	elif ext == "csv":
		api_result = pd.read_csv(StringIO(api_result.decode()), **kwargs)
	else:
		raise NotImplementedError("unable to parse API call result as {}".format(ext))

	return api_result


# definitions of weather dataset features
def pull_definitions(dataset="daily-summaries", bbox=(71.351,-178.217,18.925,179.769), place="Country:151"):

	api_result = API_call("https://www.ncei.noaa.gov/access/services/search/v1/data?dataset={}&bbox={},{},{},{}&place={}&limit={}&offset={}".format(dataset, *bbox, place, 1000, 0))

	N = len(api_result["results"][0]["dataTypes"])
	definitions = [
		[api_result["results"][i]["dataTypes"][j]["id"], api_result["results"][i]["dataTypes"][j]["name"]]
		for i in range(len(api_result["results"])) for j in range(len(api_result["results"][i]["dataTypes"]))
	]
	definitions = pd.DataFrame(definitions, columns=["ID", "NAME"])
	definitions = definitions.drop_duplicates()

	logging.getLogger(__name__).debug("pulled definitions shape: {}".format(definitions.shape))
	save_df(definitions, "definitions.pkl")


# interesting weather stations
def pull_stations(dataset="daily-summaries", bbox=(71.351,-178.217,18.925,179.769), place="Country:151", limit=1000, n_jobs=-1, progress=True):

	# get results count
	api_results = [API_call("https://www.ncei.noaa.gov/access/services/search/v1/data?dataset={}&bbox={},{},{},{}&place={}&limit={}&offset={}".format(dataset, *bbox, place, limit, 0))]

	count = api_results[0]["count"]
	logging.getLogger(__name__).debug("search returned {} results".format(count))

	# retrieve remaining results based on count
	api_results += jb.Parallel(n_jobs=n_jobs, backend="multiprocessing")(
		jb.delayed(API_call)("https://www.ncei.noaa.gov/access/services/search/v1/data?dataset={}&bbox={},{},{},{}&place={}&limit={}&offset={}".format(dataset, *bbox, place, limit, limit*i))
		for i in tqdm(range(1, count//limit if count%limit==0 else count//limit+1), desc="Pull weather stations", disable=not progress)
	)

	# parse stations
	stations = [api_results[i]["results"][j]["stations"][0]["id"] for i in range(len(api_results)) for j in range(len(api_results[i]["results"]))]
	stations = list(dict.fromkeys(stations)) # remove duplicates

	logging.getLogger(__name__).debug("{} stations".format(len(stations)))
	save_pkl(stations, "stations.pkl")


# split stations in batches
def pull_stations_batches(batch_size=10):

	stations = read_pkl("stations.pkl")

	stations_batches = []
	for i in range(len(stations)//batch_size):
		stations_batches += [stations[i*batch_size : (i+1)*batch_size]]

	logging.getLogger(__name__).debug("{} stations batches".format(len(stations_batches)))
	save_pkl(stations_batches, "stations_batches.pkl")


# API call for data
def pull_data(filename, start_date, end_date, stations, dataset="daily-summaries", bbox=(71.351,-178.217,18.925,179.769), place="Country:151", **kwargs):

	logging.getLogger(__name__).debug("target \"{}\": stations for API call: {}".format(filename, stations))

	# formatting stations
	stations_str = ""
	for station in stations:
		stations_str += station + ","
	stations = stations_str[:-1]

	url = "https://www.ncei.noaa.gov/access/services/data/v1?dataset={}&startDate={}&endDate={}&stations={}&includeStationName=1&includeStationLocation=1&bbox={},{},{},{}&place={}&units=metric".format(dataset, start_date, end_date, stations, *bbox, place)

	try:
		data = API_call(url, ext="csv", parse_dates=["DATE"])
	except Exception as ex:
		logging.getLogger(__name__).error("target \"{}\": exception encountered".format(filename))
		logging.getLogger(__name__).debug("target \"{}\": {}".format(filename, ex))
		return

	logging.getLogger(__name__).debug("target \"{}\": pulled data shape: {}".format(filename, data.shape))
	save_df(data, "{}".format(filename), **kwargs)


# pull weather dataset from ncei.noaa.gov; output files are multiple so it is not suitable to use with pull_target
def pull_weather(start_date, end_date, n_jobs=-1, progress=True):

	stations_batches = read_pkl("stations_batches.pkl")

	# pull dataset
	## many logs are generated within this parallel pool, because of this log level is reduced to debug redirecting all outputs to log file
	jb.Parallel(n_jobs=n_jobs, backend="multiprocessing")(
		jb.delayed(pull_target)(filename="weather_{}.pkl.xz".format(i), pull_func=partial(pull_data, filename="weather_{}.pkl.xz".format(i), log_level="debug"), log_level="debug", start_date=start_date, end_date=end_date, stations=stations_batch)
		for i,stations_batch in enumerate(tqdm(stations_batches, desc="Pull weather dataset", disable=not progress))
	)


# pull filename if needed (high-level function)
def pull_target(filename, pull_func, log_level="info", **kwargs):
	if os.path.isfile("{}".format(filename)):
		if log_level == "debug":
			logging.getLogger(__name__).debug("\"{}\" found. To refresh the file delete it and re-run the script.".format(filename))
		else:
			logging.getLogger(__name__).info("\"{}\" found. To refresh the file delete it and re-run the script.".format(filename))
	else:
		pull_func(**kwargs)


# main
def run(progress=True):
	pull_target(filename="definitions.pkl", pull_func=pull_definitions)
	gc.collect()
	pull_target(filename="stations.pkl", pull_func=pull_stations, progress=progress)
	gc.collect()
	pull_target(filename="stations_batches.pkl", pull_func=pull_stations_batches)
	gc.collect()
	pull_weather(start_date=pd.Timestamp("1991-12-31").date(), end_date=pd.Timestamp("2016-01-02").date(), progress=progress)
	gc.collect()


# logger

logging.getLogger(__name__).setLevel(logging.INFO)

# default
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logging.getLogger(__name__).addHandler(ch)

# errors
fh = logging.FileHandler("downloader.log", mode="w")
fh.setLevel(logging.ERROR)
fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logging.getLogger(__name__).addHandler(fh)


if __name__ == "__main__":
	try:
		if "debug" in sys.argv:
			logging.getLogger(__name__).setLevel(logging.DEBUG)
			ch.setLevel(logging.DEBUG)
			ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
			logging.getLogger(__name__).debug("Running in debug mode.")
			run(progress=False)
		else:
			run()
	except Exception as ex:
		logging.getLogger(__name__).critical(ex)
		sys.exit(1)
