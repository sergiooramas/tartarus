import glob
import json

folders = glob.glob("../millionsongdataset_echonest/*")
for folder in folders:
	files = glob.glob(folder)
	for file in files:
		data = json.load(open(file))