import json
import csv

with open('heros.json') as json_data:
    data = json.load(json_data)
    print(data)

file  = open('heros.csv', "wb")
writer = csv.writer(file, delimiter=',')

for hero in data:
    writer.writerow([hero["id"], hero["localized_name"]])

file.close()
