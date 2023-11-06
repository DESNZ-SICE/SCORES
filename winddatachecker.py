import numpy as np
import datetime
import csv

datafolder = datafolder = "C:/Users/SA0011/Documents/data/"

windatafilename = datafolder + "offshore_wind/1.csv"
rowcounter = 0

with open(windatafilename, "rU") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        rowcounter += 1
        d = datetime.datetime(int(row[0]), int(row[1]), int(row[2]), int(row[3]))
        if rowcounter > 1:
            timedelta = d - lastdate
            if timedelta.seconds > 3700:
                print("Missing data")
                print(d)
        lastdate = d
