import os, sys, json
import numpy as np

prefix = "mdpvK"

def makeCSV(jsonDictTemp):
    csvList = []
    labels = ["convN", "alpha", "discount", "epsilon", "fedP", "syncBackups", "stochasticPolicy", "envSeed", "trial"]
    itemKeys = ['sims', 'endScoreStoch', 'comment', 'avgRew', 'episodes', 'endScoreBell', 'endScoreRand', 'aggs', 'backups', 'endScoreStochBell', 'endScore']
    arrayKeys = ['diffs', 'rewards', 'avgRewArr', 'epsToBackup']
    summaryKeys = ['finalDiff', "threshEp", "threshBack"]

    csvList.append(labels + itemKeys + summaryKeys)

    for key, value in jsonDictTemp.items():
        line = key[1:-1].split(", ")
        for itemKey in itemKeys:
            if itemKey not in value:
                line.append(0)
            else:
                line.append(value[itemKey])

        # finalDiff, finalRew, threshEp, threshBack
        line.append(np.mean(value["diffs"][-10:]))
        line.append(value["episodes"])
        line.append(value["backups"])
        csvList.append(line)

    np.savetxt(prefix + ".csv", csvList, delimiter=",", fmt="%s")

if __name__ == "__main__":
    dirPath = sys.argv[1]
    filePath = sys.argv[2]
    fileList = os.listdir(dirPath)

    jsonDict = {}

    for filename in fileList:
        if filename.startswith(prefix) and filename.endswith(".json"):
            with open(dirPath + "/" + filename, 'r') as file:
                data = json.load(file)
                jsonDict.update(data)

    with open(filePath, 'w') as file:
        json.dump(jsonDict, file, indent=4)

    makeCSV(jsonDict)

