import os, sys, json

if __name__ == "__main__":
    dirPath = sys.argv[1]
    filePath = sys.argv[2]
    fileList = os.listdir(dirPath)

    jsonDict = {}

    for filename in fileList:
        if filename.endswith(".json"):
            with open(dirPath + "/" + filename, 'r') as file:
                data = json.load(file)
                jsonDict.update(data)

    with open(filePath, 'w') as file:
        json.dump(jsonDict, file, indent=4)

