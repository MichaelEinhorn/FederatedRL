import os
from time import sleep
import sys
import subprocess

from contextlib import redirect_stdout

# with open('submitLog.txt', 'w') as logf:
#     with redirect_stdout(logf):
if True:
        # submits jobs to cluster based on variations of a template sh script

        fname = 'tCoin.sh'
        if len(sys.argv) >= 2:
            fname = sys.argv[1]

        with open(fname, 'r') as file:
            data = file.read()
            
        print(data)

        modelName = "impalaVectorV1"
        prefix = f"{modelName}-t#-k#-n#"

        if len(sys.argv) >= 3:
            replaceEx = sys.argv[2] == "-r"
        else:
            replaceEx = False
            
        i = 0

        # dirPath = "~/scratch/RL"
        dirPath = "/storage/home/hcoda1/2/meinhorn6/scratch/RL/models"

        nList = [1, 2, 4, 8]
        kList = [1,10,100]

        # retry on failure
        for _ in range(2):
            if True:
                for trial in [0, 1, 2]:
                    if True:
                        for N in nList:
                            for K in kList:
                                if N == 1 and K > 1:
                                    continue

                                if True:
                                    
                                    epoch = 4000 / K
                                        
                                    tprefix = prefix.replace("t#", "t" + str(trial))
                                    tprefix = tprefix.replace("k#", "k" + str(K))
                                    tprefix = tprefix.replace("n#", "n" + str(N))
                                    tprefix = tprefix.replace(".", "")
                                    
                                    result = subprocess.run(['squeue','--format="%.18i %.9P %j %.2t %.10M %.6D %R"', '-u', 'meinhorn6'], capture_output=True, text=True).stdout
                                    strOut = result
                                    result = subprocess.run(['squeue','--format="%.18i %.9P %j %.2t %.10M %.6D %R"', '-u', 'smaguluri3'], capture_output=True, text=True).stdout
                                    strOut = result + strOut

                                    # does a file exist that contains tprefix in the name
                                    fileExists = False
                                    for file in os.listdir(dirPath):
                                        if tprefix in file:
                                            fileExists = True
                                            break

                                    if not replaceEx and (fileExists or (tprefix in strOut)):
                                        print("skipping " + tprefix)
                                        continue
                                        
                                    tdata = data.replace("p#", tprefix)
                                    tdata = tdata.replace("t#", str(trial))
                                    tdata = tdata.replace("k#", str(K))
                                    tdata = tdata.replace("n#", str(N))
                                    tdata = tdata.replace("e#", str(epoch))
                                        
                                    print(tdata)
                                    textfile = open("autoCoin.sh", "w")
                                    a = textfile.write(tdata)
                                    textfile.close()
                                    #os.system("sbatch autoExp.sh")
                                    result = subprocess.run(['sbatch', 'autoCoin.sh'], capture_output=True, text=True).stdout
                                    strOut = result
                                    print(strOut)
                                    
                                    
                                    result = subprocess.run(['squeue','--format="%.18i %.9P %j %.2t %.10M %.6D %R"', '-u', 'meinhorn6'], capture_output=True, text=True).stdout
                                    strOut = result
                                    result = subprocess.run(['squeue','--format="%.18i %.9P %j %.2t %.10M %.6D %R"', '-u', 'smaguluri3'], capture_output=True, text=True).stdout
                                    strOut = result + strOut

                                    print(strOut)
                                    nlines = strOut.count('\n')
                                    print(nlines)

                                    while nlines > 5:
                                        result = subprocess.run(['squeue','--format="%.18i %.9P %j %.2t %.10M %.6D %R"', '-u', 'meinhorn6'], capture_output=True, text=True).stdout
                                        strOut = result
                                        result = subprocess.run(['squeue','--format="%.18i %.9P %j %.2t %.10M %.6D %R"', '-u', 'smaguluri3'], capture_output=True, text=True).stdout
                                        strOut = result + strOut
                                        
                                        print(strOut)
                                        nlines = strOut.count('\n')
                                        print(nlines)
                                        sleep(90)
                                    sleep(1)
                                    i += 1
