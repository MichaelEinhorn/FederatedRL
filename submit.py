import os
from time import sleep
import sys
import subprocess

from contextlib import redirect_stdout

# with open('submitLog.txt', 'w') as logf:
#     with redirect_stdout(logf):
if True:
        # submits jobs to cluster based on variations of a template sh script

        fname = 'tExp.sh'
        if len(sys.argv) >= 2:
            fname = sys.argv[1]

        with open(fname, 'r') as file:
            data = file.read()
            
        print(data)

        prefix = "mdpv4-t#-sb#-f#-a#-e#"

        if len(sys.argv) >= 3:
            replaceEx = sys.argv[2] == "-r"
        else:
            replaceEx = False
            
        i = 0

        # dirPath = "~/scratch/RL"
        dirPath = "/storage/home/hcoda1/2/meinhorn6/scratch/RL"

        # retry on failure
        for _ in range(2):
            for trial in [0, 1, 2]:
                for epsilon in [0, 1]:
                    for fedP in [1, 2, 4, 8, 16]:
                        for syncBackups in [10000, 1000, 100, 10, 1]:
                            if fedP == 1:
                                syncBackups = -1

                            for alpha in [1, 0.5, 0.2, 0.1, 0.01]:
                                
                                alpha = alpha * fedP
                                if alpha >= 1:
                                    continue
                                    
                                tprefix = prefix.replace("t#", "t" + str(trial))
                                tprefix = tprefix.replace("sb#", "sb" + str(syncBackups))
                                tprefix = tprefix.replace("f#", "f" + str(fedP))
                                tprefix = tprefix.replace("a#", "a" + str(alpha))
                                tprefix = tprefix.replace("e#", "e" + str(epsilon))
                                tprefix = tprefix.replace(".", "")
                                
                                result = subprocess.run(['squeue','--format="%.18i %.9P %j %.2t %.10M %.6D %R"', '-u', 'meinhorn6'], capture_output=True, text=True).stdout
                                strOut = result
                                result = subprocess.run(['squeue','--format="%.18i %.9P %j %.2t %.10M %.6D %R"', '-u', 'smaguluri3'], capture_output=True, text=True).stdout
                                strOut = result + strOut

                                filePath = dirPath + "/" + tprefix + ".json"
                                if not replaceEx and (os.path.isfile(filePath) or (tprefix in strOut)):
                                    print("skipping " + tprefix)
                                    continue
                                    
                                tdata = data.replace("p#", tprefix)
                                tdata = tdata.replace("t#", str(trial))
                                tdata = tdata.replace("sb#", str(syncBackups))
                                tdata = tdata.replace("f#", str(fedP))
                                tdata = tdata.replace("a#", str(alpha))
                                tdata = tdata.replace("e#", str(epsilon))
                                    
                                print(tdata)
                                textfile = open("autoExp.sh", "w")
                                a = textfile.write(tdata)
                                textfile.close()
                                #os.system("sbatch autoExp.sh")
                                result = subprocess.run(['sbatch', 'autoExp.sh'], capture_output=True, text=True).stdout
                                strOut = result
                                print(strOut)
                                
                                
                                result = subprocess.run(['squeue','--format="%.18i %.9P %j %.2t %.10M %.6D %R"', '-u', 'meinhorn6'], capture_output=True, text=True).stdout
                                strOut = result
                                result = subprocess.run(['squeue','--format="%.18i %.9P %j %.2t %.10M %.6D %R"', '-u', 'smaguluri3'], capture_output=True, text=True).stdout
                                strOut = result + strOut

                                print(strOut)
                                nlines = strOut.count('\n')
                                print(nlines)

                                while nlines > 40:
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
