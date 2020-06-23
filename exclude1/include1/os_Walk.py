import os
path=r"C:\Users\M1053110\Desktop\MyFolder"
allfiles = []
print(list(os.walk(path))) #give us list of touples
for i in list(os.walk(path)):
    print(i)
    print("-------------")
for dir,folds,files in list(os.walk(path)):
    if len(files) !=0:
        for each_file in files:
            if each_file.endswith("txt"):
                allfiles.append(os.path.join(dir,each_file))
                print(os.path.join(dir,each_file))
                print("--------------")

# dir="abc"
# each_file="xyz.txt"
# new="mmm"
# print(os.path.join(dir,each_file,new))

allfiles


#Example
import os
path="C:\\Users\\M1053110\\Desktop\\MyFolder"
for x in os.walk(path):
    print(x)
    numFiles = len(x[2])
    #emplty list/donot list that folder for  empty folders ie folders without files
    listSubDir = ['.\\' if s == '' else s for s in [x[0].split('\\')[-1]] * numFiles ]
    pathList = [s for s in [x[0]] * numFiles ]
    inFileNameList = x[2]
    zipList = zip(pathList, inFileNameList)
    pathFileNameList = ['\\'.join(p) for p in zipList]
    print(listSubDir)
