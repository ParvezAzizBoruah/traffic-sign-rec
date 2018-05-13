import os
for i in range(43):
    mypath = r'C:\Users\Parvez\Desktop\gtj\\'+str(i)
    if not os.path.isdir(mypath):
        os.makedirs(mypath)