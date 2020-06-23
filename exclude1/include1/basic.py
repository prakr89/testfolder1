from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
help(Sequential)
model=Sequential([Dense()])
###File operation
import os
name="This is 1"
with open(r"C:\Users\M1053110\mycodings\test1.txt","r+") as file:
    for p in file:
        print(p)

###basic application of function on a list
list_a = [1, 2, 3]
list_b = [10, 20, 30]
map(lambda x, y: x + y, list_a, list_b)

multiply3=lambda x: 3*x
def multiply3(x):
  return x * 3
list(map(multiply3, [1, 2, -3, 4]))

[x*3 for x in list_a]

###fabonacci series
def fib(n):
    a, b = 0, 1
    while n!=0:            # First iteration:
        #yield a            # yield 0 to start with and then
        a, b = b, a + b    # a will now be 1, and b will also be 1, (0 + 1)
        n=n-1

def rec_fib(n):
    if n > 1:
        return rec_fib(n-1) + rec_fib(n-2)
    elif n==0:
        return 0
    else:
        return 1
    return n
rec_fib(1)

###Clearing a list
del list1[:]
list1.clear()

###removing second max
new_list.remove(max(new_list))
list1.sort(reverse=True)[-2].remove()

###Find closest absolute element to value zero
list1=[9,-9,8,-2,2]
def closest(l):
    mini=min(list1,key=abs)
    return mini
closest(list1)

###reversing string using recursion
def reverse(str1):
    if str1 == '':
        return str1
    else:
        return reverse(str1[1:]) + str1[0]

def reverse(s):
    return s[-1] + reverse(s[0:-1]) if s else s

print (reverse('retupmoc'))


###Find GCD of more than two numbers
gcd(a, b, c) = gcd(a, gcd(b, c))
             = gcd(gcd(a, b), c)
             = gcd(gcd(a, c), b)


def find_gcd(x, y):
    while y:
        x, y = y, x % y
    return x
find_gcd(8, 42)

l = [2, 4, 6, 8, 16]
num1=l[0]
num2=l[1]
gcd=find_gcd(num1,num2)
for i in range(2,len(l)):
    gcd=find_gcd(gcd,l[i])
print(gcd)

NOTE
gcd = lambda a,b : b, gcd(a%b)
>>> gcd(10,20)
>>> 10

###Creating df from file
import pandas as pd
emp = pd.read_csv(r'C:\Users\M1053110\mycodings\emp.csv')
to_list = emp.name.tolist()[1:]
print(to_list)

###
use
str.isupper()

#Row number start from 0 and 1 to 8 is skipped then take show 10 rows from here
df=pd.read_csv(r"C:\Users\M1053110\mycodings\emp.csv" , header=0, sep=',', skiprows=range(1,9), nrows=10)
df
df[df.tdate.notnull()]
df.tdate.isna()==False
newDf = df[df.gender.astype(str) == 'm']
#or
newDf = df[df.gender == 'm']
newDf

###String formating
line='My name is praveen.I age is 25'
line2='fg25 '
print(line2.isalnum())
print(line,end='.')
name='Praveen'
age=25
print("My name is {:*^5.3} \n and age is {}".format(name,age))
print(f"My name is {name:.3} and age is {age}")

###Global Local
global eggs
def spam2():
    # global eggs
    eggs=1
    eggs=3
eggs=23
#global eggs=23 --global cannot be initiated
spam2()
def xyz():
    eggs=5
    return print(eggs)
xyz()
print (eggs) --5 23


a = [(1,2,3),('a','b','c'),(7,8,9)]
b = [[1,2,3],['a','b','c'],[7,8,9]]
c = ((1,2,3),('a','b','c'),(7,8,9))
[i[0] for i in a]
[i[0] for i in b]
[i[0] for i in c]
[i[1] for i in b][1]
b[1]

###Queue in python
enqueue(5) is inserting 5 from rear back
dequeue(5) is deleting 5 from front
peak() return top front current
size() retun size
queue is FIFO and stack is LIFO
class Queue:
    def __init__(self):
       self.items=[]
    def enqueue(self,item):
       self.items.append(item)
    def dequeue(self):
       return self.items.pop(0)
    def isEmpty(self):
       return self.items==[]
    def __len__(self):
       return len(self.items)
q=Queue()
q.enqueue(5)

l=[]
global front
globl rear
front,rear=-1,-1

def isFull():
    global front
    global rear
    if(rear==maxsize):
        return 1
    else:
        return 0

def isEmpty():
    global front
    global rear
    if(front==-1 and rear==-1):
        return 1
    elif(front==rear):
        return 1
    else:
        return 0

def enqueue(n):
    global front
    global rear
    if(isEmpty()==1):
        front=0
        rear=0
        l[rear]=n
        rear=rear+1
    else:
        l[rear]=n
        rear=rear+1

def dequeue():
    global front
    global rear
    if(isEmpty()==1):
        return 1
    else:
        front=front+1


###=Python questions
                ##Q=Find if digit in a string
import re
password = "hello11"
matches = re.findall('[0-9]', password)
if len(matches) < 1:
    print ("Password must contain at least one integer")
else:
    print ("Password is valid")
    print(matches)
##OP=['1', '1']
def RepresentsInt(text):
    try:
        int(text)
        return True
    except ValueError:
        return False
RepresentsInt("I am praveen")
##OP=False
print("12345".isdigit())
print("12345a".isdigit())

                ##Q=Count repeating items only in a list
list1 = [1,1,1,1,2,2,2,2,3,3,4,5,5,"b","Aa","aa","Aa"]
dic={item:item.count() for item in list1 if count(item)>1}


dict = {item:list1.count(item) for item in list1 if list1.count(item) > 1}
dict
##OP={1: 4, 2: 4, 3: 2, 5: 2, 'Aa': 2}
dic = {}
for i in list(set(list1)):
    b = list1.count(i)
    dic[i] = b
print(dic)
#OP={1: 4, 2: 4, 3: 2, 4: 1, 5: 2, 'Aa': 2, 'aa': 1, 'b': 1}
rep_item, count = dict.keys(), dict.values()
rep_item
##OP=[1, 2, 3, 5, 'A']

                ##Q=Count max repeating items only in a list/string
count=max(dic.values())
for item in dic.keys():
    dic[item]=count
    break
print("item "+str(item)+" is most repeated")
##OP=item 1 is most repeated

                ##Q=Count number of a specific item in a list/string
word = "babulibobablingo"
letter = 'b'
if letter in word:
    print(word.count(letter,8,len(word)))
##OP=2

                ##Q=Count total number of repeating element in list and string
inputstr = "Heyiiiiaaaammmpk"
list1 =[1,1,1,1,2,2,2,2,3,3,4,5,5,"b","Aa","aa","Aa"]
print(len(inputstr)-len(list(set(inputstr.lower()))))
##OP=9

                ##Q=Removing duplictes and preserve order in list and string
list1 =[1,1,1,1,2,2,2,2,3,3,4,5,5,"b","Aa","aa","Aa"]
newlist=[]
for i in list1:
  if i not in newlist:
    newlist.append(i)
newlist

inputstr = "Heyiiiiaaaammmpk"
x=['H', 'e', 'y', 'i', 'a', 'm', 'p', 'k']
print(''.join(sorted(set(inputstr), key=inputstr.index,reverse=True)))
#OP=kpmaiyeH
from collections import OrderedDict
list(OrderedDict.fromkeys(inputstr))
#OP=['H', 'e', 'y', 'i', 'a', 'm', 'p', 'k']
print("".join(OrderedDict.fromkeys(inputstr)))
#OP=Heyiampk

                ##Q=join list to string
a = ['b','c','d']
strng = ''
print(''.join(a))
#OP=bcd
for i in a:
   strng +=str(i)
print (strng)

                ##Q=Adding two list
[a[i]+b[i] for i in range(len(a))]
np.add(first, second)
[sum(i) for i in zip(first,second)]
list(map(lambda x,y: x+y, a,b))
third = np.array(first) + np.array(second)
third

                ##Q=find random number
import numpy as np
import random
test_list = [1, 4, 5, 2, 7]
np.random.seed(4)
random_num = np.random.choice(test_list)
# using random.choice() to get a random number from that list
# We need to use random.seed() and random.choice function together to
# produces the same element every time. Let see this with an example
random.seed(4)
random_num = random.choice(test_list)
# using random.randint() to get a random number
rand_idx = random.randint(0, len(test_list)-1)
random_num = test_list[rand_idx]


                ##Q=find n random number
import random
a = [1,2,3,4,5,6,7,8,9]
np.random.shuffle(a)
a[:4] # prints 4 random variables
import random
a = [1,2,3,4,5,6,7,8,9]
random.shuffle(a)
a[:4] # prints 4 random variables
np.random.choice(mylist, 3, replace=False)


                ##Q=Construct 2d array
import numpy as np
np.random.randint(0,10, size=(2, 4))
#from 0 to 9
array([[5, 0, 2, 1],
       [3, 2, 2, 9]])
import random
rows = 3
columns = 4
[[random.randrange(22, 37, 2) for x in range(columns)] for y in range(rows)]

                ##Q=DO bubble sorting

def bs(x):
    Done = False
    while Done == False:
        Done = True
        for i in range(len(x) - 1):
            if x[i] > x[i + 1]:
                Done = False
                x[i],x[i + 1]= x[i + 1],x[i]
    return x
bs([3,4,2,1])
if __name__ == '__main__':
    sorted=bs([int(x) for x in input("Enter with spaces").split(" ")])
    print(sorted)

result = []
for a in range(1,8):
    number = int(input("please enter a number: "))
    result.append(number)
print result



                ##Q=Do Quick sorting
import random
def quicksort(arr):
    more,less=[],[]
    if len(arr) < 2:
        return arr #base case
    else:
        pivot = random.choice(arr)
        for i in arr:
            if i < pivot:less.append(i)
            if i > pivot :more.append(i)
        return (quicksort(less) + [pivot] + quicksort(more))
quicksort(list1)
list1=[3,4,2,1]




                ##Q=Find element using binary search
def binarysearch(list1,val):
    if len(list1)==0 or (len(list1)==1 and list1[0]!=val):return "Not Found"
    else:
        midindex=int(len(list1)/2)
        mid=list1[midindex]
        if mid==val :return "Found"
        if mid>val:return binarysearch(list1[:midindex],val)
        if mid<val:return binarysearch(list1[midindex+1:],val)
binarysearch([1,2,3,4],6)

##TwoSum Problem
def twoSum(arr, target):
    index_map = {}
    for i in arr:
        pair = target - i
        if pair in index_map:
            return [index_map[pair], i]
        index_map[i] = i
    return None

def twoSum(arr, target):
    dict = {}
    results = []
    for i in arr:
        if (target - i) in dict:
            if dict[target - i]:
                continue
            results.append([i, target - i])
            dict[target - i] = True
        else:
            dict[i] = False
    return results
arr = [1,4,8,5,5,9,15]
target = 13
print("Solution: ", twoSum(arr, target))


                ##Q=Find palandrome or not
def findpalan(input1):
    try:
        input=str(input1)
        if input1==input[::-1]:
            print("it is palandrome")
        else:
            print("it is not a palandrome")
    except:
        print("Please enter correct")

findpalan("madam")

def findpalan(input1):
    try:
        input=str(input1)
        if input1==input[::-1]:
            x=""
        else:
            x="not"
    except:
        print("Please enter string or int")
    return x

if __name__ == '__main__':
    print("This is "+findpalan(input("pls enter"))+" palandrome")


###Manipulating a datafrmae
import pandas as pd
import numpy as np
pd.options.display.max_columns=None
pd.options.display.max_rows=None
raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'last_name': ['Mil_ler', 'Jaco_bson', 'Ali',None, 'Coo.ze'],
        'age': [42,None, 36, 24, 73],
        'gender':['M','F','M',None,'M'],
        'preTestScore': [4, 24, 31, 2, 3],
        'postTestScore': [25, 94, 57, 62, 70],
        "years":["10+ years",None,"<1 year","<2 year","10"]}
df=pd.DataFrame(raw_data)
df[:3][['first_name','last_name']]
df[:3][df.columns[0:2]]
df.iloc[:3,:2]
df.iloc[0:3,0:2]
  first_name	last_name
0	Jason	     Miller
1	Molly	     Jacobson
2	Tina	     Ali
df.loc[:3,df.columns[0:2]]
df.loc[0:3,['first_name','last_name']]
    first_name	last_name
0   	Jason	Miller
1   	Molly	Jacobson
2   	Tina	Ali
3   	Jake	Milner

df[df.age.notnull()][0:3][['first_name','last_name']]
df.gender.fillna('')
df[df.Last_Name.notnull()]
df[(df[column_name].notnull()) & (df[column_name]!=u'')]
df.loc[df['first_name']=='Jason']
df.loc[df['gender'].isnull(),['first_name','age']]
df.loc[df['gender'].notnull(),['first_name','age']]
df.loc[df['gender'].notnull(),:]
df.loc[df.gender.isna()==False]
newdf = df.loc[df["column_name"].isnull()]
newdf = df.loc[df["column_name"]=='None']
df.dropna()
df.fillna(0)
df.gender.replace(np.nan, 0)
df1 = df.replace(np.nan, '', regex=True)
df.replace(['None', 'nan'], np.nan, inplace=True)
df.replace(r'^\s*$', np.nan, regex=True, inplace = True)
##till now we were not having issue because we were doing number operation but issue when string matching as blank is not a string so
df[df["first_name"].fillna('').str.startswith("J")]
df[df["first_name"].str.contains("K", case=False, na=False)]
df["first_name"].fillna('').str.contains('(\d+[A-Z]+\d+)')/or/match("^A.*")

##Donot use inplace in original df try assign to new column and drop original col later after your conversion is properly done.
##Check varitites of values we have in one column
df.years.unique
0      10+ years
1
2      <1 year
3      <1 year
4           10
Name: years, dtype: object>
NOTE dtype=object means could value is mix of string/number with at least null values.

df["newYrs"]=df["years"].str.replace(r"\+ years","")
df["newYrs"]=df["newYrs"].str.replace(r" year","")
df["newYrs"]=df["newYrs"].str.replace(r"<1",str(0))
df["newYrs"]=df["newYrs"].str.replace(r"<","")
df["newYrs"]=df["newYrs"].replace(np.nan,0)
df["newYrs"]=pd.to_numeric(df["newYrs"])
df["newYrs"]


df["new1Last_name"]=df["last_name"].str.replace(r"[_]","").fillna('')
df["new2Last_name"]=df["last_name"].str.replace(r"[.]","")
df["new2Last_name"]


dfnew=df.replace(r"[._]","").fillna('')[["first_name","last_name"]]
dfnew

df.info()
#apply some function to a column
def F2(x):
    return x+2
df['postTestScore'].apply(lambda x: F2(x))

x=df['postTestScore'].tolist()
list(map(F2,x))

NOTE
null values represents "no value" or "nothing", it's not even an empty string or zero.
It can be used to represent that nothing useful exists.
NaN stands for "Not a Number", it's usually the result of a mathematical operation t
hat doesn't make sense, e.g. 0.0/0.0.
#If you want to count the missing values in each column, try:
df.isnull()[["gender","age"]]
df.isnull().sum() or df.isnull().sum(axis=0)

valuesB = [(1, 'ketchup','bob', 1.20,"C",20),
           (2, 'rutabaga', 'bob', 3.35,"C",None),
           (3, 'fake vegan meat', 'rob', 13.99,"F",40),
           (4, 'cheesey poofs', 'tim', 3.99,"F",50),
           (5, 'ice cream', 'tim', 4.95,"C",60),
           (6, 'protein powder', 'tom', 49.95,None,70)]
ordersDF = pd.DataFrame(valuesB,columns=['id', 'product_name', 'customer', 'price',"Status",'number'])
ordersDF
ordersDF["orders"]=ordersDF.groupby(['customer','Status'])['number'].cumsum()
#Please note that cumsum will give entie rows bt agg will give aggregated rows
TotalOrdersDF = ordersDF.groupby(['customer', 'Status']).agg({'price':{'price_mean': 'mean', 'price_max': lambda x: max(x) - 1}, 'number':{'orders': 'sum'}})
TotalOrdersDF.columns = TotalOrdersDF.columns.get_level_values(1)
TotalOrdersDF.sort_values(["orders","customer"], ascending=[True, False])
TotalOrdersDF[TotalOrdersDF.orders>40]
TotalOrdersDF

empsal=[("ak","A",8),("bk","A",9),("ck","A",20),("dk","A",23),("ek","A",10),("fk","A",15),("gk","B",8),("hk","B",11),("ik","B",22),("jk","B",7),("kk","B",38),("lk","B",18)]
empsalDF = pd.DataFrame(empsal,columns=["ename","dept","sal"])
empsalDFnew=empsalDF.groupby(["dept"])[["sal"]].agg(['sum','cumsum'])
empsalDFnew["ratio"]=empsalDFnew["cumsum"]/empsalDFnew["sum"]
empsalDFnew[empsalDFnew["cumsum"]/empsalDFnew["sum"]<0.5]

empsalDF[empsalDF.groupby(["dept"])["sal"].cumsum().div(sum(empsalDF.groupby(["dept"])[["sal"]]))<0.5][["ename","sal","dept"]]
empsalDF.groupby(["dept"])[["sal"]].mean()

[["ename","sal","dept"]]
###Union of dataframes
# by default:
# join is a column-wise left join
# pd.merge is a column-wise inner join
# pd.concat is a row-wise outer join


import pandas as pd
data1 = {'Name':['Jai', 'Princi', 'moni', 'Anuj'],
        'Age':[27, 24, 14, 32],
        'Address':['Nagpur', 'Kanpur', 'Allahabad', 'Kannuaj'],
        'Qualification':['Msc', 'MA', 'B.A', 'Phd']}
data2 = {'Name':['Abhi', 'Ayushi', 'Dhiraj', 'Hitesh'],
        'Age':[17, 14, 12, 52],
        'Address':['Nagpur', 'Kanpur', 'Allahabad', 'Kannuaj'],
        'Qualification':['Btech', 'B.A', 'Bcom', 'B.hons']}
data3 = {'Name':['Abhi', 'moni', 'Dhiraj', 'Princi'],
        'Age':[17, 14, 12, 52],
        'Address':['Nagpur', 'Kanpur', 'Allahabad', 'Kannuaj'],
        'Qualification':['M.tech', 'B.A', 'Bcom', 'B.hons']}
df1 = pd.DataFrame(data1,index=data1['Name'],columns=['Age','Address','Qualification'])
df2 = pd.DataFrame(data2, index=data2['Name'],columns=['Age','Address','Qualification'])
df3 = pd.DataFrame(data3, index=data3['Name'],columns=['Age','Address','Qualification'])
# df['ingredient']=df.index
# df = df1.reset_index(drop=True)
# df1 = df1.set_index('A')
pd.concat([df1,df2,df3],keys=['grp-a','grp-b','grp-c'])
pd.concat([df2,df3],axis=1,join='outer',sort=False)
#Below will ignore index as "Name" and index will default 0,1,2,3
pd.concat([df1,df2,df3], axis=0, ignore_index=True,sort=False)


###General join in dataframe
pd.merge(df3,df2,on='Age')
pd.merge(df3,df2,left_on=['Age'],right_on=["Age"])
pd.merge(df2,df3,on=['Age','Qualification'],how='inner')
pd.merge(df1,pd.merge(df2,df3,on=['Age','Qualification'],how='inner'),on=['Age','Qualification'],how='inner')
pd.merge(df2[['Age','Qualification']],df3,on=['Age','Qualification'],how='inner')
pd.merge(df1,pd.merge(df2,df3,on=['Age','Qualification'],how='inner'),on=['Age','Qualification'],how='inner')
