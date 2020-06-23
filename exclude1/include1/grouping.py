import re
SearchFrom="""cat mat rat Mr Tom Mr. Som Mrs Cary 456-34-1 Ms pery 123-45-1 praveen123-Kumar@gmail.com Praveen.Kumar@perian.edu http://www.google.com https://udemy.edu www.tcil.in"""
patt=re.compile(r'(http:|https:)?(/{1,2})?(www.)?(\w+)(\.\w+)')
#it have some flags as well ex:-
#patt=re.compile(r'(http:|https:)?/?/?(www.)?(\w+)(\.\w+)',re.IGNORECASE)
##findall will give list of all matches in with respective groups
#matches=patt.findall(SearchFrom)
rowwise_matches=re.findall(patt,SearchFrom)
rowwise_matches[0]

"""
[('', '', '', 'gmail', '.com'),
 ('', '', '', 'Praveen', '.Kumar'),
 ('', '', '', 'perian', '.edu'),
 ('http:', '//', 'www.', 'google', '.com'),
 ('https:', '//', '', 'udemy', '.edu'),
 ('', '', 'www.', 'tcil', '.in')]
"""
patt=re.compile("\bcat.") #will match cat at the begining of string
patt=re.compile("cat\b")  #will match cat at the end of string


##finditer will give generator object
columnwise_matches=re.finditer(patt,SearchFrom)
columnwise_matches
for match in columnwise_matches:
    print(match.group(3))
"""
None
None
None
www.
None
www.
"""
##search gives only first match of all groups as iterator from everywhere but "match" will fail evenif start character from input string donot match
firstrowonly_matches=patt.search(SearchFrom)
firstrowonly_matches #<re.Match object; span=(79, 88), match='gmail.com'>
firstrowonly_matches.groups() #(None, None, None, 'gmail', '.com')

patt = re.compile("dog")
patt.match("dog has dog") #<re.Match object; span=(0, 3), match='dog'>
patt.search("dog has dog")#<re.Match object; span=(0, 3), match='dog'>
patt.search("dog has dog",4,15) #<re.Match object; span=(8, 11), match='dog'>
patt.fullmatch("dog has dog") #No match as pattern should be same lenght as input ie exact match ie dog has dog

#patt=re.compile(r'M(r|s|rs)\.?\s?\w+')
#patt=re.compile(r'(Mr|Ms|Mrs)\.?\s?\w+')
#patt=re.compile(r'\d{3}-\d{2}-\d{1}')
patt=re.compile(r'([A-Za-z0-9-.]+@[A-Za-z]+.)(com|edu)')
matches=patt.finditer(SearchFrom)
matches
for match in matches:
    print(match.group(0))
    print(match)


##Example to get account numbers
inFileNameList=["AMITABH BACHAN - 10339 - RED INC.xlsx","KARAN COFFEE - X00448 - NOKIA SOLUTION.xlsx"]
step1
tmpFNameList = [s.split(' - ') if ' - ' in s else s.split('_') if ' - ' in s else s.split('.') for s in inFileNameList ]
tmpFNameList
[['AMITABH BACHAN', '10339', 'RED INC.xlsx'],
 ['KARAN COFFEE', 'X00448', 'NOKIA SOLUTION.xlsx']]

step2
k=[[i for i in l if i[1:].isdigit()] for l in tmpFNameList]
k
[['10339'], ['X00448']]
step3
accntNumList = [l2[0] for l2 in k]
accntNumList
['10339', 'X00448']

##use regex search in a comprehensive list
import re
l = ['this', 'is', 'just me', 'a', 'test me','khos']
regex = re.compile(r'^t.is$') #['this', '', '', '', '', '']
regex = re.compile('^[kj]'+'.*'+'e$') #['', '', 'just me', '', '', '']
regex = re.compile('^[kj]'+'.*'+'e$')
regex = re.compile('[^kj]'+'.*'+'e$') #['', '', '', '', 'test me', '']
matches = [string if re.search(regex, string) else "" for string in l]
matches

##get digits from string
r = "456results string789"
s = ''.join(x for x in r if x.isdigit())
s


##split a string using pattern
s = 'km125'
re.findall(r'[A-Za-z]+|\d+', s)
[i for i in re.split(r'(\d+)', s) if i]
#[A-Za-z]+ matches one or more alphabets. | or \d+ one or more digits

import re
snew="""
45 meters?
45, meters?
45?
45 ?
45 meters you?
45 you  ?
45, and you?"""
snew=["45 meters?","45, meters?","45?","45 ?","45 meters you?","45 you  ?","45, and you?"]
#Sea=re.compile(r'^(?!.*you).*\?$')
Sea=re.compile(r"\d+[^?]*you|(\d+[^?]*\?)")
#it have some flags as well ex:-
#Sea=re.compile(r'(http:|https:)?/?/?(www.)?(\w+)(\.\w+)',re.IGNORECASE)
matches1=Sea.findall(snew)
matches3=Sea.search(snew,re.M)
matches3
matches1
for match in matches1:
    print(match)


#######################################NOTE short:########################################
three ways of writing:-
matches=patt.findall(searchfrom)
matches=re.findall(patt,searchfrom)
matches=re.compile(r"xyz").findall(searchfrom)

re.search=output only first search either in group of strings/or group of elements in list
re.findall=output row wise matches result in touple then use like matches[0][1]
re.finditer=output column wise matches result ie column1=match.group(1)

compile(r'(http:|https:)?(/{1,2})?(www.)?(\w+)(\.\w+)')
df.replace(['None', 'nan'], np.nan, inplace=True)
[re.sub(r'[^A-Z0-9]','',s) for s in Names]
when(regexp_replace(col('BusinessName'), '[^A-Z0-9]', '').rlike(".*" + "www" + ".*"), lit('x')))
nums.map(lambda x: x*x)
words.filter(lambda x: x.startswith('s'))

lambda x : True if (x > 10 and x < 20) else False
lambda x: 1 if x>0 else 0 if x ==0 else -1


#mtd1 to small operations like date extraction etc use UDF
from pyspark.sql.types import *
def amount_spent(quantity, price):
   return quantity * price
amount_spent_udf = F.udf(amount_spent, DoubleType())
customersDF1 = customersDF.withColumn('amount_spent', amount_spent_udf(customersDF['quantity'], customersDF['price'])).withColumn("ranking",F.row_number().over(Window.partitionBy("category").orderBy(F.desc("amount_spent"))).cast(IntegerType())).where(F.col("ranking")<=3)

string_to_datetime = F.udf(lambda x: dt.strptime(x, '%Y-%m-%d'), DateType())
customersDF3 = customersDF2.withColumn('datetime', string_to_datetime( customersDF2['date']))

#mtd2 for creating big fuctions and transform dataframe to another form
from pyspark.sql.dataframe import DataFrame
def dfmap(self, f):
    return f(self)
DataFrame.dfmap = dfmap
def F2(df, inputCol1="D", inputCol2="F", outputCol="Dclean"):
    df = (df
        .withColumn(outputCol,
            F.when(F.col(inputCol1).isNull(),
                F.concat(F.col(inputCol2), F.lit("MISSING")) )
            .otherwise(F.concat(F.col(inputCol2), F.col(inputCol1))))
    )
    return df
df=df_increment.select(selected_cols) \
.dfmap(lambda df: F2(df, inputCol1="DB", inputCol2="FS", outputCol="D_cleaned"))

#in pandas
df['team'].apply(lambda x: F2(x))

Create index i1 on table table9(col3) as ‘COMPACT’ with deferred rebuild [row format delimited fields terminated by'\n' stored as textfile];
Alter index i1 on table9 rebuild;
show formatted index on table9;
show partitons tabple2
show table extended like 'tbl_name' partition (dt='20131023');
Show databases extended like "p*";
desc formatted table0
create table if not exists table11(col1 int,col2 string,col3 string)partitioned by (year int)
clustered by(col2) into 4 buckets row format delimited fields terminated by',' collection items terminated by':' lines terminated by'\n' tblproperties(‘skip.header.line.count’=’3’ & ‘skip.footer.line.count’=’3’) stored as textfile;
"""
.-Matches any char except newline
\d-digits(0-9)
\D-except d
\w-(a-z A-Z 0-9 _)
\W-except w
\s-Whitespaces ie spaces,tab,newline)
\S-except s
\bcat|cat\b-will match cat the begining and end of string
\Bcat|cat\B-will do opposite of \b
\Acat-will match cat at the begining
\Zcat-will match cat at the end
[.-] match only one dash or dot,as mentioned in character set
[a-zA-Z] matches a to z or A-Z
[^a-c] matches not in a-c
[^abc] matches not in abc
[a^z] matches one of a,^,z
^[a-c] matches starting in a-c
[a-c]$ matches ending in a-c
quantifiers:-
? 0 or one
* 0 or more
+ 1 or more
{3}=matches exactly 3 quantity
{3,4}=3 min and 4 max
"""
