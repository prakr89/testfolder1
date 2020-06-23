#Spark with RDD and sqlcontext version less than 2.0
import findspark
findspark.init()
import pandas as pd
import pyspark
import subprocess
from pyspark import StorageLevel
from pyspark.sql import types as T
from pyspark.sql.window import Window as W
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark import sql
import pyspark.sql.functions as F
conf = SparkConf().setAll([("spark.app.name", "simpleApplicationTests")])
sc = SparkContext(conf=SparkConf().setAppName("MyApp").setMaster("local"))
#sc = SparkContext.getOrCreate(conf=SparkConf().setAppName("MyApp").setMaster("local"))
sqlContext =SQLContext(sc)
sc.stop()
#Demo1
nums = sc.parallelize([1,2,3,4])
xsquared=nums.map(lambda x: x*x).collect()
for i in xsquared:
    print(i)
words = sc.parallelize(["scala","java","hadoop","spark","akka"])
filtered = words.filter(lambda x: x.startswith('s'))
count=filtered.count()
count
response_body = "total words are with s : " + str(count)
print (response_body)


#Demo2
sc.parallelize([3,4,5]).map(lambda x: range(1,x)).collect()
#[range(1, 3), range(1, 4), range(1, 5)]
sc.parallelize([3,4,5]).flatMap(lambda x: range(1,x)).collect()
#[1, 2, 1, 2, 3, 1, 2, 3, 4]


#Demo3
abhay_marks = [("physics",85),("maths",75),("chemistry",95)]
ankur_marks = [("physics",65),("maths",75),("chemistry",85)]
abhay = sc.parallelize(abhay_marks)
ankur = sc.parallelize(ankur_marks)
abhay.union(ankur).sortByKey().distinct().collect()

[('chemistry', 95),
 ('chemistry', 85),
 ('maths', 75),
 ('physics', 85),
 ('physics', 65)]

#Demo4
string.punctuation
punc='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
import re,string
def clean(txt1):
    txt2=txt1.encode("utf-8").lower()
    txt3=txt2.replace("--"," ")
    return txt3
txt4=sc.textFile(r"C:\Users\M1053110\mycodings\paragraph.txt")
counts = txt4.flatMap(lambda line:clean(line).split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b).collect()
counts
##counts.saveAsTextFile("hdfs://...")
#sc.textFile ("file:///opt/spark/CHANGES.txt")
#sc.uiWebUrl
"""
counts.pprint() --in scala
ssc.start()
ssc.awaitTermination()

lambda x : True if (x > 10 and x < 20) else False
lambda x: 1 if x>0 elif 0 if x ==0 else -1
"""
#Demo5
df = sc.parallelize([[25, 'Prem', 'M', '12-21-2006 11:00:05','abc', '1'],
                      [20, 'Kate', 'F', '05-30-2007 10:05:00', 'asdf', '2'],
                      [40, 'Cheng', 'M', '12-30-2017 01:00:01', 'qwerty', '3']]).\
    toDF(["age","name","sex","datetime_in_strFormat","initial_col_name","col_in_strFormat"])
df = df.withColumn("struct_col", struct('age', 'name', 'sex')).\
    drop('age', 'name', 'sex')
df.show()
#Convert the timestamp from string (i.e. 'datetime_in_strFormat') to datetime (i.e. 'datetime_in_tsFormat')
df = df.withColumn('datetime_in_tsFormat',
                   unix_timestamp(col('datetime_in_strFormat'), 'MM-dd-yyyy hh:mm:ss').cast("timestamp"))
df.show()
df.printSchema()
df = df.withColumn('name', col('struct_col.name')).\
    withColumn('age', col('struct_col.age')).\
    withColumn('sex', col('struct_col.sex')).\
    drop('struct_col')
df.show()
df.printSchema()

#Demo6 Monkey patch since word 'transform' give error as unrecognised
from pyspark.sql.dataframe import DataFrame
def dfmap(self, f):
    return f(self)
DataFrame.dfmap = dfmap

def with_something(df, something):
    return df.withColumn("something", F.lit(something))
def withIDPlusOne(df ,ID):
    return df.withColumn("ID2", F.col(ID) + 1)

dfnew = sc.parallelize([(2,'spark'),(5,'hadoop')]).toDF(['ID','Name'])
weirdDf = dfnew.dfmap(lambda df: with_something(df, "crazy"))
weirdDf = weirdDf.dfmap(lambda df: withIDPlusOne(df, "ID"))
weirdDf.show()
+---+------+---------+---+
| ID|  Name|something|ID2|
+---+------+---------+---+
|  2| spark|    crazy|  3|
|  5|hadoop|    crazy|  6|
+---+------+---------+---+

#Demo7
dfemp=sqlContext.read.format("csv").option("header", "true").option("inferSchema", "true").option("sep", ",").load(r"C:\Users\M1053110\mycodings\emp.csv")
dfemp.show()
display(dfemp)




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

###Kafka
"""
from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession,SQLContext
from pyspark.streaming.kafka import KafkaUtils
from pyspark.streaming import StreamingContext
zkQuorum, topic = sys.argv[1:]
kvs = KafkaUtils.createStream(ssc, zkQuorum, "spark-streaming-consumer", {topic: 1})
lines = kvs.map(lambda x: x[1])
lines.pprint()
counts = lines.flatMap(lambda line: line.split(" ")) \
                  .map(lambda word: (word, 1)) \
                  .reduceByKey(lambda a, b: a+b)

lines = sc.textFile(path)
parts = lines.map(lambda l: l.split("\t"))
weblogs_hit = parts.map(lambda p: Row(hit_timestamp=p[1], swid=p[13], ip_address=p[7], url=p[12], user_agent=p[43], city=p[49], country = p[50], state = p[52]))

## create a Data Frame from the fields we parsed
schema_weblogs_hit = sqlContext.createDataFrame(weblogs_hit)

## register Data Frame as a temporary table
schema_weblogs_hit.registerTempTable("weblogs_hit")

## do some basic formatting and convert some values to uppercase
rows = sqlContext.sql("SELECT hit_timestamp, swid, ip_address, url, user_agent, UPPER(city) AS city, UPPER(country) AS country, UPPER(state) AS state from weblogs_hit")

df.groupBy('page').count().sort(F.desc('count')).show()

"""
spark-submit \
    --properties-file spark-defaults.conf \
    --name CLAE_prm_test \
    --class Main xyz.py \
    --py-files PyFiles.zip --conf spark.dynamicAllocation.enabled=true --executor-memory 1G



###End
