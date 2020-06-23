import findspark
findspark.init()
import pandas as pd
import pyspark
import subprocess
import re
from pyspark import StorageLevel
# from pyspark.sql.functions import *
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('CheckPyspark').master("local").getOrCreate()

# spark = SparkSession.builder.config(conf=conf).getOrCreate()
#from pyspark.sql import SQLContext
# sqlContext = SQLContext(spark)
spark.stop()

###Creating a df and manipulating it
valuesA = [(1, 'bob', 3462543658686),
           (2, 'rob', 9087567565439),
           (3, 'tim', 5436586999467),
           (4, 'tom', 8349756853250)]
customersDF = spark.createDataFrame(valuesA,('id', 'name', 'credit_card_number'))

valuesB = [(1, 'ketchup','bob', 1.20,"C",20),
           (2, 'rutabaga', 'bob', 3.35,"C",None),
           (3, 'fake vegan meat', 'rob', 13.99,"F",40),
           (4, 'cheesey poofs', 'tim', 3.99,"F",50),
           (5, 'ice cream', 'tim', 4.95,"C",60),
           (6, 'protein powder', 'tom', 49.95,None,70)]
ordersDF = spark.createDataFrame(valuesB,['id', 'product_name', 'customer', 'price',"Status",'number'])
ordersDF.show()
ordersDF.select(F.concat(F.col("id"), F.lit(" "), F.col("customer"))).show(2)
ordersDF.groupby(F.col('customer')).agg(F.count('customer').alias('count')).show()
TotalOrdersDF = ordersDF.groupby('customer', 'status').agg(F.sum('number').alias('orders')).orderBy('orders', 'customer', ascending=False)
TotalOrdersDF.filter(F.col('orders') > 40).show()
ordersDF.agg(F.corr("price","number").alias('correlation')).show()

df.na.drop()
df.na.fill(0)
df.where(col("a").isNull())
df.where(col("a").isNotNull())
# from pyspark.sql.functions import isnan
# df.where(isnan(col("a")))
# df = spark.createDataFrame([(1.0, float('nan')), (float('nan'), 2.0)], ("a", "b"))
# df.select(isnan("a").alias("r1"), isnan(df.a).alias("r2")).collect()
# [Row(r1=False, r2=False), Row(r1=True, r2=True)]

###coalesce-Returns the first column that is not null else second column
cDf = sqlContext.createDataFrame([(None, None), (1, None), (None, 2)], ("a", "b"))
cDf.show()
+----+----+
|   a|   b|
+----+----+
|null|null|
|   1|null|
|null|   2|
+----+----+
cDf.select(coalesce(cDf["a"], cDf["b"])).show()
+-------------+
|coalesce(a,b)|
+-------------+
|         null|
|            1|
|            2|
+-------------+
cDf.select('*', coalesce(cDf["a"], lit(0.0))).show()
+----+----+---------------+
|   a|   b|coalesce(a,0.0)|
+----+----+---------------+
|null|null|            0.0|
|   1|null|            1.0|
|null|   2|            0.0|
+----+----+---------------+
tmp = df.withColumn('c', coalesce(df['a'],df['b']).cast(FloatType()))
NOTE:- We can use when/otherwise generally for this purpose
df.withColumn("flag_new",
F.when(F.col('DATE_RECEIVED')<F.unix_timestamp() & df_inc_prm_date.name==F.col('name'),"D")
.otherwise("A")).drop(F.col('flag')).withColumnRenamed("flag_new","flag")


###UDF creating from def function and using it

def square(x):
    return x**2
from pyspark.sql.types import FloatType
square_price_float = F.udf(lambda z: square(z), FloatType())
ordersDF.select("*",square_price_float("price").alias("squared price")).show()

d_np = pd.DataFrame({'int_arrays': [[1,2,3], [4,5]]})
df_np = spark.createDataFrame(d_np)
from pyspark.sql.types import ArrayType
def square_list(x):
    return [float(val)**2 for val in x]
square_float_udf = F.udf(lambda y: square_list(y), ArrayType(FloatType()))
df_np.withColumn('doubled', square_float_udf('int_arrays')).show()
+----------+---------------+
|int_arrays|        doubled|
+----------+---------------+
| [1, 2, 3]|[1.0, 4.0, 9.0]|
|    [4, 5]|   [16.0, 25.0]|
+----------+---------------+

df2 = pd.DataFrame([[0,1,0],[1,0,0],[1,1,1]],columns = ['Foo','Bar','Baz'])
spark_df = spark.createDataFrame(df2)
def get_profile(foo, bar, baz):
    if foo == 1:
        return 'Foo'
    elif bar == 1:
        return 'Bar'
    elif baz == 1 :
        return 'Baz'
from pyspark.sql.types import StringType
spark_udf = F.udf(get_profile, StringType())
spark_df = spark_df.withColumn('get_profile',spark_udf('Foo', 'Bar', 'Baz'))
spark_df.show()
+---+---+---+-----------+
|Foo|Bar|Baz|get_profile|
+---+---+---+-----------+
|  0|  1|  0|        Bar|
|  1|  0|  0|        Foo|
|  1|  1|  1|        Foo|
+---+---+---+-----------+



import sys
from datetime import datetime as dt
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import *
customersDF = spark.createDataFrame([("Geoffrey", "2016-04-22", "A", "apples", 1, 50.00),
("Geoffrey", "2016-05-03", "B", "Lamp", 2, 38.00),
("Geoffrey", "2016-05-03", "D", "Solar Pannel", 1, 29.00),
("Geoffrey", "2016-05-03", "A", "apples", 3, 50.00),
("Geoffrey", "2016-05-03", "C", "Rice", 5, 15.00),
("Geoffrey", "2016-06-05", "A", "apples", 5, 50.00),
("Geoffrey", "2016-06-05", "A", "bananas", 5, 55.00),
("Geoffrey", "2016-06-15", "Y", "Motor skate", 7, 68.00),
("Geoffrey", "2016-06-15", "E", "Book: The noose", 1, 125.00),
("Yann", "2016-04-22", "B", "Lamp", 1, 38.00),
("Yann", "2016-05-03", "Y", "Motor skate", 1, 68.00),
("Yann", "2016-05-03", "D", "Recycle bin", 5, 27.00),
("Yann", "2016-05-03", "C", "Rice", 15, 15.00),
("Yann", "2016-04-02", "A", "bananas", 3, 55.00),
("Yann", "2016-04-02", "B", "Lamp", 2, 38.00),
("Yann", "2016-04-03", "E", "Book: Crime and Punishment", 5, 100.00),
("Yann", "2016-04-13", "E", "Book: The noose", 5, 125.00),
("Yann", "2016-04-27", "D", "Solar Pannel", 5, 29.00),
("Yann", "2016-05-27", "D", "Recycle bin", 5, 27.00),
("Yann", "2016-05-27","A", "bananas", 3, 55.00),
("Yann", "2016-05-01", "Y", "Motor skate", 1, 111.00),
("Yann", "2016-06-07", "Z", "space ship", 1, 227.00),
("Yoshua", "2016-02-07", "Z", "space ship", 2, 227.00),
("Yoshua", "2016-02-14", "A", "bananas", 9, 55.00),
("Yoshua", "2016-02-14", "B", "Lamp", 2, 38.00),
("Yoshua", "2016-02-14", "A", "apples", 10, 55.00),
("Yoshua", "2016-03-07", "Z", "space ship", 5, 227.00),
("Yoshua", "2016-04-07", "Y", "Motor skate", 4, 100.00),
("Yoshua", "2016-04-07", "D", "Recycle bin", 5, 27.00),
("Yoshua", "2016-04-07", "C", "Rice", 5, 15.00),
("Yoshua", "2016-04-07", "A", "bananas", 9, 55.00),
("Jurgen", "2016-05-01", "Z", "space ship", 1, 227.00),
("Jurgen", "2016-05-01", "A", "bananas", 5, 55.00),
("Jurgen", "2016-05-08", "A", "bananas", 5, 55.00),
("Jurgen", "2016-05-08", "Y", "Motor skate", 1, 125.00),
("Jurgen", "2016-06-05", "A", "bananas", 5, 55.00),
("Jurgen", "2016-06-05", "C", "Rice", 5, 15.00),
("Jurgen", "2016-06-05", "Y", "Motor skate", 2, 80.00),
("Jurgen", "2016-06-05", "D", "Recycle bin", 5, 27.00),
],["customer_name", "date", "category", "product_name", "quantity", "price"])
#customersDF.show()
def amount_spent(quantity, price):
   return quantity * price
amount_spent_udf = F.udf(amount_spent, DoubleType())
customersDF1 = customersDF.withColumn('amount_spent', amount_spent_udf(customersDF['quantity'], customersDF['price'])).withColumn("ranking",F.row_number().over(Window.partitionBy("category").orderBy(F.desc("amount_spent"))).cast(IntegerType())).where(F.col("ranking")<=3)


customersDF.createOrReplaceTempView("customersDF")
spark.udf.register("amount_spent_udf_SQL", amount_spent, "double")
customersDFsql1=spark.sql("""select *,amount_spent_udf_SQL(quantity,price) amount_spent from customersDF""" )
customersDFsql1.createOrReplaceTempView("customersDFsql1")
customersDFsql2=spark.sql("""select * ,avg(amount_spent) over(partition by category ORDER BY CAST(price AS int) RANGE BETWEEN 12 PRECEDING AND 12 FOLLOWING ) mov_avgg from customersDFsql1""" )
#NOTE:-Here consider that rows only for avg whose price column is +12 and -12 range

window_01 = Window.partitionBy("category").orderBy("date", "customer_name").rowsBetween(-sys.maxsize, 0)
window_02 =Window.partitionBy('category').orderBy("price").rangeBetween(-12,+12)
#function to calculate number of seconds from number of days
days = lambda i: i * 86400
window_03 =Window.partitionBy('category').orderBy("new_date").rangeBetween(-days(7), 0)


customersDF2 = customersDFsql1.withColumn("cumulative_sum", F.sum(customersDFsql1['amount_spent']).over(window_01)).withColumn("previous_price", F.lag(customersDFsql1['price']).over(Window.partitionBy("category").orderBy("date", "category"))).withColumn("SpentDiffFromMax", F.avg(customersDFsql1['price']).over(window_02)-customersDFsql1['price'])
#customersDF2.filter(F.col("category")=="Y").show()

# Create a new column called datetime,month,year,week and drop the date column
# import org.apache.spark.sql.functions._
# val temp = df.withColumn("modified", from_unixtime(unix_timestamp(col("modified"), "MM/ddyy"), "yyyy-MM-dd")).withColumn("created", to_utc_timestamp(unix_timestamp(col("created"), "MM/dd/yy HH:mm").cast(TimestampType), "UTC"))
#customersDF3 = customersDF.withColumn('datetime', F.col("date").cast("timestamp"))
#dd-MMM-yyyy=01-APR-2015
#NOTE:-above will only work if date format has dash "-" but use below code for any format
customersDF3 = customersDF2.withColumn('new_date', F.date_format(customersDF['date'], 'yyyy-MM-dd').cast("timestamp").cast("long")).withColumn('max', F.max("amount_spent").over(window_03))
/or/
customersDF2.createOrReplaceTempView("customersDF2")
customersDF.createOrReplaceTempView("customersDF")
# spark.sparkContext.setCheckpointDir('C:\hadoop\checkpoint')
# spark.conf.set("spark.cleaner.referenceTracking.cleanCheckpoints", "true")

#We have to use every subquery to avoid any column name not resoved/analysisException(as it is thrown becuase spark is lazy evaluted and start from starting..even caching cannot help as it is not an action) while using funtioned coulmn again in same query.

customersDF3=spark.sql("""select *,max(amount_spent) OVER(PARTITION BY cDF2.category ORDER BY CAST(newdate as timestamp) RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND CURRENT ROW) AS movingmax from (select *,to_date(cast(unix_timestamp(date,'yyyy-MM-dd')as timestamp)) as newdate from customersDF2) cDF2""")


"""
current_date()
current_timestamp()
date_trunc("hour", date_col )--lets you to truncate the datetime into the first day, month, year, hour, minute, second, week, month and even quarter

trunc(date_col, "hour")
dayofweek(date_col)
dayofmonth(date_col)
date_add(date,5)
add_months(date, 2)
datediff(date_col1,date_col2)
months_between(date_col1,date_col2)
unix_timestamp(date_col1,"yyyy-MM-dd")--specify the input format of data provided to get unixepoch

from_unixtime(date_col)--takes a number of seconds from unix epoch as parameter and converts it into a timestamp

from_utc_timestamp(date_col, "Europe/Paris")--change into Europe/Paris
to_utc_timestamp(date_col", "Europe/Paris")--change from Europe/Paris to UTC

to_date(cast(unix_timestamp(date,"yyyy-MM-dd") as timestamp)--to get general date format after casting it to timestamp
date_format(to_date(cast(unix_timestamp(date,"yyyy-MM-dd") as timestamp)), "yyyy/MM/dd")--to get yyyy/MM/dd format

last_day(date)---last day of that month
minute(date)
month(date)

Examples-
customersDF3=spark.sql("""select *, to_date(unix_timestamp(date, "yyyy-MM-dd").cast("timestamp")) as dt1 from customersDF""" )
customersDF3=spark.sql("""select *, to_date(cast(unix_timestamp(date,"yyyy-MM-dd") as timestamp)) as dt1 from customersDF""" )
customersDF3=spark.sql("""select *, date_add(from_unixtime(unix_timestamp(date,"yyyy-MM-dd")),5) as dt1 from customersDF""" )
customersDF3=spark.sql("""select *, date_format(to_date(cast(unix_timestamp(date,"yyyy-MM-dd") as timestamp)), "yyyy/MM/dd") as dt1 from customersDF""" )
customersDF3=spark.sql("""select unix_timestamp() as dt1""" )

"""
customersDF3=spark.sql("""select *,case when cDF.dt1=1 then "sunday" when cDF.dt1=2 then "monday" when cDF.dt1=3 then "tuesday" when cDF.dt1=4 then "wednesday" when cDF.dt1=5 then "thursday" when cDF.dt1=6 then "friday" else "saturday" end as nameofDay from(select *, dayofweek(to_date(cast(unix_timestamp(date,"yyyy-MM-dd") as timestamp))) as dt1 from customersDF) cDF""" )

customersDF3=spark.sql("""select count(cDF.customer_name),date_format(cDF.dt1, 'E') as dayname,date_format(cDF.dt1, 'u') as daynumber from(select *, to_date(cast(unix_timestamp(date,"yyyy-MM-dd") as timestamp)) as dt1 from customersDF) cDF group by dayname,daynumber order by dayname""" )
#NOTE:-groupy must contain every column mentioned in select except aggregate
#Syntax for case
customersDF3.show()


SELECT *,CASE when col1="x" and col2=2 then "z"
              when col1="a" or col2=2 then col3
              when col1="b" and col2=2 then
                                    case when col1="x" and col2=2 then "z" else "k" end
              end as col9 from emp




# customersDF4 = customersDF3.withColumn('day', F.dayofweek("new_date"))
# customersDF4 = customersDF3.withColumn('year', F.year( customersDF3['datetime'] )) \
# .withColumn('month', F.month( customersDF3['datetime'] )) \
# .withColumn('week', F.weekofyear( customersDF3['datetime']))
# over(partition by url, service order by ts range between interval 5 minutes preceding and current row)
# ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
#for salary type column use range
# RANGE BETWEEN 2000 PRECEDING AND 1000 FOLLOWING


###Union is dataframe
df1 = spark.createDataFrame([[1,1],[2,2]],['a','b'])
# different column order will give bad result.
df2 = spark.createDataFrame([[3,333],[4,444]],['b','a'])
df3 = spark.createDataFrame([[555,5],[666,6]],['b','a'])
df1.union(df2).union(df3).orderBy(F.desc("a")).show()
NOTE:dataframe = spark.read.csv([path1, path2, path3])

###General join in DataFrame
df1 = spark.createDataFrame([(2,1,1,3) ,(3,5,1,4),(4,1,2,1),(5,2,1,8)], ("id: int, a : Int, b : Int,k : Int"))
df2 = spark.createDataFrame([(2,'fs','a') ,(5,'fa','f')], ("id","c","d"))
df1.join(df2,df1.id == df2.id ,how="outer").select(["a","b",df2["*"],F.col("k")]).show()
df1.join(df2, df1.id == df2.id).select([c for c in df1.columns if c not in ['b','id']]).show()
# you can also use .drop(df1.id)
# customersDF.crossJoin(ordersDF).show()
The left_anti does the exact opposite of left_semi as it filters out all entries from the left table which donot have a corresponding entry in the right table
customersDF.join(ordersDF, customersDF.name == ordersDF.customer, "left_anti").show()



'''
conf = SparkConf().setAll([('spark.executor.memory', '3g'),
                           ('spark.executor.cores', '8'),
                           ('spark.cores.max', '24'),
                           ('spark.driver.memory', '9g'),
                           ("spark.app.name", "simpleApplicationTests"),
                           ("spark.master", "spark://10.0.2.12:7077")])
sc = SparkContext.getOrCreate(conf=conf);
spark.conf.set("spark.sql.shuffle.partitions", 6)
spark.conf.set("spark.executor.memory", "2g")
spark.conf.getAll()
conf.get("spark.app.name")
conf.get("spark.home")
sorted(conf.getAll(), key=lambda p: p[0])
SparkContext.sc._conf
sc.getConf.getAll.foreach(println)

for item in sorted(sc._conf.getAll()): print(item)
conf = spark.sparkContext._conf.setAll([('spark.executor.memory', '4g'), ('spark.app.name', 'Spark Updated Conf'), ('spark.executor.cores', '4'), ('spark.cores.max', '4'), ('spark.driver.memory','4g')])
##Hadoop configuration
hadoopConf = {}
iterator = sc._jsc.hadoopConfiguration().iterator()
while iterator.hasNext():
    prop = iterator.next()
    hadoopConf[prop.getKey()] = prop.getValue()
for item in sorted(hadoopConf.items()): print(item)

###Environment variables:
import os
for item in sorted(os.environ.items()): print(item)

###Find spark instance
spark.sparkContext.uiWebUrl
http://D1ML17767.mindtree.com:4041
sc.uiWebUrl
http://D1ML17767.mindtree.com:4040
'''


###Reading file in spark
dfemp = spark.read.format("csv") \
  .option("header", "true") \
  .option("sep", "|") \
  .load(r"C:\Users\M1053110\mycodings\emp.csv").show()
  .option("inferSchema", "true") \
  .option("ignoreLeadingWhiteSpace","True") \
  .option("ignoreTrailingWhiteSpace","True") \
dfemp.select('tdate','name').fillna(-1).dropDuplicates().show()
dfemp.createOrReplaceTempView("inv")
dfemp_new=spark.sql("""select dept, sum(salary) OVER(PARTITION BY dept) sum_sal from inv""")
dfemp_new.show()
# dropna()
# display(dfemp)
# df.describe().show()
dfemp_new.printSchema()


###RDD operations in sparksession
###Monkey patch
from pyspark.sql.dataframe import DataFrame
def dfmap(self, f):
    return f(self)
DataFrame.dfmap = dfmap


"""
#val df = Seq("funny","person").toDF("something")
Above is scala code

df=sc.parallelize(["1","2"],["pk","nk"]).toDF(["ID","Name"])
weirdDf = df.dfmap(withGreeting).dfmap(withFarewell)
#/or/ df.select("something").dfmap(withGreeting).dfmap(withFarewell)
weirdDf.show()
#for above RDD(spark<2.0) codes you need:-
from pyspark.sql import SQLContext
from pyspark import sql
sqlContext = sql.SQLContext(sc)
"""

data1 = {'PassengerId': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
         'Name': {0: 'Owen', 1: 'Florence', 2: 'Laina', 3: 'Lily', 4: 'William'},
         'Sex': {0: 'male', 1: 'female', 2: 'female', 3: 'female', 4: 'male'},
         'Survived': {0: 0, 1: 1, 2: 1, 3: 1, 4: 0}}
df1_pd = pd.DataFrame(data1, columns=data1.keys())
df1 = spark.createDataFrame(df1_pd)
df1.select(['PassengerId', 'Name']).sort('PassengerId', ascending=False).show()
df1.filter(df1.Sex == 'female').show()
def plus1_udf(x):
    return x + 1
"""
#to get output in integer only you need to import
from pyspark.sql.types import IntegerType
plus1_udf_int = udf(lambda z: plus1_udf(z), IntegerType())
"""
plus1 = spark.udf.register("plus1", plus1_udf)
new_df = df1.select(plus1("PassengerId"),"Name").show()



###Regular expression in spark
s="My uname is \Praveen-Kumar_ojha89, age 23."
print(regexp_replace(s, '[^A-Za-z0-9 ]', ''))

allrecords_final = spark.read.format("csv") \
  .option("inferSchema", "true") \
  .option("header", "true") \
  .option("sep", ",") \
  .option("multiLine","true") \
  .option("ignoreLeadingWhiteSpace","True") \
  .option("ignoreTrailingWhiteSpace","True") \
  .option("encoding","UTF-8")\
  .load(r"C:\Users\M1053110\mycodings\TSP.csv")


import re
Names=["TIGERHILL CAPITAL","TIGER HILL CAPTIAL","TIGERHILL CAPITAL INC.","TIGER HILL CAPTIAL INC."]
Names = [re.sub(r'[^A-Z0-9]','',s) for s in Names]
NamesRegex = '|'.join(Names)
NamesRegex

allrecords_final_1=allrecords_final.withColumn('compressedBN',regexp_replace(col('BusinessName'), '[^A-Z0-9]', '')) \
.withColumn('Party', \
when(regexp_replace(col('BusinessName'), '[^A-Z0-9]', '').rlike(NamesRegex), lit('x')))
allrecords_final_1.show()


#PartyNames=["My uname is \Praveen-Kumar_ojha89, age 23.","My uname is \Praveen-Kumar_ojha99, age 24."]
#print(re.escape(s))


allrecords_final_1=allrecords_final.withColumn('Party', \
#when(col('sample').contains('www'), lit('x')))
#when(col('sample').rlike(".*" + "www" + ".*"), lit('x')))
#when(col('sample').rlike("^(http:|https:)" +".*"+ "www" +".*"+ "com$"), lit('x')))
#when(col('sample').like("www"+"%tc_l%"), lit('x')))
#when(col('sample').rlike('(http:|https:)?(/{1,2})?(www.)+(\w+)(\.\w+)'), lit('x')))
#when(col('sample').rlike("^(http:|https:)(/{1,2})?(www.)+(\w+)(\.\w+)"), lit('x')))
when(col('sample').rlike('\d+$'), lit('x')))
allrecords_final_1.show()
spark.sql("select 'xyz123' rlike '\\\\d+$'").show()

2

SparkContext is used for basic RDD API on both Spark1.x and Spark2.x

SparkSession is used for DataFrame API and Struct Streaming API on Spark2.x

SQLContext & HiveContext are used for DataFrame API on Spark1.x and deprecated from Spark2.x
