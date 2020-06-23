import re
filenameClean = re.sub(r'[^A-Za-z0-9 /\-\.]', '', to_list[1])
filenameClean = re.sub(r'\.+', r'.', filenameClean)

##Monkey Patch is necessary to use dataframe as input in some function##
##Monkey Patch DataFrame base class with a transformer to accept Spark Functions##
from pyspark.sql.dataframe import DataFrame
def transform(self, f):
    return f(self)
DataFrame.transform = transform

def withGreeting(df: DataFrame): DataFrame = {
  df.withColumn(
"greeting", lit("hello world"))
}
def withFarewell(df: DataFrame): DataFrame = {
  df.withColumn(
"farewell", lit("goodbye"))
}


##Scala
val df = Seq("funny","person").toDF("something")
val weirdDf = df.transform(withGreeting).transform(withFarewell)
weirdDf.show()
+---------+-----------+--------+
|something|   greeting|farewell|
+---------+-----------+--------+
|    funny|hello world| goodbye|
|   person|hello world| goodbye|
+---------+-----------+--------+

#/or/python
df.select("name").transform(withGreeting).transform(withFarewell)


"""
###confidential bashrc
# .bashrc
# Source global definitions
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!

__conda_setup="$('/data2/myuser2/miniconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/data2/myuser2/miniconda/etc/profile.d/conda.sh" ]; then
        . "/data2/myuser2/miniconda/etc/profile.d/conda.sh"
    else
        export PATH="/data2/myuser2/miniconda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
export JAVA_HOME="/usr/java/jdk1.8.0_121"
export HADOOP_HOME="/opt/cloudera/parcels/CDH-5.15.0-1.cdh5.15.0.p0.21/lib/hadoop"

export HADOOP_CONF_DIR="/etc/spark2/conf/yarn-conf"

export SPARK_HOME="/opt/cloudera/parcels/SPARK2/lib/spark2"

export PYTHONPATH="/opt/cloudera/parcels/SPARK2-2.2.0.cloudera1-1.cdh5.12.0.p2832.251268/lib/spark2/python/lib/py4j-0.10.4-
src.zip:/opt/cloudera/parcels/SPARK2-2.2.0.cloudera1-1.cdh5.12.0.p2832.251268/lib/spark2/python/"

export PYSPARK_PYTHON="/opt/cloudera/parcels/Anaconda-4.2.0/bin/python"

# .bash_profile
# Get the aliases and functions
if [ -f ~/.bashrc ]; then
        . ~/.bashrc
fi

# User specific environment and startup programs

PATH=$PATH:$HOME/.local/bin:$HOME/bin

export PATH
"""

from pyspark.sql.window import Window
aggs = extended.groupBy("group_id", "date", "hour").count()
aggs.withColumn(
    "agg_count", 
    sum("count").over(Window.partitionBy("group_id", "date").orderBy("hour")))


from pyspark.sql import functions as F
from pyspark.sql.functions import col, count, hour, sum
extended = (df
  .withColumn("event_time", col("event_time").cast("timestamp"))
  .withColumn("date", col("event_time").cast("date"))
  .withColumn("hour", hour(col("event_time"))))

df.select(F.date_format('timestamp','yyyy-MM-dd').alias('day')).groupby('day').count().show()

df = sc.parallelize([
    ("XXXX", "2017-10-25 01:47:02.717013"),
    ("XXXX", "2017-10-25 14:47:25.444979"),
    ("XXXX", "2017-10-25 14:49:32.21353"),
    ("YYYY", "2017-10-25 14:50:38.321134"),
    ("YYYY", "2017-10-25 14:51:12.028447"),
    ("ZZZZ", "2017-10-25 14:51:24.810688"),
    ("YYYY", "2017-10-25 14:37:34.241097"),
    ("ZZZZ", "2017-10-25 14:37:24.427836"),
    ("XXXX", "2017-10-25 22:37:24.620864"),
    ("YYYY", "2017-10-25 16:37:24.964614")
]).toDF(["group_id", "event_time"])

the result is

+--------+----------+----+-----+---------+
|group_id|      date|hour|count|agg_count|
+--------+----------+----+-----+---------+
|    XXXX|2017-10-25|   1|    1|        1|
|    XXXX|2017-10-25|  14|    2|        3|
|    XXXX|2017-10-25|  22|    1|        4|
|    ZZZZ|2017-10-25|  14|    2|        2|
|    YYYY|2017-10-25|  14|    3|        3|
|    YYYY|2017-10-25|  16|    1|        4|
+--------+----------+----+-----+---------+


def get_weekday(date):
    import datetime
    import calendar
    month, day, year = (int(x) for x in date.split('/'))
    weekday = datetime.date(year, month, day)
    return calendar.day_name[weekday.weekday()]

spark.udf.register('get_weekday', get_weekday)
df.createOrReplaceTempView("weekdays")
df = spark.sql("select DateTime, PlayersCount, get_weekday(Date) as Weekday from weekdays")


funcWeekDay =  udf(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%w'))
from pyspark.sql.functions import udf,col
    from datetime import datetime
df=df.withColumn('shortdate',col('date').substr(1, 10))\
     .withColumn('weekDay', funcWeekDay(col('shortdate')))\
     .drop('shortdate')

df.toPandas()
    id  date                     weekDay
0   1   2017-11-01 22:05:01 -0400   3
1   2   2017-11-02 03:15:16 -0500   4
2   3   2017-11-03 19:32:24 -0600   5
3   4   2017-11-04 07:47:44 -0700   6


SELECT Salary,EmpName
FROM
(
SELECT Salary,EmpName,DENSE_RANK() OVER(ORDER BY Salary DESC) Rno from EMPLOYEE
) tbl
WHERE Rno=3

select min(CustomerID) from (SELECT distinct CustomerID FROM Customers order by CustomerID desc LIMIT 4) as A;

select t.deptno, max(t.salary) as maxs from table t where
t.salary not in (select max(salary) from table t2 where t2.deptno = t.deptno) group by t.deptno;
---------------------------
###Creating windows
###Example using sql:-
spark.sql(
    """SELECT *, mean(some_value) OVER (
        PARTITION BY id
        ORDER BY CAST(start AS timestamp)
        RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND CURRENT ROW
     ) AS mean FROM df""").show()

## +---+----------+----------+------------------+
## | id|     start|some_value|              mean|
## +---+----------+----------+------------------+
## |  1|2015-01-01|      20.0|              20.0|
## |  1|2015-01-06|      10.0|              15.0|
## |  1|2015-01-07|      25.0|18.333333333333332|
## |  1|2015-01-12|      30.0|21.666666666666668|
## |  2|2015-01-01|       5.0|               5.0|
## |  2|2015-01-03|      30.0|              17.5|
## |  2|2015-02-01|      20.0|              20.0|
## +---+----------+----------+------------------+

###Example without using sql:-
w = (Window()
   .partitionBy(col("id"))
   .orderBy(col("start").cast("timestamp").cast("long"))
   .rangeBetween(-days(7), 0))

df.select(col("*"), mean("some_value").over(w).alias("mean")).show()

## +---+----------+----------+------------------+
## | id|     start|some_value|              mean|
## +---+----------+----------+------------------+
## |  1|2015-01-01|      20.0|              20.0|
## |  1|2015-01-06|      10.0|              15.0|
## |  1|2015-01-07|      25.0|18.333333333333332|
## |  1|2015-01-12|      30.0|21.666666666666668|
## |  2|2015-01-01|       5.0|               5.0|
## |  2|2015-01-03|      30.0|              17.5|
## |  2|2015-02-01|      20.0|              20.0|
## +---+----------+----------+------------------+


#Executing an SQL query over a pandas dataset
import pandas as pd
import pandasql as ps

df = pd.DataFrame([[1234, 'Customer A', '123 Street', np.nan],
               [1234, 'Customer A', np.nan, '333 Street'],
               [1233, 'Customer B', '444 Street', '333 Street'],
              [1233, 'Customer B', '444 Street', '666 Street']], columns=
['ID', 'Customer', 'Billing Address', 'Shipping Address'])

q1 = """SELECT ID FROM df """

print(ps.sqldf(q1, locals()))

     ID
0  1234
1  1234
2  1233
3  1233

#Querying from Microsoft SQL to a Pandas Dataframe
import pandas as pd, pyodbc
con_string = 'DRIVER={SQL Server};SERVER='+ <server> +';DATABASE=' + <database>
cnxn = pyodbc.connect(con_string)
query = """
  SELECT <field1>, <field2>, <field3>
  FROM result
"""
result_port_map = pd.read_sql(query, cnxn)
result_port_map.columns.tolist()
