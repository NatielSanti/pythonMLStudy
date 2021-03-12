from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
    spark = SparkSession.builder.getOrCreate()

    df = spark.read.parquet('../resources/washing.parquet')
    df.createOrReplaceTempView('df')

    df.createOrReplaceTempView("washing")

    spark.sql("SELECT * FROM washing").show()

    result = spark.sql("select voltage,ts from washing where voltage is not null order by ts asc")
    result_rdd = result.rdd.sample(False, 0.1).map(lambda row: (row.ts, row.voltage))
    print(result_rdd.__sizeof__())
    print(result_rdd)
    result_array_ts = result_rdd.map(lambda ts_voltage: ts_voltage[0]).collect()
    result_array_voltage = result_rdd.map(lambda ts_voltage: ts_voltage[1]).collect()
    print(result_array_ts[:15])
    print(result_array_voltage[:15])

    plt.plot(result_array_ts, result_array_voltage)
    plt.xlabel("time")
    plt.ylabel("voltage")
    plt.show()

    result_df = spark.sql("""
    select hardness,temperature,flowrate from washing
        where hardness is not null and 
        temperature is not null and 
        flowrate is not null
    """)
    result_rdd = result_df.rdd.sample(False, 0.1).map(lambda row: (row.hardness, row.temperature, row.flowrate))
    result_array_hardness = result_rdd.map(
        lambda hardness_temperature_flowrate: hardness_temperature_flowrate[0]).collect()
    result_array_temperature = result_rdd.map(
        lambda hardness_temperature_flowrate: hardness_temperature_flowrate[1]).collect()
    result_array_flowrate = result_rdd.map(
        lambda hardness_temperature_flowrate: hardness_temperature_flowrate[2]).collect()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result_array_hardness, result_array_temperature, result_array_flowrate, c='r', marker='o')
    ax.set_xlabel('hardness')
    ax.set_ylabel('temperature')
    ax.set_zlabel('flowrate')
    plt.show()

    plt.hist(result_array_hardness)
    plt.show()