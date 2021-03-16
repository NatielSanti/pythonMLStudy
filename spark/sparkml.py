import os

from pyspark.python.pyspark.shell import spark
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, IntegerType

if __name__ == '__main__':
    # Создали схему
    schema = StructType([
        StructField('x', IntegerType(), True),
        StructField('y', IntegerType(), True),
        StructField('z', IntegerType(), True)])
    # Считали файлы из системы
    file_list = os.listdir('../resources/HMP_Dataset/Brush_teeth')
    # Отфильтровали только нужные нам файлы из папки
    # file_list_filtered = [s for s in file_list if '_' in s]
    # Создаёт пустой общий датафрейм
    df = None
    # Проходим по категориям (папкам)
    data_files = os.listdir('../resources/HMP_Dataset/Brush_teeth')
    # Проходим по файлам в папках
    for data_file in data_files:
        print(data_file)
        # Создаём временный датафрейм из файлов в папках
        temp_df = spark.read.option("header", "false").option("delimeter", " ") \
            .csv('../resources/HMP_Dataset/Brush_teeth/' + data_file, schema=schema)
        temp_df = temp_df.withColumn("class", lit('Brush_teeth'))
        temp_df = temp_df.withColumn("source", lit(data_file))
        # Добавляем временный датафрейм в общий датафрейм
        if df is None:
            df = temp_df
        else:
            df = df.union(temp_df)

    df.show(100)
