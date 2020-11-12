package com.recommender.scala


import org.apache.spark.sql.{Dataset, SparkSession}

object runRecommender {
  def main(args :Array[String]): Unit = {
    val spark  = SparkSession.builder().appName("Recommender").master("local[*]").getOrCreate()
    //spark.sparkContext.setCheckpointDir("hdfs:///tmp/")

    val base = "hdfs://0.0.0.0:19000/Audioscrobber_data/"
    val rawUserArtistData = spark.read.textFile(base + "user_artist_data.txt")
    val rawArtistData = spark.read.textFile(base + "artist_data.txt")
    val rawArtistAlias = spark.read.textFile(base + "artist_alias.txt")
    val Recommender = new Recommender(spark)
    Recommender.preparation(rawUserArtistData, rawArtistData,rawArtistAlias )

}}
