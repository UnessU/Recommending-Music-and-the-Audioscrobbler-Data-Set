package com.recommender.scala

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.ml.recommendation.{ALS, ALSModel}

import scala.util.Random

class Recommender(private val spark :SparkSession) {

    import spark.implicits._

  def preparation( rawUserArtistData : Dataset[String],
                     rawArtistData : Dataset[String],
                     rawArtistAlias : Dataset[String]) :Unit ={
      //rawUserArtistData.take(5).foreach(println)

      val userArtistDf = rawUserArtistData.map{ line =>
        val Array(user ,artist, _*) = line.split(' ')
        (user.toInt, artist.toInt)
      }.toDF("user", "artist")

      userArtistDf.agg( min("user"), max("user"), min("artist"), max("artist")).show()


      val artistById = artistByID(rawArtistData)
      val artistAliasDf = artistAliasDF(rawArtistAlias)

      val (badID, goodID) = artistAliasDf.head

      // $"<col>" : reference the names of the columns in the data frame
      //we can use groupBy("<col>")
      artistById.filter($"id" isin(badID, goodID)).show()

    }

  //Building Artist by ID data frame
  def  artistByID(rawArtistData : Dataset[String]): DataFrame = {
    rawArtistData.flatMap{ line =>
      val (id , name ) = line.span(_ !='\t') //span splits the line by it's first tab by consuming charcters that are not tabs
      if(name.isEmpty) {
        None
      } else {
        try {
          Some((id.toInt, name.trim)) // trim function extract the String among Space and Tabualtionss..
        }catch{
          case _: NumberFormatException => None
        }
      }
    }.toDF("id", "name")
  }

  //Mapping "bad" to "good" artist IDs
  def artistAliasDF(rawArtistAlias : Dataset[String]) : Map[Int, Int] = {
    rawArtistAlias.flatMap{ line =>
      val Array(artist, alias) = line.split('\t')
      if(artist.isEmpty){
        None
      } else{
        Some((artist.toInt, alias.toInt))
      }
    }.collect().toMap
  }

  //to convert all artist IDs to a canonical ID
  //broadcast variable = makes Spark send and hold in memory just one copy for each executor in the cluster,  rather than shipping a copy of it with tasks
 def buildCount(rawUserArtistData : Dataset[String], bArtistAlias : Broadcast[Map[Int, Int]]) : DataFrame = {
   rawUserArtistData.map{ line =>
     val Array(userID, artistID, count) = line.split(' ').map( x => x.toInt)
     val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID) //get Artist alias if it exists, otherwise get original artist
     (userID, finalArtistID, count )
   }.toDF("user", "artist", "count")
 }

 def buildingModel(rawUserArtistData : Dataset[String],
                   rawArtistData : Dataset[String],
                   rawArtistAlias : Dataset[String]) : Unit ={
   val bArtistAlias = spark.sparkContext.broadcast(artistAliasDF(rawArtistAlias))
   //suggests to Spark that this DataFrame should be temporarily stored after being computed and kept in memory in the cluster.
   val trainData = buildCount(rawUserArtistData, bArtistAlias).cache()

   val model = new ALS().
     setSeed(Random.nextLong()) //Use random seed
     .setImplicitPrefs(true)
     .setRank(10)
     .setRegParam(0.01)
     .setAlpha(1.0)
     .setMaxIter(5)
     .setUserCol("user")
     .setItemCol("artist")
     .setRatingCol("count")
     .setPredictionCol("prediction")
     .fit(trainData)

 }






}
