package com.recommender.scala

import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

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







}
