package com.recommender.scala

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.ml.recommendation.{ALS, ALSModel}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class Recommender(private val spark :SparkSession) {

  import spark.implicits._

  def preparation(rawUserArtistData: Dataset[String],
                  rawArtistData: Dataset[String],
                  rawArtistAlias: Dataset[String]): Unit = {
    //rawUserArtistData.take(5).foreach(println)

    val userArtistDf = rawUserArtistData.map { line =>
      val Array(user, artist, _*) = line.split(' ')
      (user.toInt, artist.toInt)
    }.toDF("user", "artist")

    userArtistDf.agg(min("user"), max("user"), min("artist"), max("artist")).show()


    val artistById = artistByID(rawArtistData)
    val artistAliasDf = artistAliasDF(rawArtistAlias)

    val (badID, goodID) = artistAliasDf.head

    // $"<col>" : reference the names of the columns in the data frame
    //we can use groupBy("<col>")
    artistById.filter($"id" isin(badID, goodID)).show()

  }

  //Building Artist by ID data frame
  def artistByID(rawArtistData: Dataset[String]): DataFrame = {
    rawArtistData.flatMap { line =>
      val (id, name) = line.span(_ != '\t') //span splits the line by it's first tab by consuming charcters that are not tabs
      if (name.isEmpty) {
        None
      } else {
        try {
          Some((id.toInt, name.trim)) // trim function extract the String among Space and Tabualtionss..
        } catch {
          case _: NumberFormatException => None
        }
      }
    }.toDF("id", "name")
  }

  //Mapping "bad" to "good" artist IDs
  def artistAliasDF(rawArtistAlias: Dataset[String]): Map[Int, Int] = {
    rawArtistAlias.flatMap { line =>
      val Array(artist, alias) = line.split('\t')
      if (artist.isEmpty) {
        None
      } else {
        Some((artist.toInt, alias.toInt))
      }
    }.collect().toMap
  }

  //to convert all artist IDs to a canonical ID
  //broadcast variable = makes Spark send and hold in memory just one copy for each executor in the cluster,  rather than shipping a copy of it with tasks
  def buildCounts(rawUserArtistData: Dataset[String], bArtistAlias: Broadcast[Map[Int, Int]]): DataFrame = {
    rawUserArtistData.map { line =>
      val Array(userID, artistID, count) = line.split(' ').map(x => x.toInt)
      val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID) //get Artist alias if it exists, otherwise get original artist
      (userID, finalArtistID, count)
    }.toDF("user", "artist", "count")
  }

  def buildingModel(rawUserArtistData: Dataset[String],
                    rawArtistData: Dataset[String],
                    rawArtistAlias: Dataset[String]): Unit = {
    val bArtistAlias = spark.sparkContext.broadcast(artistAliasDF(rawArtistAlias))
    //suggests to Spark that this DataFrame should be temporarily stored after being computed and kept in memory in the cluster.
    val trainData = buildCounts(rawUserArtistData, bArtistAlias).cache()

    val model = new ALS().
      setSeed(Random.nextLong()) //Use random seed
      .setImplicitPrefs(true)
      .setRank(10)
      .setRegParam(0.01) //standard overfitting parameter, values that are too high hurts factorization accuracy
      .setAlpha(1.0)
      .setMaxIter(5) //The number of iterations that the factorization runs.
      .setUserCol("user")
      .setItemCol("artist")
      .setRatingCol("count")
      .setPredictionCol("prediction")
      .fit(trainData)

    model.userFactors.select("features").show(1, truncate = false)
    //for now let's, extract the IDs of artists that this user has listened to and print their names.
    val userID = 2093760
    val existingArtistsIDs = trainData.
      filter($"user" === userID) //Find lines whose user is 2093760
      .select("artist").as[Int].collect() //Collect data set of Int artist ID.

    val artistbyID = artistByID(rawArtistData)

    artistbyID.filter($"id" isin (existingArtistsIDs: _*)).show() //Filter in those artists; note :_* varargs syntax (* Repeated Parameters)

    val topRecommendations = makeRecommendations(model, userID, 5) //Extracting the artist IDs for the recommendations,
    topRecommendations.show()

    val recommendedArtistIDs = topRecommendations.select("artist").as[Int].collect() //Look up artist names.

    artistbyID.filter($"id" isin (recommendedArtistIDs: _*)).show()

    model.userFactors.unpersist()
    model.itemFactors.unpersist()

  }

  def makeRecommendations(model: ALSModel, userID: Int, howMany: Int): DataFrame = {
    val toRecommend = model.itemFactors.
      select($"id".as("artist")). //select all artist IDs
      withColumn("user", lit(userID)) //Pair with target user ID
    model.transform(toRecommend).
      select("artist", "prediction")
      .orderBy($"prediction".desc)
      .limit(howMany) // top scored artist based on the ALS predictions
  }

  //AUC(Area Under Curve)  will be used here as a common
  // and broad measure of the quality of the entire model output.
  //let's use the function provided by Spark MMLib.
  def areaUnderCurve(positiveData: DataFrame,
                     bAllArtistIDs: Broadcast[Array[Int]],
                     predictFunction: (DataFrame => DataFrame)): Double = {

    // What this actually computes is AUC, per user. The result is actually something
    // that might be called "mean AUC".

    // Take held-out data as the "positive".
    // Make predictions for each of them, including a numeric score
    val positivePredictions = predictFunction(positiveData.select("user", "artist")).
      withColumnRenamed("prediction", "positivePrediction")

    // BinaryClassificationMetrics.areaUnderROC is not used here since there are really lots of
    // small AUC problems, and it would be inefficient, when a direct computation is available.

    // Create a set of "negative" products for each user. These are randomly chosen
    // from among all of the other artists, excluding those that are "positive" for the user.
    val negativeData = positiveData.select("user", "artist").as[(Int, Int)].
      groupByKey { case (user, _) => user }
      .flatMapGroups { case (userID, userIDAndPosArtistIDs) =>
        val random = new Random()
        val posItemIDSet = userIDAndPosArtistIDs.map { case (_, artist) => artist }.toSet
        val negative = new ArrayBuffer[Int]()
        val allArtistIDs = bAllArtistIDs.value
        var i = 0
        // Make at most one pass over all artists to avoid an infinite loop.
        // Also stop when number of negative equals positive set size
        while (i < allArtistIDs.length && negative.size < posItemIDSet.size) {
          val artistID = allArtistIDs(random.nextInt(allArtistIDs.length)) //return an artistID randomly where the index(random.nextInt(allArtistIDs.length)) is
          // a random nember between 0 and allArtistIDs.length

          // Only add new distinct IDs
          if (!posItemIDSet.contains(artistID)) {
            negative += artistID
          }
          i += 1
        }
        // Return the set with user ID added back
        negative.map(artistID => (userID, artistID))
      }.toDF("user", "artist")

    // Make predictions on the rest:
    val negativePredictions = predictFunction(negativeData).
      withColumnRenamed("prediction", "negativePrediction")

    // Join positive predictions to negative predictions by user, only.
    // This will result in a row for every possible pairing of positive and negative
    // predictions within each user.
    val joinedPredictions = positivePredictions.join(negativePredictions, "user").
      select("user", "positivePrediction", "negativePrediction").cache()

    // Count the number of pairs per user
    val allCounts = joinedPredictions.
      groupBy("user").agg(count(lit("1")).as("total")).
      select("user", "total")
    // Count the number of correctly ordered pairs per user
    val correctCounts = joinedPredictions.
      filter($"positivePrediction" > $"negativePrediction").
      groupBy("user").agg(count("user").as("correct")).
      select("user", "correct")

    // Combine these, compute their ratio, and average over all users
    val meanAUC = allCounts.join(correctCounts, Seq("user"), "left_outer").
      select($"user", (coalesce($"correct", lit(0)) / $"total").as("auc")).
      agg(mean("auc")).
      as[Double].first()

    joinedPredictions.unpersist()

    meanAUC
  }

  def evaluation(rawUserArtistData: Dataset[String],
                 rawArtistAlias: Dataset[String]): Unit = {

    val bArtistAlias = spark.sparkContext.broadcast(artistAliasDF(rawArtistAlias))
    val allData = buildCounts(rawUserArtistData, bArtistAlias)

    val Array(trainData, testData) = allData.randomSplit(Array(0.9, 0.1))
    trainData.cache()
    testData.cache()

    val allArtistIDs = allData.select("artist").as[Int].distinct().collect()
    val bAllArtistIDs = spark.sparkContext.broadcast(allArtistIDs)

    val mostPlayedArtistToEveryUser = areaUnderCurve(testData, bAllArtistIDs,predictMostPLayedArtist(trainData))
    print(mostPlayedArtistToEveryUser)

    val evaluations =
      for(rank <- Seq(5, 30);
          regParam <- Seq(1.0, 0.0001);
          alpha <- Seq(1.0, 40.0))
        yield{
          val model = new ALS()
            .setSeed(Random.nextLong())
            .setImplicitPrefs(true)
            .setRank(rank)
            .setRegParam(regParam)
            .setAlpha(alpha)
            .setMaxIter(20)
            .setUserCol("user")
            .setItemCol("artist")
            .setRatingCol("count")
            .setPredictionCol("prediction")
            .fit(trainData)

          val auc = areaUnderCurve(testData, bAllArtistIDs, predictMostPLayedArtist(trainData))

          model.userFactors.unpersist() //Free up model resources immediately.
          model.itemFactors.unpersist()

          (auc, (rank, regParam, alpha))
        }
      evaluations.sorted.reverse.foreach(println) //Sort by first value (AUC), descending, and print.

      trainData.unpersist()
      testData.unpersist()

  }


  def predictMostPLayedArtist(train: DataFrame)(allData: DataFrame) = {
    val listenCounts = train.groupBy("artist")
      .agg(sum("count").as("prediction"))
      .select("artist", "prediction")

    allData.join(listenCounts, Seq("artist"), "left_outer").
      select("user", "artist", "prediction")
  }

  def recommend(rawUserArtistData: Dataset[String],
                rawArtistData: Dataset[String],
                rawArtistAlias: Dataset[String]) : Unit ={
    val bArtistAlias = spark.sparkContext.broadcast(artistAliasDF(rawArtistAlias))
    val allData = buildCounts(rawUserArtistData, bArtistAlias).cache()

    val model = new ALS()
      .setSeed(Random.nextLong())
      .setImplicitPrefs(true)
      .setRank(10)
      .setRegParam(1.0)
      .setAlpha(40.0)
      .setMaxIter(20)
      .setUserCol("user")
      .setItemCol("artist")
      .setRatingCol("count")
      .setPredictionCol("prediction")
      .fit(allData)

    allData.unpersist()

    val userID = 2093760
    val topRecommendation = makeRecommendations(model, userID, 5)

    val recommendedArtistIDs = topRecommendation.select("artist").as[Int].collect()

    val artistById = artistByID(rawArtistData)
    artistById.join(spark.createDataset(recommendedArtistIDs).toDF("id"), "id").select("name").show()

    model.userFactors.unpersist()
    model.itemFactors.unpersist()

  }



}
