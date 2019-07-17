package com.trendyol.recomengine.ml;

import org.apache.commons.io.FileUtils;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;

/**
 * Produces a machine learning model that recommends products to users.
 */
public class Main {

    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.println("Expected arguments: <review-path> <model-path>");
            System.exit(0);
        }
        
        String reviewPath = args[0];
        String modelPath = args[1];

        SparkSession spark = SparkSession
                .builder()
                .appName("recom-engine-ml")
                .getOrCreate();

        Logger.getRootLogger().setLevel(Level.ERROR);

        Dataset<Row> reviews = loadReviews(spark, reviewPath);

        Dataset<Row>[] splits = reviews.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];

        ALS als = new ALS().setMaxIter(5).setRegParam(0.1).setUserCol("userId")
                .setItemCol("productId").setRatingCol("score");

        ALSModel model = als.fit(training);
        model.setColdStartStrategy("drop");
        try {
            FileUtils.deleteDirectory(new File(modelPath));
            model.save(modelPath);
        } catch (IOException e) {
            e.printStackTrace();
        }

        double rmse = getRMSE(test, model);
        System.out.println("Root mean square error = " + rmse);

        spark.stop();
    }

    /**
     * Loads the reviews from the specified path into a Dataframe.
     *
     * @param sparkSession  SparkSession object.
     * @param directoryPath The path of the directory which has stored review data.
     * @return A Dataframe that contains user reviews.
     */
    private static Dataset<Row> loadReviews(SparkSession sparkSession, String directoryPath) {
        JavaRDD<Review> reviewsRDD = sparkSession
                .read()
                .textFile(directoryPath)
                .javaRDD().map(Review::parseReview);
        Dataset<Row> reviews = sparkSession.createDataFrame(reviewsRDD, Review.class);
        return reviews;
    }

    /**
     * Calculates the root mean square error of the model that is produced.
     *
     * @param reviewData The data that has score value in it.
     * @param alsModel   The model that is to be evaluated.
     * @return Root mean square error of the model.
     */
    private static double getRMSE(Dataset<Row> reviewData, ALSModel alsModel) {
        Dataset<Row> predictions = alsModel.transform(reviewData);
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setMetricName("rmse")
                .setLabelCol("score")
                .setPredictionCol("prediction");
        return evaluator.evaluate(predictions);
    }

    /**
     * Stores the review data.
     */
    public static class Review implements Serializable {
        private int userId;
        private int productId;
        private float score;
        private long timestamp;

        /**
         * Initializes the member fields.
         *
         * @param userId    is the user's id whose reviews.
         * @param productId is the product's id whose is reviewed.
         * @param score     is the score that the user give to the product. It's an integer between 1 and 5.
         * @param timestamp is a UNIX timestamp that indicates when the product is reviewed.
         */
        Review(int userId, int productId, float score, long timestamp) {
            this.userId = userId;
            this.productId = productId;
            this.score = score;
            this.timestamp = timestamp;
        }

        /**
         * Parses the review data and returns a Review object.
         *
         * @param str The review data that is separated by "::" with this format: userId::productId::score::timestamp
         * @return Parsed review.
         */
        static Review parseReview(String str) {
            String[] fields = str.split(",");
            if (fields.length != 4) {
                throw new IllegalArgumentException("Each line must contain 4 fields");
            }
            int userId = Integer.parseInt(fields[0]);
            int productId = Integer.parseInt(fields[1]);
            float score = Float.parseFloat(fields[2]);
            long timestamp = Long.parseLong(fields[3]);
            return new Review(userId, productId, score, timestamp);
        }

        public int getUserId() {
            return userId;
        }

        public int getProductId() {
            return productId;
        }

        public float getScore() {
            return score;
        }

        public long getTimestamp() {
            return timestamp;
        }
    }
}
