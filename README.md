# Recommendation Engine - ML Component

## Description
This repository contains the machine learning component of the recommender engine. The component is written using Apache Spark.

The code in this repository has a simple structure. It does the following things:
* Loads the user reviews from a directory which Flume saves the reviews. The path of the directory is taken from arguments.
* Trains a learning model. (We use ALS algorithm to do that.)
* Saves the learning model to a path, which is specified in the arguments.
* Calculates Root Mean Square Error of the model and writes to the output.

## Usage
### Run
This component cannot be run by itself. In order to run this component, all of the project components must be run using `docker-compose`. See [Recommender Engine - Docker Files](https://github.com/trendyol-data-eng-summer-intern-2019/recom-engine-docker)

If you want to change the code and run, after you change the code do the following steps:

* Go to the root directory of the repository.
* Run `mvn clean package`
* Get the recom-engine-docker repository from [here](https://github.com/trendyol-data-eng-summer-intern-2019/recom-engine-docker).
* Move the jar file that is created from `target/recom-engine-ml-1.0-SNAPSHOT.jar` to `images/spark/master/target` under recom-engine-docker's root directory.
* Go to the root diretory of recom-engine-docker and run the command `docker-compose up`.

### Watch the Outputs
There is just one output of this component, the RMSE of the learning model. If you want to see the output, do the following steps:

* First run all of the components with `docker-compose`.
* Go to the worker's web page from [http://localhost:8082](http://localhost:8082).
* In this page, you can see the running drivers, click [stdout]() link of the ml application, then you can watch the outputs of the application.

## Members
- [Oğuzhan Bölükbaş](https://github.com/oguzhan-bolukbas)
- [Sercan Ersoy](https://github.com/sercanersoy)
- [Yasin Uygun](https://github.com/yasinuygun)
