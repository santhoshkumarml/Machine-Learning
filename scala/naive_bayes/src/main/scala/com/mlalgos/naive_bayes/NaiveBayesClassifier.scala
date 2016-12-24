package com.mlalgos.naive_bayes

import com.mlalgos.common.SupervisedModel
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

final class NaiveBayesClassifier(private var prior: INDArray, private var cpt : INDArray) extends SupervisedModel{
  /**
    * Train the model
    *
    * @param train_data training data
    * @param train_result training result
    */
  override def fit(train_data: INDArray, train_result: INDArray): Unit = {
    val (train_samples, no_of_features) = (train_data.rows(), train_data.columns())
    val no_of_classes = train_result.rows()

    prior = Nd4j.zeros(no_of_classes)
    cpt = Nd4j.ones(no_of_features, no_of_classes) //Add one smoothing
    //train_data.forallRC()

  }

  /**
    * Predict result for the dataSet
    *
    * @param iNDArray feature vector
    */
  override def predict(iNDArray: INDArray): Any = {
  }
}