import org.nd4s.Implicits._
import org.scalatest._


class TestNaiveBayesClassifier extends FlatSpec {

  "NDArray for all" should "print correctly" in {
    val ndArray =
      Array(
        Array(1, 2, 3),
        Array(4, 5, 6),
        Array(7, 8, 9)
      ).toNDArray

    ndArray.forall(x => {
      print(x)
      x > 3
    }
    )
  }

}
