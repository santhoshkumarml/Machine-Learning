import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import scala.collection.immutable.List
import scala.io.Source

final object DataUtils {
  def read_train_data(fileName: String): List[Array[Int]] = {
    Source.fromFile(fileName).getLines().map(_.stripLineEnd.split(" ", -1).map(_.toInt)).toList
  }

  def read_test_data(fileName: String): List[Array[Int]] = {
    Source.fromFile(fileName).getLines().map(_.stripLineEnd.split(" ", -1).map(_.toInt)).toList
  }
}