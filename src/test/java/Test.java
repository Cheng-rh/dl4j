import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.FileReader;
import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: sssd
 * Date: 2018/3/29 14:34
 * Version: V1.0
 * To change this template use File | Settings | File Templates.
 * Description:
 */
public class Test {

    public static void main(String[] args) {
        INDArray indArray = Nd4j.create(3, 2 ,5);
        INDArray zeros = Nd4j.zeros(2, 5);
        int[] origin = new int[3];
        System.out.println(indArray);
        System.out.println("---------------------");
        System.out.println(zeros);
        System.out.println("---------------------");
        System.out.println(origin.length);
    }

    @org.junit.Test
    public void test(){
        INDArray feature = Nd4j.create(2, 3, 5);
        int[] origin = new int[3];
        origin[0] = 1;
        origin[1] = 2;
        origin[2] = 2;
        feature.putScalar(origin, 3);
        System.out.println(feature);
    }

    @org.junit.Test
    public void readFile() throws Exception{
        String path = "C:\\Users\\sssd\\Desktop\\LSTM\\sourcedata.txt";
        List<String> lines = FileUtils.readLines(new File(path));
        int i = 1;
        for (String line : lines) {
            String[] split = line.split("\t");
            System.out.println("第" + i + "行" + split[0] + "-------" + split[1]);
            i++;
        }
    }

}
