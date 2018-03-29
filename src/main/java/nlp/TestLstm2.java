package nlp;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import scala.Tuple2;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created with IntelliJ IDEA.
 * User: sssd
 * Date: 2018/3/29 13:20
 * Version: V1.0
 * To change this template use File | Settings | File Templates.
 * Description:   用来测试 分区 读取数据的影响
 */
public class TestLstm2 implements Serializable {

    private static JavaSparkContext jsc;

    private static Broadcast<Map<String, Object>> broadcasTokenizerVarMap;

    private static AtomicInteger lineNum = new AtomicInteger(0);

    public TestLstm2() {
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local").setAppName("deeplearning4j-lstm");
        jsc = new JavaSparkContext(sparkConf);
        Map<String, Object> TokenizerVarMap = new HashMap<String, Object>();
        TokenizerVarMap.put("numWords", 1);     //min count of word appearence
        TokenizerVarMap.put("nGrams", 1);       //language model parameter
        TokenizerVarMap.put("tokenizer", DefaultTokenizerFactory.class.getName());  //tokenizer implemention
        TokenizerVarMap.put("tokenPreprocessor", CommonPreprocessor.class.getName());
        TokenizerVarMap.put("useUnk", true);    //unlisted words will use usrUnk
        TokenizerVarMap.put("vectorsConfiguration", new VectorsConfiguration());
        TokenizerVarMap.put("stopWords", new ArrayList<String>());  //stop words
        broadcasTokenizerVarMap = jsc.broadcast(TokenizerVarMap);
    }



}
