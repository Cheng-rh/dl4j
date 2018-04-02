import com.huaban.analysis.jieba.JiebaSegmenter;
import com.huaban.analysis.jieba.SegToken;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.spark.text.functions.TextPipeline;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created with IntelliJ IDEA.
 * User: sssd
 * Date: 2018/3/29 14:34
 * Version: V1.0
 * To change this template use File | Settings | File Templates.
 * Description:
 */
public class Test {

    private static JavaSparkContext jsc;

    static {
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]").setAppName("deeplearning4j-lstm");
        jsc = new JavaSparkContext(sparkConf);
    }

    public static void main(String[] args) {
        INDArray indArray = Nd4j.create(3, 2, 5);
        INDArray zeros = Nd4j.zeros(2, 5);
        int[] origin = new int[3];
        System.out.println(indArray);
        System.out.println("---------------------");
        System.out.println(zeros);
        System.out.println("---------------------");
        System.out.println(origin.length);
    }


    @org.junit.Test
    public void test() {
        INDArray feature = Nd4j.create(2, 3, 5);
        int[] origin = new int[3];
        origin[0] = 1;
        origin[1] = 2;
        origin[2] = 2;
        feature.putScalar(origin, 3);
        System.out.println(feature);
    }

    @org.junit.Test
    public void readFile() throws Exception {
        String path = "C:\\Users\\sssd\\Desktop\\LSTM\\sourcedata.txt";
        List<String> lines = FileUtils.readLines(new File(path));
        int i = 1;
        for (String line : lines) {
            String[] split = line.split("\t");
            System.out.println("第" + i + "行" + split[0] + "-------" + split[1]);
            i++;
        }
    }

    @org.junit.Test
    public void testPipeline() throws Exception {
        Map<String, Object> TokenizerVarMap = new HashMap<String, Object>();
        TokenizerVarMap.put("numWords", 1);     //min count of word appearence
        TokenizerVarMap.put("nGrams", 1);       //language model parameter
        TokenizerVarMap.put("tokenizer", DefaultTokenizerFactory.class.getName());  //tokenizer implemention
        TokenizerVarMap.put("tokenPreprocessor", CommonPreprocessor.class.getName());
        TokenizerVarMap.put("useUnk", true);    //unlisted words will use usrUnk
        TokenizerVarMap.put("vectorsConfiguration", new VectorsConfiguration());
        TokenizerVarMap.put("stopWords", new ArrayList<String>());  //stop words
        Broadcast<Map<String, Object>> mapBroadcast = jsc.broadcast(TokenizerVarMap);

        ArrayList<String> list = new ArrayList<String>();
        list.add("我 是 中国 人。");
        list.add("中国 是 个 好 地方。");
        list.add("中国 人 生活 在 中国。");

        ArrayList<String> list1 = new ArrayList<String>();
        list1.add("我 是 中国 人");

        JavaRDD<String> rdd = jsc.parallelize(list);
        TextPipeline pipeline = new TextPipeline(rdd, mapBroadcast);
        pipeline.buildVocabCache();
        pipeline.buildVocabWordListRDD();
        JavaRDD<List<VocabWord>> res = pipeline.getVocabWordListRDD();


        List<List<VocabWord>> collect = res.collect();
        for (List<VocabWord> words : collect) {
            System.out.println(words.toString());
        }


    }

    @org.junit.Test
    public void modelPre() {
        String str = "我是中国人";
        JiebaSegmenter segmenter = new JiebaSegmenter();
        List<SegToken> tokens = segmenter.process(str, JiebaSegmenter.SegMode.INDEX);
        ArrayList<String> list = new ArrayList<String>();
        for (SegToken token : tokens) {
            String word = token.word;
            list.add(word);
        }
        String res = StringUtils.join(list.toArray(), " ");
        System.out.println(res);

    }

    public void testStr(){

        String str = "[VocabWord{wordFrequency=1.0, index=9, word='我', codeLength=4}, " +
                "VocabWord{wordFrequency=2.0, index=1, word='是', codeLength=3}, " +
                "VocabWord{wordFrequency=3.0, index=0, word='中国', codeLength=2}, " +
                "VocabWord{wordFrequency=1.0, index=10, word='人。', codeLength=4}]";
    }

}
