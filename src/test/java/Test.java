import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
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
import java.io.FileWriter;
import java.io.IOException;
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

        savaPipeLine(res);

/*        List<List<VocabWord>> collect = res.collect();
        for (List<VocabWord> words : collect) {
            System.out.println(words.toString());
        }*/


    }

    public static void savaPipeLine(JavaRDD<List<VocabWord>> textList) {
        List<List<VocabWord>> collect = textList.collect();
        StringBuffer bf = new StringBuffer();
        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter("C:\\Users\\sssd\\Desktop\\pipeLine.txt");
            JSONArray objects = new JSONArray();
            for (List<VocabWord> list : collect) {
                for (VocabWord vocabWord : list) {
                    objects.add(vocabWord.toJSON());
                }
                bf.append(objects.toString()).append("\n");
                System.out.println(bf.toString());
                objects = new JSONArray();
            }
            bf.deleteCharAt(bf.length() - 1);
            fileWriter.write(bf.toString());
            fileWriter.flush();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                fileWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @org.junit.Test
    public void readPipeLine(){
        ArrayList<JSONArray> allLines = new ArrayList<JSONArray>();
        try {
            List<String> lines = FileUtils.readLines(new File("C:\\Users\\sssd\\Desktop\\pipeLine.txt"));
            for (String line : lines) {
                JSONArray jsonArray = (JSONArray) JSON.parse(line);
                allLines.add(jsonArray);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        ArrayList<String> list = new ArrayList<String>();
        list.add("我");
        list.add("中国");

        INDArray features = Nd4j.zeros(1, 1, 10);
        int[] origin = new int[3];
        origin[0] = 0;
        int j = 0;
        int sign = 0;
        for (String tempStr : list) {
            origin[2] = j;
            for (JSONArray line : allLines) {
                for (Object o : line) {
                    System.out.println(o.toString());
                    JSONObject jsonObject = (JSONObject)JSON.parse(o.toString());
                    if (jsonObject.getString("word").equals(tempStr)) {
                        features.putScalar(origin, jsonObject.getDouble("index"));
                        sign = 1;
                        break;
                    }
                }
                if (sign == 1){
                    sign = 0;
                    break;
                }
            }
            j++;
        }

        System.out.println(features.toString());
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
