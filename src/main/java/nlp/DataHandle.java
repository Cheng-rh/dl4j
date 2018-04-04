package nlp;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import org.apache.commons.io.FileUtils;
import org.apache.spark.Accumulator;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.spark.text.functions.TextPipeline;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import scala.Tuple2;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created with IntelliJ IDEA.
 * User: sssd
 * Date: 2018/4/3 16:31
 * Version: V1.0
 * To change this template use File | Settings | File Templates.
 * Description:      文本数据处理
 */
public class DataHandle implements Serializable {

    /**
     * JavaSparkContext
     */
    public static JavaSparkContext jsc;

    /**
     * 基本路径
     */
    public static String basePath;

    /**
     * pipeline 初始化参数
     */
    public static Broadcast<Map<String, Object>> broadcasTokenizerVarMap;

    /**
     * 统计词汇表的长度
     */
    public static volatile Accumulator<Integer> VOCAB_SIZE;

    /**
     * 设置文本向量特征维度
     */
    public static volatile Broadcast<Integer> maxlength;

    /**
     * 设置文本标签的维度
     */
    public static volatile Broadcast<Integer> numLabel;

    /**
     * 初始化文本标签种类
     */
    public static volatile Broadcast<List<String>> LABEL;


    public DataHandle(JavaSparkContext jsc, String inputPath) {
        // 初始化 jsc
        this.jsc = jsc;
        // 初始化读取文件的基本路径
        this.basePath = inputPath;
        // 统计词汇表的长度
        this.VOCAB_SIZE = this.jsc.accumulator(0, "wordSize");
        // 广播文本向量维度
        maxlength = this.jsc.broadcast(663);
        // 广播文本标签维度
        numLabel = this.jsc.broadcast(2);
        // 广播文本标签种类
        List<String> type = new ArrayList<String>();
        type.add("a");
        type.add("b");
        LABEL = this.jsc.broadcast(type);
    }

    /**
     * Java RDD 获取文本数据
     *
     * @return
     */
    public static JavaRDD<DataSet> getDataSet() {
        JavaRDD<String> javaRDD = jsc.textFile(basePath + "/src/main/resources/lstm/data.txt");
        initTokenizer();
        JavaRDD<Tuple2<List<VocabWord>, Integer>> pipeLineData = pipeLine(javaRDD);
        savaPipeLine(pipeLineData);
        JavaRDD<DataSet> data = chang2DataSet(pipeLineData).persist(StorageLevel.MEMORY_AND_DISK_SER_2());
        return data;
    }

    /**
     * 初始化 pipeline 参数并进行广播
     */
    public static void initTokenizer() {
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

    /**
     * pipeLine 处理
     *
     * @param javaRDDText 文本内容或标签
     * @return
     */
    public static JavaRDD<Tuple2<List<VocabWord>, Integer>> pipeLine(JavaRDD<String> javaRDDText) {
        JavaRDD<Tuple2<List<VocabWord>, Integer>> resMap = null;
        try {
            TextPipeline pipeline = new TextPipeline(javaRDDText, broadcasTokenizerVarMap);
            pipeline.buildVocabCache();
            pipeline.buildVocabWordListRDD();
            JavaRDD<List<VocabWord>> wordListRDD = pipeline.getVocabWordListRDD();
            resMap = wordListRDD.map(new Function<List<VocabWord>, Tuple2<List<VocabWord>, Integer>>() {
                public Tuple2<List<VocabWord>, Integer> call(List<VocabWord> vocabWords) throws Exception {
                    VOCAB_SIZE.add(vocabWords.size() - 1);
                    Tuple2<List<VocabWord>, Integer> tuple2 = new Tuple2<List<VocabWord>, Integer>(
                            new ArrayList<VocabWord>(vocabWords.subList(1, vocabWords.size())),
                            LABEL.getValue().indexOf(vocabWords.get(0).getWord()));
                    return tuple2;
                }
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
        return resMap;
    }

    /**
     * 将文本内容对应的pipeLine 参数进行保存
     *
     * @param textList 经过pipeLine 处理过的文本内容
     */
    public static void savaPipeLine(JavaRDD<Tuple2<List<VocabWord>, Integer>> textList) {
        System.out.println("------------------ 开始保存bpeLine ----------------------");
        List<Tuple2<List<VocabWord>, Integer>> collect = textList.collect();
        StringBuffer bf = new StringBuffer();
        FileWriter fileWriter = null;

        try {
            fileWriter = new FileWriter(basePath + "\\src\\main\\resources\\lstm\\pipText.txt");
            JSONArray objects = new JSONArray();
            for (Tuple2<List<VocabWord>, Integer> tuple2 : collect) {
                List<VocabWord> list = tuple2._1;
                for (VocabWord vocabWord : list) {
                    objects.add(vocabWord.toJSON());
                }
                bf.append(objects.toString()).append("\n");
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


    /**
     * 封装模型需要的数据格式
     *
     * @param combine 将pipeLIne处理数据合并后的数据
     * @return
     */
    public static JavaRDD<DataSet> chang2DataSet(JavaRDD<Tuple2<List<VocabWord>, Integer>> combine) {
        JavaRDD<DataSet> res = combine.map(new Function<Tuple2<List<VocabWord>, Integer>, DataSet>() {
            public DataSet call(Tuple2<List<VocabWord>, Integer> tuple) throws Exception {
                List<VocabWord> wordList = tuple._1;
                INDArray features = Nd4j.create(1, 1, maxlength.getValue());
                INDArray labels = Nd4j.create(1, numLabel.getValue(), maxlength.getValue());
                INDArray featureMask = Nd4j.zeros(1, maxlength.getValue());
                INDArray labelMask = Nd4j.zeros(1, maxlength.getValue());
                int[] origin = new int[3];
                int[] mask = new int[2];
                origin[0] = 0;
                mask[0] = 0;
                int j = 0;
                for (VocabWord vocabWord : wordList) {
                    origin[2] = j;
                    features.putScalar(origin, vocabWord.getIndex());
                    mask[1] = j;
                    featureMask.putScalar(mask, 1.0);
                    ++j;
                }
                int lastIndex = wordList.size();
                int index = tuple._2;
                labels.putScalar(new int[]{0, index, lastIndex - 1}, 1.0);
                labelMask.putScalar(new int[]{0, lastIndex - 1}, 1.0);
                return new DataSet(features, labels, featureMask, labelMask);
            }
        });
        return res;
    }


    /**
     * 转换数据为指定格式
     *
     * @param list 分此后的数据
     * @return
     */
    public static INDArray str2INDArray(ArrayList<String> list) {
        ArrayList<JSONArray> allLines = new ArrayList<JSONArray>();
        try {
            List<String> lines = FileUtils.readLines(new File(basePath + "\\src\\main\\resources\\lstm\\pipText.txt"));
            for (String line : lines) {
                JSONArray jsonArray = (JSONArray) JSON.parse(line);
                allLines.add(jsonArray);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        INDArray features = Nd4j.zeros(1, 1, 663);
        int[] origin = new int[3];
        origin[0] = 0;
        int j = 0;
        int sign = 0;
        for (String tempStr : list) {
            origin[2] = j;
            for (JSONArray line : allLines) {
                for (Object o : line) {
                    JSONObject jsonObject = (JSONObject) JSON.parse(o.toString());
                    if (jsonObject.getString("word").equals(tempStr)) {
                        features.putScalar(origin, jsonObject.getDouble("index"));
                        sign = 1;
                        break;
                    }
                }
                if (sign == 1) {
                    sign = 0;
                    break;
                }
            }
            j++;
        }
        return features;
    }
}
