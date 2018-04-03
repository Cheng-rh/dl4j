package nlp;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import org.apache.commons.io.FileUtils;
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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created with IntelliJ IDEA.
 * User: sssd
 * Date: 2018/4/3 16:31
 * Version: V1.0
 * To change this template use File | Settings | File Templates.
 * Description:      文本数据预处理
 */
public class DataHandle {

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
     * 统计词汇表长度
     */
    public static AtomicInteger VOCAB_SIZE = new AtomicInteger(0);

    /**
     * 文本特征维度
     */
    public static Integer maxlength = 663;

    /**
     * 记录文本行数
     */
    public static AtomicInteger lineNum = new AtomicInteger(0);

    /**
     * 样本标签的维度
     */
    public static Integer numLabel = 2;


    public DataHandle(JavaSparkContext jsc, String inputPath) {
        this.jsc = jsc;
        this.basePath = inputPath;
    }

    /**
     * Java RDD 获取文本数据
     *
     * @return
     */
    public static JavaRDD<DataSet> getDataSet() {
        JavaRDD<Tuple2<String, String>> sourceData = readFile(jsc, basePath + "/src/main/resources/lstm/data.txt").persist(StorageLevel.MEMORY_AND_DISK_SER_2());
        JavaRDD<String> label = readLabel(sourceData);
        JavaRDD<String> text = readText(sourceData);
        initTokenizer();
        JavaRDD<List<VocabWord>> labelList = pipeLine(label);
        JavaRDD<List<VocabWord>> textList = pipeLine(text);
        savaPipeLine(textList, "pipText");
        savaPipeLine(labelList, "pipLabel");
        findMaxlength(textList);
        JavaRDD<Tuple2<List<VocabWord>, VocabWord>> combine = combine(labelList, textList);
        JavaRDD<DataSet> data = chang2DataSet(combine).persist(StorageLevel.MEMORY_AND_DISK_SER_2());
        sourceData.unpersist();
        return data;
    }

    /**
     * 读取文本数据
     *
     * @param jsc  JavaSparkContext
     * @param path 文本存放路径
     * @return
     */
    public static JavaRDD<Tuple2<String, String>> readFile(JavaSparkContext jsc, String path) {
        JavaRDD<String> javaRDD = jsc.textFile(path);
        JavaRDD<Tuple2<String, String>> rdd = javaRDD.map(new Function<String, Tuple2<String, String>>() {
            public Tuple2<String, String> call(String s) throws Exception {
                String[] split = s.split("\t");
                Tuple2<String, String> tuple2 = new Tuple2<String, String>(split[0], split[1]);
                return tuple2;
            }
        });
        return rdd;
    }

    /**
     * 读取文本中的标签
     *
     * @param data javaRDD 文本数据
     * @return
     */
    public static JavaRDD<String> readLabel(JavaRDD<Tuple2<String, String>> data) {
        JavaRDD<String> labelRdd = data.map(new Function<Tuple2<String, String>, String>() {
            public String call(Tuple2<String, String> stringStringTuple2) throws Exception {
                String label = stringStringTuple2._1;
                return label;
            }
        });
        return labelRdd;
    }

    /**
     * 读取文本中的文本内容
     *
     * @param data JavaRDD文本数据
     * @return
     */
    public static JavaRDD<String> readText(JavaRDD<Tuple2<String, String>> data) {
        JavaRDD<String> textRdd = data.map(new Function<Tuple2<String, String>, String>() {
            public String call(Tuple2<String, String> stringStringTuple2) throws Exception {
                String text = stringStringTuple2._2;
                return text;
            }
        });
        return textRdd;
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
    public static JavaRDD<List<VocabWord>> pipeLine(JavaRDD<String> javaRDDText) {
        JavaRDD<List<VocabWord>> textVocab = null;
        try {
            TextPipeline pipeline = new TextPipeline(javaRDDText, broadcasTokenizerVarMap);
            pipeline.buildVocabCache();
            pipeline.buildVocabWordListRDD();
            textVocab = pipeline.getVocabWordListRDD();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return textVocab;
    }

    /**
     * 将文本内容对应的pipeLine 参数进行保存
     *
     * @param textList 经过pipeLine 处理过的文本内容
     * @param fileName 文本保存名字
     */
    public static void savaPipeLine(JavaRDD<List<VocabWord>> textList, String fileName) {
        System.out.println("------------------ 开始保存bpeLine ----------------------");
        List<List<VocabWord>> collect = textList.collect();
        StringBuffer bf = new StringBuffer();
        FileWriter fileWriter = null;
        try {
            if (fileName.equals("pipText")) {
                fileWriter = new FileWriter(basePath + "\\src\\main\\resources\\lstm\\pipText.txt");
            } else if (fileName.equals("pipLabel")) {
                fileWriter = new FileWriter(basePath + "\\src\\main\\resources\\lstm\\pipText.txt");
            }

            JSONArray objects = new JSONArray();
            for (List<VocabWord> list : collect) {
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
     * 统计文本特征最大维度和 文本词汇表长度
     *
     * @param textList
     */
    public static void findMaxlength(JavaRDD<List<VocabWord>> textList) {
        JavaRDD<Integer> map = textList.map(new Function<List<VocabWord>, Integer>() {
            public Integer call(List<VocabWord> vocabWords) throws Exception {
                int size = vocabWords.size();
                VOCAB_SIZE.addAndGet(size);
                if (maxlength < size) {
                    maxlength = size;
                }
                return size;
            }
        });
        map.count();
        System.out.println("词表的长度为：" + VOCAB_SIZE);
        System.out.println("预料的最大维度为：" + maxlength);
    }

    /**
     * 将pipeLine 处理过的文本内容和文本标签进行合并
     *
     * @param label pipeLine 处理过的文本标签
     * @param text  pipeLine 处理过的文本内容
     * @return
     */
    public static JavaRDD<Tuple2<List<VocabWord>, VocabWord>> combine(JavaRDD<List<VocabWord>> label, JavaRDD<List<VocabWord>> text) {
        final List<List<VocabWord>> labels = label.collect();
        JavaRDD<Tuple2<List<VocabWord>, VocabWord>> res = text.map(new Function<List<VocabWord>, Tuple2<List<VocabWord>, VocabWord>>() {
            public Tuple2<List<VocabWord>, VocabWord> call(List<VocabWord> vocabWords) throws Exception {
                int num = lineNum.getAndIncrement();
                VocabWord labelVocabWord = labels.get(num).get(0);
                Tuple2<List<VocabWord>, VocabWord> tuple2 = new Tuple2<List<VocabWord>, VocabWord>(vocabWords, labelVocabWord);
                if (lineNum.equals(labels.size() - 1)) {
                    lineNum = new AtomicInteger(0);
                }
                return tuple2;
            }
        });
        return res;
    }

    /**
     * 封装模型需要的数据格式
     *
     * @param combine 将pipeLIne处理数据合并后的数据
     * @return
     */
    public static JavaRDD<DataSet> chang2DataSet(JavaRDD<Tuple2<List<VocabWord>, VocabWord>> combine) {
        JavaRDD<DataSet> res = combine.map(new Function<Tuple2<List<VocabWord>, VocabWord>, DataSet>() {
            public DataSet call(Tuple2<List<VocabWord>, VocabWord> tuple) throws Exception {
                List<VocabWord> wordList = tuple._1;
                VocabWord labelList = tuple._2;
                INDArray features = Nd4j.create(1, 1, maxlength);
                INDArray labels = Nd4j.create(1, numLabel, maxlength);
                INDArray featureMask = Nd4j.zeros(1, maxlength);
                INDArray labelMask = Nd4j.zeros(1, maxlength);
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
                int index = labelList.getIndex();
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
            List<String> lines = FileUtils.readLines(new File(basePath + "\\src\\main\\resources\\lstm\\pipeLine.txt"));
            for (String line : lines) {
                JSONArray jsonArray = (JSONArray) JSON.parse(line);
                allLines.add(jsonArray);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        INDArray features = Nd4j.zeros(1, 1, maxlength);
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

    /**
     * 读取 pipeLine 文件
     *
     * @param path  label 文件的保存路径
     */
    public static HashMap<Double,String> readPipeLine(String path) {
        HashMap<Double, String> map = new HashMap<Double, String>();
        try {
            List<String> lines = FileUtils.readLines(new File(path));
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
