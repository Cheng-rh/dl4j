package nlp;

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.text.functions.TextPipeline;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import scala.Tuple2;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created with IntelliJ IDEA.
 * User: sssd
 * Date: 2018/3/28 11:49
 * Version: V1.0
 * To change this template use File | Settings | File Templates.
 * Description:   基于lstm的文本情感识别及其spark实现
 */
public class TestLstm implements Serializable {

    private static JavaSparkContext jsc;

    private static Broadcast<Map<String, Object>> broadcasTokenizerVarMap;

    private static AtomicInteger lineNum = new AtomicInteger(0);

    private static Integer maxlength = 0;

    private static Integer numLabel = 2;

    private static Integer VOCAB_SIZE = 0;

    private static Integer batchSize = 36;

    private static Integer totalEpoch = 4;

    private static Integer lstmLayerSize = 256;

    private static Integer nOut = 2;

    public TestLstm() {
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


    public JavaRDD<Tuple2<String, String>> readFile(String path) {
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

    public JavaRDD<String> readLabel(JavaRDD<Tuple2<String, String>> data) {
        JavaRDD<String> labelRdd = data.map(new Function<Tuple2<String, String>, String>() {
            public String call(Tuple2<String, String> stringStringTuple2) throws Exception {
                String label = stringStringTuple2._1;
                return label;
            }
        });
        labelRdd.count();
        return labelRdd;
    }

    public JavaRDD<String> readText(JavaRDD<Tuple2<String, String>> data) {
        JavaRDD<String> textRdd = data.map(new Function<Tuple2<String, String>, String>() {
            public String call(Tuple2<String, String> stringStringTuple2) throws Exception {
                String text = stringStringTuple2._2;
                return text;
            }
        });
        return textRdd;
    }


    public JavaRDD<List<VocabWord>> pipeLine(JavaRDD<String> javaRDDText) {
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

    public JavaRDD<Tuple2<List<VocabWord>, VocabWord>> combine(JavaRDD<List<VocabWord>> label, JavaRDD<List<VocabWord>> text) {
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

    public void findMaxlength(JavaRDD<List<VocabWord>> textList) {
        JavaRDD<Integer> map = textList.map(new Function<List<VocabWord>, Integer>() {
            public Integer call(List<VocabWord> vocabWords) throws Exception {
                int size = vocabWords.size();
                VOCAB_SIZE += size;
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

    public JavaRDD<DataSet> chang2DataSet(JavaRDD<Tuple2<List<VocabWord>, VocabWord>> combine) {
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

    public void modelTrain(JavaRDD<DataSet> trainData, String savaPath) throws Exception {
        MultiLayerConfiguration netconf = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .iterations(1)
                .learningRate(0.1)
                .learningRateScoreBasedDecayRate(0.5)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .regularization(true)
                .l2(5 * 1e-4)
                .updater(Updater.ADAM)
                .list()
                .layer(0, new EmbeddingLayer.Builder().nIn(VOCAB_SIZE).nOut(lstmLayerSize).activation(Activation.IDENTITY).build())
                .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize).activation(Activation.SOFTSIGN).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(lstmLayerSize).nOut(nOut).build())
                .pretrain(false).backprop(true)
                .setInputType(InputType.recurrent(VOCAB_SIZE))
                .build();

        ParameterAveragingTrainingMaster trainMaster = new ParameterAveragingTrainingMaster.Builder(batchSize)
                .workerPrefetchNumBatches(0)
                .saveUpdater(true)
                .averagingFrequency(5)
                .batchSizePerWorker(batchSize)
                .build();

        SparkDl4jMultiLayer sparknet = new SparkDl4jMultiLayer(jsc, netconf, trainMaster);
        sparknet.setListeners(Collections.<IterationListener>singletonList(new ScoreIterationListener(1)));
        for (int numEpoch = 0; numEpoch < totalEpoch; ++numEpoch) {
            sparknet.fit(trainData);
            Evaluation evaluation = sparknet.evaluate(trainData);
            double accuracy = evaluation.accuracy();
            System.out.println("====================================================================");
            System.out.println("Epoch " + numEpoch + " Has Finished");
            System.out.println("Accuracy: " + accuracy);
            System.out.println("====================================================================");
        }
        MultiLayerNetwork network = sparknet.getNetwork();

        FileSystem fileSystem = FileSystem.get(jsc.hadoopConfiguration());
        Path path = new Path(savaPath);
        if (fileSystem.exists(path)) {
            fileSystem.delete(path, true);
        }
        FSDataOutputStream outputStream = fileSystem.create(path);
        ModelSerializer.writeModel(network, outputStream, true);
    }

    public static void main(String[] args) {
        TestLstm testLstm = new TestLstm();
        String path = "file:///C:/Users/sssd/Desktop/LSTM/data.txt";
        String savePath = "file:///C:/Users/sssd/Desktop/LSTM/model.txt";
        JavaRDD<Tuple2<String, String>> data = testLstm.readFile(path).persist(StorageLevel.MEMORY_AND_DISK_SER_2());
        JavaRDD<String> label = testLstm.readLabel(data);
        JavaRDD<String> text = testLstm.readText(data);
        data.unpersist();
        JavaRDD<List<VocabWord>> labelList = testLstm.pipeLine(label);
        JavaRDD<List<VocabWord>> textList = testLstm.pipeLine(text);
        testLstm.findMaxlength(textList);
        JavaRDD<Tuple2<List<VocabWord>, VocabWord>> combine = testLstm.combine(labelList, textList);
        JavaRDD<DataSet> trainData = testLstm.chang2DataSet(combine).persist(StorageLevel.MEMORY_AND_DISK_SER_2());
        combine.unpersist();
        try {
            testLstm.modelTrain(trainData, savePath);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}
