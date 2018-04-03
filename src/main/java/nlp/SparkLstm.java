package nlp;


import com.huaban.analysis.jieba.JiebaSegmenter;
import com.huaban.analysis.jieba.SegToken;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
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
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created with IntelliJ IDEA.
 * User: sssd
 * Date: 2018/3/30 16:04
 * Version: V1.0
 * To change this template use File | Settings | File Templates.
 * Description:    spark + dl4j + lstm 实现文本分类
 */
public class SparkLstm {

    /**
     * 文本转换词汇表的维度
     */
    private static Integer maxlength = 663;

    /**
     * 词汇表的长度
     */
    private static AtomicInteger VOCAB_SIZE = new AtomicInteger(0);

    /**
     * JavaSparkContext
     */
    private static JavaSparkContext jsc;

    /**
     * 文件基本路径
     */
    private static String basePath = System.getProperty("user.dir");

    /**
     * 模型训练批大小
     */
    private static Integer batchSize = 36;

    static {
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]").setAppName("deeplearning4j-lstm");
        jsc = new JavaSparkContext(sparkConf);
    }

    /**
     * 模型训练
     *
     * @param args
     * @throws Exception
     */
    public void modelTrain(String[] args) throws Exception {

        // 获取训练集和测试集
        DataHandle dataHandle = new DataHandle(jsc, basePath);
        JavaRDD<DataSet> dataSet = dataHandle.getDataSet();
        this.VOCAB_SIZE = dataHandle.VOCAB_SIZE;
        this.maxlength = dataHandle.maxlength;
        JavaRDD<DataSet>[] javaRDDS = dataSet.randomSplit(new double[]{0.7, 0.3}, 11l);
        JavaRDD<DataSet> trainData = javaRDDS[0];
        JavaRDD<DataSet> testData = javaRDDS[1];

        // lstm 的初始化参数
        Integer totalEpoch = 10;
        Integer lstmLayerSize = 256;
        Integer nOut = 2;

        // 构建模型参数
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
                .layer(0, new EmbeddingLayer.Builder().nIn(VOCAB_SIZE.get()).nOut(lstmLayerSize).activation(Activation.IDENTITY).build())
                .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize).activation(Activation.SOFTSIGN).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(lstmLayerSize).nOut(nOut).build())
                .pretrain(false).backprop(true)
                .setInputType(InputType.recurrent(VOCAB_SIZE.get()))
                .build();

        ParameterAveragingTrainingMaster trainMaster = new ParameterAveragingTrainingMaster.Builder(batchSize)
                .workerPrefetchNumBatches(0)
                .saveUpdater(true)
                .averagingFrequency(5)
                .batchSizePerWorker(batchSize)
                .build();

        //lstm 模型训练
        SparkDl4jMultiLayer sparknet = new SparkDl4jMultiLayer(jsc, netconf, trainMaster);
        sparknet.setListeners(Collections.<IterationListener>singletonList(new ScoreIterationListener(1)));
        for (int numEpoch = 0; numEpoch < totalEpoch; numEpoch++) {
            sparknet.fit(trainData);
        }

        // lstm 模型验证
        Evaluation evaluation = sparknet.evaluate(testData);
        double accuracy = evaluation.accuracy();
        System.out.println("====================================================================");
        System.out.println("Accuracy: " + accuracy);
        System.out.println("====================================================================");

        // lstm 保存训练模型
        MultiLayerNetwork network = sparknet.getNetwork();
        ModelSerializer.writeModel(network, new File(basePath + "/src/main/resources/lstm/lstm-model.zip"), true);
    }


    public void modelPre(String str) throws Exception {

        // 文本分词
        JiebaSegmenter segmenter = new JiebaSegmenter();
        List<SegToken> tokens = segmenter.process(str, JiebaSegmenter.SegMode.INDEX);
        ArrayList<String> list = new ArrayList<String>();
        for (SegToken token : tokens) {
            String word = token.word;
            list.add(word);
        }

        // 文本数据预处理
        DataHandle dataHandle = new DataHandle(jsc, basePath);
        INDArray testArray = dataHandle.str2INDArray(list);

        // 加载模型并预测
        MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(basePath + "\\src\\main\\resources\\lstm\\lstm-model.zip");
        INDArray output = net.output(testArray, Layer.TrainingMode.TEST);
        INDArray row = output.getRow(0).getRow(0);
        INDArray mean = row.mean(1);
        double res = mean.getDouble(0);
        System.out.println("");
    }


    public static void main(String[] args) throws Exception {
        new SparkLstm().modelTrain(args);
        String str = "老者还知道你很好奇，知道你有疑问，他还允许你提问，然后会用他会用他渊博的见识，耐心地为你分析，合情合理合情合理，还不乱结论下结论，好像在和你互动。";
        String str1 = "蒙牛牛你果然是个傻逼ps:你不冷啊还短裤咯";
//        new SparkLstm().modelPre(str1);
    }

}
