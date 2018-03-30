package nlp;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.*;

/**
 * GravesLSTM + Spark character modelling example
 * Example: Train a LSTM RNN to generates text, one character at a time.
 * Training here is done on Spark
 * <p>
 * See dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/character/GravesLSTMCharModellingExample.java
 * for the single-machine version of this example
 * <p>
 * To run the example locally: Run the example as-is. The example is set up to use Spark local by default.
 * NOTE: Spark local should only be used for development/testing. For data parallel training on a single machine
 * (for example, multi-GPU systems) instead use ParallelWrapper (which is faster than using Spark for training on a single machine).
 * See for example MultiGpuLenetMnistExample in dl4j-cuda-specific-examples
 * <p>
 * To run the example using Spark submit (for example on a cluster): pass "-useSparkLocal false" as the application argument,
 * OR first modify the example by setting the field "useSparkLocal = false"
 *
 * @author Alex Black
 */
public class SparkLSTMCharacterExample {
    private static final Logger log = LoggerFactory.getLogger(SparkLSTMCharacterExample.class);

    /**
     * 根据字符串下标转换为字符串
     */
    private static Map<Integer, Character> INT_TO_CHAR = getIntToChar();

    /**
     * 根据字符串转换为对应的下标。
     */
    private static Map<Character, Integer> CHAR_TO_INT = getCharToInt();

    /**
     *  下标词汇表的大小
     */
    private static final int N_CHARS = INT_TO_CHAR.size();

    /**
     * 字符串词汇表的大小
     */
    private static int nOut = CHAR_TO_INT.size();

    /**
     * 切割字符串的长度
     */
    private static int exampleLength = 1000;                    //Length of each training example sequence to use

    @Parameter(names = "-useSparkLocal", description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
    private boolean useSparkLocal = true;

    @Parameter(names = "-batchSizePerWorker", description = "Number of examples to fit each worker with")
    private int batchSizePerWorker = 8;   //How many examples should be used per worker (executor) when fitting?

    @Parameter(names = "-numEpochs", description = "Number of epochs for training")
    private int numEpochs = 1;

    /**
     * 类中 main  方法
     *
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        // 调用方法入口
        new SparkLSTMCharacterExample().entryPoint(args);
    }

    /**
     *  程序调用的接口
     *
     * @param args
     * @throws Exception
     */
    protected void entryPoint(String[] args) throws Exception {
        //Handle command line arguments  JCommander： 用来处理解析命令行参数的的Java框架
        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);    // 解析参数
        } catch (ParameterException e) {
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try {
                Thread.sleep(500);
            } catch (Exception e2) {
            }
            throw e;
        }

        Random rng = new Random(12345);
        // 设置 lstm 网络神经元的个数
        int lstmLayerSize = 200;                    //Number of units in each GravesLSTM layer
        // 设置反向传播的时间长度
        int tbpttLength = 50;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
        // 每次训练生成的样本数
        int nSamplesToGenerate = 4;                    //Number of samples to generate after each training epoch
        // 生成样本的长度
        int nCharactersToSample = 300;                //Length of each sample to generate
        String generationInitialization = null;        //Optional character initialization; a random character is used if null
        // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
        // Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default

        //Set up network configuration:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)    //设置的随机梯度下降法
            .iterations(1)  // 对于一个神经网络而言，一次迭代（iteration）指的是一个学习步骤，亦即模型参数的一次更新
            .learningRate(0.1) // 用于设定学习速率（learning rate），即每次迭代时对于权重的调整幅度，亦即步幅，
            .seed(12345) // 该参数将一组随机生成的权重确定为初始权重
            .regularization(true)   //正则化用来防止过拟合的一种方法。
            .l2(0.001) //用L2正则化来防止个别权重对总体结果产生过大的影响
            .weightInit(WeightInit.XAVIER)  //权重初始化
            .updater(Updater.RMSPROP)   //To configure: .updater(new RmsProp(0.95))  //加速衰减系数，防止梯度变化过大，训练过早结束的参数更新方法
            .list()  //函数可指定网络中层的数量；它会将您的配置复制n次，建立分层的网络结构。
            .layer(0, new GravesLSTM.Builder().nIn(CHAR_TO_INT.size()).nOut(lstmLayerSize).activation(Activation.TANH).build()) //第一层是LSTM，输入大小独立字符数，输出大小是200
            .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize).activation(Activation.TANH).build())  //第二层还是LSTM层，输入输出节点都是200
            .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
                .nIn(lstmLayerSize).nOut(nOut).build())  //使用交叉熵作为损失函数
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength) //使用截断式bptt，截断长度为50，即正反向参数更新参考的长度都是50
            .pretrain(false).backprop(true)
            .build();


        //-------------------------------------------------------------
        //Set up the Spark-specific configuration
        /* How frequently should we average parameters (in number of minibatches)?
        Averaging too frequently can be slow (synchronization + serialization costs) whereas too infrequently can result
        learning difficulties (i.e., network may not converge) */

        //参数平均化的频率，3批平均一次
        int averagingFrequency = 3;

        // 初始化sparkConf 和 sparkContext
        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        sparkConf.setAppName("LSTM Character Example");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        // 获取训练集的RDD<DataSet>
        JavaRDD<DataSet> trainingData = getTrainingData(sc);  //获取数据得到训练RDD。


        //Set up the TrainingMaster. The TrainingMaster controls how learning is actually executed on Spark
        //Here, we are using standard parameter averaging
        //For details on these configuration options, see: https://deeplearning4j.org/spark#configuring
        int examplesPerDataSetObject = 1;  //每个DataSet对象有一个例子
        ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
            .workerPrefetchNumBatches(2)    //Asynchronously prefetch up to 2 batches  //异步获取2批数据
            .averagingFrequency(averagingFrequency)    //参数平均化的频率是3
            .batchSizePerWorker(batchSizePerWorker)  //每个worker处理批的大小是8
            .build();
        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, conf, tm);  //把参数传入spark的网络配置
        sparkNetwork.setListeners(Collections.<IterationListener>singletonList(new ScoreIterationListener(1)));//设置监听器,singletonList返回一个包含具体对象的不可变list

        //Do training, and then generate and print samples from network
        for (int i = 0; i < numEpochs; i++) {
            //Perform one epoch of training. At the end of each epoch, we are returned a copy of the trained network
            MultiLayerNetwork net = sparkNetwork.fit(trainingData);  //训练网络

            //Sample some characters from the network (done locally)
            log.info("Sampling characters from network given initialization \"" +
                (generationInitialization == null ? "" : generationInitialization) + "\"");
            String[] samples = sampleCharactersFromNetwork(generationInitialization, net, rng, INT_TO_CHAR,
                nCharactersToSample, nSamplesToGenerate);
            for (int j = 0; j < samples.length; j++) {
                log.info("----- Sample " + j + " -----");
                log.info(samples[j]);
            }
        }

        //Delete the temp training files, now that we are done with them
        tm.deleteTempFiles(sc);

        log.info("\n\nExample complete");
    }


    /**
     * 获取训练数据集
     *
     * @param sc JavaSparkContext sc
     * @return
     * @throws IOException
     */
    public static JavaRDD<DataSet> getTrainingData(JavaSparkContext sc) throws IOException {
        //Get data. For the sake of this example, we are doing the following operations:
        // File -> String -> List<String> (split into length "sequenceLength" characters) -> JavaRDD<String> -> JavaRDD<DataSet>

        List<String> list = getShakespeareAsList(exampleLength);   // 获取list，其中每个的长度都是1000的字符串。
        // 并行化数据
        JavaRDD<String> rawStrings = sc.parallelize(list);
        // 将 CHAR_TO_INT 广播出去
        Broadcast<Map<Character, Integer>> bcCharToInt = sc.broadcast(CHAR_TO_INT);
        // 将JavaRDD<String> 的数据转换为JavaRDD<DataSet>
        return rawStrings.map(new StringToDataSetFn(bcCharToInt));
    }

    /**
     * 内部类
     */
    private static class StringToDataSetFn implements Function<String, DataSet> {
        /**
         * 获取得到的广播变量 （CHAR_TO_INT）根据字符串找到对应的下标。
         */
        private final Broadcast<Map<Character, Integer>> ctiBroadcast;

        /**
         * 构造器，初始化广播变量
         * @param characterIntegerMap
         */
        private StringToDataSetFn(Broadcast<Map<Character, Integer>> characterIntegerMap) {
            this.ctiBroadcast = characterIntegerMap;
        }

        public DataSet call(String s) throws Exception {
            // 获取广播变量中的数据内容
            Map<Character, Integer> cti = ctiBroadcast.getValue();
            // 获取字符串的长度（相当于其中的一行字符串）
            int length = s.length();
            // 初始化三维的矩阵
            // spark的数据弄成nd4j的数据，第一个参数代表有1个元素，第二个参数代表这个矩阵元素的行即字符索引数，第三个参数代表这个矩阵元素的列即字符的长度
            INDArray features = Nd4j.zeros(1, N_CHARS, length - 1);
            INDArray labels = Nd4j.zeros(1, N_CHARS, length - 1);
            // 将一行string 转换为 char 数组
            char[] chars = s.toCharArray();
            int[] f = new int[3];
            int[] l = new int[3];
            for (int i = 0; i < chars.length - 2; i++) {
                f[1] = cti.get(chars[i]); //在广播变量里搜索字符的索引，放到f的第二个位置
                f[2] = i;  // 把字符串下标的地址放在放到f的第三个位置
                l[1] = cti.get(chars[i + 1]);   //在广播变量里搜索下一个字符的索引，放到l的第二个字符
                l[2] = i;

                features.putScalar(f, 1.0); // 这里看出f第一个位置不放数字的原因是nd4j高维数组只有1个元素，f代表位置索引，1代表把f代表的位置置为1，one-hot一般都是这个套路
                labels.putScalar(l, 1.0); // 同理把标签放好
            }
            return new DataSet(features, labels);
        }
    }

    //This function downloads (if necessary), loads and splits the raw text data into "sequenceLength" strings

    /**
     * 获取每个长度为1000的字符串list
     *
     * @param sequenceLength 设置字符串list的长度
     * @return
     * @throws IOException
     */
    private static List<String> getShakespeareAsList(int sequenceLength) throws IOException {
        //The Complete Works of William Shakespeare
        //5.3MB file in UTF-8 Encoding, ~5.4 million characters
        //https://www.gutenberg.org/ebooks/100
        String url = "https://s3.amazonaws.com/dl4j-distribution/pg100.txt";
        String tempDir = System.getProperty("java.io.tmpdir");
        String fileLocation = tempDir + "/Shakespeare.txt";    //Storage location from downloaded file
        File f = new File(fileLocation);
        if (!f.exists()) {
            FileUtils.copyURLToFile(new URL(url), f);
            System.out.println("File downloaded to " + f.getAbsolutePath());
        } else {
            System.out.println("Using existing text file at " + f.getAbsolutePath());
        }

        if (!f.exists()) throw new IOException("File does not exist: " + fileLocation);    //Download problem?

        // 根据文件路径，读取文件的全部内容
        String allData = getDataAsString(fileLocation);

        List<String> list = new ArrayList<String>();
        int length = allData.length();
        int currIdx = 0;
        // 将长度为 length 的字符串按长度为 sequenceLength 进行切分，并添加到 list 中
        while (currIdx + sequenceLength < length) {
            int end = currIdx + sequenceLength;
            String substr = allData.substring(currIdx, end);
            currIdx = end;
            list.add(substr);
        }
        return list;
    }

    /**
     * Load data from a file, and remove any invalid characters.
     * Data is returned as a single large String
     */
    /**
     * 根据给定的地址获取全部的数据
     *
     * @param filePath 给定文件的路径
     * @return
     * @throws IOException
     */
    private static String getDataAsString(String filePath) throws IOException {
        // 读取指定路径下的所有文件内容
        List<String> lines = FileUtils.readLines(new File(filePath), Charset.forName("UTF-8"));
        StringBuilder sb = new StringBuilder();
        for (String line : lines) {
            // 将内容转换为 chars 的数组
            char[] chars = line.toCharArray();
            for (int i = 0; i < chars.length; i++) {
                // 如果 CHAR_TO_INT 中包含该字符，则将其拼接到 sb 中输出
                if (CHAR_TO_INT.containsKey(chars[i])) sb.append(chars[i]);
            }
            sb.append("\n");
        }

        return sb.toString();
    }

    /**
     * Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
     * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
     * Note that the initalization is used for all samples
     *
     * @param initialization     String, may be null. If null, select a random character as initialization for all samples
     * @param charactersToSample Number of characters to sample from network (excluding initialization)
     * @param net                MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
     */
    private static String[] sampleCharactersFromNetwork(String initialization, MultiLayerNetwork net, Random rng,
                                                        Map<Integer, Character> intToChar, int charactersToSample, int numSamples) {
        //Set up initialization. If no initialization: use a random character
        if (initialization == null) {
            int randomCharIdx = rng.nextInt(intToChar.size());
            initialization = String.valueOf(intToChar.get(randomCharIdx));
        }

        //Create input for initialization
        INDArray initializationInput = Nd4j.zeros(numSamples, intToChar.size(), initialization.length());
        char[] init = initialization.toCharArray();
        for (int i = 0; i < init.length; i++) {
            int idx = CHAR_TO_INT.get(init[i]);
            for (int j = 0; j < numSamples; j++) {
                initializationInput.putScalar(new int[]{j, idx, i}, 1.0f);
            }
        }

        StringBuilder[] sb = new StringBuilder[numSamples];
        for (int i = 0; i < numSamples; i++) sb[i] = new StringBuilder(initialization);

        //Sample from network (and feed samples back into input) one character at a time (for all samples)
        //Sampling is done in parallel here
        net.rnnClearPreviousState();
        INDArray output = net.rnnTimeStep(initializationInput);
        output = output.tensorAlongDimension(output.size(2) - 1, 1, 0);    //Gets the last time step output

        for (int i = 0; i < charactersToSample; i++) {
            //Set up next input (single time step) by sampling from previous output
            INDArray nextInput = Nd4j.zeros(numSamples, intToChar.size());
            //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
            for (int s = 0; s < numSamples; s++) {
                double[] outputProbDistribution = new double[intToChar.size()];
                for (int j = 0; j < outputProbDistribution.length; j++)
                    outputProbDistribution[j] = output.getDouble(s, j);
                int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution, rng);

                nextInput.putScalar(new int[]{s, sampledCharacterIdx}, 1.0f);        //Prepare next time step input
                sb[s].append(intToChar.get(sampledCharacterIdx));    //Add sampled character to StringBuilder (human readable output)
            }

            output = net.rnnTimeStep(nextInput);    //Do one time step of forward pass
        }

        String[] out = new String[numSamples];
        for (int i = 0; i < numSamples; i++) out[i] = sb[i].toString();
        return out;
    }

    /**
     * Given a probability distribution over discrete classes, sample from the distribution
     * and return the generated class index.
     *
     * @param distribution Probability distribution over classes. Must sum to 1.0
     */
    private static int sampleFromDistribution(double[] distribution, Random rng) {
        double d = rng.nextDouble();
        double sum = 0.0;
        for (int i = 0; i < distribution.length; i++) {
            sum += distribution[i];
            if (d <= sum) return i;
        }
        //Should never happen if distribution is a valid probability distribution
        throw new IllegalArgumentException("Distribution is invalid? d=" + d + ", sum=" + sum);
    }

    /**
     * A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc
     */
    /**
     * 获取英文字符串数组
     *
     * @return
     */
    private static char[] getValidCharacters() {
        List<Character> validChars = new LinkedList<Character>();
        for (char c = 'a'; c <= 'z'; c++) validChars.add(c);
        for (char c = 'A'; c <= 'Z'; c++) validChars.add(c);
        for (char c = '0'; c <= '9'; c++) validChars.add(c);
        char[] temp = {'!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t'};
        for (char c : temp) validChars.add(c);
        // 将 list 转换为 字符串数组
        char[] out = new char[validChars.size()];
        int i = 0;
        for (Character c : validChars) out[i++] = c;
        return out;
    }

    /**
     * 根据下标int 转换为 char
     *
     * @return
     */
    public static Map<Integer, Character> getIntToChar() {
        Map<Integer, Character> map = new HashMap<Integer, Character>();
        char[] chars = getValidCharacters();
        for (int i = 0; i < chars.length; i++) {
            map.put(i, chars[i]);
        }
        return map;
    }

    /**
     * 将字符串转换为int
     *
     * @return
     */
    public static Map<Character, Integer> getCharToInt() {
        Map<Character, Integer> map = new HashMap<Character, Integer>();
        // 获取英文的字符串
        char[] chars = getValidCharacters();
        // map 加载数据，字符串----字符串对应的下标。
        for (int i = 0; i < chars.length; i++) {
            map.put(chars[i], i);
        }
        return map;
    }
}
