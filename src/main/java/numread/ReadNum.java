package numread;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created with IntelliJ IDEA.
 * User: hui
 * Date: 2018/3/27 21:57
 * Version: V1.0
 * To change this template use File | Settings | File Templates.
 * Description:   dl4j 手写体数字识别 （1）
 */
public class ReadNum {

    public static void main(String[] args) throws Exception{

        // 初始化模型参数
        int nChannels = 1;      //black & white picture, 3 if color image
        int outputNum = 10;     //number of classification
        int batchSize = 64;     //mini batch size for sgd
        int nEpochs = 10;       //total rounds of training
        int iterations = 1;     //number of iteration in each traning round
        int seed = 123;         //random seed for initialize weights

        // 获取训练姐和测试集
        DataSetIterator mnistTrain = null;
        DataSetIterator mnistTest = null;
        mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);
        mnistTest = new MnistDataSetIterator(batchSize, false, 12345);

        // 初始化模型结构
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)   // 定型的迭代次数
                .regularization(true).l2(0.0005)  // regularization表示采用正则化，L2正则化
//                步幅，亦即在搜索空间中移动时改变参数向量的速度（学习速率越大，得到最终结果的速度越快，但有可能错过最佳值；速率较小，所需的定型时间可能会大幅增加） 优化函数的变化速率
                .learningRate(0.01)//.biasLearningRate(0.02)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation("identity")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation("identity")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation("relu")
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false)
                .cnnInputSize(28, 28, 1);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1));

        // 进行模型训练
        for( int i = 0; i < nEpochs; ++i ) {
            // 进行模型训练
            model.fit(mnistTrain);
            System.out.println("*** Completed epoch " + i + "***");
            System.out.println("Evaluate model....");
            Evaluation eval = new Evaluation(outputNum);
            while(mnistTest.hasNext()){
                DataSet ds = mnistTest.next();
                // 模型预测输出
                INDArray output = model.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), output);
            }
            System.out.println(eval.stats());
            mnistTest.reset();
        }

    }

}
