package mods.thecomputerizer.javanet.neuralnet;

import mods.thecomputerizer.javanet.render.ImageRender;
import mods.thecomputerizer.javanet.training.AbstractTrainable;
import mods.thecomputerizer.javanet.util.FunctionHelper;
import mods.thecomputerizer.javanet.util.MNIST;
import mods.thecomputerizer.javanet.util.MNIST.DigitData;
import mods.thecomputerizer.javanet.layer.Layer;
import mods.thecomputerizer.javanet.util.NNIO;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.nd4j.linalg.activations.impl.ActivationGELU;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.nd4j.linalg.api.buffer.DataType.FLOAT;

/**
 * TODO Train & Test!
 */
public class NeuralNet extends AbstractTrainable {
    
    private static final Logger LOGGER = LoggerFactory.getLogger("JavaNet NeuralNet");
    
    public static Builder builder(int ... layers) {
        if(layers.length<=1) throw new RuntimeException("Neural network must have at least 2 layers!");
        return new Builder(layers);
    }
    
    private final Layer[] layers;
    
    /**
     * Initialize Layer & LayerConnection arrays
     */
    public NeuralNet(Layer[] layers, IWeightInit biasInit, IWeightInit weightInit) {
        this.layers = layers;
        for(int i=0;i<layers.length;i++) layers[i].initializeNeurons(i,biasInit,weightInit);
        getOutputLayer().setFunction(new ActivationSoftmax());
        this.layers[1].setFunction(new ActivationGELU());
        this.layers[2].setFunction(new ActivationGELU());
        load(NNIO.getTrainingData("trained_data"));
    }
    
    /**
     * Assumes the scores have not yet been derivated
     */
    public void backPropagate(INDArray errors) {
        this.layers[this.layers.length-1].backPropagate(errors); //Start back propagating from the output layer
    }
    
    private INDArray feedForward(INDArray inputs, boolean training) {
        return getInputLayer().feedForward(inputs,training);
    }
    
    /**
     * Runs the inputs through the network and checks the output against the expected output
     * Returns the margin of error for each output neuron
     */
    public INDArray forwardCost(INDArray inputs, INDArray expected, boolean training) {
        INDArray outputs = feedForward(inputs,training);
        if(outputs.length()!=expected.length())
            throw new RuntimeException("Training output mismatch! Expected "+expected.length()+" "+
                                       "values but got "+outputs.length());
        return outputs;
    }
    
    public Layer getInputLayer() {
        return this.layers[0];
    }
    
    public Layer getOutputLayer() {
        return this.layers[this.layers.length-1];
    }
    
    public int getTrainingDataSize() {
        return getOutputLayer().getTrainingSize();
    }
    
    @Override public void load(@Nullable INDArray data) {
        if(Objects.isNull(data)) {
            LOGGER.info("Skipping load for empty training data set");
            return;
        }
        for(Layer layer : this.layers) layer.load(data);
    }
    
    public INDArray savedTrainingData() {
        INDArray data = Nd4j.create(FLOAT,getTrainingDataSize());
        store(data);
        return data;
    }
    
    @Override public void store(INDArray data) {
        for(Layer layer : this.layers) layer.store(data);
    }
    
    public void test() {
        List<DigitData> digits = MNIST.readTesting();
        LOGGER.info("Running MNIST test with {} digits",digits.size());
        List<DigitData> wrong = new ArrayList<>();
        int right = 0;
        for(int i=0;i<digits.size();i++) {
            DigitData digit = digits.get(i);
            int previous = right;
            right = test(digit,i+1,right);
            if(previous==right) wrong.add(digit);
        }
        double percent = (((double)right)/((double)digits.size()))*100d;
        LOGGER.info("Finished MNIST test with success rate of {}%",percent);
        if(!wrong.isEmpty()) ImageRender.INSTANCE.loadAndDisplay(wrong.getFirst().getImageData());
    }
    
    private int test(DigitData digit, int index, int right) {
        INDArray outputs = forwardCost(digit.getData(),digit.getExpectedActivation(),false);
        int expected = digit.getExpected();
        int actual = FunctionHelper.maxIndex(outputs);
        if(index%25==0) LOGGER.info("Testing cycle {}: Expected = {} Actual = {}",index,expected,actual);
        return expected==actual ? right+1 : right;
    }
    
    public void train(int cycles) {
        LOGGER.info("Training data size is {}",getTrainingDataSize());
        List<DigitData> digits = MNIST.readTraining();
        LOGGER.info("Running MNIST training with {} digits for {} cycles",digits.size(),cycles);
        for(int c=0;c<cycles;c++)
            for(int i=0;i<digits.size();i++) train(digits.get(i),(digits.size()*c)+i+1);
        LOGGER.info("Finished MNIST training cycle! Writing data to file");
        NNIO.writeTrainingData("trained_data",savedTrainingData());
    }
    
    private void train(DigitData digit, int index) {
        INDArray expected = digit.getExpectedActivation();
        INDArray outputs = forwardCost(digit.getData(),expected,true);
        INDArray costs = Nd4j.loss().softmaxCrossEntropy(expected,outputs,null);
        if(index%1000==0)
            LOGGER.info("Training cycle {}: Cost = {}\n\t\texpected = {}\n\t\tactual = {}\n",index,
                        costs.meanNumber(),expected,outputs);
        backPropagate(outputs.sub(expected));
    }
    
    /**
     * Builder to simplify setting up the structure of the neural network
     */
    public static class Builder {
        
        private final int initialLayer;
        private final int finalLayer;
        private final int[] hiddenLayers;
        private IWeightInit biasInit;
        private IWeightInit weightInit;
        
        Builder(int ... layers) {
            this.initialLayer = layers[0];
            this.finalLayer = layers[layers.length-1];
            this.hiddenLayers = new int[layers.length-2];
            System.arraycopy(layers,1,this.hiddenLayers,0,layers.length-2);
        }
        
        private void addParents(Layer[] layers) {
            for(int i=layers.length-2;i>=0;i--) {
                Layer l = layers[i];
                l.setNext(layers[i+1]);
            }
        }
        
        public NeuralNet build() {
            Layer[] layers = new Layer[this.hiddenLayers.length+2];
            layers[0] = new Layer(this.initialLayer);
            for(int i=0;i<hiddenLayers.length;i++) layers[i+1] = new Layer(layers[i],this.hiddenLayers[i]);
            layers[layers.length-1] = new Layer(layers[layers.length-2],this.finalLayer);
            addParents(layers);
            return new NeuralNet(layers,this.biasInit,this.weightInit);
        }
        
        public Builder setBiasInit(IWeightInit init) {
            this.biasInit = init;
            return this;
        }
        
        public Builder setWeightInit(IWeightInit init) {
            this.weightInit = init;
            return this;
        }
    }
}
