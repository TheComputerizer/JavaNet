package mods.thecomputerizer.javanet.neuralnet;

import mods.thecomputerizer.javanet.neuron.Neuron;
import mods.thecomputerizer.javanet.training.AbstractTrainable;
import mods.thecomputerizer.javanet.util.FunctionHelper;
import mods.thecomputerizer.javanet.util.MNIST;
import mods.thecomputerizer.javanet.util.MNIST.DigitData;
import mods.thecomputerizer.javanet.layer.Layer;
import mods.thecomputerizer.javanet.util.NNIO;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

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
    public NeuralNet(Layer[] layers, double weightRadius, double biasRadius) {
        this.layers = layers;
        try(Random random = new DefaultRandom()) {
            for(int i=0;i<layers.length;i++) {
                Layer l = layers[i];
                l.initializeNeurons(i);
                l.randomize(random,biasRadius,weightRadius);
            }
            getOutputLayer().setFunctions(v -> v,v -> v);
            getHiddenLayers()[1].setFunctions(FunctionHelper::relu,FunctionHelper::reluDerivative);
        } catch(Exception ex) {
            throw new RuntimeException("Failed to initialize CpuNativeRandom",ex);
        }
        load(NNIO.getTrainingData("trained_data"));
    }
    
    /**
     * Assumes the scores have not yet been derivated
     */
    public void backPropagate(RealVector errors) {
        this.layers[this.layers.length-1].backPropagate(errors); //Start back propagating from the output layer
    }
    
    private RealVector feedForward(RealVector inputs) {
        return getInputLayer().feedForward(inputs);
    }
    
    /**
     * Runs the inputs through the network and checks the output against the expected output
     * Returns the margin of error for each output neuron
     */
    public RealVector forwardCost(double[] inputs, RealVector expected) {
        RealVector outputs = feedForward(new ArrayRealVector(inputs));
        if(outputs.getDimension()!=expected.getDimension())
            throw new RuntimeException("Training output mismatch! Expected "+expected.getDimension()+" "+
                                       "values but got "+outputs.getDimension());
        return outputs;
    }
    
    public Layer[] getHiddenLayers() {
        Layer[] hidden = new Layer[this.layers.length-2];
        System.arraycopy(this.layers,1,hidden,0,this.layers.length-2);
        return hidden;
    }
    
    public Layer getInputLayer() {
        return this.layers[0];
    }
    
    public Layer getOutputLayer() {
        return this.layers[this.layers.length-1];
    }
    
    public int getTrainingDataSize() {
        Neuron last = getOutputLayer().getLastNeuron();
        return last.getTotalIndex()+last.getInputWeights().length+1;
    }
    
    @Override public void load(RealVector data) {
        int dataSize = data.getDimension();
        if(dataSize==0) {
            LOGGER.info("Skipping load for empty training data set");
            return;
        }
        for(Layer layer : this.layers) layer.load(data);
    }
    
    public RealVector savedTrainingData() {
        RealVector data = new ArrayRealVector(getTrainingDataSize());
        store(data);
        return data;
    }
    
    @Override public void store(RealVector data) {
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
    }
    
    private int test(DigitData digit, int index, int right) {
        RealVector outputs = forwardCost(digit.getData(),new ArrayRealVector(digit.getExpectedActivation()));
        int expected = digit.getExpected();
        int actual = FunctionHelper.maxIndex(outputs.toArray());
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
        RealVector expected = new ArrayRealVector(digit.getExpectedActivation());
        RealVector outputs = forwardCost(digit.getData(),expected);
        RealVector costs = FunctionHelper.crossEntropyLoss(outputs,expected);
        if(index%1000==0)
            LOGGER.info("Training cycle {}: Cost = {}\n\t\texpected = {}\n\t\tactual = {}\n",index,
                        FunctionHelper.average(costs),expected,outputs);
        backPropagate(outputs.subtract(expected));
    }
    
    /**
     * Builder to simplify setting up the structure of the neural network
     */
    public static class Builder {
        
        private final int initialLayer;
        private final int finalLayer;
        private final int[] hiddenLayers;
        private double weightRadius;
        private double biasRadius;
        
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
            return new NeuralNet(layers,this.weightRadius,this.biasRadius);
        }
        
        public Builder setBiasRadius(double radius) {
            this.biasRadius = radius;
            return this;
        }
        
        public Builder setWeightRadius(double radius) {
            this.weightRadius = radius;
            return this;
        }
    }
}
