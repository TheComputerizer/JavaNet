package mods.thecomputerizer.javanet;

import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import mods.thecomputerizer.javanet.MNIST.DigitData;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.cpu.nativecpu.rng.CpuNativeRandom;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * TODO Needs to be finished before it can work
 * Weights are currently randomized
 * Bias values are currently randomized
 */
public class NeuralNet {
    
    private static final Logger LOGGER = LoggerFactory.getLogger("JavaNet NeuralNet");
    
    public static Builder builder(int ... layers) {
        if(layers.length<=1) throw new RuntimeException("Neural network must have at least 2 layers!");
        return new Builder(layers);
    }
    
    private final LayerConnection[] connections;
    private final Layer[] layers;
    
    /**
     * Initialize Layer & LayerConnection arrays
     */
    public NeuralNet(Layer inputLayer, Layer outputLayer, Layer[] hiddenLayers, double weightRadius, double biasRadius) {
        Layer[] layers = new Layer[hiddenLayers.length+2];
        layers[0] = inputLayer;
        layers[layers.length-1] = outputLayer;
        int i=1;
        for(Layer layer : hiddenLayers) {
            layers[i] = layer;
            i++;
        }
        Random random;
        try(Random r = new CpuNativeRandom()) {
            random = r;
            for(Layer layer : layers) layer.populateBias(random,biasRadius);
        } catch(Exception ex) {
            throw new RuntimeException("Failed to initialize CpuNativeRandom",ex);
        }
        this.layers = layers;
        this.connections = new LayerConnection[layers.length-1];
        connectLayers(random,weightRadius);
        loadTrainingData(NNIO.getTrainingData("trained_data"));
    }
    
    /**
     * Set up connections between layers
     */
    private void connectLayers(Random random, double weightRadius) {
        for(int i=0;i<this.connections.length;i++)
            this.connections[i] = this.layers[i+1].setConnection(this.layers[i],random,weightRadius);
    }
    
    /**
     * Takes in data stored as a vector and loads it into the appropriate weights & biases
     */
    public void loadTrainingData(ArrayRealVector data) {
        if(data.getDimension()==0) return;
        int index = 0;
        for(Layer layer : this.layers) index = layer.loadTrainingData(data,index);
    }
    
    /**
     * Inputs the values into the initial layer and returns the output of the final layer
     */
    public double[] run(double ... values) {
        return runLayer(0,values);
    }
    
    /**
     * Use the output of the layer at the given index to run the next layer.
     * Once the final layer is reached the values are returned
     */
    private double[] runLayer(int index, double ... values) {
        if(index>=this.connections.length) return values;
        LayerConnection connection = this.connections[index];
        return runLayer(index+1,connection.run(values));
    }
    
    /**
     * Runs the inputs through the network and checks the output against the expected output
     * Returns the margin of error
     */
    public double score(double[] inputs, double ... expectedOutputs) {
        double[] outputs = run(inputs);
        if(outputs.length!=expectedOutputs.length)
            throw new RuntimeException("Training output mismatch! Expected "+expectedOutputs.length+" "+
                                       "values but got "+outputs.length);
        double sum = 0d;
        for(int i=0;i<outputs.length;i++) sum+=FastMath.pow(outputs[i]-expectedOutputs[i],2d);
        return sum;
    }
    
    public void test() {
        List<DigitData> digits = MNIST.readTesting();
        LOGGER.info("Running MNIST test with {} digits",digits.size());
        int i=1;
        for(DigitData digit : digits) {
            double score = score(digit.getData(),digit.getExpectedActivation());
            if(i%5==0) LOGGER.info("Test {}: Score = {}",i,score);
            i++;
        }
    }
    
    public void train() {
        List<DigitData> digits = MNIST.readTraining();
        LOGGER.info("Running MNIST training with {} digits",digits.size());
        int i=1;
        for(DigitData digit : digits) {
            train(digit,i);
            i++;
        }
    }
    
    private void train(DigitData digit, int index) {
        double[] output = run(digit.getData());
        double[] expected = digit.getExpectedActivation();
        if(output.length!=expected.length)
            throw new RuntimeException("Training output mismatch! Expected "+expected.length+" "+
                                       "values but got "+output.length);
        if(index%5==0) LOGGER.info("Training cycle {}: Score = {}",index,score);
    }
    
    public ArrayRealVector writeTrainingData() {
        List<Double> dataList = new DoubleArrayList();
        for(Layer layer : this.layers) layer.writeTrainingData(dataList);
        return new ArrayRealVector(dataList.toArray(new Double[0]));
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
        
        public NeuralNet build() {
            return new NeuralNet(new Layer(this.initialLayer),new Layer(this.finalLayer),buildHiddenLayers(),
                                 this.weightRadius,this.biasRadius);
        }
        
        Layer[] buildHiddenLayers() {
            if(this.hiddenLayers.length==0) return new Layer[]{};
            Layer[] layers = new Layer[this.hiddenLayers.length];
            for(int i=0;i<this.hiddenLayers.length;i++) layers[i] = new Layer(this.hiddenLayers[i]);
            return layers;
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
