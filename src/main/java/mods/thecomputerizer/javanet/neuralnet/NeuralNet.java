package mods.thecomputerizer.javanet.neuralnet;

import mods.thecomputerizer.javanet.training.AbstractTrainable;
import mods.thecomputerizer.javanet.util.MNIST;
import mods.thecomputerizer.javanet.util.MNIST.DigitData;
import mods.thecomputerizer.javanet.layer.Layer;
import mods.thecomputerizer.javanet.layer.LayerConnection;
import mods.thecomputerizer.javanet.util.NNIO;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * TODO Train & Test!
 */
public class NeuralNet extends AbstractTrainable {
    
    private static final Logger LOGGER = LoggerFactory.getLogger("JavaNet NeuralNet");
    private static final int OFFSET_AVERAGE = 50; //Apply backpropagation offsets every x training cycles
    
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
        try(Random r = new DefaultRandom()) {
            random = r;
            for(Layer layer : layers) layer.populateBias(random,biasRadius);
        } catch(Exception ex) {
            throw new RuntimeException("Failed to initialize CpuNativeRandom",ex);
        }
        this.layers = layers;
        this.connections = new LayerConnection[layers.length-1];
        connectLayers(random,weightRadius);
        initScope(0);
        load(NNIO.getTrainingData("trained_data"));
    }
    
    /**
     * Averages & applies offset gradients gathered via back propagation
     */
    RealVector applyOffsets(RealVector trainingData, List<RealVector> offsets, int dataSize, boolean ignoreSize) {
        if(offsets.isEmpty()) return trainingData;
        int offsetSize = offsets.size();
        if(ignoreSize || offsetSize>=OFFSET_AVERAGE) {
            RealVector averageOffset = new ArrayRealVector(dataSize);
            boolean somethingChanged = false;
            for(int i=0;i<dataSize;i++) {
                double average = averageAt(offsets,offsetSize,i);
                if(average!=0d) {
                    averageOffset.setEntry(i,average);
                    somethingChanged = true;
                }
            }
            if(somethingChanged) {
                trainingData = trainingData.add(averageOffset);
                load(trainingData);
            }
            offsets.clear();
        }
        return trainingData;
    }
    
    double averageAt(List<RealVector> offsets, int size, int index) {
        double total = 0d;
        for(RealVector offset : offsets) total+=(offset.getEntry(index));
        return total/((double)size);
    }
    
    /**
     * Assumes the scores have not yet been derivated
     */
    public void backPropagate(RealVector offset, double ... scores) {
        double[] dScores = new double[scores.length];
        for(int i=0;i<scores.length;i++) dScores[i] = FastMath.sqrt(scores[i])*2d; //We just need the difference
        this.layers[this.layers.length-1].backPropagate(offset,dScores); //Start back propagating from the output layer
    }
    
    /**
     * Set up connections between layers
     */
    private void connectLayers(Random random, double weightRadius) {
        for(int i=0;i<this.connections.length;i++)
            this.connections[i] = this.layers[i+1].setConnection(this.layers[i],random,weightRadius);
    }
    
    @Override public int dataSize() {
        int size = 0;
        for(Layer layer : this.layers) size+=layer.dataSize();
        return size;
    }
    
    public double getCost(double ... scores) {
        double sum = 0;
        for(double score : scores) sum+=score;
        return FastMath.sqrt(sum/(scores.length-1));
    }
    
    public Layer getOutputLayer() {
        return this.layers[this.layers.length-1];
    }
    
    @Override public int initScope(int index) {
        for(Layer layer : this.layers) index = layer.initScope(index);
        return index;
    }
    
    @Override protected void loadFrom(RealVector data, int index) {
        if(data.getDimension()==0) {
            LOGGER.debug("Skipping load for empty training data set");
            return;
        }
        for(Layer layer : this.layers) layer.load(data);
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
        //Activation values in the output layer need to be updated so back propagation can work
        double[] finalValues = runLayer(index+1,connection.run(values));
        for(int i=0;i<finalValues.length;i++) getOutputLayer().getNeurons()[i].setActivationValue(finalValues[i]);
        return finalValues;
    }
    
    public RealVector saveTrainingData() {
        RealVector data = new ArrayRealVector(dataSize());
        store(data);
        return data;
    }
    
    /**
     * Runs the inputs through the network and checks the output against the expected output
     * Returns the margin of error for each output neuron
     */
    public double[] scores(double[] inputs, double ... expectedOutputs) {
        double[] outputs = run(inputs);
        if(outputs.length!=expectedOutputs.length)
            throw new RuntimeException("Training output mismatch! Expected "+expectedOutputs.length+" "+
                                       "values but got "+outputs.length);
        double[] scores = new double[outputs.length];
        for(int i=0;i<outputs.length;i++) scores[i] = FastMath.pow(outputs[i]-expectedOutputs[i],2d);
        return scores;
    }
    
    @Override protected void storeFrom(RealVector data, int index) {
        for(Layer layer : this.layers) layer.store(data);
    }
    
    public void test() {
        List<DigitData> digits = MNIST.readTesting();
        LOGGER.info("Running MNIST test with {} digits",digits.size());
        int i=1;
        for(DigitData digit : digits) {
            double[] scores = scores(digit.getData(),digit.getExpectedActivation());
            if(i%25==0) {
                LOGGER.info("Test {}: Cost = {} (from {})", i, getCost(scores), Arrays.toString(scores));
            }
            i++;
        }
    }
    
    public void train() {
        RealVector trainingData = saveTrainingData();
        LOGGER.info("Training data size is {}",trainingData.getDimension());
        List<DigitData> digits = MNIST.readTraining();
        LOGGER.info("Running MNIST training with {} digits",digits.size());
        int i=1;
        List<RealVector> offsets = new ArrayList<>();
        int size = dataSize();
        for(DigitData digit : digits) {
            trainingData = train(trainingData,digit,i,size,offsets);
            i++;
        }
        if(!offsets.isEmpty()) trainingData = applyOffsets(trainingData,offsets,size,true);
        LOGGER.info("Finished MNIST training cycle! Writing data to file");
        //NNIO.writeTrainingData("trained_data",trainingData);
    }
    
    private RealVector train(RealVector trainingData, DigitData digit, int index, int dataSize, List<RealVector> offsets) {
        double[] scores = scores(digit.getData(),digit.getExpectedActivation());
        if(index%250==0) {
            LOGGER.info("Training cycle {}: Cost = {} (from {})",index,getCost(scores),Arrays.toString(scores));
        }
        RealVector offset = new ArrayRealVector(dataSize);
        backPropagate(offset,scores);
        offsets.add(offset);
        return applyOffsets(trainingData,offsets,dataSize,false);
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
