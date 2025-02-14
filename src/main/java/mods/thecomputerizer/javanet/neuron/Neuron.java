package mods.thecomputerizer.javanet.neuron;

import lombok.Getter;
import lombok.Setter;
import mods.thecomputerizer.javanet.layer.Layer;
import mods.thecomputerizer.javanet.training.AbstractTrainable;
import org.apache.commons.math3.linear.RealVector;
import org.nd4j.linalg.api.rng.Random;

import java.util.Objects;

@Getter @Setter
public class Neuron extends AbstractTrainable {
    
    private static final double MOMENTUM = 0.5d;
    private static final double LEARNING_RATE = 0.001d;
    
    private final Layer layer; //Parent layer of this neuron
    private double previousBiasGradient;
    private final double[] previousWeightGradients;
    private final double[] inputWeights; //Connections to previous layer
    private final double[] outputWeights; //Connections to the next layer
    private int layerIndex; //Index of this neuron in its layer
    private int totalIndex; //Index of this neuron in the entire network
    /**
     * Bias marks a cutoff for how high the weighted sum needs to be before the neuron can be considered meaningfully
     * active. Needs to be applied before normalization occurs
     */
    private double bias;
    
    public Neuron(Layer layer, int inputCount, int outputCount) {
        this.layer = layer;
        this.inputWeights = new double[inputCount];
        this.outputWeights = new double[outputCount];
        this.previousWeightGradients = new double[inputCount];
    }
    
    public boolean isInputLayer() {
        return this.layer.isInput();
    }
    
    /**
     * Load bias and input weight values from a vector & update output weight for the previous layer
     */
    @Override public void load(RealVector data) {
        if(isInputLayer()) return;
        this.bias = data.getEntry(this.totalIndex);
        for(int i=0;i<this.inputWeights.length;i++) this.inputWeights[i] = data.getEntry(this.totalIndex+i+1);
        updatePreviousLayerOutputWeights();
    }
    
    public void applyBiasGradient(double gradient) {
        double update = (gradient*LEARNING_RATE)+(this.previousBiasGradient*MOMENTUM);
        this.bias = this.bias-update;
        this.previousBiasGradient = update;
    }
    
    public void applyInputWeightGradients(RealVector gradients) {
        double[] array = gradients.toArray();
        for(int i=0;i<array.length;i++) {
            double value = array[i];
            double update = (value*LEARNING_RATE)+(this.previousWeightGradients[i]*MOMENTUM);
            this.inputWeights[i] = this.inputWeights[i]-update;
            this.previousWeightGradients[i] = update;
        }
        updatePreviousLayerOutputWeights();
    }
    
    /**
     * Store bias and input weight values to a vector
     */
    @Override public void store(RealVector data) {
        if(isInputLayer()) return;
        data.setEntry(this.totalIndex,this.bias);
        for(int i=0;i<this.inputWeights.length;i++)
            data.setEntry(this.totalIndex+i+1,this.inputWeights[i]);
    }
    
    public void randomize(Random random, double biasRadius, double weightRadius) {
        this.bias = initRandomly(random,biasRadius);
        for(int i=0;i<this.inputWeights.length;i++) this.inputWeights[i] = initRandomly(random,weightRadius);
        updatePreviousLayerOutputWeights();
    }
    
    /**
     * Update output weight for the previous layer after the input weights for this layer have been changed
     */
    private void updatePreviousLayerOutputWeights() {
        Layer previous = this.layer.getPrevious();
        if(Objects.nonNull(previous)) {
            Neuron[] previousNeurons = previous.getNeurons();
            for(int i=0;i<this.inputWeights.length;i++) //Update output weights for previous layer
                previousNeurons[i].outputWeights[this.layerIndex] = this.inputWeights[i];
        }
    }
}
