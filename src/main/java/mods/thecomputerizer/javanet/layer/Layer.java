package mods.thecomputerizer.javanet.layer;

import lombok.Getter;
import lombok.Setter;
import mods.thecomputerizer.javanet.neuron.Neuron;
import mods.thecomputerizer.javanet.training.AbstractTrainable;
import mods.thecomputerizer.javanet.util.FunctionHelper;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.nd4j.linalg.api.rng.Random;

import java.util.Objects;

@Getter
public class Layer extends AbstractTrainable {
    
    private final Neuron[] neurons;
    private final Layer previous;
    @Setter private Layer next;
    private int index; //Index of this layer in the network
    private UnivariateFunction forwardFunc;
    private UnivariateFunction backwardFunc; //Should be at least the 1st derivative of the forward function
    private RealVector activationValues;
    
    public Layer(int size) {
        this(null,size);
    }
    
    public Layer(Layer previous, int size) {
        this.previous = previous;
        this.neurons = new Neuron[size];
        this.forwardFunc = FunctionHelper::sigmoid;
        this.backwardFunc = FunctionHelper::sigmoidDerivative;
    }
    
    private void applyBiasGradients(RealVector gradients) {
        double[] gradientArray = gradients.toArray();
        for(int i=0;i<gradientArray.length;i++) this.neurons[i].applyBiasGradient(gradientArray[i]);
    }
    
    /**
     * The weight matrix (row,col) is (output,input).
     * Each row corresponds to the input weights for a single neuron in the output layer.
     * This is assumed to be called in the output layer.
     */
    private void applyWeightGradients(RealVector previousActivations, RealVector gradients) {
        for(int i=0;i<this.neurons.length;i++)
            this.neurons[i].applyInputWeightGradients(previousActivations.copy().mapMultiply(gradients.getEntry(i)));
    }
    
    /**
     * Bias values for the neurons of THIS layer.
     * Actual bias values are stored in each neuron and should not be stored here
     */
    private RealVector assembleBiases() {
        RealVector biases = new ArrayRealVector(this.neurons.length);
        for(int i=0;i<this.neurons.length;i++) biases.setEntry(i,this.neurons[i].getBias());
        return biases;
    }
    
    private RealMatrix assembleWeights() {
        RealMatrix weights = new Array2DRowRealMatrix(this.neurons.length,this.previous.neurons.length);
        for(int i=0;i<weights.getColumnDimension();i++)
            for(int j=0;j<weights.getRowDimension();j++)
                weights.setEntry(j,i,this.previous.neurons[i].getOutputWeights()[j]);
        return weights;
    }
    
    public void backPropagate(RealVector errors) {
        // We don't care about the bias values for the input layer, and it doesn't have any input weights to consider,
        // so we should stop here.
        if(isInput()) return;
        
        // Compute local errors for neurons in this layer.
        // For an output neuron, gradients should be the loss derivative.
        // For a hidden neuron, gradients should be the weighted error from the next layer.
        RealVector gradients = errors.copy();
        if(!isOutput()) gradients.ebeMultiply(derivatives()); // error * derivative of each output activation
        applyBiasGradients(gradients); //Apply as is to apply for bias values
        
        // The gradient of a single weight is simply the activation value of the incoming neuron
        // multiplied by the gradient of the outgoing neuron.
        applyWeightGradients(this.previous.activationValues,gradients);
        
        // The "hard" part where we need to pass error data to the previous layer for its back propagation step.
        // The number of errors needs to match the number of neurons for the propagating layer, so that is first.
        RealVector previousErrors = new ArrayRealVector(this.previous.neurons.length);
        // Now we need to use the gradients we calculated for this layer to get a sum for each input neuron.
        // Each input error value should be the sum of each gradient value multiplied by the weight of the connection
        // between the input and the output that the gradient corresponds to.
        for(int p=0;p<previousErrors.getDimension();p++) { // p for previous because why not
            Neuron previousNeuron = this.previous.getNeurons()[p];
            double sum = 0d;
            // I find using output weights easier to conceptualize here
            double[] outputWeights = previousNeuron.getOutputWeights();
            for(int o=0;o<outputWeights.length;o++) // and of course o for output
                sum+=(outputWeights[o]*gradients.getEntry(o));
            previousErrors.setEntry(p,sum);
        }
        
        // Recursively back-propagate to the previous layer.
        this.previous.backPropagate(previousErrors);
    }
    
    /**
     * Map the current activation values of this layer to their derivatives
     */
    private RealVector derivatives() {
        return this.activationValues.copy().map(this.backwardFunc);
    }
    
    /**
     * For each output neuron, get the sum of each input neuron * the weight of the connection + output bias.
     * Apply activation function (sigmoid, reLU, tan, etc.)
     */
    public RealVector feedForward(RealVector activations) {
        if(isInput()) {
            this.activationValues = activations;
            return this.next.feedForward(activations);
        }
        RealVector biases = this.assembleBiases();
        RealMatrix weights = assembleWeights();
        this.activationValues = weights.operate(activations).add(biases).map(this.forwardFunc);
        return isOutput() ? FunctionHelper.softmax(this.activationValues) : this.next.feedForward(this.activationValues);
    }
    
    public Neuron getLastNeuron() {
        return this.neurons[this.neurons.length-1];
    }
    
    /**
     * Instantiates all the neurons in this layer.
     * Assumes the structure of the net and the next layer for this object have already been set as necessary
     */
    public void initializeNeurons(int index) {
        int previousSize = Objects.nonNull(this.previous) ? this.previous.neurons.length : 0;
        int nextSize = Objects.nonNull(this.next) ? this.next.neurons.length : 0;
        for(int i=0;i<this.neurons.length;i++) this.neurons[i] = new Neuron(this,previousSize,nextSize);
        setIndex(index);
    }
    
    public boolean isInput() {
        return Objects.isNull(this.previous);
    }
    
    public boolean isOutput() {
        return Objects.isNull(this.next);
    }
    
    @Override public void load(RealVector data) {
        for(Neuron neuron : this.neurons) neuron.load(data);
    }
    
    public void randomize(Random random, double biasRadius, double weightRadius) {
        for(Neuron neuron : this.neurons) neuron.randomize(random,biasRadius,weightRadius);
    }
    
    public void setFunctions(UnivariateFunction forward, UnivariateFunction backward) {
        this.forwardFunc = forward;
        this.backwardFunc = backward;
    }
    
    private void setIndex(int index) {
        this.index = index;
        int total = 0;
        if(Objects.nonNull(this.previous)) {
            Neuron[] previousNeurons = this.previous.neurons;
            Neuron last = previousNeurons[previousNeurons.length-1];
            total = last.getTotalIndex()+last.getInputWeights().length+(this.previous.isInput() ? 0 : 1);
        }
        for(int i=0;i<this.neurons.length;i++) {
            Neuron n = this.neurons[i];
            n.setLayerIndex(i);
            n.setTotalIndex(total);
            //The total will still end up as 0 if this is the input layer since it doesn't need to save bias
            total+=(n.getInputWeights().length+(isInput() ? 0 : 1));
        }
    }
    
    @Override public void store(RealVector data) {
        for(Neuron neuron : this.neurons) neuron.store(data);
    }
}