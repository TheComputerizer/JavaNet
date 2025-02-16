package mods.thecomputerizer.javanet.layer;

import lombok.Getter;
import lombok.Setter;
import mods.thecomputerizer.javanet.training.AbstractTrainable;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Objects;

import static org.nd4j.linalg.api.buffer.DataType.DOUBLE;

@Getter
public class Layer extends AbstractTrainable {
    
    private static final double MOMENTUM = 0.5d;
    private static final double LEARNING_RATE = 0.001d;
    
    private final Layer previous;
    private final INDArray biases;
    private final INDArray biasUpdates;
    private final INDArray weights;
    private final INDArray weightUpdates;
    private final int size;
    @Setter private Layer next;
    @Setter private int index; //Index of this layer in the network used for storing & loading training data
    @Setter private IActivation function;
    private INDArray activationValues;
    
    public Layer(int size) {
        this(null,size);
    }
    
    public Layer(Layer previous, int size) {
        this.previous = previous;
        this.size = size;
        this.function = new ActivationSigmoid();
        this.biases = Nd4j.create(DOUBLE,size);
        this.biasUpdates = Nd4j.zeros(DOUBLE, size);
        this.weights = isInput() ? null : Nd4j.create(DOUBLE,size,this.previous.size);
        this.weightUpdates = isInput() ? null : Nd4j.zeros(DOUBLE,size,this.previous.size);
    }
    
    /**
     * To be a bit more efficient we can use separate update arrays for biases and weight.
     * This makes it possible to calculate gradient values for the entire network against the original
     * bias & weight values while being able to easily update the values after the recursion is finished
     */
    public void backPropagate(INDArray errors) {
        // We don't care about the bias values for the input layer, and it doesn't have any input weights to consider,
        // so we should stop here.
        if(isInput()) return;
        
        // Compute local errors for neurons in this layer.
        // For an output neuron, gradients should be the loss derivative.
        // For a hidden neuron, gradients should be the weighted error from the next layer.
        if(isDifferentiable(this.function)) // error * derivative of each output activation unless using softmax
            // or any other non-differentiable activation function
            errors = applyBackwards(this.function,this.activationValues,errors);
        
        //Apply as is to apply for bias values
        this.biasUpdates.muli(MOMENTUM).addi(errors.mul(LEARNING_RATE));
        
        // The gradient of a single weight is simply the activation value of the incoming neuron
        // multiplied by the gradient of the outgoing neuron.
        this.weightUpdates.muli(MOMENTUM).addi(getPreviousError(errors,this.previous.activationValues));
        
        errors = errors.reshape(1,errors.length()).mmul(this.weights).getRow(0);
        
        // Apply the queued updates to the weight and the bias values now that the original values are no longer needed
        this.biases.subi(this.biasUpdates);
        this.weights.subi(this.weightUpdates);
        
        // Recursively back-propagate to the previous layer.
        this.previous.backPropagate(errors);
    }
    
    /**
     * For each output neuron, get the sum of each input neuron * the weight of the connection + output bias.
     * Apply activation function (sigmoid, reLU, tan, etc.)
     */
    public INDArray feedForward(INDArray activations, boolean training) {
        if(isInput()) {
            this.activationValues = activations;
            return this.next.feedForward(activations,training);
        }
        this.activationValues = applyForward(this.function,this.weights.mmul(activations).addi(biases),training);
        return isOutput() ? this.activationValues : this.next.feedForward(this.activationValues,training);
    }
    
    protected INDArray getPreviousError(INDArray errors, INDArray previousActivations) {
        // Reshape errors to be a column vector and activationValues to be a row vector
        INDArray errorsColumn = errors.reshape(errors.length(), 1);
        INDArray activationsRow = previousActivations.reshape(1,previousActivations.length());
        
        // Compute the outer product and scale it by the learning rate
        return errorsColumn.mmul(activationsRow).mul(LEARNING_RATE);
    }
    
    protected int getTrainingIndex() {
        return isInput() ? 0 : this.previous.getTrainingSize();
    }
    
    /**
     * Recursively
     */
    public int getTrainingSize() {
        return isInput() ? 0 : this.previous.getTrainingSize()+(int)this.biases.length()+(int)this.weights.length();
    }
    
    /**
     * Instantiates all the bias and weight values for this layer
     * Assumes the structure of the net and the next layer for this object have already been set as necessary
     */
    public void initializeNeurons(int index, IWeightInit biasInit, IWeightInit weightInit) {
        if(isInput()) return;
        int previousSize = this.previous.size;
        this.biases.addi(initWeight(Nd4j.ones(this.size),biasInit,this.size,1,this.size));
        this.weights.addi(initWeight(Nd4j.ones(this.size,previousSize),weightInit,previousSize,this.size,this.size,previousSize));
        setIndex(index);
    }
    
    public boolean isInput() {
        return Objects.isNull(this.previous);
    }
    
    public boolean isOutput() {
        return Objects.isNull(this.next);
    }
    
    /**
     * Load from 4D array (layer,type,output,input)
     */
    @Override public void load(INDArray data) {
        if(isInput()) return;
        
        //Reset update values
        this.biasUpdates.muli(0d);
        this.weightUpdates.muli(0d);
        
        long index = getTrainingIndex();
        
        //Add from data matrix to load
        for(int i=0;i<this.size;i++) {
            this.biases.putScalar(i,data.getDouble(index));
            index++;
        }
        for(int i=0;i<this.size;i++) {
            for(int j=0;j<this.previous.size;j++) {
                this.weights.putScalar(i,j,data.getDouble(index));
                index++;
            }
        }
    }
    
    @Override public void store(INDArray data) {
        if(isInput()) return;
        long index = getTrainingIndex();
        //Add from data matrix to load
        for(long l=0;l<this.biases.length();l++) {
            data.putScalar(index,this.biases.getDouble(l));
            index++;
        }
        INDArray flat = this.weights.ravel();
        for(long l=0;l<flat.length();l++) {
            data.putScalar(index,flat.getDouble(l));
            index++;
        }
    }
}