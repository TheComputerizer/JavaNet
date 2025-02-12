package mods.thecomputerizer.javanet.training;

import org.apache.commons.math3.linear.RealVector;

public interface Trainable {
    
    /**
     * Compute the backwards propagation in a "forwards" manner.
     * That is to say; when finding gradient data for a single non-final neuron, we don't care about the parent neurons.
     * In this way the gradient value of a neuron can be calculated via its output neurons after a training cycle.
     */
    double backPropagate(RealVector offset, double ... scores);
    
    /**
     * Input index is total the amount of training data read in so far.
     * Returns the new total after initialization
     */
    int initScope(int index);
    void load(RealVector data);
    int dataSize();
    void store(RealVector data);
}