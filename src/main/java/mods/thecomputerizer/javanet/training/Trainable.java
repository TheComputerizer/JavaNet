package mods.thecomputerizer.javanet.training;

import org.apache.commons.math3.linear.RealVector;

public interface Trainable {
    
    /**
     * Input index is total the amount of training data read in so far.
     * Returns the new total after initialization
     */
    int initScope(int index);
    void load(RealVector data);
    int dataSize();
    void store(RealVector data);
}