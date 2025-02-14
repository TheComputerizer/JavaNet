package mods.thecomputerizer.javanet.training;

import org.apache.commons.math3.linear.RealVector;

public interface Trainable {
    
    void load(RealVector data);
    void store(RealVector data);
}