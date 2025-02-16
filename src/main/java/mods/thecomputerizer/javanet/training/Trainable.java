package mods.thecomputerizer.javanet.training;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Trainable {
    
    void load(INDArray data);
    void store(INDArray data);
}