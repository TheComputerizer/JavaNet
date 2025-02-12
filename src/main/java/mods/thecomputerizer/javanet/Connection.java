package mods.thecomputerizer.javanet;

import lombok.Getter;
import lombok.Setter;
import org.apache.commons.math3.linear.ArrayRealVector;

@Getter @Setter
public class Connection {
    
    private final Neuron parent; // input neuron
    private double weight;
    
    public Connection(Neuron parent, double weight) {
        this.parent = parent;
        this.weight = weight;
    }
    
    /**
     * Loads the weight of the connection and returns the next index of the vector
     */
    public int loadTrainingData(ArrayRealVector data, int index) {
        this.weight = data.getEntry(index);
        return index+1;
    }
}
