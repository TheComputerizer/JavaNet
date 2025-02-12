package mods.thecomputerizer.javanet.neuron;

import lombok.Getter;
import lombok.Setter;

@Getter @Setter
public class Connection {
    
    private final Neuron parent; // input neuron
    private final Neuron output;
    private int trainingIndex;
    private double weight;
    
    public Connection(Neuron parent, Neuron output, double weight) {
        this.parent = parent;
        this.output = output;
        this.weight = weight;
        parent.addForwardConnection(this);
    }
}
