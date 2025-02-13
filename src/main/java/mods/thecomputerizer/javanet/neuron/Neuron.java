package mods.thecomputerizer.javanet.neuron;

import lombok.Getter;
import lombok.Setter;
import mods.thecomputerizer.javanet.training.AbstractTrainable;
import org.apache.commons.math3.linear.RealVector;
import org.nd4j.linalg.api.rng.Random;

import java.util.ArrayList;
import java.util.List;

@Getter @Setter
public class Neuron extends AbstractTrainable {
    
    private double activationValue; //Between 0-1
    private List<Connection> forwardConnections; //Needed for back propagation
    private Connection[] parentConnections; //Connections where this neuron is the output
    /**
     * Bias marks a cutoff for how high the weighted sum needs to be before the neuron can be considered meaningfully
     * active. Needs to be applied before normalization occurs
     */
    private double bias;
    
    public Neuron() {
        this.forwardConnections = new ArrayList<>();
        this.parentConnections = new Connection[0];
    }
    
    /**
     * Add a connection from a neuron in the parent layer with a random weight
     */
    public Connection addConnection(int index, Neuron from, Random random, double radius) {
        return addConnection(index,from,(random.nextDouble()*radius*2d)-radius);
    }
    
    /**
     * Add a weighted connection from a neuron in the parent layer
     */
    public Connection addConnection(int index, Neuron from, double weight) {
        Connection connection = new Connection(from,this,weight);
        this.parentConnections[index] = connection;
        return connection;
    }
    
    public void addForwardConnection(Connection connection) {
        this.forwardConnections.add(connection);
    }
    
    @Override public int dataSize() {
        return this.parentConnections.length+1;
    }
    
    @Override public int initScope(int index) {
        int ret = super.initScope(index);
        for(int i=0;i<this.parentConnections.length;i++) this.parentConnections[i].setTrainingIndex(index+i+1);
        return ret;
    }
    
    @Override protected void loadFrom(RealVector data, int index) {
        this.bias = data.getEntry(index);
        for(Connection connection : this.parentConnections)
            connection.setWeight(data.getEntry(connection.getTrainingIndex()));
    }
    
    /**
     * Initializes the connections array with the number of neurons in the parent layer
     */
    public void populateConnections(int count) {
        this.parentConnections = new Connection[count];
    }
    
    public void setRandomBias(Random random, double range) {
        this.bias = (random.nextDouble()*range*2d)-range;
    }
    
    @Override protected void storeFrom(RealVector data, int index) {
        data.setEntry(index,this.bias);
        for(Connection connection : this.parentConnections)
            data.setEntry(connection.getTrainingIndex(),connection.getWeight());
    }
}
