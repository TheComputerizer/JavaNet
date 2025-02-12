package mods.thecomputerizer.javanet;

import lombok.Getter;
import lombok.Setter;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.nd4j.linalg.api.rng.Random;

import java.util.List;

@Getter @Setter
public class Neuron {
    
    private double activationValue; //Between 0-1
    private Connection[] connections; //Connections where this neuron is the output
    /**
     * Bias marks a cutoff for how high the weighted sum needs to be before the neuron can be considered meaningfully
     * active. Needs to be applied before normalization occurs
     */
    private double bias;
    
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
        Connection connection = new Connection(from,weight);
        this.connections[index] = connection;
        return connection;
    }
    
    /**
     * Loads the bias of this neuron and the weights of its connections before returning the next index of the vector
     */
    public int loadTrainingData(ArrayRealVector data, int index) {
        this.bias = data.getEntry(index);
        index++;
        for(Connection connection : this.connections) index = connection.loadTrainingData(data,index);
        return index;
    }
    
    /**
     * Initializes the connections array with the number of neurons in the parent layer
     */
    public void populateConnections(int count) {
        this.connections = new Connection[count];
    }
    
    public void setRandomBias(Random random, double range) {
        this.bias = (random.nextDouble()*range*2d)-range;
    }
    
    public void writeTrainingData(List<Double> data) {
        data.add(this.bias);
        for(Connection connection : this.connections) data.add(connection.getWeight());
    }
}
