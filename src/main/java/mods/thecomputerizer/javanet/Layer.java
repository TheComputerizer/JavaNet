package mods.thecomputerizer.javanet;

import lombok.Getter;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.nd4j.linalg.api.rng.Random;

import java.util.List;
import java.util.Objects;

@Getter
public class Layer {
    
    private final Neuron[] neurons;
    private LayerConnection connection;
    
    public Layer(int size) {
        this.neurons = new Neuron[size];
    }
    
    /**
     * Get the gradient
     */
    public double backPropagate() {
    
    }
    
    /**
     * Loads the weights & biases of this layer from a vector and returns the next index of the vector
     */
    public int loadTrainingData(ArrayRealVector data, int index) {
        for(Neuron neuron : this.neurons) index = neuron.loadTrainingData(data,index);
        if(Objects.nonNull(this.connection)) this.connection.reloadConnections();
        return index;
    }
    
    public void populateBias(Random random, double radius) {
        for(int i=0;i<this.neurons.length;i++) {
            Neuron neuron = new Neuron();
            neuron.setRandomBias(random,radius);
            this.neurons[i] = neuron;
        }
    }
    
    public LayerConnection setConnection(Layer parent, Random random, double weightRadius) {
        this.connection = new LayerConnection(parent,this,random,weightRadius);
        return this.connection;
    }
    
    public void writeTrainingData(List<Double> dataList) {
        for(Neuron neuron : this.neurons) neuron.writeTrainingData(dataList);
    }
}
