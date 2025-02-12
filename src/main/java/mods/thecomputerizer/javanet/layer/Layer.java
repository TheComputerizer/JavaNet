package mods.thecomputerizer.javanet.layer;

import lombok.Getter;
import lombok.Setter;
import mods.thecomputerizer.javanet.neuron.Neuron;
import mods.thecomputerizer.javanet.training.AbstractTrainable;
import mods.thecomputerizer.javanet.util.FunctionHelper;
import org.apache.commons.math3.linear.RealVector;
import org.nd4j.linalg.api.rng.Random;

import java.util.Objects;

@Getter
public class Layer extends AbstractTrainable {
    
    private final Neuron[] neurons;
    @Setter private Layer nextLayer;
    private LayerConnection connection;
    
    public Layer(int size) {
        this.neurons = new Neuron[size];
    }
    
    /**
     * Assumes the scores have already been derivated
     */
    @Override public double backPropagate(RealVector offset, double ... dScores) {
        double[] outputScores = new double[dScores.length]; //Scores for every neuron in this layer
        for(int i=0;i<this.neurons.length;i++) {
            Neuron neuron = this.neurons[i];
            double score = dScores[i]*FunctionHelper.sigmoidDerivative(neuron.getActivationValue());
            outputScores[i] = score;
            offset.setEntry(neuron.getStartingIndex(),score); //I think this just works as is for the bias gradient?
        }
        if(Objects.nonNull(this.connection)) {
            Layer from = this.connection.getFrom();
            Neuron[] neuronsFrom = from.getNeurons();
            double[] parentScores = new double[neuronsFrom.length]; //Scores for every neuron in the parent layer
            for(int i=0;i<neuronsFrom.length;i++) {
                Neuron neuron = from.neurons[i];
                parentScores[i] = neuron.backPropagate(offset,outputScores);
            }
            from.backPropagate(offset,parentScores); //Propagate to the parent layer
        }
        return 0d; //We don't need to care about the output here
    }
    
    @Override public int dataSize() {
        int size = 0;
        for(Neuron neuron : this.neurons) size+=neuron.dataSize();
        return size;
    }
    
    @Override public int initScope(int index) {
        for(Neuron neuron : this.neurons) index = neuron.initScope(index);
        return index;
    }
    
    @Override protected void loadFrom(RealVector data, int index) {
        for(Neuron neuron : this.neurons) neuron.load(data);
        if(Objects.nonNull(this.connection)) this.connection.reloadConnections();
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
    
    @Override protected void storeFrom(RealVector data, int index) {
        for(Neuron neuron : this.neurons) neuron.store(data);
    }
}
