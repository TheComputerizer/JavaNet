package mods.thecomputerizer.javanet.layer;

import lombok.Getter;
import lombok.Setter;
import mods.thecomputerizer.javanet.neuron.Connection;
import mods.thecomputerizer.javanet.neuron.Neuron;
import mods.thecomputerizer.javanet.training.AbstractTrainable;
import mods.thecomputerizer.javanet.util.FunctionHelper;
import org.apache.commons.math3.linear.RealVector;
import org.nd4j.linalg.api.rng.Random;

import java.util.List;
import java.util.Objects;

@Getter
public class Layer extends AbstractTrainable {
    
    private static final double BIAS_TRAINING_FACTOR = 0.01d;
    private static final double WEIGHT_TRAINING_FACTOR = 0.1d;
    
    private final Neuron[] neurons;
    @Setter private Layer nextLayer;
    private LayerConnection connection;
    
    public Layer(int size) {
        this.neurons = new Neuron[size];
    }
    
    /**
     * Assumes the scores have already been derivated
     */
    public void backPropagate(RealVector offset, double ... dScores) {
        if(Objects.isNull(this.nextLayer))
            this.connection.getFrom().backPropagate(offset,backPropagateOutput(offset,dScores));
        else {
            double[] outputs = new double[this.neurons.length];
            for(int i=0;i<this.neurons.length;i++) {
                Neuron neuron = this.neurons[i];
                List<Connection> connections = neuron.getForwardConnections();
                double sum = 0d;
                for(int c=0;c<connections.size();c++) {
                    double dActivation = FunctionHelper.sigmoidDerivative(neuron.getActivationValue());
                    sum+=(dScores[c]*connections.get(c).getWeight()*dActivation);
                }
                outputs[i] = sum;
                //I think this just works as is for the bias gradient?
                offset.setEntry(neuron.getStartingIndex(),outputs[i]*BIAS_TRAINING_FACTOR);
            }
            if(Objects.nonNull(this.connection)) {
                Layer from = this.connection.getFrom();
                Neuron[] neuronsFrom = from.getNeurons();
                double[] actualOutputsForReal = new double[neuronsFrom.length];
                for(int i=0;i<neuronsFrom.length;i++) {
                    Neuron neuron = from.neurons[i];
                    double sum = 0d;
                    List<Connection> connections = neuron.getForwardConnections();
                    for(int c=0;c<outputs.length;c++) {
                        Connection forward = connections.get(c);
                        sum+=(outputs[c]*forward.getWeight());
                        double value = outputs[c]*neuron.getActivationValue()*WEIGHT_TRAINING_FACTOR;
                        offset.setEntry(forward.getTrainingIndex(),value);
                    }
                    actualOutputsForReal[i] = sum*FunctionHelper.sigmoidDerivative(neuron.getActivationValue());
                }
                from.backPropagate(offset,actualOutputsForReal);
            }
        }
    }
    
    private double[] backPropagateOutput(RealVector offset, double ... dScores) {
        double[] outputs = new double[neurons.length];
        for(int i=0;i<this.neurons.length;i++) {
            Neuron neuron = this.neurons[i];
            outputs[i] = dScores[i]*FunctionHelper.sigmoidDerivative(neuron.getActivationValue());
            //I think this just works as is for the bias gradient?
            offset.setEntry(neuron.getStartingIndex(),outputs[i]*BIAS_TRAINING_FACTOR);
        }
        Layer from = this.connection.getFrom();
        Neuron[] neuronsFrom = from.getNeurons();
        for(int i=0;i<neuronsFrom.length;i++) {
            Neuron neuron = from.neurons[i];
            List<Connection> connections = neuron.getForwardConnections();
            for(int c=0;c<outputs.length;c++) {
                double value = outputs[c]*neuron.getActivationValue()*WEIGHT_TRAINING_FACTOR;
                offset.setEntry(connections.get(c).getTrainingIndex(),value);
            }
        }
        return outputs;
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
