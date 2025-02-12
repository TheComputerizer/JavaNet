package mods.thecomputerizer.javanet.layer;

import lombok.Getter;
import mods.thecomputerizer.javanet.neuron.Connection;
import mods.thecomputerizer.javanet.neuron.Neuron;
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.nd4j.linalg.api.rng.Random;

public class LayerConnection {
    
    @Getter private final Layer from;
    private final Layer to;
    private final Array2DRowRealMatrix weights;
    @Getter private final RealVector activationValues;
    private final RealVector biases;
    
    public LayerConnection(Layer from, Layer to, Random random, double weightRadius) {
        this.from = from;
        this.to = to;
        from.setNextLayer(to);
        int parentCount = from.getNeurons().length;
        int targetCount = to.getNeurons().length;
        this.weights = new Array2DRowRealMatrix(targetCount,from.getNeurons().length);
        this.activationValues = new ArrayRealVector(parentCount);
        this.biases = new ArrayRealVector(targetCount);
        populate(random,weightRadius);
    }
    
    private void populate(Random random, double weightRadius) {
        Neuron[] neuronsTo = this.to.getNeurons();
        Neuron[] neuronsFrom = this.from.getNeurons();
        for(int row = 0;row<neuronsTo.length;row++)
            populate(row,neuronsTo[row],neuronsFrom,random,weightRadius);
        for(int col=0;col<neuronsFrom.length;col++)
            this.activationValues.setEntry(col,neuronsFrom[col].getActivationValue());
    }
    
    private void populate(int row, Neuron neuron, Neuron[] neuronsFrom, Random random, double weightRadius) {
        int parentCount = neuronsFrom.length;
        neuron.populateConnections(parentCount);
        for(int col=0;col<parentCount;col++) {
            Neuron from = neuronsFrom[col];
            Connection connection = neuron.addConnection(col,from,random,weightRadius);
            this.weights.setEntry(row,col,connection.getWeight());
        }
    }
    
    /**
     * Set the activation values of the parent layer
     */
    private void inputActivationValues(double ... values) {
        Neuron[] neuronsFrom = this.from.getNeurons();
        if(values.length!=neuronsFrom.length)
            throw new RuntimeException("Input activation value mismatch! Expected "+neuronsFrom.length+
                                       " values but got "+values.length);
        for(int i=0;i<values.length;i++) {
            double value = values[i];
            neuronsFrom[i].setActivationValue(value);
            this.activationValues.setEntry(i,value);
        }
    }
    
    public void reloadConnections() {
        Neuron[] neuronsTo = this.to.getNeurons();
        for(int row=0;row<neuronsTo.length;row++) {
            Neuron neuron = neuronsTo[row];
            this.biases.setEntry(row,neuron.getBias());
            Connection[] connections = neuron.getParentConnections();
            for(int col=0;col<connections.length;col++)
                this.weights.setEntry(row,col,connections[col].getWeight());
        }
    }
    
    /**
     * Set the activation values of the parent layer and then use those to calculate the output values
     * Currently uses Sigmoid to normalize outputs
     */
    public double[] run(double ... values) {
        inputActivationValues(values);
        return this.weights.operate(this.activationValues).add(this.biases).map(new Sigmoid()).toArray();
    }
}