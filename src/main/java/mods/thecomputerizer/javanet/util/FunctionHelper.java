package mods.thecomputerizer.javanet.util;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.rng.Random;

import java.util.Arrays;

public class FunctionHelper {
    
    private static final double ALPHA = 0.01d;  // Leaky factor
    private static final double EPSILON = 1e-15; //Cross-Entropy Loss clipping to avoid log(1) & log(0)
    
    public static double average(RealVector vector) {
        return sum(vector)/((double)vector.getDimension());
    }
    
    public static double averageColumn(RealMatrix matrix, int column) {
        double size = matrix.getRowDimension();
        if(size==0d) return 0d;
        double sum = 0d;
        for(double value : matrix.getColumn(column)) sum+=value;
        return sum/size;
    }
    
    public static double averageRow(RealMatrix matrix, int row) {
        double size = matrix.getColumnDimension();
        if(size==0d) return 0d;
        double sum = 0d;
        for(double value : matrix.getRow(row)) sum+=value;
        return sum/size;
    }
    
    public static RealVector crossEntropyLoss(RealVector outputs, RealVector targets) {
        RealVector costs = new ArrayRealVector(outputs.getDimension());
        for(int i=0;i<outputs.getDimension();i++)
            costs.setEntry(i,crossEntropyLoss(outputs.getEntry(i),targets.getEntry(i)));
        return costs;
    }
    
    public static double crossEntropyLoss(double output, double target) {
        output = FastMath.max(EPSILON,FastMath.min(1d-EPSILON,output));
        return -(target*FastMath.log(output)+(1d-target)*FastMath.log(1d-output));
    }
    
    public static double heInit(Random random, double fanIn, double fanOut) {
        return random.nextGaussian()*FastMath.sqrt(2d/(fanIn+fanOut));
    }
    
    
    public static int maxIndex(double[] values) {
        if(values.length==0) return -1;
        if(values.length==1) return 0;
        int index = 0;
        for(int i=1;i<values.length;i++)
            if(values[i]>values[index]) index = i;
        return index;
    }
    
    public static double relu(double value) {
        return FastMath.max(value,0d);
    }
    
    public static double reluDerivative(double value) {
        return value>0d ? 1d : 0d;
    }
    
    public static double sigmoid(double value) {
        return new Sigmoid().value(value);
    }
    
    /**
     * dS/dx = S(x)*(1-S(x))
     */
    public static double sigmoidDerivative(double value) {
        return value*(1d-value);
    }
    
    public static RealVector softmax(RealVector input) {
        // For numerical stability subtract the max value
        RealVector expValues = input.copy().map(FastMath::exp);
        double sum = sum(expValues);
        expValues.map(value -> value/sum);
        return expValues;
    }
    
    public static double sum(RealVector vector) {
        return Arrays.stream(vector.toArray()).sum();
    }
}