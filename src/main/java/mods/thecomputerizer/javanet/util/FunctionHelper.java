package mods.thecomputerizer.javanet.util;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;

import java.util.Collection;

import static org.apache.commons.math3.util.FastMath.E;

public class FunctionHelper {
    
    private static final double ALPHA = 0.01d;  // Leaky factor
    
    public static double average(Collection<Double> values) {
        double size = values.size();
        if(size==0d) return 0d;
        double sum = 0d;
        for(double value : values) sum+=value;
        return sum/size;
    }
    
    public static double average(RealVector vector) {
        double size = vector.getDimension();
        if(size==0d) return 0d;
        double sum = 0d;
        for(double value : vector.toArray()) sum+=value;
        return sum/size;
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
        RealVector costs = outputs.copy();
        for(int i=0;i<outputs.getDimension();i++)
            costs.setEntry(i,crossEntropyLoss(outputs.getEntry(i),targets.getEntry(i)));
        return costs;
    }
    
    public static double crossEntropyLoss(double output, double target) {
        return -(target*FastMath.log(output)+(1-target)*FastMath.log(1-output));
    }
    
    public static double relu(double value) {
        return value>=0d ? value : ALPHA*value;
    }
    
    public static double reluDerivative(double value) {
        return value>=0d ? 1d : ALPHA;
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
    
    public static double sigmoidInverse(double value) {
        return FastMath.log(E,value/(1d-value));
    }
}