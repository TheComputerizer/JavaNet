package mods.thecomputerizer.javanet.util;

import org.apache.commons.math3.util.FastMath;

import static org.apache.commons.math3.util.FastMath.E;

public class FunctionHelper {
    
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