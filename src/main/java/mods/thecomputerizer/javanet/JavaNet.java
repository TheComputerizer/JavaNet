package mods.thecomputerizer.javanet;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class JavaNet {
    
    private static final Logger LOGGER = LoggerFactory.getLogger("JavaNet Main");
    
    public static void main(String ... args) {
        LOGGER.info("Beginning to take over the world! I mean run some tests");
        digitNet();
    }
    
    static void digitNet() {
        LOGGER.info("Running digit recognizer test");
        NeuralNet net = NeuralNet.builder(784, 16, 16, 10).setBiasRadius(10d).setWeightRadius(10d).build();
        int digit = new ArrayRealVector(net.run()).getMaxIndex();
        LOGGER.info("Final guess is {}! Did I get it right?",digit);
    }
}