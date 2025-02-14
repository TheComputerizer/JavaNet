package mods.thecomputerizer.javanet;

import mods.thecomputerizer.javanet.neuralnet.NeuralNet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class JavaNet {
    
    private static final Logger LOGGER = LoggerFactory.getLogger("JavaNet Main");
    
    public static void main(String ... args) {
        LOGGER.info("----- START MAIN -----");
        if(args.length==0) {
            LOGGER.info("No args were specified so I guess I won't be doing anything");
            return;
        }
        String arg = args[0];
        LOGGER.info("Beginning to take over the world! I mean do some {}",arg);
        digitNet(!"testing".equalsIgnoreCase(arg));
        LOGGER.info("----- END MAIN -----");
    }
    
    static NeuralNet defaultNeuralNet() {
        return NeuralNet.builder(784,256,256,10).setBiasRadius(0.25d).setWeightRadius(0.25d).build();
    }
    
    static void digitNet(boolean training) {
        LOGGER.info("Running digit recognizer {} sequence",training ? "training" : "testing");
        NeuralNet neuralNet = defaultNeuralNet();
        if(training) neuralNet.train();
        else neuralNet.test();
        LOGGER.info("Finished running {} sequence",training ? "training" : "testing");
    }
}