package mods.thecomputerizer.javanet.training;

import org.nd4j.linalg.api.rng.Random;

public abstract class AbstractTrainable implements Trainable {
    
    protected double initRandomly(Random random, double range) {
        return (random.nextDouble()*range*2d)-range;
    }
}