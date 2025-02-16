package mods.thecomputerizer.javanet.training;

import org.deeplearning4j.nn.weights.IWeightInit;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.deeplearning4j.nn.weights.IWeightInit.DEFAULT_WEIGHT_INIT_ORDER;

public abstract class AbstractTrainable implements Trainable {
    
    protected INDArray applyBackwards(IActivation activationFunc, INDArray array, INDArray gradients) {
        return activationFunc.backprop(array,gradients).getFirst();
    }
    
    protected INDArray applyForward(IActivation activationFunc, INDArray array, boolean training) {
        return activationFunc.getActivation(array,training);
    }
    
    protected INDArray initWeight(INDArray view, IWeightInit init, int input, int output, long ... sizes) {
        return init.init(input,output,sizes,DEFAULT_WEIGHT_INIT_ORDER,view);
    }
    
    /**
     * There are probably more, but I only care about softmax for now
     */
    protected boolean isDifferentiable(IActivation function) {
        return !(function instanceof ActivationSoftmax);
    }
}