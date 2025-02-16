package mods.thecomputerizer.javanet.training;

import org.deeplearning4j.nn.weights.IWeightInit;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.deeplearning4j.nn.weights.IWeightInit.DEFAULT_WEIGHT_INIT_ORDER;

public abstract class AbstractTrainable implements Trainable {
    
    protected INDArray applyBackwards(IActivation activationFunc, INDArray array, INDArray gradients) {
        return activationFunc.backprop(array,gradients).getFirst();
    }
    
    protected INDArray applyForward(IActivation activationFunc, INDArray array, boolean training) {
        return activationFunc.getActivation(array,training);
    }
    
    protected void assignVectorToMatrix(INDArray matrix, INDArray vector) {
        matrix.assign(vector.reshape(matrix.rows(),matrix.columns()));
    }
    
    protected INDArray initWeight(INDArray view, IWeightInit init, int input, int output, long ... sizes) {
        return init.init(input,output,sizes,DEFAULT_WEIGHT_INIT_ORDER,view);
    }
    
    protected INDArrayIndex interval(long start, long end) {
        return NDArrayIndex.interval(start,end);
    }
    
    protected INDArrayIndex[] intervalAsArray(long start, long end) {
        return new INDArrayIndex[]{interval(start,end)};
    }
    
    /**
     * There are probably more, but I only care about softmax for now
     */
    protected boolean isDifferentiable(IActivation function) {
        return !(function instanceof ActivationSoftmax);
    }
    
    protected INDArray subset(INDArray data, long start, long end) {
        return data.get(interval(start,end));
    }
}