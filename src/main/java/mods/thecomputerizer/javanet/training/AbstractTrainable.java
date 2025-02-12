package mods.thecomputerizer.javanet.training;

import lombok.Getter;
import org.apache.commons.math3.linear.RealVector;

@Getter
public abstract class AbstractTrainable implements Trainable {
    
    private int startingIndex;
    
    @Override public int initScope(int index) {
        this.startingIndex = index;
        return index+dataSize();
    }
    
    @Override public final void load(RealVector data) {
        loadFrom(data,this.startingIndex);
    }
    
    protected abstract void loadFrom(RealVector data, int index);
    
    @Override public final void store(RealVector data) {
        storeFrom(data,this.startingIndex);
    }
    
    protected abstract void storeFrom(RealVector data, int index);
}