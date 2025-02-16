package mods.thecomputerizer.javanet.util;

import org.nd4j.linalg.api.ndarray.INDArray;

public class FunctionHelper {
    
    public static int maxIndex(INDArray values) {
        if(values.isEmpty()) return -1;
        long length = values.length();
        if(length==1) return 0;
        long index = 0;
        for(long l=1L;l<length;l++)
            if(values.getDouble(l)>values.getDouble(index)) index = l;
        return (int)index;
    }
}