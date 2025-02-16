package mods.thecomputerizer.javanet.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;

import static java.lang.Double.BYTES;

/**
 * I/O Helper methods for storing/retrieving training data
 */
public class NNIO {
    
    private static final Logger LOGGER = LoggerFactory.getLogger("JavaNet I/O");
    private static final Path WORKING_PATH = Paths.get("");
    
    @SuppressWarnings("ResultOfMethodCallIgnored")
    private static File getFile(String path, boolean create) {
        File file = new File(WORKING_PATH.toAbsolutePath().toFile(),path);
        if(create && !file.exists()) {
            try {
                file.createNewFile();
            } catch(IOException ex) {
                LOGGER.error("Failed to create new file at {}",file.getAbsolutePath(),ex);
            }
        }
        return file;
    }
    
    public static INDArray getTrainingData(String path) {
        byte[] bytes = readFromFile(path);
        return bytes.length==0 ? null : toVector(bytes);
    }
    
    public static byte[] readFromFile(String path) {
        File file = getFile(path+".bytes",false);
        if(!file.exists()) {
            LOGGER.warn("No file exists at {}",file.getAbsolutePath());
            return new byte[]{};
        }
        try(FileInputStream stream = new FileInputStream(file)) {
            return stream.readAllBytes();
        } catch(IOException ex) {
            LOGGER.error("Failed to read bytes from {}",path,ex);
        }
        return new byte[]{};
    }
    
    public static byte[] toBytes(INDArray data) {
        double[] values = data.toDoubleVector();
        ByteBuffer buffer = ByteBuffer.allocate(values.length*BYTES);
        for(double value : values) buffer.putDouble(value);
        return buffer.array();
    }
    
    public static INDArray toVector(byte[] bytes) {
        DoubleBuffer buffer = ByteBuffer.wrap(bytes).asDoubleBuffer();
        double[] asArray = new double[buffer.remaining()];
        buffer.get(asArray); //DoubleBuffer#toArray doesn't work if the buffer is direct
        return Nd4j.createFromArray(asArray);
    }
    
    public static void writeTrainingData(String path, INDArray data) {
        writeToFile(path,toBytes(data));
    }
    
    public static void writeToFile(String path, byte[] bytes) {
        try(FileOutputStream stream = new FileOutputStream(getFile(path+".bytes",true))) {
            LOGGER.info("Writing bytes to file {}",path);
            stream.write(bytes);
        } catch(IOException ex) {
            LOGGER.error("Failed to write bytes to {}",path,ex);
        }
    }
}
