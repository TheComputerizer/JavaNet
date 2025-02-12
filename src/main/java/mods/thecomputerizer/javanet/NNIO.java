package mods.thecomputerizer.javanet;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * I/O Helper methods for storing/retrieving training data
 */
public class NNIO {
    
    private static final Logger LOGGER = LoggerFactory.getLogger("JavaNet I/O");
    
    public static ArrayRealVector getTrainingData(String path) {
        return toVector(readFromFile(path));
    }
    
    public static byte[] readFromFile(String path) {
        try(FileInputStream stream = new FileInputStream(path+".bytes")) {
            return stream.readAllBytes();
        } catch(IOException ex) {
            LOGGER.error("Failed to read bytes from {}",path,ex);
        }
        return new byte[]{};
    }
    
    public static byte[] toBytes(ArrayRealVector vector) {
        double[] values = vector.toArray();
        return ByteBuffer.allocate(values.length*8).array();
    }
    
    public static ArrayRealVector toVector(byte[] bytes) {
        return new ArrayRealVector(ByteBuffer.wrap(bytes).asDoubleBuffer().array());
    }
    
    public static void writeTrainingData(String path, ArrayRealVector data) {
        writeToFile(path,toBytes(data));
    }
    
    public static void writeToFile(String path, byte[] bytes) {
        try(FileOutputStream stream = new FileOutputStream(path+".bytes")) {
            stream.write(bytes);
        } catch(IOException ex) {
            LOGGER.error("Failed to write bytes to {}",path,ex);
        }
    }
}
