package mods.thecomputerizer.javanet.util;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
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
    
    public static RealVector getTrainingData(String path) {
        return toVector(readFromFile(path));
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
    
    public static byte[] toBytes(RealVector vector) {
        double[] values = vector.toArray();
        ByteBuffer buffer = ByteBuffer.allocate(values.length*8);
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes); //ByteBuffer#toArray doesn't work if the buffer is direct
        return bytes;
    }
    
    public static RealVector toVector(byte[] bytes) {
        DoubleBuffer buffer = ByteBuffer.wrap(bytes).asDoubleBuffer();
        double[] asArray = new double[buffer.remaining()];
        buffer.get(asArray); //DoubleBuffer#toArray doesn't work if the buffer is direct
        return new ArrayRealVector(asArray);
    }
    
    public static void writeTrainingData(String path, RealVector data) {
        writeToFile(path,toBytes(data));
    }
    
    public static void writeToFile(String path, byte[] bytes) {
        try(FileOutputStream stream = new FileOutputStream(getFile(path+".bytes",true))) {
            stream.write(bytes);
        } catch(IOException ex) {
            LOGGER.error("Failed to write bytes to {}",path,ex);
        }
    }
}
