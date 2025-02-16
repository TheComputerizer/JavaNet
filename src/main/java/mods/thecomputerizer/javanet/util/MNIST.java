package mods.thecomputerizer.javanet.util;

import au.com.bytecode.opencsv.CSVReader;
import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import static org.nd4j.linalg.api.buffer.DataType.FLOAT;

public class MNIST {
    
    private static final Logger LOGGER = LoggerFactory.getLogger("JavaNet MNIST");
    
    private static List<String> flatten(List<String[]> parsedCSV) {
        List<String> elements = new ArrayList<>();
        for(String[] line : parsedCSV) elements.addAll(Arrays.asList(line));
        return elements;
    }
    
    private static List<DigitData> parseFrom(List<String> elements) {
        List<DigitData> digits = new ArrayList<>();
        double[] data = new double[784];
        int counter = 0;
        int expected = 0;
        for(String element : elements) {
            if(counter==0) expected = Integer.parseInt(element);
            else data[counter-1] = Double.parseDouble(element);
            counter++;
            if(counter>784) {
                digits.add(new DigitData(expected,Arrays.copyOf(data,data.length)));
                data = new double[data.length];
                counter = 0;
            }
        }
        return digits;
    }
    
    public static List<DigitData> readTesting() {
        return read("testing/mnist_test");
    }
    
    public static List<DigitData> readTraining() {
        return read("training/mnist_train");
    }
    
    public static List<DigitData> read(String path) {
        try(InputStream stream = MNIST.class.getClassLoader().getResourceAsStream(path+".csv")) {
            if(Objects.isNull(stream)) return List.of();
            try(InputStreamReader reader = new InputStreamReader(stream)) {
                return parseFrom(flatten(new CSVReader(reader).readAll()));
            }
        } catch(IOException ex) {
            LOGGER.error("Failed to read CSV from {}",path,ex);
        }
        return List.of();
    }
    
    @Getter
    public static class DigitData {
        
        private final int expected;
        private final INDArray expectedActivation;
        private final INDArray data;
        
        private DigitData(int expected, double[] data) {
            this.expected = expected;
            this.expectedActivation = Nd4j.zeros(FLOAT,10);
            this.expectedActivation.putScalar(expected,1f);
            this.data = Nd4j.createFromArray(data).divi(255d).castTo(FLOAT);
        }
        
        /**
         * Convert digit data stored as a 1D array of doubles to a 3D array index by x, y, & rbg values
         */
        public INDArray getImageData() {
            INDArray rgbByPos = Nd4j.create(FLOAT,28,28,3); //x,y,rbg
            for(int x=0;x<28;x++)
                for(int y=0;y<28;y++) indexToRGB(rgbByPos,x,y);
            return rgbByPos;
        }
        
        private void indexToRGB(INDArray rgbByPos, long x, long y) {
            float dataAtIndex = this.data.getFloat((x*28)+y);
            for(long c=0;c<3;c++) rgbByPos.putScalar(new long[]{x,y,c},dataAtIndex);
        }
    }
}
