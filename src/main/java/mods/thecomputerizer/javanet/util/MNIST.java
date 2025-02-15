package mods.thecomputerizer.javanet.util;

import au.com.bytecode.opencsv.CSVReader;
import lombok.Getter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

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
    
    public static class DigitData {
        
        @Getter private final int expected;
        @Getter private final double[] expectedActivation;
        private final double[] data;
        
        private DigitData(int expected, double[] data) {
            this.expected = expected;
            this.expectedActivation = new double[10];
            this.expectedActivation[expected] = 1d;
            this.data = data;
        }
        
        public double[] getData() {
            double[] normalized = new double[this.data.length];
            for(int i=0;i<normalized.length;i++) normalized[i] = normalize(this.data[i]);
            return normalized;
        }
        
        private double normalize(double value) {
            return (value/255d);
        }
    }
}
