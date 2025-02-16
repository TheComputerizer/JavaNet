package mods.thecomputerizer.javanet.render;

import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.util.Objects;

import static javax.swing.WindowConstants.EXIT_ON_CLOSE;

public class ImageRender {
    
    public static final ImageRender INSTANCE = new ImageRender("MNIST Render");
    
    private final JFrame frame;
    private final JLabel label;
    private final Java2DNativeImageLoader arrayImageLoader;
    private BufferedImage image;
    
    private ImageRender(String frameTitle) {
        this.arrayImageLoader = new Java2DNativeImageLoader();
        this.frame = new JFrame(frameTitle);
        this.frame.setDefaultCloseOperation(EXIT_ON_CLOSE);
        this.label = new JLabel();
        this.frame.add(this.label);
        this.frame.setLayout(null);
        this.frame.setVisible(false);
    }
    
    private void drawImage() {
        if(Objects.isNull(this.image)) return;
        int width = this.image.getWidth();
        int height = this.image.getHeight();
        this.frame.setSize(width,height);
        this.label.setBounds(0,0,width,height);
        this.label.setIcon(new ImageIcon(this.image));
        this.frame.setVisible(true);
    }
    
    public void loadAndDisplay(INDArray imageData) {
        this.image = this.arrayImageLoader.asBufferedImage(imageData);
        drawImage();
        this.frame.invalidate();
        this.frame.repaint();
    }
}
