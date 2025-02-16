package mods.thecomputerizer.javanet.render;

import mods.thecomputerizer.javanet.util.SystemHelper;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;


import static java.awt.BorderLayout.SOUTH;
import static java.awt.Image.SCALE_SMOOTH;
import static javax.swing.SwingConstants.BOTTOM;
import static javax.swing.WindowConstants.EXIT_ON_CLOSE;

public class ImageRender {
    
    private static final double BASE_SCALE_FACTOR = 0.5d;
    public static final ImageRender INSTANCE = new ImageRender("MNIST Render");
    
    private final JFrame frame;
    private JButton button;
    private final List<BufferedImage> images;
    private int imageIndex;
    
    private ImageRender(String frameTitle) {
        this.frame = new JFrame(frameTitle);
        this.frame.setDefaultCloseOperation(EXIT_ON_CLOSE);
        this.frame.setVisible(false);
        this.images = new ArrayList<>();
        this.imageIndex = -1;
    }
    
    private void display() {
        setImage();
        this.frame.pack();
        this.frame.setLocationRelativeTo(null);
        this.frame.setVisible(true);
    }
    
    private int getScaleFactor(int width, int height) {
        int displayHeight = SystemHelper.getDefaultDisplayHeight();
        int displayWidth = SystemHelper.getDefaultDisplayWidth();
        boolean horizontal = displayWidth>=displayHeight;
        double size = horizontal ? displayHeight : displayWidth;
        double scaleWith = horizontal ? height : width;
        return (int)((size/scaleWith)*BASE_SCALE_FACTOR);
    }
    
    /**
     * Scale to display height while keeping the aspect ratio.
     * Round down to ensure the image is the same size or slightly smaller than the display size
     */
    private Image getScaledToDisplayHeight(BufferedImage image, int width, int height) {
        int scale = getScaleFactor(width,height);
        return image.getScaledInstance(width*scale,height*scale,SCALE_SMOOTH);
    }
    
    public void loadAndDisplay(Collection<BufferedImage> images) {
        this.images.clear();
        this.images.addAll(images);
        this.imageIndex = 0;
        this.button = nextButton();
        display();
    }
    
    private JButton nextButton() {
        JButton button = new JButton("Next ("+(this.imageIndex+1)+"/"+this.images.size()+")");
        button.addMouseListener(new MouseListener() {
            @Override public void mouseClicked(MouseEvent e) {}
            @Override public void mousePressed(MouseEvent e) {}
            @Override public void mouseReleased(MouseEvent e) {
                nextImage();
            }
            @Override public void mouseEntered(MouseEvent e) {}
            @Override public void mouseExited(MouseEvent e) {}
        });
        button.setSize(28*getScaleFactor(28,28),20);
        button.setVerticalAlignment(BOTTOM);
        return button;
    }
    
    public void nextImage() {
        int previous = this.imageIndex;
        this.imageIndex++;
        if(this.imageIndex>=this.images.size()) this.imageIndex = 0;
        if(this.imageIndex!=previous) {
            this.frame.getContentPane().removeAll();
            this.button.setText("Next ("+(this.imageIndex+1)+"/"+this.images.size()+")");
        }
        display();
    }
    
    private void setImage() {
        if(this.imageIndex<0 || this.imageIndex>=this.images.size()) return;
        BufferedImage image = this.images.get(this.imageIndex);
        Image scaled = getScaledToDisplayHeight(image,image.getWidth(),image.getHeight());
        setImage(this.frame.getContentPane(),new JLabel(new ImageIcon(scaled)));
    }
    
    private void setImage(Container container, JLabel label) {
        container.add(this.button,SOUTH);
        container.add(label);
    }
}
