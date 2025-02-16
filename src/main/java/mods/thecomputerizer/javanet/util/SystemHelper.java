package mods.thecomputerizer.javanet.util;

import java.awt.*;

public class SystemHelper {
    
    public static DisplayMode getDefaultDisplay() {
        return GraphicsEnvironment.getLocalGraphicsEnvironment().getDefaultScreenDevice().getDisplayMode();
    }
    
    public static int getDefaultDisplayHeight() {
        return getDefaultDisplay().getHeight();
    }
    
    public static int getDefaultDisplayWidth() {
        return getDefaultDisplay().getWidth();
    }
}