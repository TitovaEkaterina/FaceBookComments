/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package facebookcomments_ml;

import static java.lang.Math.sqrt;
import java.util.List;

/**
 *
 * @author titova_ekaterina
 */
public class RMSE {
    
    public static double calcRMSE( List<Double> predictVector,  List<Double> testvector){
    
        double s = 0;  
        for(int i = 0; i<testvector.size(); i++){
            s += (double) (testvector.get(i) - predictVector.get(i))*(testvector.get(i) - predictVector.get(i));
        }
        return sqrt(s/testvector.size());
    }
}
