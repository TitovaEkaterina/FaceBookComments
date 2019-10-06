/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package facebookcomments_ml;

import java.util.List;

/**
 *
 * @author titova_ekaterina
 */
public class R2 {
    
    public static double calcR2( List<Double> predictVector,  List<Double> testvector){

            double up = 0;  
            double down = 0;

            Statistic st = Statistics.calcMeanAndSig(testvector);

            for(int i = 0; i<predictVector.size(); i++){
                up += (double) ( testvector.get(i) - predictVector.get(i))*( testvector.get(i) - predictVector.get(i));
                down += (double) ( testvector.get(i) - st.getMean())*( testvector.get(i) - st.getMean());
            }
            return 1 - (up/down);
        }
}
