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
    
    public static double calcR2( double[] predictVector,  double[] testvector){

            double up = 0;  
            double down = 0;

            Statistic st = Statistics.calcMeanAndSig(testvector);

            for(int i = 0; i<predictVector.length; i++){
                up += (double) ( testvector[i] - predictVector[i])*( testvector[i] - predictVector[i]);
                down += (double) ( testvector[i] - st.getMean())*( testvector[i] - st.getMean());
            }
            return 1 - (up/down);
        }
}
