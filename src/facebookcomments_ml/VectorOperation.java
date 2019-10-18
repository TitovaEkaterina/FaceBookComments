/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package facebookcomments_ml;

import org.jblas.DoubleMatrix;

/**
 *
 * @author titova_ekaterina
 */
public class VectorOperation {
    
    public static void normalizationVector(DoubleMatrix vector){

        for(int i = 0; i<vector.columns-1; ++i){
            
            Statistic st = Statistics.calcMeanAndSig(vector.getColumn(i));
            
            for (int j=0; j<vector.rows; ++j){
                if (st.getSigma() != 0){
                    vector.put(vector.rows*i + j, (vector.get(vector.rows*i + j) - st.getMean())/st.getSigma());
                } else {
                   //vector.put(vector.rows*i + j, 1);
                }
            }
        }
    }
}
