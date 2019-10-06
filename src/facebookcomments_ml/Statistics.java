/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package facebookcomments_ml;

import java.util.List;
import org.jblas.DoubleMatrix;

/**
 *
 * @author titova_ekaterina
 */
public class Statistics{
    
    public static Statistic calcMeanAndSig( List<Double> vector ){
    
        double sum = 0;
        double summSquare = 0;
        for (int i = 0; i < vector.size(); ++i){
            sum += vector.get(i);
            summSquare +=  vector.get(i)*vector.get(i);
        }
        Statistic st = new Statistic();
        st.setMean(sum/vector.size());
        st.setSigma(Math.sqrt(summSquare/vector.size() - st.getMean()*st.getMean()));
        
        return st;
    }
    
    public static Statistic calcMeanAndSig( DoubleMatrix vector ){
    
        double sum = 0;
        double summSquare = 0;
        for (int i = 0; i < vector.rows; ++i){
            sum += vector.get(i);
            summSquare +=  vector.get(i)*vector.get(i);
        }
        Statistic st = new Statistic();
        st.setMean(sum/vector.rows);
        st.setSigma(Math.sqrt(summSquare/vector.rows - st.getMean()*st.getMean()));
        
        return st;
    }
    
}
