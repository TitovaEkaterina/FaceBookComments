/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package facebookcomments_ml;

import static java.lang.Math.abs;
import static java.lang.Math.sqrt;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 *
 * @author titova_ekaterina
 */


public class LinReg {
    
    private int countOfEpoch;
    private double alpha;
    private int batchSize;
    
    private DoubleMatrix W;
    
    public LinReg(int countOfEpoch,double alpha, int batchSize ){
        this.alpha = alpha;
        this.countOfEpoch = countOfEpoch;
        this.batchSize = batchSize;
    }    
    
    public void fit(List<List<Double>> X,List<Double> Y){
        
        
        double[][] t = new double[X.size()][X.get(0).size()];
        for(int i=0; i<X.size(); ++i){
            for(int j=0; j<X.get(i).size(); ++j){
                t[i][j] = X.get(i).get(j);
            }
        }
        
        DoubleMatrix new_XMatrix = new DoubleMatrix(t);
        DoubleMatrix new_Y = new DoubleMatrix(Y);
        //List<List<Double>> new_X = new ArrayList(X);
        
        double[] tempW = new double[X.get(0).size()];
        for(int i = 0; i<X.get(0).size(); ++i){
            Random r = new Random();
            double randomValue = r.nextDouble();
            tempW[i] = randomValue;
        }
        W = new DoubleMatrix(tempW);
        
        this.gradientDescent(new_XMatrix,new_Y);
       
    }
    
    public List<Double> predict(List<List<Double>> XTest){

        //List<List<Double>> new_X = new ArrayList(XTest);
        
        double[][] t = new double[XTest.size()][XTest.get(0).size()];
        for(int i=0; i<XTest.size(); ++i){
            for(int j=0; j<XTest.get(i).size(); ++j){
                t[i][j] = XTest.get(i).get(j);
            }
        }
        
        DoubleMatrix new_X = new DoubleMatrix(t);
        
        VectorOperation.normalizationVector(new_X);
        
       
        DoubleMatrix predY = predict_value(W, new_X);
        List<Double> predict_Y = new ArrayList<>();
        for(int i=0; i<predY.rows; i++){
            predict_Y.add(predY.get(i));
        }
        return predict_Y;
    }
    
    public List<Double> getW(){
        List<Double> new_W = new ArrayList<>();
        for(int i=0; i<W.rows; i++){
            new_W.add(W.get(i));
        }
        
        return new_W;
    }
    
    private DoubleMatrix predict_value(DoubleMatrix W, DoubleMatrix X){     
        return X.mmul(W);
    }
    
    
    private DoubleMatrix gradientDescent(DoubleMatrix X,DoubleMatrix Y){
        
        VectorOperation.normalizationVector(X);
        DoubleMatrix lastDiff = new DoubleMatrix(this.batchSize);
        int k = 0;
        
        while (k<this.countOfEpoch) {
            
            DoubleMatrix newW = new DoubleMatrix(this.W.data);
            
            for (int i = 0; i < X.rows; i+=this.batchSize){ 
                
                DoubleMatrix bachX;
                DoubleMatrix bachY;
                if (i + this.batchSize < X.rows) {
                    bachX = X.getRange(i, i+this.batchSize, 0, X.columns);
                    bachY = Y.getRange(i, i+this.batchSize, 0, 1);
                } else {
                    bachX = X.getRange(i, X.rows, 0, X.columns);
                    bachY = Y.getRange(i, X.rows, 0, 1);
                }
                DoubleMatrix current_predict_value = predict_value(newW, bachX);
                DoubleMatrix diff = (bachY.sub(current_predict_value)).div(MatrixFunctions.sqrt((bachY.sub(current_predict_value)).mul(bachY.sub(current_predict_value))));
                diff = diff.mul(alpha);
                newW = newW.add(((diff.transpose().mmul(bachX)).div(bachY.rows)).transpose());
                W = newW;
                  
            }
            ++k;
        }
        return W;
    }
    
}

