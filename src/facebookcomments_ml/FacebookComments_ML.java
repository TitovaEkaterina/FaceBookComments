/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package facebookcomments_ml;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author titova_ekaterina
 */

public class FacebookComments_ML {

    /**
     * @param args the command line arguments
     */
    
            
    public static void main(String[] args) throws IOException {
    
        String trainFileName = "/home/kate_t/ML_Prod/FaceBookComments/data/Dataset/Training/Features_Variant_1.csv";

        int FoldsCounts = 0;
        List<Double> RMSEMetrix = new ArrayList<>();
        List<Double> R2Metrix = new ArrayList<>();
        List<List<Double>> dataSet = new ArrayList<>();
        List<List<String>> records = new ArrayList<>();
       
        try (BufferedReader br = new BufferedReader(new FileReader(trainFileName))) {
            String line;
            while ((line = br.readLine()) != null) {
                FoldsCounts++;
                String[] values = line.split(",");
                String[] valuesWithOnes = new String[values.length + 1];
                for(int i=0; i<values.length - 1; ++i){
                    valuesWithOnes[i] = values[i];
                }
                valuesWithOnes[values.length - 1] = "1";
                valuesWithOnes[values.length] = values[values.length - 1];
                records.add(Arrays.asList(valuesWithOnes));
            }
        } catch (Exception ex) {
            Logger.getLogger(FacebookComments_ML.class.getName()).log(Level.SEVERE, null, ex);
        } 
        FoldsCounts/=5;
        Collections.shuffle(records, new Random()); 
        
        

        List<Double> dataY = new ArrayList<>();
        List<List<Double>> dataX = new ArrayList<>();
        for(int i=0; i<records.size(); ++i){
            List<String> tempVector = records.get(i);
            List<Double> currentX = new ArrayList<>();
            for(int j=0;j<tempVector.size() - 1; ++j){
                currentX.add(Double.parseDouble(tempVector.get(j)));
            }
            dataX.add(currentX);
            dataY.add(Double.parseDouble(tempVector.get(tempVector.size()-1)));
        }
        

        List<Double> data_train_Y = new ArrayList<>();
        List<List<Double>> data_train_X = new ArrayList<>();
        
        List<Double> data_test_Y = new ArrayList<>();
        List<List<Double>> data_test_X = new ArrayList<>();
        
        for(int i = 0; i<5; ++i){
        
            data_train_X.clear();
            data_test_X.clear();
            data_train_Y.clear();
            data_test_Y.clear();

            for(int j=0;j<dataX.size(); ++j){
                if(j < FoldsCounts*i || j >= FoldsCounts*(i+1)) {
                    data_train_X.add(dataX.get(j));
                    data_train_Y.add(dataY.get(j));
                } else {
                    data_test_X.add(dataX.get(j));
                    data_test_Y.add(dataY.get(j));
                }
            }

            LinReg model = new LinReg( 200, 0.9, 1000);  

            model.fit(data_train_X, data_train_Y);

            List<Double> YPred = model.predict(data_train_X);
            
            double RMSE_train = RMSE.calcRMSE(YPred, data_train_Y);
            double R2_train = R2.calcR2(YPred, data_train_Y);
            
            System.out.println("RMSE trening fold "+i+" = "+RMSE_train);
            
            System.out.println("R2 trening fold "+i+" = " + R2_train);
            
         
            List<Double>  YPredTest = model.predict(data_test_X);
            
            double RMSE_test = RMSE.calcRMSE(YPredTest, data_test_Y);
            double R2_test = R2.calcR2(YPredTest, data_test_Y);
            
            
            System.out.println("RMSE test fold "+i+" = "+RMSE_test);
            
            System.out.println("R2 test fold "+i+" = " + R2_test);
            System.out.println();
            System.out.println();
            System.out.println();

            RMSEMetrix.add(RMSE_test);
            R2Metrix.add(R2_test);
            dataSet.add(model.getW());
        }



        Statistic stR2 = Statistics.calcMeanAndSig(R2Metrix );
        Statistic stRMSE = Statistics.calcMeanAndSig(RMSEMetrix);
        
        System.out.println("RMSE Mean = " + stRMSE.getMean());
        System.out.println("RMSE Sigma = " + stRMSE.getSigma());
        System.out.println("R2 Mean = " + stR2.getMean());
        System.out.println("R2 Sigma = " + stR2.getSigma());
        

        FileWriter csvWriter = new FileWriter("new2.csv");
        csvWriter.append(",");
        csvWriter.append("1");
        csvWriter.append(",");
        csvWriter.append("2");
        csvWriter.append(",");
        csvWriter.append("3");
        csvWriter.append(",");
        csvWriter.append("4");
        csvWriter.append(",");
        csvWriter.append("5");
        csvWriter.append(",");
        csvWriter.append("E");
        csvWriter.append(",");
        csvWriter.append("SD");
        csvWriter.append("\n");

        csvWriter.append("RMSE,"+RMSEMetrix.get(0)+","+RMSEMetrix.get(1)+","+RMSEMetrix.get(2)+","+RMSEMetrix.get(3)+","+RMSEMetrix.get(4)+","+stRMSE.getMean()+","+stRMSE.getSigma()+",\n");
        csvWriter.append("R2,"+R2Metrix.get(0)+","+R2Metrix.get(1)+","+R2Metrix.get(2)+","+R2Metrix.get(3)+","+R2Metrix.get(4)+","+stR2.getMean()+","+stR2.getSigma()+",\n");

        
        for (int i = 0; i < dataSet.get(0).size(); i++) {
            double W_M = 0;
            double W_Sig = 0;
            for (int k = 0; k < dataSet.size(); ++k) {
                W_M += dataSet.get(k).get(i);
                W_Sig += dataSet.get(k).get(i) * dataSet.get(k).get(i);
            }
            W_M = W_M / dataSet.size();
            W_Sig = sqrt(W_Sig / dataSet.size() - pow(W_M,2));
            csvWriter.append("W[" + i + "]," + dataSet.get(0).get(i) + "," + dataSet.get(1).get(i) + "," + dataSet.get(2).get(i) + "," + dataSet.get(3).get(i) + "," + dataSet.get(4).get(i) + "," + W_M + "," + W_Sig + ",\n");
        }

        csvWriter.flush();
        csvWriter.close();
    }
    
}
