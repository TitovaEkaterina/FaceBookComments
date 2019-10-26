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
import org.jblas.DoubleMatrix;

/**
 *
 * @author titova_ekaterina
 */
public class FacebookComments_ML {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {

        String trainFileName = "/home/boyko_mihail/NetBeansProjects/ML_Facebook_LinearRegression/ML_2019_FaceBookComments_LinearRegression/Dataset/Dataset/Training/Features_Variant_1.csv";

        int FoldsCounts = 0;
        double[] RMSEMetrix = new double[5];
        double[] R2Metrix = new double[5];
        double[][] dataSet = new double[5][];
        List<List<String>> records = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(trainFileName))) {
            String line;
            while ((line = br.readLine()) != null) {
                FoldsCounts++;
                String[] values = line.split(",");
                String[] valuesWithOnes = new String[values.length + 1];
                for (int i = 0; i < values.length - 1; ++i) {
                    valuesWithOnes[i] = values[i];
                }
                valuesWithOnes[values.length - 1] = "1";
                valuesWithOnes[values.length] = values[values.length - 1];
                records.add(Arrays.asList(valuesWithOnes));
            }
        } catch (Exception ex) {
            Logger.getLogger(FacebookComments_ML.class.getName()).log(Level.SEVERE, null, ex);
        }
        FoldsCounts /= 5;
        Collections.shuffle(records, new Random());

        double[] dataY = new double[records.size()];
        double[][] dataX = new double[records.size()][records.get(0).size() - 1];
        
        for (int i = 0; i < records.size(); ++i) {
            List<String> tempVector = records.get(i);
            double[] currentX = new double[tempVector.size() - 1];
            for (int j = 0; j < tempVector.size() - 1; ++j) {
                currentX[j] = Double.parseDouble(tempVector.get(j));
            }
            dataX[i] = currentX;
           
            dataY[i] = Double.parseDouble(tempVector.get(tempVector.size() - 1));
        }

        double[] data_train_Y;
        double[][] data_train_X;

        double[] data_test_Y;
        double[][] data_test_X;

        for (int i = 0; i < 5; ++i) {

            data_train_X = new  double[dataX.length - dataX.length/5][];
            data_test_X = new  double[dataX.length/5][];
            data_train_Y = new  double[dataX.length - dataX.length/5];
            data_test_Y = new  double[dataX.length/5];
            
            int indexTrain  = 0;
            int indexTest  = 0;
            for (int j = 0; j < dataX.length; ++j) {
                if (j < FoldsCounts * i || j >= FoldsCounts * (i + 1)) {
                    data_train_X[indexTrain] = dataX[j];
                    data_train_Y[indexTrain] = dataY[j];
                    ++indexTrain;
                } else {
                    data_test_X[indexTest] = dataX[j];
                    data_test_Y[indexTest] = dataY[j];
                    ++indexTest;
                }
            }
            
            DoubleMatrix data_train_X_matrix = new  DoubleMatrix(data_train_X);
            DoubleMatrix data_test_X_matrix = new  DoubleMatrix(data_test_X);
            DoubleMatrix data_train_Y_matrix = new  DoubleMatrix(data_train_Y);
            DoubleMatrix data_test_Y_matrix = new  DoubleMatrix(data_test_Y);

            LinReg model = new LinReg(750, 0.9, 2500, LossFuctions.RMSE);

            model.fit(data_train_X_matrix, data_train_Y_matrix);

            double[] YPred = model.predict(data_train_X_matrix);

            double RMSE_train = RMSE.calcRMSE(YPred, data_train_Y);
            double R2_train = R2.calcR2(YPred, data_train_Y);

            System.out.println("RMSE trening fold " + i + " = " + RMSE_train);

            System.out.println("R2 trening fold " + i + " = " + R2_train);

            double[] YPredTest = model.predict(data_test_X_matrix);

            double RMSE_test = RMSE.calcRMSE(YPredTest, data_test_Y);
            double R2_test = R2.calcR2(YPredTest, data_test_Y);

            System.out.println("RMSE test fold " + i + " = " + RMSE_test);

            System.out.println("R2 test fold " + i + " = " + R2_test);
            System.out.println();
            System.out.println();
            System.out.println();

            RMSEMetrix[i] = RMSE_test;
            R2Metrix[i] = R2_test;
            dataSet[i] = model.getW();
        }

        Statistic stR2 = Statistics.calcMeanAndSig(R2Metrix);
        Statistic stRMSE = Statistics.calcMeanAndSig(RMSEMetrix);

        System.out.println("RMSE Mean = " + stRMSE.getMean());
        System.out.println("RMSE Sigma = " + stRMSE.getSigma());
        System.out.println("R2 Mean = " + stR2.getMean());
        System.out.println("R2 Sigma = " + stR2.getSigma());

        FileWriter csvWriter = new FileWriter("new_RMSE.csv");
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

        csvWriter.append("RMSE," + RMSEMetrix[0] + "," + RMSEMetrix[1] + "," + RMSEMetrix[2] + "," + RMSEMetrix[3] + "," + RMSEMetrix[4] + "," + stRMSE.getMean() + "," + stRMSE.getSigma() + ",\n");
        csvWriter.append("R2," + R2Metrix[0] + "," + R2Metrix[1] + "," + R2Metrix[2] + "," + R2Metrix[3] + "," + R2Metrix[4] + "," + stR2.getMean() + "," + stR2.getSigma() + ",\n");

        for (int i = 0; i < dataSet[0].length; i++) {
            double W_M = 0;
            double W_Sig = 0;
            for (int k = 0; k < dataSet.length; ++k) {
                W_M += dataSet[k][i];
                W_Sig += dataSet[k][i] * dataSet[k][i];
            }
            W_M = W_M / dataSet.length;
            W_Sig = sqrt(W_Sig / dataSet.length - pow(W_M, 2));
            csvWriter.append("W[" + i + "]," + dataSet[0][i] + "," + dataSet[1][i] + "," + dataSet[2][i] + "," + dataSet[3][i] + "," + dataSet[4][i] + "," + W_M + "," + W_Sig + ",\n");
        }

        csvWriter.flush();
        csvWriter.close();
    }

}
