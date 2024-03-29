/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package facebookcomments_ml;

import java.util.ArrayList;
import java.util.Collections;
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

    public LinReg(int countOfEpoch, double alpha, int batchSize) {
        this.alpha = alpha;
        this.countOfEpoch = countOfEpoch;
        this.batchSize = batchSize;
    }

    public void fit(DoubleMatrix X, DoubleMatrix Y) {

        double[] tempW = new double[X.columns];
        for (int i = 0; i < X.columns; ++i) {
            Random r = new Random();
            double randomValue = r.nextDouble();

            tempW[i] = (r.nextInt() % 2 == 0 ? 1 : -1) * randomValue;
        }
        W = new DoubleMatrix(tempW);

        this.gradientDescent(X, Y);

    }

    public double[] predict(DoubleMatrix X) {

        VectorOperation.normalizationVector(X);

        DoubleMatrix predY = predict_value(W, X);
        double[] predict_Y = new double[predY.rows];
        for (int i = 0; i < predY.rows; i++) {
            predict_Y[i] = predY.get(i);
        }
        return predict_Y;
    }

    public double[] getW() {
        double[] new_W = new double[W.rows];
        for (int i = 0; i < W.rows; i++) {
            new_W[i] = W.get(i);
        }
        return new_W;
    }

    private DoubleMatrix predict_value(DoubleMatrix W, DoubleMatrix X) {
        return X.mmul(W);
    }

    private DoubleMatrix gradientDescent(DoubleMatrix X, DoubleMatrix Y) {

        VectorOperation.normalizationVector(X);
        DoubleMatrix lastDiff = new DoubleMatrix(this.batchSize);
        int k = 0;

        List<Integer> listOfIndexes = new ArrayList<>(0);
        for (int i = 0; i < X.rows; ++i) {
            listOfIndexes.add(i);
        }

        while (k < this.countOfEpoch) {

            Collections.shuffle(listOfIndexes, new Random());
            DoubleMatrix newW = new DoubleMatrix(this.W.data);

            for (int i = 0; i < X.rows; i += this.batchSize) {

                double[][] data_bach_X = new double[(i + this.batchSize) < listOfIndexes.size() ? this.batchSize : listOfIndexes.size() - i][];
                double[] data_bach_Y = new double[(i + this.batchSize) < listOfIndexes.size() ? this.batchSize : listOfIndexes.size() - i];

                int index = 0;

                for (int t = i; t < (i + this.batchSize) && t < listOfIndexes.size(); ++t) {
                    data_bach_X[index] = X.getRow(listOfIndexes.get(t)).data;
                    data_bach_Y[index] = Y.getRow(listOfIndexes.get(t)).data[0];
                    index++;
                }

                DoubleMatrix bachX = new DoubleMatrix(data_bach_X);
                DoubleMatrix bachY = new DoubleMatrix(data_bach_Y);

                DoubleMatrix current_predict_value = predict_value(newW, bachX);
                DoubleMatrix diff;

                diff = (bachY.sub(current_predict_value));
//                    diff = (bachY.sub(current_predict_value)).div(MatrixFunctions.sqrt((bachY.sub(current_predict_value)).mul(bachY.sub(current_predict_value))));
                diff = diff.mul(alpha);

                newW = newW.add(((diff.transpose().mmul(bachX)).div(bachY.rows)).transpose());

                W = newW;

                int indexCurrent = 0;
                if (lastDiff.rows == diff.rows) {
                    indexCurrent = (new Random()).nextInt(diff.rows);
                } else if (lastDiff.rows < diff.rows) {
                    indexCurrent = (new Random()).nextInt(lastDiff.rows);
                }

                if (alpha > 0.0000000000000000000000015) {
                    if (lastDiff.get(indexCurrent) * diff.get(indexCurrent) <= 0) {
                        alpha *= 0.9999999;
                        lastDiff = diff;
                    } else {
                        alpha *= 1.000001;
                        lastDiff = diff;
                    }
                }
            }
            ++k;
        }
        return W;
    }

}
