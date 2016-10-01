/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package decisiontreeclassifier;

import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.Id3;
import weka.core.Instance;
import weka.core.SerializationHelper;


/**
 *
 * @author TOSHIBA PC
 */
public class MyWeka {
    
    private Instances data;
    private Instances train;
    private Instances test;
    private Classifier classifier;
    private Evaluation eval;
            
    public MyWeka(){
    }
    
    //****************** File Arff dan CSV Reader ******************//
    
    public void ReadFileArff(String filename) throws IOException{
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        data = new Instances(reader);
        reader.close();
        
        // setting class attribute
        data.setClassIndex(data.numAttributes() - 1);
    }
    
    public void ReadFileCsv(String filename) throws Exception{
        DataSource source = new DataSource(filename);
        data = source.getDataSet();
        
        // setting class attribute
        if (data.classIndex() == -1)
          data.setClassIndex(data.numAttributes() - 1);
    }
    
    //****************** Filtering ******************//
    
    public void RemoveAttribute(String attribute) throws Exception {
        String[] options = new String[2];
        options[0] = "-R";                                    
        options[1] = attribute;   
        
        Remove remove = new Remove();                         
        remove.setOptions(options);                           
        remove.setInputFormat(data);                          
        data = Filter.useFilter(data, remove);   
    }
    
    public void Resample(String bias, String percent) throws Exception{
        String[] options = new String[4];
        options[0] = "-B";
        options[1] = bias;
        options[2] = "-Z";
        options[3] = percent;
        
        Resample sample = new Resample();
        sample.setOptions(options);
        sample.setInputFormat(data);
        data = Filter.useFilter(data, sample);     
    }
    
    //****************** Build Classifier ******************//
    
    public void BuildClassifierID3() throws Exception{
        classifier = new Id3();
        classifier.buildClassifier(train);
    }
    
    public void BuildClassifierJ48(String confidence) throws Exception {
        //Assume always prune
        
        String[] options = new String[2];
        options[0] = "-C";
        options[1] = confidence;
        
        // new instance of tree
        classifier = new J48();
        classifier.setOptions(options);     
        classifier.buildClassifier(train);   
    }
    
    public void BuildClassifierNaiveBayes() throws Exception{
        classifier = new NaiveBayes();
        classifier.buildClassifier(train);
    }
    
    
    public void EvaluateModel() throws Exception{
        eval = new Evaluation(train);
        eval.evaluateModel(classifier, test); 
    }
    
    public void FullTraining() {
        train = data;
        test = data;
    }
    
    public void crossValidate(int folds) throws Exception {
        eval = new Evaluation(data);
        eval.crossValidateModel(classifier, data, folds, new Random(1));     
    }
    
    public void SplitPrecentage(double percent) {
        int trainSize = (int) Math.round(data.numInstances() * percent/ 100);
        int testSize = data.numInstances() - trainSize;
        
        train = new Instances(data, 0, trainSize);
        test = new Instances(data, trainSize, testSize);   
    }
    
    //****************** Save and Load Data ******************//
    
    public void saveModel(String filename) throws Exception {
        SerializationHelper.write(filename, classifier);
    }
    
    public void loadModel(String filename) throws Exception {
        classifier = (Classifier) SerializationHelper.read(filename);
    }
    
    //****************** Classify Unseen Data ******************//
    
    public void ClassifyUnseenData(String[] attributes) throws Exception {
        Instance newInstance = new Instance(data.numAttributes());
        newInstance.setDataset(data);
        for (int i = 0; i < data.numAttributes()-1; i++) {
            newInstance.setValue(i, attributes[i]);
        }
        
        double clsLabel = classifier.classifyInstance(newInstance);
        newInstance.setClassValue(clsLabel);
        
        String result = data.classAttribute().value((int) clsLabel);
        
        System.out.println("Hasil Classify Unseen Data: " + result);
    }
    
   
}
