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
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.Id3;


/**
 *
 * @author TOSHIBA PC
 */
public class MyWeka {
    
    private Instances data;
    private Id3 treeId3;
    private J48 treeJ48;
    private NaiveBayes model;
    
    public MyWeka(){
       treeId3 = new Id3(); 
       treeJ48 = new J48();
       model = new NaiveBayes();
    }
    
    public void ReadFileArff(String filename) throws IOException{
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        data = new Instances(reader);
        reader.close();
        
        // setting class attribute
        data.setClassIndex(data.numAttributes() - 1);
    }
    
    public void ReadFileCsv(String filename) throws Exception{
        DataSource source = new DataSource(filename);
        Instances data = source.getDataSet();
        
        // setting class attribute
        if (data.classIndex() == -1)
          data.setClassIndex(data.numAttributes() - 1);
    }
    
    public void RemoveAttribute(String attribute) throws Exception {
        String[] options = new String[2];
        options[0] = "-R";                                    
        options[1] = attribute;                                    
        Remove remove = new Remove();                         
        remove.setOptions(options);                           
        remove.setInputFormat(data);                          
        data = Filter.useFilter(data, remove);   
    }
    
    public void Resample(String bias, String percentage) throws Exception{
        String[] options = new String[4];
        options[0] = "-B";
        options[1] = bias;
        options[2] = "-Z";
        options[3] = percentage;
        Resample sample = new Resample();
        sample.setOptions(options);
        sample.setInputFormat(data);
        data = Filter.useFilter(data, sample);     
    }
    
    public void BuildClassifierID3() throws Exception{
        treeId3.buildClassifier(data);
    }
    
    public void BuildClassifierJ48(String confidence) throws Exception {
        //Assume always prune
        
        String[] options = new String[2];
        options[0] = "-C";
        options[1] = confidence;
        
               // new instance of tree
        treeJ48.setOptions(options);     // set the options
        treeJ48.buildClassifier(data);   // build classifier
    }
    
    public void BuildClassifierNaiveBayes() throws Exception{
        model.buildClassifier(data);
    }
    
    
}
