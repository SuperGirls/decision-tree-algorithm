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
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.Id3;
import weka.core.Attribute;
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
    
    
    private int optCls = 1;
    private int optTest = 1;
    private int folds = 0;
    private double percent;
    private String confidence;
    private String testFilename;
            
    public MyWeka(){
    }
    
    //****************** File Arff dan CSV Reader ******************//
    
    public void readFileArff(String filename) throws IOException{
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        data = new Instances(reader);
        reader.close();
        
        // setting class attribute
        data.setClassIndex(data.numAttributes() - 1);
    }
    
    public void readFileCsv(String filename) throws Exception{
        DataSource source = new DataSource(filename);
        data = source.getDataSet();
        
        // setting class attribute
        if (data.classIndex() == -1)
          data.setClassIndex(data.numAttributes() - 1);
    }
    
    //****************** Filtering ******************//
    
    public void removeAttribute(String attribute) throws Exception {
        String[] options = new String[2];
        options[0] = "-R";                                    
        options[1] = attribute;   
        
        Remove remove = new Remove();                         
        remove.setOptions(options);                           
        remove.setInputFormat(data);                          
        data = Filter.useFilter(data, remove);   
    }
    
    public void resample(String bias, String percent) throws Exception{
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
    
    public void buildClassifierID3() throws Exception{
        classifier = new Id3();
        classifier.buildClassifier(train);
        //System.out.println("This is the ID3 classifier");
        //System.out.println(classifier);
    }
    
    public void buildClassifierJ48(String confidence) throws Exception {
        //Assume always prune
        
        String[] options = new String[2];
        options[0] = "-C";
        options[1] = confidence;
        
        // new instance of tree
        classifier = new J48();
        classifier.setOptions(options);     
        classifier.buildClassifier(train);   
    }
    
    public void buildClassifierNaiveBayes() throws Exception{
        classifier = new NaiveBayes();
        classifier.buildClassifier(train);
    }
    
    public void buildClassifierMyID3() throws Exception{
        classifier = new MyID3();
        classifier.buildClassifier(train);
        //System.out.println("This is the My ID3 classifier");
        //System.out.println(classifier);
    }
    
    public void evaluateModel() throws Exception{
        eval = new Evaluation(train);
        eval.evaluateModel(classifier, test); 
    }
    
    //****************** Test Option ******************//
    
    public void fullTraining() {
        train = data;
        test = data;
    }
    
    public void setTestCase(String filename) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        test = new Instances(reader);
        reader.close();
        
        // setting class attribute
        test.setClassIndex(data.numAttributes() - 1);
    }
    
    public void crossValidate(int folds) throws Exception {
        eval = new Evaluation(data);
        eval.crossValidateModel(classifier, data, folds, new Random(1));     
    }
    
    public void splitPercentage(double percent) {
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
        optCls = 0;
        optTest = 0;
    }
    
    //****************** Classify Unseen Data ******************//
    
    public void classifyUnseenData(String[] attributes) throws Exception {
        Instance newInstance = new Instance(data.numAttributes());
        newInstance.setDataset(data);
        for (int i = 0; i < data.numAttributes()-1; i++) {
            if(Attribute.NUMERIC == data.attribute(i).type()){
                Double value = Double.valueOf(attributes[i]);
                newInstance.setValue(i, value);
            } else {
                newInstance.setValue(i, attributes[i]);
            }
        }
        
        double clsLabel = classifier.classifyInstance(newInstance);
        newInstance.setClassValue(clsLabel);
        
        String result = data.classAttribute().value((int) clsLabel);
        
        System.out.println("Hasil Classify Unseen Data: " + result);
    }
    
    public void printDataSummary() {
        System.out.println("\nSummary\n======\n");
        System.out.println(data.toSummaryString());
    }
    
    public void printTrainingData () {
        for (int i=0; i < data.numInstances(); i++) {
            System.out.print(i);
            System.out.print(": ");
            System.out.println(data.instance(i));
        }
    }
    
    //****************** Menuuu ******************//
    
    public void printMainMenu(){
        System.out.println("\nMenu");
        System.out.println("0. Keluar");
        System.out.println("1. Masukkan Data Train");
        System.out.println("2. Filtering");
        System.out.println("3. Pilih Classifier");
        System.out.println("4. Pilih Test Option");
        System.out.println("5. Start Classifying");
        System.out.println("6. Classify Unseen Data");
        System.out.println("7. Print Data Summary");
        System.out.println("8. Save Model");
        System.out.println("9. Load Model");
        System.out.println("10. Show Training Data");
    }
    
    public void inputDataTrain() throws IOException, Exception{
        Scanner input = new Scanner(System.in);
        
        System.out.print("\nMasukkan data train: ");
        String filename = input.nextLine();
        
        if(filename.endsWith("arff")) {
            readFileArff(filename);
        } else if(filename.endsWith("csv")){
            readFileCsv(filename);
        }
        
    }
    
    public void filtering() throws Exception{
        Scanner input = new Scanner(System.in);

        System.out.println("\nFiltering:");
        System.out.println("1. Remove Attribute");
        System.out.println("2. Resample");
        System.out.print("Pilihan opsi: ");
        
        int option = input.nextInt();
        input.nextLine();
        
        if(option == 1) {
            System.out.print("Masukkan Index Atributte: ");
            String attribute = input.nextLine();
            removeAttribute(attribute);
        } else if(option == 2) {
            System.out.print("Masukkan Bias: ");
            String bias = input.nextLine();
            System.out.print("Masukkan Persentase: ");
            String percent = input.nextLine();
            resample(bias, percent);
        }
    }
    
    public void chooseClassifier(){
        Scanner input = new Scanner(System.in);
        
        System.out.println("\nClassifier:");
        System.out.println("1. ID3");
        System.out.println("2. C4.5");
        System.out.println("3. Naive Bayes");
        System.out.println("4. My ID3");
        System.out.print("Pilihan opsi: ");
        
        int option = input.nextInt();
        input.nextLine();
        
        switch (option) {
            case 1: 
                optCls = 1;
                break;
            case 2: 
                System.out.print("Masukkan nilai confidence: ");
                confidence = input.nextLine();
                optCls = 2;
                break;
            case 3: 
                optCls = 3;
                break;
            case 4: 
                optCls = 4;
                break;
        }

    }
    
    public void chooseTestOption(){
        Scanner input = new Scanner(System.in);
        
        System.out.println("\nTest Options:");
        System.out.println("1. Full Training");
        System.out.println("2. Cross Validation");
        System.out.println("3. Presentage Split");
        System.out.println("4. Supplied Test Case");
        
        System.out.print("Pilihan opsi: ");
        int option = input.nextInt();
        input.nextLine();
        
        if(option == 1) {
            optTest = 1;
        } else if(option == 2) {
            System.out.print("Masukkan nilai fold: ");
            folds = input.nextInt();
            input.nextLine();
            optTest = 2;
        } else if (option == 3) {
            System.out.print("Masukkan nilai persen train data: ");
            percent = input.nextDouble();
            input.nextLine();
            optTest = 3;
        } else if (option == 4) {
            System.out.print("Masukkan nilai persen train data: ");
            percent = input.nextDouble();
            input.nextLine();
            optTest = 4;
        }   
    }
    
    public void startClassify() throws Exception{
        if(optTest == 1) {
            fullTraining();
        } else if(optTest == 2) {
            train = data;
        } else if(optTest == 3) {
            splitPercentage(percent);
        } else if(optTest == 4) {
            setTestCase(testFilename);
            train = data;
        }

        if(optCls == 1) {
            buildClassifierID3();    
        } else if(optCls == 2) {
            buildClassifierJ48(confidence);
        } else if(optCls == 3) {
            buildClassifierNaiveBayes();
        } else if(optCls == 4) {
            buildClassifierMyID3();
        }
        
        if(optTest == 2){
            crossValidate(folds);
        } else {
            evaluateModel();
        }  
        
        //Print Result
        System.out.println(eval.toSummaryString("\nSummary\n======\n", false));   
        System.out.println(eval.toClassDetailsString("\nStatistic\n======\n"));
        System.out.println(eval.toMatrixString("\nConfusion Matrix\n======\n"));
        
    }
    
    public void startClassifyUnseen() throws Exception{
        Scanner input = new Scanner(System.in);
        
        System.out.print("\nMasukkan nilai atribut (pisahkan dengan spasi): ");
        String in = input.nextLine();
        
        if(optTest == 1) {
            fullTraining();
        } else if(optTest == 2) {
            train = data;
        } else if(optTest == 3) {
            splitPercentage(percent);
        } else if(optTest == 4) {
            setTestCase(testFilename);
            train = data;
        }

        if(optCls == 1) {
            buildClassifierID3();    
        } else if(optCls == 2) {
            buildClassifierJ48(confidence);
        } else if(optCls == 3) {
            buildClassifierNaiveBayes();
        } else if(optCls == 4) {
            buildClassifierMyID3();
        }
        
        String[] attributes = in.split(" ");
        classifyUnseenData(attributes);  
    }
    
    
        
}
