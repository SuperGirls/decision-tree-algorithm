/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package decisiontreeclassifier;

import java.util.ArrayList;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author TOSHIBA PC
 */
public class MyID3 extends Classifier {
    
    private MyID3[] m_Successors; // The node's successors.
    private Attribute m_Attribute; // Attribute used for splitting.
    private double m_ClassValue; // Class value if node is leaf.
    private double[] m_Distribution; // Class distribution if node is leaf.
    private Attribute m_ClassAttribute; // Class attribute of dataset.
    
    public MyID3 () {
        // do nothing
    }
    
    public void buildClassifier (Instances data) throws Exception {
        try {
            testCapability(data);
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return;
        }
        
        data = new Instances(data);
        data.deleteWithMissingClass();
        
        makeTree(data);
    }
    
    public int allExamplesClassValue (Instances data) {
        // Count how many examples have class value X, for each class X
        int[] labelNums = new int[data.numClasses()];
        for (int i = 0; i < labelNums.length; i++) {
            labelNums[i] = 0;
        }

        for (int i = 0; i < data.numInstances(); i++) {
            labelNums[(int) data.instance(i).classValue()]++;
        }
        
        // Check whether all examples have the same class
        // If yes, then return the class value
        // Otherwise, return -1
        for (int i = 0; i < labelNums.length; i++) {
            if (labelNums[i] == data.numInstances()) {
                return i;
            }
        }
        
        return -1;
    }
    
    public Attribute bestAttribute (Instances data) {
        double[] informationGain = calculateInformationGain(data);
        int bestAttributeIndex = -1;
        double highestInformationGain = 0;
        for (int i = 0; i < informationGain.length; i++) {
            if (i != data.classIndex() && informationGain[i] >= highestInformationGain) {
                bestAttributeIndex = i;
                highestInformationGain = informationGain[i];
            }
        }
        return data.attribute(bestAttributeIndex);
    }
    
    @Override
    public double classifyInstance (Instance instance) throws Exception {
        ArrayList<Attribute> usedAttributes = new ArrayList<Attribute>();
        return recursiveClassifyInstance (instance, usedAttributes);
    }
    
    public double recursiveClassifyInstance (Instance instance, ArrayList<Attribute> usedAttributes) throws Exception {
        if (instance.hasMissingValue()) {
            throw new Exception("My Id3: no missing values, " + "please.");
        }
        
        if (m_Attribute == null) {
            return m_ClassValue;
        } else {
                System.out.println(m_Attribute);
                System.out.println(instance);
                System.out.println(m_Attribute.index() + " " + instance.value(m_Attribute));
            Instance simplifiedInstance = new Instance(instance);
            for (int i = usedAttributes.size()-1; i >= 0; i--) {
                simplifiedInstance.deleteAttributeAt(usedAttributes.get(i).index());
            }
            usedAttributes.add(m_Attribute);
            return m_Successors[(int) simplifiedInstance.value(m_Attribute)].recursiveClassifyInstance(instance, usedAttributes);
        }
    }

    public double computeProbability (Instances data, int classValue) {
        int numInstancesWithCorrespondingClass = 0;
        for (int i = 0; i < data.numInstances(); i++) {
            if (data.instance(i).classValue()== classValue) {
                numInstancesWithCorrespondingClass++;
            }
        }
        return (double) numInstancesWithCorrespondingClass / data.numInstances();
    }
    
    public double computeProbability (Instances data, int classValue, Attribute att, int attributeValueIndex) {
        Instances[] splitData = splitData(data, att);
        if (splitData[attributeValueIndex].numInstances() == 0) {
            return 0;
        }
        
        int numInstancesWithCorrespondingClass = 0;
        for (int i = 0; i < splitData[attributeValueIndex].numInstances(); i++) {
            if (splitData[attributeValueIndex].instance(i).classValue() == classValue) {
                numInstancesWithCorrespondingClass++;
            }
        }
        
        return (double) numInstancesWithCorrespondingClass / splitData[attributeValueIndex].numInstances();
    }
    
    public double computeEntropy (Instances data) {
        double entropy = 0;
        for (int i = 0; i < data.numClasses(); i++) {
            double probability = computeProbability(data, i);
            
            entropy += -1 * probability * ((probability > 0) ? Utils.log2(probability) : 0 );
        }
        return entropy;
    }
    
    public double computeEntropy (Instances data, Attribute att, int attributeValueIndex) {
        double entropy = 0;
        for (int i = 0; i < data.numClasses(); i++) {
            double probability = computeProbability(data, i, att, attributeValueIndex);
            entropy += -1 * probability * ((probability > 0) ? Utils.log2(probability) : 0 );
        }
        return entropy;
    }
    
    public double[] calculateInformationGain (Instances data) {
        double entropy = computeEntropy(data);
        
        double[] informationGain = new double[data.numAttributes()];
        for (int i = 0; i < data.numAttributes(); i++) {
            informationGain[i] = entropy;
            Instances[] splitData = splitData(data, data.attribute(i));
            
            for (int j = 0; j < splitData.length; j++) {
                informationGain[i] -= (double) splitData[j].numInstances() / data.numInstances() * computeEntropy(data, data.attribute(i), j);
            }

            
        }
        return informationGain;
    }
    
    public void makeTree (Instances data) {
        // If number of predicting attributes is empty, then Return the single node tree Root,
        // with label = most common value of the target attribute in the examples.
        if (data.numAttributes() <= 1) {
            // TODO: return the single node tree Root
            m_Attribute = null;
            m_Distribution = new double[data.numClasses()];
            
            for (int i = 0; i < data.numInstances(); i++) {
                m_Distribution[(int) data.instance(i).classValue()]++;
            }
            
            Utils.normalize(m_Distribution);
            m_ClassValue = Utils.maxIndex(m_Distribution);
            m_ClassAttribute = data.classAttribute();
            return;
        }
                
        // Else if all examples' classes are classValue, Return the single-node tree Root, with label = classValue.
        int classValue = allExamplesClassValue(data);
        if (classValue > -1) {
            m_Attribute = null;
            m_Distribution = new double[data.numClasses()];
            
            for (int i = 0; i < data.numInstances(); i++) {
                m_Distribution[(int) data.instance(i).classValue()]++;
            }
            
            Utils.normalize(m_Distribution);
            m_ClassValue = Utils.maxIndex(m_Distribution);
            m_ClassAttribute = data.classAttribute();
            return;
        }
        
        // Otherwise
        m_Attribute = bestAttribute(data);
        Instances[] splitData = splitData(data, m_Attribute);
        for (int i = 0; i < splitData.length; i++) {
            if (splitData[i].numInstances() == 0) {
                // Below this new branch add a leaf node with label = most common target value in the examples
                m_Attribute = null;
                m_Distribution = new double[data.numClasses()];

                for (int j = 0; j < data.numInstances(); j++) {
                    m_Distribution[(int) data.instance(j).classValue()]++;
                }

                Utils.normalize(m_Distribution);
                m_ClassValue = Utils.maxIndex(m_Distribution);
                m_ClassAttribute = data.classAttribute();
                return;
                
            } else {
                // create successors
                m_Successors = new MyID3[m_Attribute.numValues()];
                for (int j = 0; j < m_Attribute.numValues(); j++) {
                    m_Successors[j] = new MyID3();
                    Instances newData = new Instances (splitData[j]);                   
                    m_Successors[j].makeTree(newData);
                }
            }
        }
            
    }
    
    private Instances[] splitData (Instances data, Attribute att) {
        Instances[] splitData = new Instances[att.numValues()];
        for (int j = 0; j < att.numValues(); j++) {
            splitData[j] = new Instances(data, data.numInstances());
        }
        
        for (int i = 0; i < data.numInstances(); i++) {
            splitData[(int) data.instance(i).value(att)].add(data.instance(i));
        }

        for (int i = 0; i < splitData.length; i++) {
          splitData[i].compactify();
        }
        
        return splitData;
    }
    
    public void testCapability (Instances data) throws Exception {
        testMissingValue(data);
        testNumericalAttribute(data);
    }
    
    public void testMissingValue (Instances data) throws Exception{

        boolean satisfy = true;
        int i = 0;
        while (i < data.numInstances() && satisfy) {
            int j = 0;
            while (j < data.numAttributes() && satisfy) {
                satisfy = !data.instance(i).hasMissingValue();
                j++;
            }
            i++;   
        }
        if (!satisfy) {
            throw new Exception("My Id3: no missing values, " + "please.");
        }
    }
    
    public void testNumericalAttribute (Instances data) throws Exception {
        boolean satisfy = true;
        int i = 0;
        while (i < data.numInstances() && satisfy) {
            int j = 0;
            while (j < data.numAttributes() && satisfy) {
                satisfy = data.instance(i).attribute(j).isNominal();
                j++;
            }
            i++;   
        }
        if (!satisfy) {
            throw new Exception("My Id3: no numerical attribute, " + "please.");
        }
    }
    
    public String toString() {
        if ((m_Distribution == null) && (m_Successors == null)) {
            return "My ID3: No model built yet.";
        }
        return "My ID3\n\n" + toString(0);
    }
    
    private String toString(int level) {
        StringBuffer text = new StringBuffer();

        if (m_Attribute == null) {
            if (Instance.isMissingValue(m_ClassValue)) {
                text.append(": null");
            } else {
                text.append(": " + m_ClassAttribute.value((int) m_ClassValue));
            } 
        } else {
            for (int j = 0; j < m_Attribute.numValues(); j++) {
                text.append("\n");
                for (int i = 0; i < level; i++) {
                    text.append("|  ");
                }
                text.append(m_Attribute.name() + " = " + m_Attribute.value(j));
                text.append(m_Successors[j].toString(level + 1));
            }
        }
        return text.toString();
    }
    
}
