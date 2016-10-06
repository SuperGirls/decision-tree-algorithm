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
public class C45Tree {
    
    private C45Tree[] m_Successors; // The node's successors.
    private Attribute m_Attribute; // Attribute used for splitting.
    private double m_ClassValue; // Class value if node is leaf.
    private double[] m_Distribution; // Class distribution if node is leaf.
    private Attribute m_ClassAttribute; // Class attribute of dataset.
    private double m_splitPoint; // Split point if m_Attribute is numeric
    
    public C45Tree () {
        // do nothing
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
    
    public double classifyInstance (Instance instance) throws Exception {
        ArrayList<Attribute> usedAttributes = new ArrayList<Attribute>();
        return recursiveClassifyInstance (instance, usedAttributes);
    }
    
    public double recursiveClassifyInstance (Instance instance, ArrayList<Attribute> usedAttributes) {
        if (instance.hasMissingValue()) {
            
        }
        
        if (m_Attribute == null) {
            return m_ClassValue;
        } else {
            Instance simplifiedInstance = new Instance(instance);
            for (int i = usedAttributes.size()-1; i >= 0; i--) {
                simplifiedInstance.deleteAttributeAt(usedAttributes.get(i).index());
            }
            usedAttributes.add(m_Attribute);
            if (m_Attribute.isNumeric()) {
                if (simplifiedInstance.value(m_Attribute) <= m_splitPoint)
                   return m_Successors[0].recursiveClassifyInstance(instance, usedAttributes);
                else
                    return m_Successors[1].recursiveClassifyInstance(instance, usedAttributes);
            } else {
                return m_Successors[(int) simplifiedInstance.value(m_Attribute)].recursiveClassifyInstance(instance, usedAttributes);
            }
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
        m_splitPoint = calculateSplitPoint(data, m_Attribute);
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
                int numSuccessors;
                if (m_Attribute.isNumeric())
                    numSuccessors = 2;
                else
                    numSuccessors = m_Attribute.numValues();
                m_Successors = new C45Tree[numSuccessors];
                
                for (int j = 0; j < numSuccessors; j++) {
                    m_Successors[j] = new C45Tree();
                    Instances newData = new Instances (splitData[j]);
                    for (int k = 0; k < newData.numAttributes(); k++) {
                        if (newData.attribute(k).equals(m_Attribute)) {
                            newData.deleteAttributeAt(k);
                        }
                    }
                    m_Successors[j].makeTree(newData);
                }
            }
        }
    }
    
    private Instances[] splitData (Instances data, Attribute att) {
        Instances[] splitData;
    
        if (att.isNumeric()) {
            double splitPoint = calculateSplitPoint (data, att);
            
            splitData = new Instances[2];
            splitData[0] = new Instances(data);
            splitData[1] = new Instances(data);
            splitData[0].delete();
            splitData[1].delete();
        
            for (int i = 0; i < data.numInstances()-1; i++) {
                if (data.instance(i).classValue() <= splitPoint) {
                    splitData[0].add(data.instance(i));
                } else {
                    splitData[1].add(data.instance(i));
                }
            }
        } else {
            splitData = new Instances[att.numValues()];
            for (int j = 0; j < att.numValues(); j++) {
                splitData[j] = new Instances(data, data.numInstances());
            }

            for (int i = 0; i < data.numInstances(); i++) {
                splitData[(int) data.instance(i).value(att)].add(data.instance(i));
            }

            for (int i = 0; i < splitData.length; i++) {
              splitData[i].compactify();
            }
        }
        
        return splitData;
    }

    private double calculateSplitPoint (Instances data, Attribute att) {
        Instances localData = data;
        localData.sort(att);
        int idxSplit = -1;
        double currentGain = -1, gain = -1;

        for (int i = 0; i < localData.numInstances()-1; i++) {
            if (localData.instance(i).classValue() != localData.instance(i+1).classValue()) {
                Instances group1 = new Instances(localData, 0, i+1);
                Instances group2 = new Instances(localData, i+1, localData.numInstances()-i-1);
                currentGain = computeEntropy(localData) - computeEntropy(group1) - computeEntropy(group2);
            }
            if (currentGain > gain) {
                gain = currentGain;
                idxSplit = i;
            }
        }
        
        return Double.parseDouble(localData.instance(idxSplit).stringValue(att));
    }
}
