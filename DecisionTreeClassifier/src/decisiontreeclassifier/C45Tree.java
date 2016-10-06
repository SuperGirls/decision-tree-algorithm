/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package decisiontreeclassifier;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

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
    
    public double classifyInstance (Instance instance) {
        if (instance.hasMissingValue()) {
            
        }
        if (m_Attribute == null) {
            return m_ClassValue;
        } else {
            if (m_Attribute.isNumeric()) {
                if (instance.value(m_Attribute) <= m_splitPoint)
                   return m_Successors[0].classifyInstance(instance);
                else
                    return m_Successors[1].classifyInstance(instance);
            } else {
                return m_Successors[(int) instance.value(m_Attribute)].classifyInstance(instance);
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
        Instances[] splitData = splitData(data, m_Attribute);
        if (m_Attribute.isNumeric()) {
            double max = -1;
            for (int i = 0; i < splitData[0].numInstances(); i++) {
                if (splitData[0].instance(i).value(m_Attribute) > max)
                    max = splitData[0].instance(i).value(m_Attribute);
            }
            m_splitPoint = max;
        }
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
//                    for (int k = 0; k < newData.numAttributes(); k++) {
//                        if (newData.attribute(k).equals(m_Attribute)) {
//                            newData.deleteAttributeAt(k);
//                        }
//                    }
                    m_Successors[j].makeTree(newData);
                }
            }
        }
    }
    
    public Instances[] splitData (Instances data, Attribute att) {
        Instances[] splitData;
    
        if (att.isNumeric()) {
            double splitPoint = calculateSplitPoint (data, att);
            
            splitData = new Instances[2];
            for (int j = 0; j < 2; j++) {
                splitData[j] = new Instances(data, data.numInstances());
            }

            for (int i = 0; i < data.numInstances(); i++) {
                if (data.instance(i).value(att) <= splitPoint) {
                    splitData[0].add(data.instance(i));
                } else {
                    splitData[1].add(data.instance(i));
                }
            }

            for (int i = 0; i < splitData.length; i++) {
              splitData[i].compactify();
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

    public double calculateSplitPoint (Instances data, Attribute att) {
        Instances localData = data;
        localData.sort(att);
        int idxSplit = -1;
        double currentGain = -1, gain = -1;

        for (int i = 0; i < localData.numInstances()-1; i++) {
            if (localData.instance(i).classValue() != localData.instance(i+1).classValue()) {
                Instances group1 = new Instances(localData, 0, i+1);
                Instances group2 = new Instances(localData, i+1, localData.numInstances()-i-1);
                currentGain = computeEntropy(localData) - computeEntropy(group1) - computeEntropy(group2);
                if (currentGain > gain) {
                    gain = currentGain;
                    idxSplit = i;
                }
            }
        }
        return (localData.instance(idxSplit).value(att));
    }

    public void prune (Instances data) {
        if (m_Attribute != null) {
            // Prune all subtree
            Instances[] splitData = splitData(data, m_Attribute);
            for (int i=0; i<m_Successors.length; i++) {
                m_Successors[i].prune(splitData[i]);
            }
            
            // Calculate error in this subtree
            double errorBefore = errorBeforePruning(data);
            
            // Calculate error if this is a leaf
            double errorAfter = errorAfterPruning(data);
            
            if (errorAfter < errorBefore) {
                // Make this node a leaf
                m_Successors = null;
                m_Attribute = null;
                m_Distribution = new double[data.numClasses()];

                for (int i = 0; i < data.numInstances(); i++) {
                    m_Distribution[(int) data.instance(i).classValue()]++;
                }

                Utils.normalize(m_Distribution);
                m_ClassValue = Utils.maxIndex(m_Distribution);
                m_ClassAttribute = data.classAttribute();
            }
        }
    }
    
    public double errorBeforePruning(Instances data) {
        int error = 0;
        
        // Count error
        for (int i=0; i<data.numInstances(); i++) {
            double classResult = classifyInstance(data.instance(i));
            if (data.instance(i).classValue() != classResult) {
                error++;
            }
        }
        
        return error/data.numInstances();
    }
    
    public double errorAfterPruning(Instances data) {
        // Count error
        Map<Double,Integer> classes = new HashMap<>();
        for (int i = 0; i < data.numInstances(); i++) {
            if(!classes.containsKey(data.instance(i).classValue())) {
                classes.put(data.instance(i).classValue(),1);
            }
            else {
                int temp = classes.get(data.instance(i).classValue()) + 1;
                classes.put(data.instance(i).classValue(),temp);
            }
        }
        int max = 0;
        for (int i = 0; i < classes.size(); i++) {
            if (classes.get(i) > max)
                max = classes.get(i);
        }
        
        return (data.numInstances() - max)/data.numInstances();
    }

    public String toString() {
        if ((m_Distribution == null) && (m_Successors == null)) {
            return "My C45: No model built yet.";
        }
        return "My C45\n\n" + toString(0);
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
