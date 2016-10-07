/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package decisiontreeclassifier;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author TOSHIBA PC
 */
public class MyC45 extends Classifier {
    
    protected C45Tree root;
    
    public MyC45() {
        
    }
    
    @Override
    public void buildClassifier(Instances data) {
        // Split Data
        Instances train = new Instances(data, data.numInstances());
        Instances test = new Instances(data, data.numInstances());
        for(int i=0; i<data.numInstances(); i++) {
            if (i % 4 == 0)
                test.add(data.instance(i));
            else
                train.add(data.instance(i));
        }
        test.compactify();
        train.compactify();
        
        root = new C45Tree();
        root.makeTree(data);
        root.prune(data);
        System.out.println(root);
    }
    
    public double classifyInstance (Instance instance) {
        return root.classifyInstance(instance);
    }
    
    public String toString() {
        return root.toString();
    }

}
