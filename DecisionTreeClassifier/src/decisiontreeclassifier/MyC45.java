/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package decisiontreeclassifier;

import weka.classifiers.Classifier;
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
        root = new C45Tree();
        root.makeTree(data);
//        root.prune();
    }

}
