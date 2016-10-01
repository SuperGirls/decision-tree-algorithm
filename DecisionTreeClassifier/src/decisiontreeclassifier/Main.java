/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package decisiontreeclassifier;

import java.io.IOException;
import java.util.Scanner;
/**
 *
 * @author TOSHIBA PC
 */
public class Main {
    public static void main(String[] args) throws IOException, Exception {
        Scanner input = new Scanner(System.in);
        MyWeka myWeka = new MyWeka();
        int option;
        
        System.out.println("Welcome To My Weka\n");
        
        myWeka.printMainMenu();     
        System.out.print("Pilihan opsi: ");
        option = input.nextInt();
        input.nextLine();
        
        while(option != 0) {
            
            if(option == 1) {
                myWeka.inputDataTrain();
            } else if(option == 2) {
                myWeka.filtering();
            } else if(option == 3) {
                myWeka.chooseClassifier();
            } else if(option == 4) {
                myWeka.chooseTestOption();
            } else if(option == 5) {
                myWeka.startClassify();
            } else if(option == 6) {
                myWeka.startClassifyUnseen();
            } else if(option == 7) {
                myWeka.printDataSummary();
            } else if(option == 8) {
                System.out.print("Masukkan Nama File : ");
                String filename = input.nextLine();
                myWeka.saveModel(filename);
            } else if(option == 9) {
                System.out.print("Masukkan Nama File : ");
                String filename = input.nextLine();
                myWeka.loadModel(filename);
            }
            
            myWeka.printMainMenu();     
            System.out.print("Pilihan opsi: ");
            option = input.nextInt();
            input.nextLine();
        }
    }
}
